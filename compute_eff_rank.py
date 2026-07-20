"""
compute_eff_rank.py  --  CoopCalib-TP
Computes effective rank of the latent space z for all 8 SocialVAE checkpoints.
Checkpoints are saved as 'ckpt-best' (no extension) by SocialVAE's trainer.

Usage (Windows CMD, from C:\CoopCalib):
    C:\CoopCalib\venv\Scripts\activate
    python compute_eff_rank.py

Outputs:
    experiments/results/socialvae/eff_rank_summary.json
    experiments/results/socialvae/eff_rank_table.txt
"""

import os
import sys
import json
import numpy as np

BASE    = r"C:\CoopCalib"
VAE_DIR = os.path.join(BASE, "models", "socialvae")
EXP_DIR = os.path.join(BASE, "experiments", "results", "socialvae")
OUT_JSON = os.path.join(EXP_DIR, "eff_rank_summary.json")
OUT_TXT  = os.path.join(EXP_DIR, "eff_rank_table.txt")

sys.path.insert(0, VAE_DIR)

import torch

# ---------------------------------------------------------------------------
# Effective rank  (Roy & Vetterli 2007)
# ---------------------------------------------------------------------------
def effective_rank(z_samples):
    """
    Entropy of the normalised eigenvalue spectrum of the covariance matrix.
    Returns scalar in [1, latent_dim]. Higher = richer latent space.
    """
    z = z_samples - z_samples.mean(axis=0, keepdims=True)
    cov = np.cov(z.T)
    if cov.ndim == 0:
        return 1.0
    evals = np.linalg.eigvalsh(cov)
    evals = evals[evals > 1e-10]
    if len(evals) == 0:
        return 1.0
    p = evals / evals.sum()
    return round(float(np.exp(-np.sum(p * np.log(p + 1e-12)))), 4)


# ---------------------------------------------------------------------------
# Run registry
# ---------------------------------------------------------------------------
RUNS = [
    {"name": "eth_500_v0",  "lam3": 0.0, "n_train": 500,  "ade": 0.72, "fde": 1.21},
    {"name": "eth_1000_v0", "lam3": 0.0, "n_train": 1000, "ade": 0.70, "fde": 1.22},
    {"name": "eth_1500_v0", "lam3": 0.0, "n_train": 1500, "ade": 0.74, "fde": 1.37},
    {"name": "eth_full_v0", "lam3": 0.0, "n_train": 1845, "ade": 0.73, "fde": 1.33},
    {"name": "eth_500_v3",  "lam3": 0.1, "n_train": 500,  "ade": 0.74, "fde": 1.23},
    {"name": "eth_1000_v3", "lam3": 0.1, "n_train": 1000, "ade": 0.73, "fde": 1.31},
    {"name": "eth_1500_v3", "lam3": 0.1, "n_train": 1500, "ade": 0.74, "fde": 1.42},
    {"name": "eth_full_v3", "lam3": 0.1, "n_train": 1845, "ade": 0.72, "fde": 1.33},
]

NAME_MAP = {500: "eth_500", 1000: "eth_1000", 1500: "eth_1500", 1845: "eth_full"}


# ---------------------------------------------------------------------------
# Load checkpoint  (SocialVAE saves as 'ckpt-best', no extension)
# ---------------------------------------------------------------------------
def load_checkpoint(run_name):
    """
    Loads the SocialVAE checkpoint for a given run.
    SocialVAE saves extensionless files named 'ckpt-best' and 'ckpt-last'.
    Returns the loaded object, or None if not found.
    """
    ckpt_dir = os.path.join(EXP_DIR, run_name)

    # Priority order: ckpt-best first, then ckpt-last as fallback
    candidates = [
        os.path.join(ckpt_dir, "ckpt-best"),
        os.path.join(ckpt_dir, "ckpt-last"),
        # Also try standard extensions in case trainer was changed
        os.path.join(ckpt_dir, "best.pth"),
        os.path.join(ckpt_dir, "best.pt"),
        os.path.join(ckpt_dir, "checkpoint.pth"),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for ckpt_path in candidates:
        if not os.path.exists(ckpt_path):
            continue
        size_kb = os.path.getsize(ckpt_path) // 1024
        print("  Loading: " + ckpt_path + "  (" + str(size_kb) + " KB)")
        try:
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            print("  Loaded successfully. Type: " + str(type(state)))
            return state
        except Exception as e:
            print("  [ERROR] Could not load " + ckpt_path + ": " + str(e))
            continue

    print("  [WARN] No checkpoint found in: " + ckpt_dir)
    return None


# ---------------------------------------------------------------------------
# Extract z samples from checkpoint
# ---------------------------------------------------------------------------
def extract_z(state, n_samples=2000):
    """
    Extracts or approximates latent z samples from a SocialVAE checkpoint.

    SocialVAE (social_vae.py) uses a CVAE architecture. The checkpoint
    is typically a dict with model weights. We extract the encoder's
    mu projection weights to compute effective rank of the latent space.

    Strategy:
    1. If checkpoint contains saved z tensors directly -> use them
    2. If state_dict available -> find mu/logvar layers, use weight rows as z proxy
    3. Fallback -> use all encoder FC weight rows
    """
    # Unwrap if needed
    if isinstance(state, dict):
        # Check for directly saved z
        for key in ["z_mu", "z_mean", "mu", "latent_mu", "z"]:
            if key in state and isinstance(state[key], torch.Tensor):
                z = state[key].cpu().float().numpy()
                z = z.reshape(-1, z.shape[-1])
                print("  Found saved z tensor at key '" + key + "': " + str(z.shape))
                return z

        # Unwrap state dict
        sd = state.get("model_state_dict",
             state.get("state_dict",
             state.get("model", state)))
    else:
        sd = state

    if not isinstance(sd, dict):
        print("  [WARN] Cannot parse checkpoint structure.")
        return None

    # Print all tensor keys and shapes for debugging
    tensor_keys = [(k, tuple(v.shape)) for k, v in sd.items()
                   if isinstance(v, torch.Tensor)]
    print("  Checkpoint has " + str(len(tensor_keys)) + " tensor layers.")
    print("  First 15 layers:")
    for k, s in tensor_keys[:15]:
        print("    " + k + " -> " + str(s))

    # --- Strategy 1: find mu projection layer (best proxy for latent space) ---
    mu_layer = None
    latent_dim = None

    for k, v in sd.items():
        if not isinstance(v, torch.Tensor) or v.ndim != 2:
            continue
        kl = k.lower()
        if any(tag in kl for tag in ["mu", "mean", "z_proj", "latent"]):
            mu_layer = v.cpu().float().numpy()
            latent_dim = mu_layer.shape[0]
            print("  Using mu layer: '" + k + "'  shape=" + str(mu_layer.shape))
            break

    # --- Strategy 2: any encoder FC layer ---
    if mu_layer is None:
        for k, v in sd.items():
            if not isinstance(v, torch.Tensor) or v.ndim != 2:
                continue
            kl = k.lower()
            if "enc" in kl or "encoder" in kl:
                mu_layer = v.cpu().float().numpy()
                latent_dim = mu_layer.shape[0]
                print("  Using encoder layer: '" + k + "'  shape=" + str(mu_layer.shape))
                break

    # --- Strategy 3: any FC weight with plausible latent size ---
    if mu_layer is None:
        for k, v in sd.items():
            if not isinstance(v, torch.Tensor) or v.ndim != 2:
                continue
            if 8 <= v.shape[0] <= 128:   # plausible latent dim range
                mu_layer = v.cpu().float().numpy()
                latent_dim = mu_layer.shape[0]
                print("  Using fallback layer: '" + k + "'  shape=" + str(mu_layer.shape))
                break

    if mu_layer is None:
        print("  [WARN] No suitable weight layer found.")
        return None

    # Build z samples from weight rows
    # Each row = one direction in weight space -> proxy for latent directions
    rng = np.random.default_rng(42)
    n_rows = mu_layer.shape[0]

    if n_rows >= n_samples:
        idx = rng.choice(n_rows, n_samples, replace=False)
        z = mu_layer[idx]
    else:
        # Tile with small noise to reach n_samples
        repeats = (n_samples // n_rows) + 1
        z = np.tile(mu_layer, (repeats, 1))[:n_samples]
        z += rng.normal(0, 1e-3, z.shape)

    # Standardise columns so effective rank reflects structure not scale
    col_std = z.std(axis=0)
    col_std[col_std < 1e-8] = 1.0
    z = z / col_std

    print("  z proxy shape: " + str(z.shape))
    return z


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("CoopCalib-TP  --  Effective Rank Analysis")
    print("=" * 60)

    # Confirm the 8 checkpoint directories exist
    print("\nChecking checkpoint directories under: " + EXP_DIR)
    for run in RUNS:
        d = os.path.join(EXP_DIR, run["name"])
        best = os.path.join(d, "ckpt-best")
        exists = "OK  -- ckpt-best found" if os.path.exists(best) else "MISSING"
        print("  " + run["name"] + ": " + exists)

    os.makedirs(EXP_DIR, exist_ok=True)

    results = []
    failed  = []

    for run in RUNS:
        name = run["name"]
        print("\n" + "-" * 50)
        print("[" + name + "]  lam3=" + str(run["lam3"]) + "  N=" + str(run["n_train"]))

        state = load_checkpoint(name)

        if state is None:
            failed.append(name)
            results.append({
                "name": name, "lam3": run["lam3"], "n_train": run["n_train"],
                "ade": run["ade"], "fde": run["fde"],
                "eff_rank": None, "latent_dim": None, "status": "missing"
            })
            continue

        z = extract_z(state)

        if z is None:
            failed.append(name)
            results.append({
                "name": name, "lam3": run["lam3"], "n_train": run["n_train"],
                "ade": run["ade"], "fde": run["fde"],
                "eff_rank": None, "latent_dim": None, "status": "extract_failed"
            })
            continue

        eff_r = effective_rank(z)
        print("  Effective rank: " + str(eff_r) + "  (latent_dim=" + str(z.shape[1]) + ")")

        results.append({
            "name": name, "lam3": run["lam3"], "n_train": run["n_train"],
            "ade": run["ade"], "fde": run["fde"],
            "eff_rank": eff_r, "latent_dim": z.shape[1],
            "z_shape": list(z.shape), "status": "ok"
        })

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------
    summary = {
        "description": "Effective rank of SocialVAE latent space per run",
        "method": "Roy & Vetterli (2007) -- entropy of eigenvalue spectrum",
        "proxy": "encoder mu-layer weight rows used as z proxy",
        "runs": results,
        "failed_runs": failed
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved: " + OUT_JSON)

    # -----------------------------------------------------------------------
    # Print ASCII table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EFFECTIVE RANK RESULTS")
    print("=" * 60)
    header = "{:<20} {:>5} {:>6} {:>9} {:>6} {:>6}".format(
        "Run", "lam3", "N", "EffRank", "ADE", "FDE")
    sep = "-" * 60
    print(header)
    print(sep)

    lines = ["EFFECTIVE RANK SUMMARY", "=" * 60, header, sep]

    for r in results:
        er_str = "{:.4f}".format(r["eff_rank"]) if r["eff_rank"] is not None else "FAIL"
        row = "{:<20} {:>5} {:>6} {:>9} {:>6} {:>6}".format(
            r["name"], r["lam3"], r["n_train"], er_str, r["ade"], r["fde"])
        print(row)
        lines.append(row)

    print("\nV0 vs V3 -- Effective Rank Delta (positive = V3 richer latent space):")
    lines.append("\nV0 vs V3 Delta:")

    for n in [500, 1000, 1500, 1845]:
        v0 = next((r for r in results if r["name"] == NAME_MAP[n] + "_v0"), None)
        v3 = next((r for r in results if r["name"] == NAME_MAP[n] + "_v3"), None)
        if v0 and v3 and v0["eff_rank"] is not None and v3["eff_rank"] is not None:
            delta = v3["eff_rank"] - v0["eff_rank"]
            sign  = "+" if delta >= 0 else ""
            icon  = "V3 BETTER" if delta > 0 else ("SAME" if delta == 0 else "V0 better")
            row = "  N={:<5}  V0={:.4f}  V3={:.4f}  Delta={}{}  ({})".format(
                n, v0["eff_rank"], v3["eff_rank"], sign, round(delta, 4), icon)
        else:
            row = "  N={:<5}  SKIP (one or both failed)".format(n)
        print(row)
        lines.append(row)

    with open(OUT_TXT, "w", encoding="ascii", errors="replace") as f:
        f.write("\n".join(lines))
    print("\nSaved: " + OUT_TXT)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    n_ok = len([r for r in results if r["eff_rank"] is not None])
    print("\n" + "=" * 60)
    print("Done: " + str(n_ok) + "/" + str(len(RUNS)) + " runs computed successfully.")
    if failed:
        print("Failed: " + str(failed))
    print("\nNext step: run wilcoxon_tests.py")


if __name__ == "__main__":
    main()
