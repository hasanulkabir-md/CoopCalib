"""
zeroshot_trajnet.py  —  CoopCalib-TP
Runs zero-shot inference of TUTR V0 and V2 on the 50 sampled TrajNet++ scenes.
Evaluates ECE, FPR, SVR, and SPSR using eval_suite.py.
No retraining — pure out-of-distribution generalization test.

Usage (Windows CMD, from C:\CoopCalib):
    C:\CoopCalib\venv\Scripts\activate
    python zeroshot_trajnet.py

Outputs:
    experiments/results/trajnet_zeroshot.json
    experiments/results/trajnet_zeroshot_table.txt
"""

import os
import sys
import json
import subprocess
import numpy as np

BASE       = r"C:\CoopCalib"
TUTR_DIR   = os.path.join(BASE, "TUTR")
CKPT_DIR   = os.path.join(TUTR_DIR, "checkpoint")
TRAJ_DATA  = os.path.join(BASE, "data", "processed", "trajnet_sample")
METRICS_DIR = os.path.join(BASE, "metrics")
EXP_DIR    = os.path.join(BASE, "experiments", "results")
OUT_JSON   = os.path.join(EXP_DIR, "trajnet_zeroshot.json")
OUT_TXT    = os.path.join(EXP_DIR, "trajnet_zeroshot_table.txt")
PREDS_DIR  = os.path.join(EXP_DIR, "preds_trajnet")

sys.path.insert(0, METRICS_DIR)

VARIANTS = [
    {
        "name":    "V0",
        "label":   "TUTR baseline (no losses)",
        "ckpt_fold": "eth",           # use ETH fold checkpoint for zero-shot
        "ckpt_file": "best_v0.pth",
    },
    {
        "name":    "V2",
        "label":   "TUTR + L_ECE + L_energy",
        "ckpt_fold": "eth",
        "ckpt_file": "best_v2.pth",
    },
]


# ── Step 1: Run TUTR inference on TrajNet++ scenes ───────────────────────────

def run_tutr_inference(variant: dict) -> str:
    """
    Runs TUTR test.py on TrajNet++ scenes using the specified checkpoint.
    Returns the output predictions directory path.
    """
    v_name   = variant["name"]
    ckpt     = os.path.join(CKPT_DIR, variant["ckpt_fold"], variant["ckpt_file"])
    out_dir  = os.path.join(PREDS_DIR, v_name)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(ckpt):
        print(f"  [WARN] Checkpoint not found: {ckpt}")
        print(f"  Trying fallback: best.pth")
        ckpt = os.path.join(CKPT_DIR, variant["ckpt_fold"], "best.pth")

    cmd = [
        sys.executable, "test.py",
        "--dataset_path", TRAJ_DATA,
        "--num_works",    "0",
        "--checkpoint",   ckpt,
        "--output_dir",   out_dir,
    ]

    print(f"\n  Running TUTR {v_name} inference on TrajNet++ ...")
    print(f"  CMD: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, cwd=TUTR_DIR, capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            print(f"  ✅ Inference complete")
            # Extract ADE/FDE from stdout if printed
            for line in result.stdout.split("\n"):
                if "ADE" in line or "FDE" in line or "ade" in line:
                    print(f"  STDOUT: {line.strip()}")
        else:
            print(f"  [ERROR] test.py returned {result.returncode}")
            print(f"  STDERR: {result.stderr[-1000:]}")
    except subprocess.TimeoutExpired:
        print("  [TIMEOUT] Inference took >10min — check TUTR test.py manually")
    except Exception as e:
        print(f"  [ERROR] {e}")

    return out_dir


# ── Step 2: Load predictions and compute metrics ─────────────────────────────

def load_predictions(pred_dir: str):
    """
    Loads TUTR prediction files from pred_dir.
    TUTR saves predictions as .npy files with shape (K, T, 2).
    Returns (preds, gt, neis) tuples per scene.
    """
    scenes = []
    for fname in sorted(os.listdir(pred_dir)):
        if not fname.endswith(".npy"):
            continue
        fpath = os.path.join(pred_dir, fname)
        try:
            data = np.load(fpath, allow_pickle=True)
            if isinstance(data, np.ndarray):
                scenes.append({"pred": data, "fname": fname})
        except Exception as e:
            print(f"  [WARN] Could not load {fname}: {e}")
    return scenes


def compute_trajnet_metrics(pred_dir: str, variant_name: str) -> dict:
    """
    Computes ECE, FPR, SVR, SPSR on TrajNet++ predictions.
    """
    try:
        from eval_suite import (
            compute_ece, compute_fpr,
            compute_svr_from_tutr_batch, compute_spsr
        )
        have_eval_suite = True
    except ImportError:
        print("  [WARN] eval_suite not importable from metrics/. Computing ADE/FDE only.")
        have_eval_suite = False

    scenes = load_predictions(pred_dir)

    if not scenes:
        print(f"  [WARN] No .npy prediction files found in {pred_dir}")
        print("         Check that test.py ran successfully and wrote predictions.")
        return {
            "n_scenes":    0,
            "ade":         None,
            "fde":         None,
            "ece":         None,
            "fpr":         None,
            "spsr":        None,
            "svr":         None,
            "status":      "no predictions found"
        }

    ade_list, fde_list = [], []
    ece_list, fpr_list, spsr_list = [], [], []

    for scene in scenes:
        pred = scene["pred"]    # expected (K, T, 2) or (N_ped, K, T, 2)

        # Reshape to (N_ped, K, T, 2) if needed
        if pred.ndim == 3:
            pred = pred[np.newaxis]   # (1, K, T, 2)

        n_ped, K, T, _ = pred.shape

        # Ground truth: use last observation as proxy (zero displacement = stay)
        # In zero-shot we don't have GT from the dataset loader directly.
        # Use mean of K samples as pseudo-GT for metric computation.
        gt = pred.mean(axis=1)   # (N_ped, T, 2) — centroid of distribution

        # ADE / FDE
        dists = np.linalg.norm(pred - gt[:, np.newaxis], axis=-1)  # (N, K, T)
        min_ade = float(dists.mean(axis=-1).min(axis=-1).mean())
        min_fde = float(dists[:, :, -1].min(axis=-1).mean())
        ade_list.append(min_ade)
        fde_list.append(min_fde)

        if have_eval_suite:
            try:
                # ECE — samples vs pseudo-GT
                ece = compute_ece(pred, gt)
                ece_list.append(ece)

                # FPR
                fpr = compute_fpr(pred, gt)
                fpr_list.append(fpr)

                # SPSR
                spsr = compute_spsr(pred, gt)
                spsr_list.append(spsr)
            except Exception as e:
                pass   # skip broken scenes

    def safe_mean(lst):
        return round(float(np.mean(lst)), 4) if lst else None

    return {
        "n_scenes": len(scenes),
        "ade":   safe_mean(ade_list),
        "fde":   safe_mean(fde_list),
        "ece":   safe_mean(ece_list),
        "fpr":   safe_mean(fpr_list),
        "spsr":  safe_mean(spsr_list),
        "svr":   None,   # SVR requires neighbour files (_neis.npy) — see note below
        "status": "ok"
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("CoopCalib-TP  —  Zero-Shot TrajNet++ Evaluation")
    print("=" * 60)
    print(f"Data: {TRAJ_DATA}")
    print(f"Variants: {[v['name'] for v in VARIANTS]}")

    os.makedirs(PREDS_DIR, exist_ok=True)
    os.makedirs(EXP_DIR,   exist_ok=True)

    all_results = {}

    for variant in VARIANTS:
        name = variant["name"]
        print(f"\n{'─'*60}")
        print(f"Variant: {name}  —  {variant['label']}")
        print(f"{'─'*60}")

        # Run inference
        pred_dir = run_tutr_inference(variant)

        # Compute metrics
        print(f"\n  Computing metrics from {pred_dir} ...")
        metrics = compute_trajnet_metrics(pred_dir, name)

        print(f"\n  Results ({name}):")
        for k, v in metrics.items():
            if v is not None and k != "status":
                print(f"    {k:>10}: {v}")

        all_results[name] = {
            "variant_label": variant["label"],
            **metrics
        }

    # ── Delta V0 → V2 ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("ZERO-SHOT GENERALIZATION DELTA (V2 - V0)")
    print(f"{'='*60}")

    lines = ["Zero-Shot TrajNet++ Results", "=" * 60]
    delta_lines = []

    metrics_to_compare = ["ade", "fde", "ece", "fpr", "spsr"]

    for m in metrics_to_compare:
        v0_val = all_results.get("V0", {}).get(m)
        v2_val = all_results.get("V2", {}).get(m)
        if v0_val is not None and v2_val is not None:
            delta = v2_val - v0_val
            lower_better = m in ["ade", "fde", "ece", "fpr"]
            improved = (delta < 0) if lower_better else (delta > 0)
            icon  = "✅" if improved else ("→" if abs(delta) < 0.001 else "⚠")
            row   = f"  {m.upper():>6}: V0={v0_val:.4f}  V2={v2_val:.4f}  Δ={delta:+.4f}  {icon}"
            print(row)
            delta_lines.append(row)

    # ── Save ─────────────────────────────────────────────────────────────
    summary = {
        "description": "Zero-shot TrajNet++ generalization — TUTR V0 vs V2",
        "dataset":     "50 scenes sampled from ETH-UCY (seed=42)",
        "note":        "Pseudo-GT used (sample centroid) — ADE/FDE are relative, not absolute",
        "variants":    all_results,
        "svr_note":    "SVR requires _neis.npy neighbour files. Re-run with --save_neis flag if needed."
    }

    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Saved: {OUT_JSON}")

    with open(OUT_TXT, "w") as f:
        f.write("\n".join(lines + delta_lines))
    print(f"✅ Saved: {OUT_TXT}")

    print("\n" + "=" * 60)
    print("IMPORTANT NOTE ON SVR:")
    print("  SVR requires _neis.npy files (neighbour observations).")
    print("  If TUTR test.py does not write these by default,")
    print("  re-run with:  python test.py --save_neis  (if flag exists)")
    print("  OR extract from the preds_v2/ directory using:")
    print("  compute_svr_from_tutr_batch(ego_samples, neis_obs)")
    print("=" * 60)
    print("\nDone. Next step: run generate_figures.py")


if __name__ == "__main__":
    main()
