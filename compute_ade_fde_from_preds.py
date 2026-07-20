"""
compute_ade_fde_from_preds.py  --  CoopCalib-TP
================================================
Computes minADE and minFDE from the saved .npy prediction arrays
in preds_v1/ and preds_v2/ and writes them into v1_metrics.json
and v2_metrics.json.

Array layout confirmed from run_inference.py console output (Day 6):
    samples : (N, K, T, 2)   K=20 candidates, T=12 timesteps
    gt      : (N, T, 2)

Per-fold N values (confirmed):
    ETH=364, HOTEL=1197, UNIV=24334, ZARA1=2356, ZARA2=5910

V2 full-precision values from Day 6 console (for cross-check):
    ETH   minADE=0.42214344360016204  minFDE=0.6179722167633392
    HOTEL minADE=0.12783484670054246  minFDE=0.1848722288981018
    UNIV  minADE=0.23814717596751206  minFDE=0.4358220289695466
    ZARA1 minADE=0.18918566549977950  minFDE=0.3428292922131679
    ZARA2 minADE=0.14147488292300360  minFDE=0.2578218194635588

Run from C:\\CoopCalib\\ with venv active:
    python compute_ade_fde_from_preds.py
"""

import os
import sys
import json
import numpy as np

RESULTS = os.path.join("experiments", "results")

# Confirmed from Day 6 console output (for cross-validation only)
# Script does NOT use these -- it computes from npy. They are printed
# so you can verify the computed values match.
V2_CONSOLE = {
    "eth":   (0.42214344360016204, 0.6179722167633392),
    "hotel": (0.12783484670054246, 0.18487222889810180),
    "univ":  (0.23814717596751206, 0.43582202896954660),
    "zara1": (0.18918566549977950, 0.34282929221316794),
    "zara2": (0.14147488292300360, 0.25782181946355880),
}

# Key mapping: internal fold name -> JSON key in v1/v2_metrics.json
# v1/v2_metrics.json uses "zara1" (confirmed from actual file)
FOLDS = ["eth", "hotel", "univ", "zara1", "zara2"]
JSON_KEY = {
    "eth":   "eth",
    "hotel": "hotel",
    "univ":  "univ",
    "zara1": "zara1",
    "zara2": "zara2",
}


def compute_min_ade_fde(samples, gt):
    """
    samples : (N, K, T, 2)  -- K=20 candidate trajectories
    gt      : (N, T, 2)     -- ground truth future
    Returns (minADE, minFDE) averaged over N pedestrians.
    """
    assert samples.ndim == 4, f"Expected (N,K,T,2), got {samples.shape}"
    assert gt.ndim == 3,      f"Expected (N,T,2), got {gt.shape}"
    assert samples.shape[0] == gt.shape[0], "N mismatch"
    assert samples.shape[2] == gt.shape[1], \
        f"T mismatch: samples T={samples.shape[2]}, gt T={gt.shape[1]}"

    # Euclidean distance per candidate per timestep: (N, K, T)
    diff = samples - gt[:, None, :, :]        # broadcast (N,1,T,2)
    dist = np.linalg.norm(diff, axis=-1)      # (N, K, T)

    # minADE: best candidate by mean displacement
    ade_per_k = dist.mean(axis=2)             # (N, K)
    min_ade   = ade_per_k.min(axis=1).mean()  # scalar

    # minFDE: best candidate by final displacement
    fde_per_k = dist[:, :, -1]               # (N, K)
    min_fde   = fde_per_k.min(axis=1).mean() # scalar

    return float(min_ade), float(min_fde)


def load_fold(preds_dir, fold):
    """Load samples and gt npy arrays for one fold."""
    samples_path = os.path.join(preds_dir, f"{fold}_samples.npy")
    gt_path      = os.path.join(preds_dir, f"{fold}_gt.npy")

    if not os.path.exists(samples_path):
        return None, None, f"samples not found: {samples_path}"
    if not os.path.exists(gt_path):
        return None, None, f"gt not found: {gt_path}"

    samples = np.load(samples_path)
    gt      = np.load(gt_path)
    return samples, gt, None


def process_version(preds_dir, json_path, version_label, console_ref=None):
    print(f"\n{'='*60}")
    print(f"{version_label}  |  preds: {preds_dir}")
    print(f"{'='*60}")

    if not os.path.exists(json_path):
        print(f"[ERROR] {json_path} not found -- cannot write results")
        return False

    with open(json_path) as f:
        metrics = json.load(f)

    results = {}
    all_ok  = True

    for fold in FOLDS:
        samples, gt, err = load_fold(preds_dir, fold)
        if err:
            print(f"  {fold:<8}  [SKIP] {err}")
            all_ok = False
            continue

        # Validate shape matches confirmed layout
        N_expected = {"eth": 364, "hotel": 1197,
                      "univ": 24334, "zara1": 2356, "zara2": 5910}
        if samples.shape[0] != N_expected[fold]:
            print(f"  {fold:<8}  [WARN] N={samples.shape[0]}, "
                  f"expected {N_expected[fold]}")
        if samples.shape[1] != 20:
            print(f"  {fold:<8}  [WARN] K={samples.shape[1]}, expected 20")
        if samples.shape[2] != 12:
            print(f"  {fold:<8}  [WARN] T={samples.shape[2]}, expected 12")

        min_ade, min_fde = compute_min_ade_fde(samples, gt)
        results[fold] = (min_ade, min_fde)

        # Cross-check against Day 6 console values if available
        ref_str = ""
        if console_ref and fold in console_ref:
            ref_ade, ref_fde = console_ref[fold]
            ade_err = abs(min_ade - ref_ade)
            fde_err = abs(min_fde - ref_fde)
            tol = 1e-5
            if ade_err < tol and fde_err < tol:
                ref_str = "  [MATCHES console]"
            else:
                ref_str = (f"  [DIFF vs console: "
                           f"dADE={ade_err:.2e} dFDE={fde_err:.2e}]")

        print(f"  {fold:<8}  "
              f"shape={samples.shape}  "
              f"minADE={min_ade:.6f}  minFDE={min_fde:.6f}"
              f"{ref_str}")

    # Write into JSON
    changed = False
    for fold, (min_ade, min_fde) in results.items():
        jk = JSON_KEY[fold]
        if jk not in metrics:
            print(f"  [WARN] key '{jk}' not in {json_path} -- skipping")
            continue
        if "min_ade" in metrics[jk]:
            print(f"  {fold}: already has min_ade={metrics[jk]['min_ade']} "
                  f"-- overwriting with npy-computed value")
        metrics[jk]["min_ade"] = round(min_ade, 10)
        metrics[jk]["min_fde"] = round(min_fde, 10)
        metrics[jk]["_ade_fde_source"] = (
            f"Computed from {preds_dir}/{fold}_samples.npy + "
            f"{fold}_gt.npy by compute_ade_fde_from_preds.py. "
            f"Array layout (N,K,T,2) confirmed from Day 6 "
            f"run_inference.py console output."
        )
        changed = True

    if changed:
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  Written -> {json_path}")

    # Final summary table
    print(f"\n  {'Fold':<8} {'minADE':>10} {'minFDE':>10}  "
          f"{'Ref ADE':>12} {'Match?':>8}")
    print(f"  {'-'*56}")
    for fold in FOLDS:
        jk = JSON_KEY[fold]
        if jk not in metrics or "min_ade" not in metrics[jk]:
            print(f"  {fold:<8}  MISSING")
            continue
        ade = metrics[jk]["min_ade"]
        fde = metrics[jk]["min_fde"]
        if console_ref and fold in console_ref:
            ref_ade = console_ref[fold][0]
            match   = "OK" if abs(ade - ref_ade) < 1e-5 else "DIFF"
            print(f"  {fold:<8} {ade:>10.6f} {fde:>10.6f}  "
                  f"{ref_ade:>12.6f} {match:>8}")
        else:
            print(f"  {fold:<8} {ade:>10.6f} {fde:>10.6f}  "
                  f"{'(no ref)':>12}")

    return all_ok


def main():
    print("=" * 60)
    print("compute_ade_fde_from_preds.py  --  CoopCalib-TP")
    print("Computing minADE/minFDE from saved .npy prediction arrays")
    print("=" * 60)

    # Check preds directories exist
    for d in ["preds_v1", "preds_v2"]:
        p = os.path.join(RESULTS, d)
        if not os.path.exists(p):
            sys.exit(f"[ERROR] Directory not found: {p}\n"
                     f"Run from C:\\CoopCalib\\ with venv active.")

    # V1 -- no full-precision console reference for 4/5 folds
    # ETH reference available but others are 4dp rounded -- skip cross-check
    # to avoid false DIFF warnings on rounding differences
    process_version(
        preds_dir    = os.path.join(RESULTS, "preds_v1"),
        json_path    = os.path.join(RESULTS, "v1_metrics.json"),
        version_label= "V1 (+L_ECE, lambda1=0.1)",
        console_ref  = {"eth": (0.42214344360016204, 0.6179722167633392)},
    )

    # V2 -- full-precision console reference available for all 5 folds
    process_version(
        preds_dir    = os.path.join(RESULTS, "preds_v2"),
        json_path    = os.path.join(RESULTS, "v2_metrics.json"),
        version_label= "V2 (+L_ECE +L_energy, lambda1=0.1 lambda2=0.1)",
        console_ref  = V2_CONSOLE,
    )

    print("\n" + "=" * 60)
    print("Done.")
    print("Next step:  python generate_figures.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
