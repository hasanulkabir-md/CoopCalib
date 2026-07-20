"""
CoopCalib-TP — Metric Computation WITH proper SVR
==================================================
File:  C:\\CoopCalib\\scripts\\compute_metrics_with_svr.py

Identical to compute_metrics.py EXCEPT:
  - Loads _neis.npy alongside _samples, _gt, _obs
  - Calls compute_svr_from_tutr_batch(ego_samples, neighbour_obs)
    instead of compute_svr(samples) — gives real SVR not 0.0
  - Calls compute_spsr_with_neighbours for proper SPSR
  - Accepts --preds_dir and --out_file as arguments
    so it can be pointed at preds/, preds_v1/, preds_v2/, preds_v3/

Usage (run from C:\\CoopCalib\\):
    python scripts\\compute_metrics_with_svr.py --preds_dir experiments\\results\\preds_v0_neis --out_file experiments\\results\\v0_metrics_full.json
    python scripts\\compute_metrics_with_svr.py --preds_dir experiments\\results\\preds_v1     --out_file experiments\\results\\v1_metrics_full.json
    python scripts\\compute_metrics_with_svr.py --preds_dir experiments\\results\\preds_v2     --out_file experiments\\results\\v2_metrics_full.json
    python scripts\\compute_metrics_with_svr.py --preds_dir experiments\\results\\preds_v3     --out_file experiments\\results\\v3_metrics_full.json
"""

import os
import sys
import json
import argparse
import numpy as np

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--preds_dir", type=str, required=True,
                    help="Directory containing {name}_samples.npy etc.")
parser.add_argument("--out_file",  type=str, required=True,
                    help="Output JSON path e.g. experiments/results/v1_metrics_full.json")
parser.add_argument("--ped_radius", type=float, default=0.3,
                    help="Personal space radius for SVR (default 0.3m)")
parser.add_argument("--planner_radius", type=float, default=0.5,
                    help="Collision radius for SPSR (default 0.5m)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = os.path.dirname(os.path.abspath(__file__)) + "\\.."
METRICS_DIR = os.path.join(ROOT, "metrics")
sys.path.insert(0, METRICS_DIR)

from eval_suite import (
    compute_ece,
    compute_fpr,
    compute_svr_from_tutr_batch,
    compute_spsr_with_neighbours,
)

PREDS_DIR = args.preds_dir
OUT_FILE  = args.out_file

DATASETS = ["eth", "hotel", "univ", "zara1", "zara2"]

TIER_MAP = {
    "eth":   "sparse",
    "hotel": "medium",
    "zara1": "medium",
    "zara2": "medium",
    "univ":  "dense",
}

FREEZE_THRESHOLDS = [0.5, 0.8]

# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_fold(name):
    """Load samples, gt, obs, neis for one fold. Returns None tuple if missing."""
    s_path = os.path.join(PREDS_DIR, f"{name}_samples.npy")
    g_path = os.path.join(PREDS_DIR, f"{name}_gt.npy")
    o_path = os.path.join(PREDS_DIR, f"{name}_obs.npy")
    n_path = os.path.join(PREDS_DIR, f"{name}_neis.npy")

    missing = [p for p in [s_path, g_path, o_path] if not os.path.exists(p)]
    if missing:
        print(f"  [SKIP] {name} — missing: {missing}")
        return None, None, None, None

    samples = np.load(s_path)  # (N, K, T, 2)
    gt      = np.load(g_path)  # (N, T, 2)
    obs     = np.load(o_path)  # (N, T_obs, 2)

    if os.path.exists(n_path):
        # neis shape from run_inference: (N, MaxN, T, 2)
        # compute_svr_from_tutr_batch expects: (N, T, P-1, 2)
        # So we transpose axis 1 and 2
        neis_raw = np.load(n_path)           # (N, MaxN, T, 2)
        neis = neis_raw.transpose(0, 2, 1, 3)  # (N, T, MaxN, 2)
        print(f"    neis loaded: {neis_raw.shape} → transposed to {neis.shape}")
    else:
        neis = None
        print(f"    WARNING: {name}_neis.npy not found — SVR will be 0.0")

    return samples, gt, obs, neis

# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def compute_all_metrics(name, samples, gt, obs, neis):
    results = {}

    # ECE
    results["ece"] = compute_ece(samples, gt)

    # FPR at two thresholds
    for thresh in FREEZE_THRESHOLDS:
        key = f"fpr_{int(thresh*100)}cm"
        results[key] = compute_fpr(samples, gt, freeze_thresh=thresh)

    # SVR — proper multi-agent via neis
    if neis is not None:
        results["svr"] = compute_svr_from_tutr_batch(
            samples, neis, ped_radius=args.ped_radius
        )
        print(f"    SVR (multi-agent, r={args.ped_radius}m): {results['svr']:.4f}")
    else:
        results["svr"] = 0.0
        print(f"    SVR: 0.0 (no neis — single-agent fallback)")

    # SPSR — proper with neighbours if available
    if neis is not None:
        results["spsr"] = compute_spsr_with_neighbours(
            samples, gt, neis,
            planner_radius=args.planner_radius,
            goal_radius=1.0
        )
    else:
        # fallback to simple SPSR
        from eval_suite import compute_spsr
        results["spsr"] = compute_spsr(
            samples, gt,
            planner_radius=args.planner_radius,
            goal_radius=1.0
        )

    # ADE / FDE
    diff  = samples - gt[:, None, :, :]
    dist  = np.sqrt((diff ** 2).sum(axis=-1))
    ade_k = dist.mean(axis=-1)
    fde_k = dist[:, :, -1]
    results["min_ade"] = float(ade_k.min(axis=-1).mean())
    results["min_fde"] = float(fde_k.min(axis=-1).mean())

    return results

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"\n{'='*60}")
    print(f"  CoopCalib-TP — Metrics WITH SVR")
    print(f"  preds_dir : {PREDS_DIR}")
    print(f"  out_file  : {OUT_FILE}")
    print(f"{'='*60}\n")

    per_fold = {}

    for name in DATASETS:
        print(f"\n  [{name.upper()}]")
        samples, gt, obs, neis = load_fold(name)
        if samples is None:
            continue

        print(f"    scenes={samples.shape[0]}  K={samples.shape[1]}")
        metrics = compute_all_metrics(name, samples, gt, obs, neis)
        metrics["tier"] = TIER_MAP[name]
        per_fold[name] = metrics

        print(f"    ECE      : {metrics['ece']:.4f}")
        print(f"    FPR@0.5m : {metrics['fpr_50cm']:.4f}")
        print(f"    FPR@0.8m : {metrics['fpr_80cm']:.4f}")
        print(f"    SVR      : {metrics['svr']:.4f}")
        print(f"    SPSR     : {metrics['spsr']:.4f}")
        print(f"    minADE   : {metrics['min_ade']:.4f}")
        print(f"    minFDE   : {metrics['min_fde']:.4f}")

    if not per_fold:
        print("\nNo folds computed. Check --preds_dir path.")
        return

    # Tier summary
    print(f"\n{'─'*60}")
    print("  Tier Summary")
    print(f"{'─'*60}")

    tier_groups = {"sparse": [], "medium": [], "dense": []}
    for name, m in per_fold.items():
        tier_groups[m["tier"]].append((name, m))

    tier_summary = {}
    metric_keys = ["ece", "fpr_50cm", "svr", "spsr", "min_ade", "min_fde"]

    for tier, fold_list in tier_groups.items():
        if not fold_list:
            continue
        fold_metrics = [m for _, m in fold_list]
        fold_names   = [n for n, _ in fold_list]
        agg = {}
        for key in metric_keys:
            vals = [m[key] for m in fold_metrics]
            agg[key]          = float(np.mean(vals))
            agg[key + "_std"] = float(np.std(vals))
        agg["n_folds"] = len(fold_metrics)
        tier_summary[tier] = agg
        print(f"\n  {tier.upper()} ({fold_names})")
        for key in metric_keys:
            print(f"    {key:<12}: {agg[key]:.4f} ±{agg[key+'_std']:.4f}")

    # Overall summary
    all_metrics = list(per_fold.values())
    print(f"\n{'─'*60}")
    print("  OVERALL AVERAGE")
    print(f"{'─'*60}")
    for key in metric_keys:
        vals = [m[key] for m in all_metrics]
        print(f"    {key:<12}: {np.mean(vals):.4f} ±{np.std(vals):.4f}")

    # Full results table
    print(f"\n{'─'*60}")
    print(f"  {'Subset':<8} {'ECE':>7} {'FPR':>7} {'SVR':>7} "
          f"{'SPSR':>7} {'ADE':>7} {'FDE':>7}")
    print(f"  {'─'*50}")
    for ds in DATASETS:
        if ds not in per_fold:
            continue
        m = per_fold[ds]
        print(f"  {ds:<8} {m['ece']:>7.4f} {m['fpr_50cm']:>7.4f} "
              f"{m['svr']:>7.4f} {m['spsr']:>7.4f} "
              f"{m['min_ade']:>7.4f} {m['min_fde']:>7.4f}")

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(OUT_FILE)), exist_ok=True)
    output = {
        "per_fold": per_fold,
        "tier_summary": tier_summary,
        "overall": {
            key: {
                "mean": float(np.mean([m[key] for m in all_metrics])),
                "std":  float(np.std( [m[key] for m in all_metrics]))
            }
            for key in metric_keys
        }
    }
    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {OUT_FILE}\n")


if __name__ == "__main__":
    main()
