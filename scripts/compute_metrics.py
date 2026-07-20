"""
CoopCalib-TP — Step 3: Compute Metrics per Fold + Density Tier
===============================================================
File:  C:\\CoopCalib\\scripts\\compute_metrics.py

Loads the .npy prediction files produced by run_inference.py,
runs all four metrics from eval_suite.py, and prints a summary
table per fold and per density tier.

Usage (from C:\\CoopCalib\\):
    python scripts\compute_metrics.py

Requires:
    - experiments\results\preds\{name}_samples.npy   for all 5 folds
    - metrics\eval_suite.py
    - data\processed\density_manifest.json

Output:
    Prints per-fold table + tier-aggregated table to stdout.
    Saves raw per-fold numbers to experiments\results\metrics_per_fold.json
    (Step 4 will collate these into baseline.json)
"""

import os
import sys
import json
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.abspath(__file__)) + "\\.."
PREDS_DIR = os.path.join(ROOT, "experiments", "results", "preds")
OUT_DIR   = os.path.join(ROOT, "experiments", "results")
MANIFEST  = os.path.join(ROOT, "data", "processed", "density_manifest.json")

sys.path.insert(0, os.path.join(ROOT, "metrics"))
from eval_suite import compute_ece, compute_fpr, compute_svr, compute_spsr

# ── Density tier assignment (from density_manifest.json) ─────────────────────
TIER_MAP = {
    "eth":   "sparse",
    "hotel": "medium",
    "zara1": "medium",
    "zara2": "medium",
    "univ":  "dense",
}

DATASETS = ["eth", "hotel", "univ", "zara1", "zara2"]

# ── FPR freeze threshold ──────────────────────────────────────────────────────
# Default 0.5m. If baseline FPR on dense tier < 0.05, raise to 0.8m
# (Risk 3 bypass protocol from roadmap). We compute both and report both.
FREEZE_THRESHOLDS = [0.5, 0.8]

# ── Metric computation ────────────────────────────────────────────────────────

def load_fold(name):
    """Load (samples, gt, obs) for one fold. Returns None if files missing."""
    s_path = os.path.join(PREDS_DIR, f"{name}_samples.npy")
    g_path = os.path.join(PREDS_DIR, f"{name}_gt.npy")
    o_path = os.path.join(PREDS_DIR, f"{name}_obs.npy")

    missing = [p for p in [s_path, g_path, o_path] if not os.path.exists(p)]
    if missing:
        return None, None, None

    samples = np.load(s_path)   # (N, 20, 12, 2)
    gt      = np.load(g_path)   # (N, 12, 2)
    obs     = np.load(o_path)   # (N,  8, 2)
    return samples, gt, obs


def compute_all_metrics(samples, gt, obs):
    """
    Compute ECE, FPR (×2 thresholds), SVR (single-agent), SPSR.

    SVR in single-agent mode (eval_suite returns 0.0 for 4-D input) —
    this is correct for the V0 baseline since we only have ego predictions.
    Full multi-agent SVR is enabled in V1+ when neighbour trajectories are
    concatenated. We note this in the output.

    Returns dict of metric_name -> value.
    """
    results = {}

    # ECE
    results["ece"] = compute_ece(samples, gt)

    # FPR at two thresholds
    for thresh in FREEZE_THRESHOLDS:
        key = f"fpr_{int(thresh*100)}cm"
        results[key] = compute_fpr(samples, gt, freeze_thresh=thresh)

    # SVR (single-agent — returns 0.0; noted in output)
    results["svr"] = compute_svr(samples)

    # SPSR
    results["spsr"] = compute_spsr(samples, gt)

    # Standard ADE / FDE (min over K)  — for sanity check vs train.py output
    diff     = samples - gt[:, None, :, :]        # (N, K, T, 2)
    dist     = np.sqrt((diff**2).sum(axis=-1))    # (N, K, T)
    ade_k    = dist.mean(axis=-1)                 # (N, K)
    fde_k    = dist[:, :, -1]                     # (N, K)
    results["min_ade"] = float(ade_k.min(axis=-1).mean())
    results["min_fde"] = float(fde_k.min(axis=-1).mean())

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  CoopCalib-TP — Step 3: Metric Computation")
    print("="*60)

    per_fold = {}
    missing_folds = []

    for name in DATASETS:
        samples, gt, obs = load_fold(name)
        if samples is None:
            print(f"\n  [SKIP] {name} — prediction files not found in {PREDS_DIR}")
            missing_folds.append(name)
            continue

        print(f"\n  [{name.upper()}]  N={samples.shape[0]} scenes, K={samples.shape[1]}")
        metrics = compute_all_metrics(samples, gt, obs)
        metrics["tier"] = TIER_MAP[name]
        per_fold[name] = metrics

        print(f"    ECE          : {metrics['ece']:.4f}")
        print(f"    FPR@0.5m     : {metrics['fpr_50cm']:.4f}")
        print(f"    FPR@0.8m     : {metrics['fpr_80cm']:.4f}")
        print(f"    SVR          : {metrics['svr']:.4f}  (single-agent; 0.0 expected at V0)")
        print(f"    SPSR         : {metrics['spsr']:.4f}")
        print(f"    minADE       : {metrics['min_ade']:.4f}  (sanity vs train.py output)")
        print(f"    minFDE       : {metrics['min_fde']:.4f}")

    if missing_folds:
        print(f"\n  ⚠  Missing folds: {missing_folds}")
        print("     Run run_inference.py for those folds first.")

    if not per_fold:
        print("\n  No folds available yet. Exiting.")
        return

    # ── Tier-aggregated summary ───────────────────────────────────────────────
    print("\n" + "-"*60)
    print("  Tier-Aggregated Summary")
    print("-"*60)

    tier_groups = {"sparse": [], "medium": [], "dense": []}
    for name, m in per_fold.items():
        tier_groups[m["tier"]].append(m)

    tier_summary = {}
    metric_keys = ["ece", "fpr_50cm", "fpr_80cm", "svr", "spsr", "min_ade", "min_fde"]

    for tier, fold_metrics in tier_groups.items():
        if not fold_metrics:
            continue
        agg = {}
        for key in metric_keys:
            vals = [m[key] for m in fold_metrics]
            agg[key]            = float(np.mean(vals))
            agg[key + "_std"]   = float(np.std(vals))
            agg["n_folds"]      = len(fold_metrics)
        tier_summary[tier] = agg

        folds_in_tier = [n for n, m in per_fold.items() if m["tier"] == tier]
        print(f"\n  {tier.upper()} tier  (folds: {folds_in_tier})")
        for key in metric_keys:
            if key in agg:
                print(f"    {key:<12}: {agg[key]:.4f}  ±{agg[key+'_std']:.4f}")

    # ── FPR diagnostic (Risk 3 bypass) ───────────────────────────────────────
    if "dense" in tier_summary:
        fpr_dense = tier_summary["dense"].get("fpr_50cm", None)
        if fpr_dense is not None:
            print("\n" + "-"*60)
            print("  Risk 3 Diagnostic — FPR on Dense Tier")
            print("-"*60)
            if fpr_dense <= 0.05:
                print(f"  ⚠  FPR@0.5m on DENSE = {fpr_dense:.4f}  (≤ 0.05 threshold)")
                print("     → TUTR does NOT exhibit the Freezing Predictor Effect.")
                print("     → Reframe RQ2: quantify WHY (pairwise attention enforces")
                print("       cooperative structure implicitly). Still publishable.")
                print(f"     → FPR@0.8m = {tier_summary['dense'].get('fpr_80cm', 0):.4f}")
            else:
                print(f"  ✓  FPR@0.5m on DENSE = {fpr_dense:.4f}  (> 0.05 — FPE confirmed)")
                print("     → Proceed with RQ2 as planned.")

    # ── Save per-fold JSON ────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "metrics_per_fold.json")
    with open(out_path, "w") as f:
        json.dump({"per_fold": per_fold, "tier_summary": tier_summary}, f, indent=2)

    print(f"\n  Saved: {out_path}")
    print("  Run scripts\\collate_baseline.py to produce baseline.json\n")


if __name__ == "__main__":
    main()
