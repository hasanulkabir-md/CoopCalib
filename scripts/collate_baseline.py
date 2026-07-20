"""
CoopCalib-TP — Step 4: Collate Baseline Results
================================================
File:  C:\\CoopCalib\\scripts\\collate_baseline.py

Reads experiments/results/metrics_per_fold.json (written by Step 3)
and writes the final experiments/results/baseline.json in the
canonical CoopCalib-TP format used by all downstream comparison scripts.

Usage (from C:\\CoopCalib\\):
    python scripts\collate_baseline.py

Output: experiments\results\baseline.json
"""

import os
import sys
import json
from datetime import datetime

ROOT     = os.path.dirname(os.path.abspath(__file__)) + "\\.."
OUT_DIR  = os.path.join(ROOT, "experiments", "results")
IN_PATH  = os.path.join(OUT_DIR, "metrics_per_fold.json")
OUT_PATH = os.path.join(OUT_DIR, "baseline.json")


def main():
    print("\n" + "="*60)
    print("  CoopCalib-TP — Step 4: Collate Baseline")
    print("="*60)

    assert os.path.exists(IN_PATH), \
        f"metrics_per_fold.json not found at {IN_PATH}\nRun compute_metrics.py first."

    with open(IN_PATH) as f:
        data = json.load(f)

    per_fold     = data["per_fold"]
    tier_summary = data["tier_summary"]

    baseline = {
        "version":     "V0",
        "description": "Vanilla TUTR baseline — no CoopCalib-TP loss modifications",
        "model":       "TUTR (Shi et al. ICCV 2023)",
        "dataset":     "ETH-UCY (5-fold leave-one-out)",
        "generated":   datetime.now().isoformat(timespec="seconds"),
        "metrics": {
            "per_fold":     {},
            "tier_summary": {},
            "overall":      {},
        },
        "notes": [
            "SVR=0.0 at V0 — single-agent eval only; multi-agent SVR enabled from V1+",
            "FPR computed at two thresholds (0.5m, 0.8m) for Risk-3 diagnostic",
            "minADE/minFDE included as sanity check vs TUTR train.py output",
        ],
    }

    # ── Per-fold ──────────────────────────────────────────────────────────────
    METRIC_KEYS = ["ece", "fpr_50cm", "fpr_80cm", "svr", "spsr", "min_ade", "min_fde"]

    for fold_name, fold_data in per_fold.items():
        baseline["metrics"]["per_fold"][fold_name] = {
            "tier":    fold_data["tier"],
            **{k: round(fold_data[k], 6) for k in METRIC_KEYS if k in fold_data},
        }

    # ── Tier summary ──────────────────────────────────────────────────────────
    for tier, tier_data in tier_summary.items():
        baseline["metrics"]["tier_summary"][tier] = {
            k: round(tier_data[k], 6)
            for k in tier_data
            if not k.endswith("_std") and k != "n_folds"
        }
        baseline["metrics"]["tier_summary"][tier]["n_folds"] = tier_data.get("n_folds", 0)
        # include std for key metrics
        for key in ["ece", "fpr_50cm", "min_ade", "min_fde", "spsr"]:
            std_key = key + "_std"
            if std_key in tier_data:
                baseline["metrics"]["tier_summary"][tier][std_key] = round(tier_data[std_key], 6)

    # ── Overall (mean across all available folds) ─────────────────────────────
    import numpy as np
    all_folds = list(per_fold.values())
    if all_folds:
        for key in METRIC_KEYS:
            vals = [f[key] for f in all_folds if key in f]
            if vals:
                baseline["metrics"]["overall"][key]          = round(float(np.mean(vals)), 6)
                baseline["metrics"]["overall"][key + "_std"] = round(float(np.std(vals)),  6)

    # ── Write ──────────────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(baseline, f, indent=2)

    print(f"\n  Written: {OUT_PATH}")
    print("\n  ── Overall V0 Baseline ──")
    for key in METRIC_KEYS:
        if key in baseline["metrics"]["overall"]:
            val = baseline["metrics"]["overall"][key]
            std = baseline["metrics"]["overall"].get(key + "_std", 0)
            print(f"    {key:<12}: {val:.4f}  ±{std:.4f}")

    print("\n  ── Tier Summary ──")
    for tier in ["sparse", "medium", "dense"]:
        if tier in baseline["metrics"]["tier_summary"]:
            t = baseline["metrics"]["tier_summary"][tier]
            print(f"\n  {tier.upper()}  (n={t.get('n_folds','-')} folds)")
            for key in ["ece", "fpr_50cm", "min_ade", "min_fde", "spsr"]:
                if key in t:
                    std = t.get(key+"_std", 0)
                    print(f"    {key:<12}: {t[key]:.4f}  ±{std:.4f}")

    print("\n  Day 2 PM complete. baseline.json is your Novelty Claim 1 evidence.\n")


if __name__ == "__main__":
    main()
