"""
CoopCalib-TP — Simple SPSR for all variants
============================================
File:  C:\\CoopCalib\\scripts\\compute_simple_spsr.py

Computes SPSR using the original simple implementation
(ground-truth proximity as collision proxy, no neighbour checking)
for all four variants V0-V3.

This matches the SPSR values reported in the paper's Table 2.

Usage (from C:\\CoopCalib\\):
    python scripts\\compute_simple_spsr.py
"""

import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__)) + "\\.."
sys.path.insert(0, os.path.join(ROOT, "metrics"))

from eval_suite import compute_spsr

VARIANTS = {
    "V0": os.path.join(ROOT, "experiments", "results", "preds_v0_neis"),
    "V1": os.path.join(ROOT, "experiments", "results", "preds_v1"),
    "V2": os.path.join(ROOT, "experiments", "results", "preds_v2"),
    "V3": os.path.join(ROOT, "experiments", "results", "preds_v3"),
}

DATASETS = ["eth", "hotel", "univ", "zara1", "zara2"]

TIER_MAP = {
    "eth":   "sparse",
    "hotel": "medium",
    "univ":  "dense",
    "zara1": "medium",
    "zara2": "medium",
}

print("\n" + "="*60)
print("  Simple SPSR (planner_radius=0.5m, goal_radius=1.0m)")
print("  Matches original paper Table 2 implementation")
print("="*60)

all_results = {}

for variant, pdir in VARIANTS.items():
    print(f"\n  --- {variant} ---")
    vals = {}

    for ds in DATASETS:
        s_path = os.path.join(pdir, f"{ds}_samples.npy")
        g_path = os.path.join(pdir, f"{ds}_gt.npy")

        if not os.path.exists(s_path):
            print(f"  {ds}: MISSING — {s_path}")
            continue

        s = np.load(s_path)   # (N, K, T, 2)
        g = np.load(g_path)   # (N, T, 2)

        spsr = compute_spsr(s, g, planner_radius=0.5, goal_radius=1.0)
        vals[ds] = spsr
        tier = TIER_MAP[ds]
        print(f"  {ds:<8} ({tier:<6}): {spsr:.4f}")

    if vals:
        avg = sum(vals.values()) / len(vals)
        print(f"  {'AVG':<8}         : {avg:.4f}")
        all_results[variant] = vals

# Summary table
print("\n" + "="*60)
print("  SUMMARY TABLE — Simple SPSR")
print("="*60)
print(f"\n  {'Subset':<8} {'Tier':<8}", end="")
for v in VARIANTS:
    print(f" {v:>8}", end="")
print()
print("  " + "-"*52)

for ds in DATASETS:
    tier = TIER_MAP[ds]
    print(f"  {ds:<8} {tier:<8}", end="")
    for v in VARIANTS:
        val = all_results.get(v, {}).get(ds, float("nan"))
        print(f" {val:>8.4f}", end="")
    print()

print("  " + "-"*52)
print(f"  {'AVG':<8} {'':8}", end="")
for v in VARIANTS:
    vals = list(all_results.get(v, {}).values())
    avg = sum(vals) / len(vals) if vals else float("nan")
    print(f" {avg:>8.4f}", end="")
print()

# Save results
out_path = os.path.join(ROOT, "experiments", "results", "spsr_simple_all_variants.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n  Saved: {out_path}")

# Cohen's d between V0 and V2 (replicating paper claim)
print("\n" + "="*60)
print("  Statistical check: Cohen's d  V0 vs V2 SPSR")
print("="*60)

if "V0" in all_results and "V2" in all_results:
    v0_vals = np.array([all_results["V0"].get(ds, 0) for ds in DATASETS])
    v2_vals = np.array([all_results["V2"].get(ds, 0) for ds in DATASETS])
    diff = v0_vals - v2_vals
    pooled_std = np.sqrt(
        (np.std(v0_vals, ddof=1)**2 + np.std(v2_vals, ddof=1)**2) / 2
    )
    cohens_d = abs(np.mean(diff)) / (pooled_std + 1e-9)
    print(f"  V0 SPSR: {v0_vals}")
    print(f"  V2 SPSR: {v2_vals}")
    print(f"  Mean diff: {np.mean(diff):.4f}")
    print(f"  Pooled std: {pooled_std:.4f}")
    print(f"  Cohen's d: {cohens_d:.4f}")
    print()
    if cohens_d > 1.0:
        print("  Paper claim (d=2.327) is in the right ballpark — large effect")
    else:
        print("  WARNING: Cohen's d does not replicate paper claim")
        print("  The SPSR result may need revision")
