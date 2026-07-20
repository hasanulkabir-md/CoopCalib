"""
compile_multiseed_stats.py
Computes mean +/- std across seeds for V0 baseline.
Reads: v0_metrics_corrected.json (seed 42)
       v0_seed1_metrics.json     (seed 1)
       v0_seed123_metrics.json   (seed 123)
Writes: v0_multiseed_summary.json
"""
import json
import numpy as np
import os

BASE = r"C:\CoopCalib\experiments\results"

FILES = {
    "seed42":  os.path.join(BASE, "v0_metrics_corrected.json"),
    "seed1":   os.path.join(BASE, "v0_seed1_metrics.json"),
    "seed123": os.path.join(BASE, "v0_seed123_metrics.json"),
}

SUBSETS  = ["eth", "hotel", "univ", "zara1", "zara2"]
METRICS  = ["ece", "fpr_50cm", "svr", "spsr", "min_ade", "min_fde"]
TIER_MAP = {"eth": "sparse", "hotel": "medium", "univ": "dense",
            "zara1": "medium", "zara2": "medium"}

# Load all three seeds
data = {}
for seed_name, path in FILES.items():
    with open(path) as f:
        data[seed_name] = json.load(f)
    print(f"Loaded {seed_name}: {path}")

print()
print("=" * 70)
print("  V0 BASELINE — 3-Seed Statistics (seeds 42, 1, 123)")
print("=" * 70)

summary = {}

for ds in SUBSETS:
    summary[ds] = {"tier": TIER_MAP[ds]}
    row_parts = []
    for metric in METRICS:
        vals = []
        for seed_name in ["seed42", "seed1", "seed123"]:
            v = data[seed_name]["per_fold"][ds][metric]
            vals.append(v)
        mean = np.mean(vals)
        std  = np.std(vals)
        summary[ds][metric] = {"mean": round(float(mean), 4),
                               "std":  round(float(std),  4),
                               "values": [round(float(v), 4) for v in vals]}
        row_parts.append(f"{metric}={mean:.4f}±{std:.4f}")

    print(f"\n  {ds.upper()} ({TIER_MAP[ds]})")
    for metric in METRICS:
        vals = [data[s]["per_fold"][ds][metric]
                for s in ["seed42", "seed1", "seed123"]]
        mean = np.mean(vals)
        std  = np.std(vals)
        print(f"    {metric:<12}: {mean:.4f} ± {std:.4f}   "
              f"[{vals[0]:.4f}, {vals[1]:.4f}, {vals[2]:.4f}]")

# Overall averages across all 5 subsets
print()
print("=" * 70)
print("  OVERALL AVERAGE (5 subsets)")
print("=" * 70)
overall = {}
for metric in METRICS:
    seed_avgs = []
    for seed_name in ["seed42", "seed1", "seed123"]:
        per_fold_vals = [data[seed_name]["per_fold"][ds][metric]
                         for ds in SUBSETS]
        seed_avgs.append(np.mean(per_fold_vals))
    mean = np.mean(seed_avgs)
    std  = np.std(seed_avgs)
    overall[metric] = {"mean": round(float(mean), 4),
                       "std":  round(float(std),  4),
                       "seed_avgs": [round(float(v), 4) for v in seed_avgs]}
    print(f"  {metric:<12}: {mean:.4f} ± {std:.4f}   "
          f"seed_avgs=[{seed_avgs[0]:.4f}, {seed_avgs[1]:.4f}, {seed_avgs[2]:.4f}]")

# Save
out = {"variant": "V0", "seeds": [42, 1, 123],
       "n_seeds": 3, "per_subset": summary, "overall": overall}
out_path = os.path.join(BASE, "v0_multiseed_summary.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print()
print(f"  Saved: {out_path}")
print()

# Print clean table for paper
print("=" * 70)
print("  PAPER TABLE — V0 mean ± std")
print("=" * 70)
print(f"  {'Subset':<8} {'ADE':>12} {'ECE':>12} {'SVR':>12} "
      f"{'SPSR':>12} {'FPR':>8}")
print("  " + "-" * 66)
for ds in SUBSETS:
    d = summary[ds]
    print(f"  {ds:<8} "
          f"{d['min_ade']['mean']:.3f}±{d['min_ade']['std']:.3f}  "
          f"{d['ece']['mean']:.3f}±{d['ece']['std']:.3f}  "
          f"{d['svr']['mean']:.3f}±{d['svr']['std']:.3f}  "
          f"{d['spsr']['mean']:.3f}±{d['spsr']['std']:.3f}  "
          f"{d['fpr_50cm']['mean']:.3f}")
print("  " + "-" * 66)
ov = overall
print(f"  {'Avg':<8} "
      f"{ov['min_ade']['mean']:.3f}±{ov['min_ade']['std']:.3f}  "
      f"{ov['ece']['mean']:.3f}±{ov['ece']['std']:.3f}  "
      f"{ov['svr']['mean']:.3f}±{ov['svr']['std']:.3f}  "
      f"{ov['spsr']['mean']:.3f}±{ov['spsr']['std']:.3f}  "
      f"0.000")