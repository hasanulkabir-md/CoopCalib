"""
compile_v2r_multiseed.py
Computes 3-seed mean +/- std for V2R and produces V0 vs V2R comparison table.
"""
import json, numpy as np, os

BASE = r"C:\CoopCalib\experiments\results"

V0_FILES  = {
    "seed42":  "v0_metrics_corrected.json",
    "seed1":   "v0_seed1_metrics.json",
    "seed123": "v0_seed123_metrics.json",
}
V2R_FILES = {
    "seed42":  "v2r_metrics_corrected.json",
    "seed1":   "v2r_seed1_metrics.json",
    "seed123": "v2r_seed123_metrics.json",
}

SUBSETS  = ["eth", "hotel", "univ", "zara1", "zara2"]
METRICS  = ["ece", "fpr_50cm", "svr", "spsr", "min_ade", "min_fde"]

def load_all(file_dict):
    data = {}
    for k, fname in file_dict.items():
        with open(os.path.join(BASE, fname)) as f:
            data[k] = json.load(f)
    return data

def compute_stats(data, seeds=("seed42","seed1","seed123")):
    stats = {}
    for ds in SUBSETS:
        stats[ds] = {}
        for m in METRICS:
            vals = [data[s]["per_fold"][ds][m] for s in seeds]
            stats[ds][m] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals)),
                "vals": [round(float(v),4) for v in vals]
            }
    # Overall
    stats["_overall"] = {}
    for m in METRICS:
        seed_avgs = [np.mean([data[s]["per_fold"][ds][m]
                              for ds in SUBSETS]) for s in seeds]
        stats["_overall"][m] = {
            "mean": float(np.mean(seed_avgs)),
            "std":  float(np.std(seed_avgs)),
            "seed_avgs": [round(float(v),4) for v in seed_avgs]
        }
    return stats

v0_data  = load_all(V0_FILES)
v2r_data = load_all(V2R_FILES)
v0_stats  = compute_stats(v0_data)
v2r_stats = compute_stats(v2r_data)

# Save V2R summary JSON
out = {"variant": "V2R", "seeds": [42,1,123], "n_seeds": 3,
       "per_subset": {ds: v2r_stats[ds] for ds in SUBSETS},
       "overall": v2r_stats["_overall"]}
out_path = os.path.join(BASE, "v2r_multiseed_summary.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved: {out_path}")

# ── FINAL COMPARISON TABLE ───────────────────────────────────────────────────
print()
print("=" * 72)
print("  DEFINITIVE V0 vs V2R — 3-Seed Mean ± Std")
print("  (Seeds 42, 1, 123 for both variants)")
print("=" * 72)

header = f"  {'Subset':<7} {'Tier':<8}"
for v in ["V0","V2R","Delta"]:
    header += f"  {'SVR('+v+')':>12}"
print(header)
print("  " + "-"*60)

TIERS = {"eth":"sparse","hotel":"medium","univ":"dense",
         "zara1":"medium","zara2":"medium"}

for ds in SUBSETS:
    v0m  = v0_stats[ds]["svr"]["mean"]
    v0s  = v0_stats[ds]["svr"]["std"]
    v2rm = v2r_stats[ds]["svr"]["mean"]
    v2rs = v2r_stats[ds]["svr"]["std"]
    delta = v2rm - v0m
    print(f"  {ds:<7} {TIERS[ds]:<8}  "
          f"{v0m:.4f}±{v0s:.4f}  {v2rm:.4f}±{v2rs:.4f}  {delta:+.4f}")

v0o  = v0_stats["_overall"]["svr"]
v2ro = v2r_stats["_overall"]["svr"]
delta_o = v2ro["mean"] - v0o["mean"]
print("  " + "-"*60)
print(f"  {'Avg':<7} {'':8}  "
      f"{v0o['mean']:.4f}±{v0o['std']:.4f}  "
      f"{v2ro['mean']:.4f}±{v2ro['std']:.4f}  {delta_o:+.4f}")

print()
print("=" * 72)
header2 = f"  {'Subset':<7} {'Tier':<8}"
for v in ["V0","V2R","Delta"]:
    header2 += f"  {'SPSR('+v+')':>12}"
print(header2)
print("  " + "-"*60)

for ds in SUBSETS:
    v0m  = v0_stats[ds]["spsr"]["mean"]
    v0s  = v0_stats[ds]["spsr"]["std"]
    v2rm = v2r_stats[ds]["spsr"]["mean"]
    v2rs = v2r_stats[ds]["spsr"]["std"]
    delta = v2rm - v0m
    print(f"  {ds:<7} {TIERS[ds]:<8}  "
          f"{v0m:.4f}±{v0s:.4f}  {v2rm:.4f}±{v2rs:.4f}  {delta:+.4f}")

v0o  = v0_stats["_overall"]["spsr"]
v2ro = v2r_stats["_overall"]["spsr"]
delta_o = v2ro["mean"] - v0o["mean"]
print("  " + "-"*60)
print(f"  {'Avg':<7} {'':8}  "
      f"{v0o['mean']:.4f}±{v0o['std']:.4f}  "
      f"{v2ro['mean']:.4f}±{v2ro['std']:.4f}  {delta_o:+.4f}")

print()
print("=" * 72)
print("  ADE COMPARISON")
print("=" * 72)
header3 = f"  {'Subset':<7} {'Tier':<8}"
for v in ["V0","V2R","Delta"]:
    header3 += f"  {'ADE('+v+')':>12}"
print(header3)
print("  " + "-"*60)
for ds in SUBSETS:
    v0m  = v0_stats[ds]["min_ade"]["mean"]
    v0s  = v0_stats[ds]["min_ade"]["std"]
    v2rm = v2r_stats[ds]["min_ade"]["mean"]
    v2rs = v2r_stats[ds]["min_ade"]["std"]
    delta = v2rm - v0m
    sig = "**" if abs(delta) > 3*max(v0s, v2rs, 1e-6) else ""
    print(f"  {ds:<7} {TIERS[ds]:<8}  "
          f"{v0m:.4f}±{v0s:.4f}  {v2rm:.4f}±{v2rs:.4f}  {delta:+.4f} {sig}")

v0o  = v0_stats["_overall"]["min_ade"]
v2ro = v2r_stats["_overall"]["min_ade"]
delta_o = v2ro["mean"] - v0o["mean"]
print("  " + "-"*60)
print(f"  {'Avg':<7} {'':8}  "
      f"{v0o['mean']:.4f}±{v0o['std']:.4f}  "
      f"{v2ro['mean']:.4f}±{v2ro['std']:.4f}  {delta_o:+.4f}")

print()
print("=" * 72)
print("  STATISTICAL SIGNIFICANCE CHECK")
print("  Rule: |delta| > 3 * max(std_V0, std_V2R) = practically significant")
print("=" * 72)
for metric_label, metric_key in [("SVR","svr"),("SPSR","spsr"),("ADE","min_ade")]:
    v0m  = v0_stats["_overall"][metric_key]["mean"]
    v2rm = v2r_stats["_overall"][metric_key]["mean"]
    v0s  = v0_stats["_overall"][metric_key]["std"]
    v2rs = v2r_stats["_overall"][metric_key]["std"]
    delta = v2rm - v0m
    threshold = 3 * max(v0s, v2rs, 1e-6)
    sig = "SIGNIFICANT" if abs(delta) > threshold else "NOT significant"
    print(f"  {metric_label:<6}: delta={delta:+.4f}  "
          f"threshold=±{threshold:.4f}  → {sig}")