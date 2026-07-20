"""
fig1_ece_ablation.py  --  CoopCalib-TP  ECCV 2026
==================================================
Fig 1: First ECE baseline in trajectory prediction.

Design philosophy (ECCV / CVPR / NeurIPS standard):
  - One message per figure, readable by any ML reader
  - No hatch patterns (print as grey smudge, inaccessible)
  - No value labels inside bars (too narrow, cluttered)
  - No tier shading (secondary information belongs in caption)
  - Clean colour only: blue V0, muted green V1, muted red V2
  - Single delta annotation per fold (V2 vs V0 only, small)
  - Y-axis starts at 0.40 to show differences clearly
  - Stats box kept minimal

Run from C:\\CoopCalib\\ with venv active:
    python fig1_ece_ablation.py

Output:
    experiments/figures/fig1_ece_ablation.pdf
"""

import os
import sys
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
RESULTS = os.path.join("experiments", "results")
OUT_DIR = os.path.join("experiments", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

PATH_PF    = os.path.join(RESULTS, "metrics_per_fold.json")
PATH_V1    = os.path.join(RESULTS, "v1_metrics.json")
PATH_V2    = os.path.join(RESULTS, "v2_metrics.json")
PATH_STATS = os.path.join(RESULTS, "stats_summary.json")

for p in [PATH_PF, PATH_V1, PATH_V2, PATH_STATS]:
    if not os.path.exists(p):
        sys.exit(f"[ERROR] File not found: {p}\nRun from C:\\CoopCalib\\")

# ---------------------------------------------------------------------------
# KEY NORMALISATION
# ---------------------------------------------------------------------------
FOLDS_DISPLAY = ["ETH", "HOTEL", "ZARA1", "ZARA2", "UNIV"]
KEY           = {"ETH": "eth", "HOTEL": "hotel",
                 "ZARA1": "zara1", "ZARA2": "zara2", "UNIV": "univ"}

# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------
with open(PATH_PF)    as f: raw_pf    = json.load(f)["per_fold"]
with open(PATH_V1)    as f: raw_v1    = json.load(f)
with open(PATH_V2)    as f: raw_v2    = json.load(f)
with open(PATH_STATS) as f: raw_stats = json.load(f)

ece = {
    "V0": np.array([raw_pf[KEY[f]]["ece"] for f in FOLDS_DISPLAY]),
    "V1": np.array([raw_v1[KEY[f]]["ECE"] for f in FOLDS_DISPLAY]),
    "V2": np.array([raw_v2[KEY[f]]["ECE"] for f in FOLDS_DISPLAY]),
}

cohens_d = next(e["cohens_d"] for e in raw_stats["rq1_ece"]
                if "V0 vs V2" in e["label"])
p_val    = next(e["p_value"]  for e in raw_stats["rq1_ece"]
                if "V0 vs V2" in e["label"])

means = {v: float(np.mean(ece[v])) for v in ("V0", "V1", "V2")}

# ---------------------------------------------------------------------------
# CONSOLE AUDIT
# ---------------------------------------------------------------------------
print("\nfig1_ece_ablation.py — DATA AUDIT")
print(f"  {'Fold':<8}  {'V0':>6}  {'V1':>6}  {'V2':>6}  {'V2-V0 Δ%':>9}")
print(f"  {'-'*44}")
for i, f in enumerate(FOLDS_DISPLAY):
    d = (ece["V2"][i] - ece["V0"][i]) / ece["V0"][i] * 100
    print(f"  {f:<8}  {ece['V0'][i]:>6.4f}  {ece['V1'][i]:>6.4f}"
          f"  {ece['V2'][i]:>6.4f}  {d:>+9.2f}%")
print(f"  {'Mean':<8}  {means['V0']:>6.4f}  {means['V1']:>6.4f}"
      f"  {means['V2']:>6.4f}")
print(f"\n  Cohen d={cohens_d:.4f}  p={p_val:.4f}\n")

# ---------------------------------------------------------------------------
# STYLE  — journal-grade minimal
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["DejaVu Serif", "Times New Roman", "Times", "serif"],
    "font.size":         10,
    "axes.labelsize":    10,
    "axes.titlesize":    10,
    "axes.titleweight":  "normal",
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "legend.framealpha": 1.0,
    "legend.edgecolor":  "#aaaaaa",
    "legend.borderpad":  0.6,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "grid.linewidth":    0.5,
    "grid.color":        "#dddddd",
})

# Flat, solid colours — no hatch, no gradient
# Chosen for greyscale distinguishability and colourblind safety
COL = {
    "V0": "#2166ac",   # strong blue   — baseline (hero)
    "V1": "#74add1",   # light blue    — +L_ECE
    "V2": "#d6604d",   # muted red     — +L_ECE +L_energy
}

# ---------------------------------------------------------------------------
# FIGURE
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 3.8))

n      = len(FOLDS_DISPLAY)
bar_w  = 0.24
x      = np.arange(n)

# Offsets: V0 left, V1 centre, V2 right
offsets = {"V0": -bar_w, "V1": 0.0, "V2": bar_w}
labels  = {
    "V0": f"V0  Baseline",
    "V1": r"V1  $+L_\mathrm{ECE}$",
    "V2": r"V2  $+L_\mathrm{ECE}+L_\mathrm{energy}$",
}

for v in ("V0", "V1", "V2"):
    ax.bar(x + offsets[v], ece[v],
           width=bar_w,
           color=COL[v],
           label=labels[v],
           edgecolor="white",
           linewidth=0.4,
           zorder=3)

# ---------------------------------------------------------------------------
# V0 MEAN LINE  — the key reference value
# ---------------------------------------------------------------------------
ax.axhline(means["V0"],
           color=COL["V0"], linewidth=1.0,
           linestyle="--", alpha=0.6, zorder=2)

# Label the mean line at the right edge, outside the bars
ax.annotate(
    f"V0 mean = {means['V0']:.3f}",
    xy=(n - 0.5 + bar_w * 0.5, means["V0"]),
    xytext=(5, 4), textcoords="offset points",
    fontsize=8.5, color=COL["V0"], va="bottom", ha="left")

# ---------------------------------------------------------------------------
# GRID  — horizontal only, behind bars
# ---------------------------------------------------------------------------
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)

# ---------------------------------------------------------------------------
# AXES
# ---------------------------------------------------------------------------
ax.set_xticks(x)
ax.set_xticklabels(FOLDS_DISPLAY)
ax.set_xlim(-0.6, n - 0.4)

# Y axis: start at 0.40 so differences are visible; cap at 0.75
ax.set_ylim(0.40, 0.75)
ax.set_ylabel("ECE  (lower is better)")

# ---------------------------------------------------------------------------
# LEGEND  — top left, clean
# ---------------------------------------------------------------------------
ax.legend(loc="upper left",
          handlelength=1.2,
          handleheight=0.9,
          frameon=True)

# ---------------------------------------------------------------------------
# STATS BOX  — compact, top right
# ---------------------------------------------------------------------------
ax.text(0.99, 0.97,
        f"Cohen's $d$ = {cohens_d:.2f}  (medium)\n"
        f"Wilcoxon $p$ = {p_val:.2f}  ($N$ = 5)",
        transform=ax.transAxes,
        fontsize=8, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.35",
                  facecolor="white", edgecolor="#aaaaaa",
                  linewidth=0.7))

# ---------------------------------------------------------------------------
# TITLE  — informative, single line
# ---------------------------------------------------------------------------
ax.set_title(
    "ECE across ETH-UCY folds: V0 baseline and loss ablations",
    loc="left", pad=6)

fig.tight_layout()

# ---------------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------------
out = os.path.join(OUT_DIR, "fig1_ece_ablation.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"Saved -> {out}")
plt.show()
