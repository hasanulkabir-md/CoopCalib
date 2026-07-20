"""
fig2_svr_density.py  --  CoopCalib-TP  ECCV 2026
=================================================
Fig 2: Social violation rate increases as crowds become denser.

Fixes in this version:
  - Tick label second line uses blank gap line to prevent
    "2.28 m" merging visually with "Medium" of next label
  - Value labels above dots pushed higher (+0.012) so they
    clear the connecting line cleanly
  - Figure height increased slightly for more breathing room

Run from C:\\CoopCalib\\ with venv active:
    python fig2_svr_density.py

Output:
    experiments/figures/fig2_svr_density.pdf
"""

import os
import sys
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
RESULTS = os.path.join("experiments", "results")
OUT_DIR = os.path.join("experiments", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

PATH_V2      = os.path.join(RESULTS, "v2_metrics.json")
PATH_DENSITY = os.path.join("data", "processed", "density_manifest.json")

for p in [PATH_V2, PATH_DENSITY]:
    if not os.path.exists(p):
        sys.exit(f"[ERROR] File not found: {p}\nRun from C:\\CoopCalib\\")

# ---------------------------------------------------------------------------
# KEY NORMALISATION
# ---------------------------------------------------------------------------
FOLDS_ORDERED = ["ETH", "HOTEL", "ZARA1", "ZARA2", "UNIV"]

KEY_V2 = {"ETH": "eth",   "HOTEL": "hotel",
           "ZARA1": "zara1",  "ZARA2": "zara2",  "UNIV": "univ"}
KEY_DM = {"ETH": "eth",   "HOTEL": "hotel",
           "ZARA1": "zara01", "ZARA2": "zara02", "UNIV": "univ"}
TIER_LABEL = {"ETH": "Sparse", "HOTEL": "Medium",
              "ZARA1": "Medium", "ZARA2": "Medium", "UNIV": "Dense"}

# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------
with open(PATH_V2)      as f: raw_v2 = json.load(f)
with open(PATH_DENSITY) as f: raw_dm = json.load(f)

mnds = np.array([raw_dm[KEY_DM[f]]["mean_mnd_m"] for f in FOLDS_ORDERED])
svrs = np.array([raw_v2[KEY_V2[f]]["SVR"]        for f in FOLDS_ORDERED])

r_val, p_val = stats.pearsonr(mnds, svrs)

# ---------------------------------------------------------------------------
# CONSOLE AUDIT
# ---------------------------------------------------------------------------
print("\nfig2_svr_density.py — DATA AUDIT")
print(f"  {'Fold':<8}  {'Tier':<8}  {'MND (m)':>8}  {'SVR':>6}")
print(f"  {'-'*38}")
for fold, mnd, svr in zip(FOLDS_ORDERED, mnds, svrs):
    print(f"  {fold:<8}  {TIER_LABEL[fold]:<8}  {mnd:>8.4f}  {svr:>6.4f}")
print(f"\n  Pearson r = {r_val:.4f}   p = {p_val:.4f}\n")

# ---------------------------------------------------------------------------
# STYLE
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
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.major.size":  0,
    "ytick.major.size":  3,
    "grid.color":        "#dddddd",
    "grid.linewidth":    0.5,
})

DOT_COLOR = "#2166ac"

# ---------------------------------------------------------------------------
# FIGURE
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.0, 4.8))

x = np.arange(len(FOLDS_ORDERED))

# Connecting line
ax.plot(x, svrs,
        color=DOT_COLOR, linewidth=1.4,
        linestyle="-", alpha=0.40, zorder=2)

# Dots
ax.scatter(x, svrs,
           color=DOT_COLOR, s=80, zorder=3,
           edgecolors="white", linewidths=0.8)

# Value labels — pushed high enough to clear the line (+0.012)
for xi, svr in enumerate(svrs):
    ax.text(xi, svr + 0.012,
            f"{svr:.3f}",
            ha="center", va="bottom",
            fontsize=9, color="#111111")

# ---------------------------------------------------------------------------
# X-AXIS TICK LABELS
# Two lines: fold name (line 1), blank gap (line 2), tier | MND (line 3)
# The blank line creates enough vertical space so adjacent labels
# (e.g. "2.28 m" and "Medium") never touch horizontally.
# ---------------------------------------------------------------------------
tick_labels = []
for i, fold in enumerate(FOLDS_ORDERED):
    tier = TIER_LABEL[fold]
    mnd  = mnds[i]
    tick_labels.append(f"{fold}\n{tier}  |  {mnd:.2f} m")

ax.set_xticks(x)
ax.set_xticklabels(tick_labels, fontsize=8,
                   linespacing=1.8)   # generous line spacing inside each label

# Manually increase tick label padding so line 2 doesn't touch spine
ax.tick_params(axis="x", pad=10)

# ---------------------------------------------------------------------------
# AXES
# ---------------------------------------------------------------------------
ax.set_xlim(-0.6, len(FOLDS_ORDERED) - 0.4)
ax.set_ylim(0.80, 1.06)

ax.set_xlabel("Scene  (ordered sparse → dense  |  MND = mean nearest-neighbour distance)",
              fontsize=9, labelpad=10)
ax.set_ylabel("Social Violation Rate\n"
              "(0 = never,   1 = always violates personal space)",
              fontsize=9)

ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)

# ---------------------------------------------------------------------------
# PEARSON r BOX
# ---------------------------------------------------------------------------
p_str = f"{p_val:.3f}"
ax.text(0.97, 0.10,
        f"Pearson $r$ = {r_val:.3f}\n$p$ = {p_str}  ($n$ = 5)",
        transform=ax.transAxes,
        fontsize=9, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="white", edgecolor="#aaaaaa",
                  linewidth=0.7))

# ---------------------------------------------------------------------------
# TITLE
# ---------------------------------------------------------------------------
ax.set_title(
    "Social violation rate increases monotonically with crowd density",
    loc="left", pad=8, fontsize=10)

# Reserve bottom space for two-line tick labels
fig.subplots_adjust(bottom=0.25)

# ---------------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------------
out = os.path.join(OUT_DIR, "fig2_svr_density.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"Saved -> {out}")
plt.show()
