"""
fig3_effrank_crossover.py  --  CoopCalib-TP  ECCV 2026
=======================================================
Fig 3: Dynamic KL prior — latent diversity and accuracy vs training size.
Minimal: two clean lines per panel, legend, nothing else.

Run from C:\\CoopCalib\\ with venv active:
    python fig3_effrank_crossover.py

Output:
    experiments/figures/fig3_effrank_crossover.pdf
"""

import os, sys, json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

RESULTS = os.path.join("experiments", "results")
OUT_DIR = os.path.join("experiments", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

PATH_ER = os.path.join(RESULTS, "socialvae", "eff_rank_summary.json")
if not os.path.exists(PATH_ER):
    sys.exit(f"[ERROR] Not found: {PATH_ER}")

# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------
with open(PATH_ER) as f:
    raw = json.load(f)

runs   = raw["runs"]
n_vals = sorted({r["n_train"] for r in runs})

sv = {}
for n in n_vals:
    v0 = [r for r in runs if r["lam3"] == 0.0 and r["n_train"] == n]
    v3 = [r for r in runs if r["lam3"] != 0.0 and r["n_train"] == n]
    if v0 and v3:
        sv[n] = {"v0_rank": v0[0]["eff_rank"], "v3_rank": v3[0]["eff_rank"],
                 "v0_ade":  v0[0]["ade"],       "v3_ade":  v3[0]["ade"]}

x       = np.array(n_vals)
v0_rank = np.array([sv[n]["v0_rank"] for n in n_vals])
v3_rank = np.array([sv[n]["v3_rank"] for n in n_vals])
v0_ade  = np.array([sv[n]["v0_ade"]  for n in n_vals])
v3_ade  = np.array([sv[n]["v3_ade"]  for n in n_vals])

print("\nDATA AUDIT")
print(f"  {'N':>6}  {'V0 rank':>9}  {'V3 rank':>9}  {'V0 ADE':>8}  {'V3 ADE':>8}")
for n in n_vals:
    s = sv[n]
    print(f"  {n:>6}  {s['v0_rank']:>9.4f}  {s['v3_rank']:>9.4f}"
          f"  {s['v0_ade']:>8.2f}  {s['v3_ade']:>8.2f}")
print()

# ---------------------------------------------------------------------------
# STYLE
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["DejaVu Serif", "Times New Roman", "Times", "serif"],
    "font.size": 10, "axes.labelsize": 9, "axes.titlesize": 10,
    "axes.titleweight": "normal",
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8.5, "legend.framealpha": 1.0,
    "legend.edgecolor": "#aaaaaa",
    "figure.dpi": 300, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.major.size": 3, "ytick.major.size": 3,
    "grid.color": "#dddddd", "grid.linewidth": 0.5,
    "lines.linewidth": 2.0, "lines.markersize": 7,
})

BLUE = "#2166ac"
RED  = "#d6604d"

# ---------------------------------------------------------------------------
# FIGURE
# ---------------------------------------------------------------------------
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.5, 3.8))
fig.subplots_adjust(wspace=0.35)

for ax, y0, y3, ylabel, title in [
    (ax_a, v0_rank, v3_rank,
     "Latent space diversity (effective rank)",
     "(a)  Latent space diversity"),
    (ax_b, v0_ade, v3_ade,
     "Prediction error — minADE (m)\n(lower is better)",
     "(b)  Trajectory prediction error"),
]:
    ax.plot(x, y0, "o-",  color=BLUE, label="No modification")
    ax.plot(x, y3, "s--", color=RED,  label="With dynamic KL prior")

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in x])
    ax.set_xlabel("Training trajectories", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    ax.set_title(title, pad=6)

fig.suptitle(
    "Dynamic KL prior constrains latent diversity at low data,"
    " matches baseline at full data",
    fontsize=10, y=1.02)

fig.tight_layout()

out = os.path.join(OUT_DIR, "fig3_effrank_crossover.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"Saved -> {out}")
plt.show()
