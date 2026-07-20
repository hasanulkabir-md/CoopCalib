"""
fig_supp_spsr.py
CoopCalib-TP — Supplementary Figure: SPSR per fold, V0 vs V2

Message: Safe Planning Success Rate is density-dependent.
         Sparse scenes are hardest; medium scenes easiest.
         The pattern is consistent across both variants.

Design standard (matches approved Figs 1-4):
  - Connected dot plot (same type as Fig 2)
  - No hatch patterns, solid colour only
  - No value labels on points
  - Y-axis starts at meaningful value (not 0)
  - Density tier shading in background
  - Plain-English legend and labels
  - 300 dpi PDF output
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import os

# ── Data (locked from experiments/results/baseline.json + v2_metrics.json) ────

# Folds ordered Sparse → Medium → Dense to tell the density story
folds  = ['ETH\n(Sparse)', 'HOTEL\n(Medium)', 'ZARA1\n(Medium)',
          'ZARA2\n(Medium)', 'UNIV\n(Dense)']
tiers  = ['Sparse',        'Medium',           'Medium',
          'Medium',         'Dense']

spsr_v0 = [0.3269, 0.9265, 0.8570, 0.8164, 0.6674]
spsr_v2 = [0.3187, 0.9198, 0.8544, 0.8078, 0.6560]

x = np.arange(len(folds))

# ── Style ─────────────────────────────────────────────────────────────────────

C_V0   = '#3266AD'   # blue  — baseline
C_V2   = '#2A9D5C'   # green — +Calibration & social loss
C_TIER = {
    'Sparse': '#F5F0E8',
    'Medium': '#EAF3F0',
    'Dense':  '#EAF0F8',
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size':   10,
    'axes.linewidth':    0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

fig, ax = plt.subplots(figsize=(7, 4.5))

# ── Density tier background shading ───────────────────────────────────────────
# Tier boundaries: ETH (0), HOTEL-ZARA1-ZARA2 (1-3), UNIV (4)
tier_spans = [
    (folds[0],  -0.5, 0.5,  'Sparse'),
    (folds[1],   0.5, 3.5,  'Medium'),
    (folds[4],   3.5, 4.5,  'Dense'),
]
for _, xlo, xhi, tier in tier_spans:
    ax.axvspan(xlo, xhi, color=C_TIER[tier], alpha=0.6, zorder=0)
    ax.text((xlo + xhi) / 2, 0.975, tier,
            transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=8,
            color='#888888', style='italic')

# ── Lines and dots ────────────────────────────────────────────────────────────

dot_kw = dict(zorder=5, clip_on=False)

ax.plot(x, spsr_v0, color=C_V0, linewidth=1.6, zorder=4)
ax.scatter(x, spsr_v0, color=C_V0, s=60, marker='o', **dot_kw,
           edgecolors='white', linewidths=0.8)

ax.plot(x, spsr_v2, color=C_V2, linewidth=1.6, zorder=4, linestyle='--')
ax.scatter(x, spsr_v2, color=C_V2, s=60, marker='s', **dot_kw,
           edgecolors='white', linewidths=0.8)

# ── Axes ──────────────────────────────────────────────────────────────────────

ax.set_xticks(x)
ax.set_xticklabels(folds, fontsize=9.5)
ax.set_xlim(-0.5, len(folds) - 0.5)

ax.set_ylabel('Safe Planning Success Rate (SPSR)', fontsize=10)
ax.set_ylim(0.25, 1.02)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax.grid(axis='y', linewidth=0.4, color='#e8e8e8', zorder=1)
ax.tick_params(axis='y', which='minor', length=2)

# ── Legend ────────────────────────────────────────────────────────────────────

h_v0 = mpatches.Patch(color=C_V0, label='Baseline')
h_v2 = mpatches.Patch(color=C_V2, label='+Calibration & social loss')

ax.legend(
    handles=[h_v0, h_v2],
    loc='lower right',
    frameon=False,
    fontsize=9,
)

# ── Title ─────────────────────────────────────────────────────────────────────

ax.set_title(
    'Safe planning success rate is density-dependent\n'
    'and consistent across variants',
    fontsize=10, pad=10,
)

plt.tight_layout()

out_dir = os.path.join('experiments', 'figures')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'fig_supp_spsr.pdf')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.close()

# ── Console summary ───────────────────────────────────────────────────────────

print("\nSPSR values:")
print(f"{'Fold':<10} {'V0':>8} {'V2':>8} {'Delta':>8}")
for fold, v0, v2 in zip(['ETH','HOTEL','ZARA1','ZARA2','UNIV'],
                         spsr_v0, spsr_v2):
    print(f"{fold:<10} {v0:>8.4f} {v2:>8.4f} {v2-v0:>+8.4f}")

print(f"\nV0 range: {min(spsr_v0):.4f} – {max(spsr_v0):.4f}")
print(f"V2 range: {min(spsr_v2):.4f} – {max(spsr_v2):.4f}")
print(f"Max |delta|: {max(abs(v2-v0) for v0,v2 in zip(spsr_v0,spsr_v2)):.4f}")
