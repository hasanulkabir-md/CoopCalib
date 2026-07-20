"""
generate_figures.py
Generates fig2_svr_density.pdf and fig_supp_spsr.pdf
from verified 3-seed Session 16 results.
All numbers hardcoded from v0_multiseed_summary.json
and v2r_multiseed_summary.json — no file reads needed.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = r"C:\CoopCalib\paper"

# ── Verified 3-seed numbers ─────────────────────────────────────────────────
# V0 SVR per subset (mean, std) — from v0_multiseed_summary.json
v0_svr = {
    "ETH":   (0.8384, 0.0028),
    "HOTEL": (0.9437, 0.0001),
    "UNIV":  (0.9893, 0.0001),
    "ZARA1": (0.8801, 0.0006),
    "ZARA2": (0.9613, 0.0002),
}
# V2R SVR per subset — from v2r_multiseed_summary.json
v2r_svr = {
    "ETH":   (0.8095, 0.0010),
    "HOTEL": (0.9370, 0.0003),
    "UNIV":  (0.9885, 0.0000),
    "ZARA1": (0.8699, 0.0023),
    "ZARA2": (0.9573, 0.0007),
}
# V0 SPSR per subset
v0_spsr = {
    "ETH":   (0.8178, 0.0091),
    "HOTEL": (0.9195, 0.0008),
    "UNIV":  (0.6603, 0.0008),
    "ZARA1": (0.9288, 0.0016),
    "ZARA2": (0.7694, 0.0002),
}
# V2R SPSR per subset
v2r_spsr = {
    "ETH":   (0.7894, 0.0111),
    "HOTEL": (0.9632, 0.0030),
    "UNIV":  (0.6855, 0.0151),
    "ZARA1": (0.9288, 0.0016),
    "ZARA2": (0.8599, 0.0270),
}

TIERS = {
    "ETH": "Sparse", "HOTEL": "Medium", "UNIV": "Dense",
    "ZARA1": "Medium", "ZARA2": "Medium"
}
TIER_LABELS = ["ETH\n(Sparse)", "HOTEL\n(Medium)", "UNIV\n(Dense)",
               "ZARA1\n(Medium)", "ZARA2\n(Medium)"]
SUBSETS = ["ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2"]

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 300,
})

# ── FIGURE 2: SVR by subset ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(SUBSETS))
w = 0.35

v0_m  = [v0_svr[s][0]  for s in SUBSETS]
v0_e  = [v0_svr[s][1]  for s in SUBSETS]
v2r_m = [v2r_svr[s][0] for s in SUBSETS]
v2r_e = [v2r_svr[s][1] for s in SUBSETS]

bars0 = ax.bar(x - w/2, v0_m, w, yerr=v0_e, label='V0 (baseline)',
               color='steelblue', alpha=0.85, capsize=4, error_kw={'linewidth':1.5})
bars2 = ax.bar(x + w/2, v2r_m, w, yerr=v2r_e, label='V2R (ranking-aware)',
               color='coral', alpha=0.85, capsize=4, error_kw={'linewidth':1.5})

# Annotate delta
for i, s in enumerate(SUBSETS):
    delta = v2r_svr[s][0] - v0_svr[s][0]
    ax.text(x[i], max(v0_svr[s][0], v2r_svr[s][0]) + 0.004,
            f'{delta:+.3f}', ha='center', va='bottom', fontsize=8.5,
            color='darkred' if delta > 0 else 'darkgreen')

ax.set_xticks(x)
ax.set_xticklabels(TIER_LABELS)
ax.set_ylabel('Social Violation Rate (SVR) ↓')
ax.set_ylim(0.75, 1.03)
ax.legend(loc='lower right', fontsize=10)
ax.set_title('SVR by Subset and Density Tier\n(3-seed mean ± std; seeds 42, 1, 123)')
ax.axhline(1.0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
ax.grid(axis='y', alpha=0.3)

# Shade by tier
tier_colors = {"Sparse": '#e8f4f8', "Medium": '#f0f8e8', "Dense": '#f8ede8'}
for i, s in enumerate(SUBSETS):
    ax.axvspan(i-0.5, i+0.5, alpha=0.15,
               color={'Sparse':'blue','Medium':'green','Dense':'red'}[TIERS[s]],
               zorder=0)

fig.tight_layout()
out2 = os.path.join(OUT, "fig2_svr_density.pdf")
fig.savefig(out2, bbox_inches='tight')
print(f"Saved: {out2}")
plt.close()

# ── FIGURE 3: SPSR by subset ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))

v0_m  = [v0_spsr[s][0]  for s in SUBSETS]
v0_e  = [v0_spsr[s][1]  for s in SUBSETS]
v2r_m = [v2r_spsr[s][0] for s in SUBSETS]
v2r_e = [v2r_spsr[s][1] for s in SUBSETS]

ax.bar(x - w/2, v0_m, w, yerr=v0_e, label='V0 (baseline)',
       color='steelblue', alpha=0.85, capsize=4, error_kw={'linewidth':1.5})
ax.bar(x + w/2, v2r_m, w, yerr=v2r_e, label='V2R (ranking-aware)',
       color='coral', alpha=0.85, capsize=4, error_kw={'linewidth':1.5})

for i, s in enumerate(SUBSETS):
    delta = v2r_spsr[s][0] - v0_spsr[s][0]
    ax.text(x[i], max(v0_spsr[s][0], v2r_spsr[s][0]) + 0.008,
            f'{delta:+.3f}', ha='center', va='bottom', fontsize=8.5,
            color='darkgreen' if delta > 0 else 'darkred')

ax.set_xticks(x)
ax.set_xticklabels(TIER_LABELS)
ax.set_ylabel('Safe Planning Success Rate (SPSR) ↑')
ax.set_ylim(0.55, 1.06)
ax.legend(loc='lower right', fontsize=10)
ax.set_title('SPSR by Subset and Density Tier\n(3-seed mean ± std; seeds 42, 1, 123)')
ax.grid(axis='y', alpha=0.3)

for i, s in enumerate(SUBSETS):
    ax.axvspan(i-0.5, i+0.5, alpha=0.15,
               color={'Sparse':'blue','Medium':'green','Dense':'red'}[TIERS[s]],
               zorder=0)

fig.tight_layout()
out3 = os.path.join(OUT, "fig_supp_spsr.pdf")
fig.savefig(out3, bbox_inches='tight')
print(f"Saved: {out3}")
plt.close()

print("\nBoth figures saved. Copy to C:\\CoopCalib\\paper\\ before compiling.")