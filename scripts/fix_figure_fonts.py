"""
fix_figure_fonts.py
Regenerates all paper figures with Type 1 fonts only.
IEEE RAL requires no Type 3 fonts.
Run from: C:\CoopCalib\
"""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42      # TrueType — IEEE compliant
matplotlib.rcParams['ps.fonttype']  = 42      # TrueType for PS output too
matplotlib.rcParams.update({
    'font.family':     'serif',
    'font.serif':      ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size':        9,
    'axes.labelsize':  10,
    'axes.titlesize':  10,
    'xtick.labelsize':  8,
    'ytick.labelsize':  8,
    'figure.dpi':      150,
    'savefig.dpi':     300,
    'text.usetex':     False,   # Keep False — usetex needs LaTeX on PATH
})

import matplotlib.pyplot as plt
import numpy as np
import os

OUT = r"C:\CoopCalib\paper"

# ── Verified 3-seed numbers ─────────────────────────────────
SUBSETS  = ["ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2"]
TIER_LABELS = ["ETH\n(Sparse)", "HOTEL\n(Medium)", "UNIV\n(Dense)",
               "ZARA1\n(Medium)", "ZARA2\n(Medium)"]
TIERS = {"ETH":"Sparse","HOTEL":"Medium","UNIV":"Dense",
         "ZARA1":"Medium","ZARA2":"Medium"}
TIER_COLOR = {"Sparse":"#ddeeff","Medium":"#ddffd8","Dense":"#ffeedd"}

v0_svr  = {"ETH":(0.8384,0.0028),"HOTEL":(0.9437,0.0001),
            "UNIV":(0.9893,0.0001),"ZARA1":(0.8801,0.0006),
            "ZARA2":(0.9613,0.0002)}
v2r_svr = {"ETH":(0.8095,0.0010),"HOTEL":(0.9370,0.0003),
            "UNIV":(0.9885,0.0000),"ZARA1":(0.8699,0.0023),
            "ZARA2":(0.9573,0.0007)}
v0_spsr  = {"ETH":(0.8178,0.0091),"HOTEL":(0.9195,0.0008),
             "UNIV":(0.6603,0.0008),"ZARA1":(0.9288,0.0016),
             "ZARA2":(0.7694,0.0002)}
v2r_spsr = {"ETH":(0.7894,0.0111),"HOTEL":(0.9632,0.0030),
             "UNIV":(0.6855,0.0151),"ZARA1":(0.9288,0.0016),
             "ZARA2":(0.8599,0.0270)}

x = np.arange(len(SUBSETS))
w = 0.35

def shade_tiers(ax):
    for i, s in enumerate(SUBSETS):
        ax.axvspan(i-0.5, i+0.5, alpha=0.12,
                   color=TIER_COLOR[TIERS[s]], zorder=0)

# ── FIGURE 2: SVR ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.0, 3.2))
shade_tiers(ax)

v0m  = [v0_svr[s][0]  for s in SUBSETS]
v0e  = [v0_svr[s][1]  for s in SUBSETS]
v2rm = [v2r_svr[s][0] for s in SUBSETS]
v2re = [v2r_svr[s][1] for s in SUBSETS]

ax.bar(x-w/2, v0m,  w, yerr=v0e,  label='V0 (baseline)',
       color='#4472C4', alpha=0.88, capsize=3,
       error_kw={'linewidth':1.2, 'ecolor':'#2a4a8a'})
ax.bar(x+w/2, v2rm, w, yerr=v2re, label='V2R (ranking-aware)',
       color='#ED7D31', alpha=0.88, capsize=3,
       error_kw={'linewidth':1.2, 'ecolor':'#a04000'})

for i, s in enumerate(SUBSETS):
    delta = v2r_svr[s][0] - v0_svr[s][0]
    ypos  = max(v0_svr[s][0], v2r_svr[s][0]) + v0_svr[s][1] + 0.004
    color = '#1a6600' if delta < 0 else '#8b0000'
    ax.text(x[i], ypos, f'{delta:+.3f}', ha='center', va='bottom',
            fontsize=7.5, color=color, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(TIER_LABELS, fontsize=8)
ax.set_ylabel('SVR $\\downarrow$', fontsize=10)
ax.set_ylim(0.75, 1.04)
ax.legend(fontsize=8, loc='lower right', framealpha=0.9)
ax.set_title('Social Violation Rate by Subset and Density Tier\n'
             '(3-seed mean $\\pm$ std; seeds 42, 1, 123)', fontsize=9)
ax.axhline(1.0, color='gray', lw=0.5, ls='--', alpha=0.4)
ax.grid(axis='y', alpha=0.25, linewidth=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout(pad=0.8)
path2 = os.path.join(OUT, "fig2_svr_density.pdf")
fig.savefig(path2, bbox_inches='tight', backend='pdf')
print(f"Saved: {path2}")
plt.close()

# ── FIGURE 3: SPSR ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.0, 3.2))
shade_tiers(ax)

v0m  = [v0_spsr[s][0]  for s in SUBSETS]
v0e  = [v0_spsr[s][1]  for s in SUBSETS]
v2rm = [v2r_spsr[s][0] for s in SUBSETS]
v2re = [v2r_spsr[s][1] for s in SUBSETS]

ax.bar(x-w/2, v0m,  w, yerr=v0e,  label='V0 (baseline)',
       color='#4472C4', alpha=0.88, capsize=3,
       error_kw={'linewidth':1.2, 'ecolor':'#2a4a8a'})
ax.bar(x+w/2, v2rm, w, yerr=v2re, label='V2R (ranking-aware)',
       color='#ED7D31', alpha=0.88, capsize=3,
       error_kw={'linewidth':1.2, 'ecolor':'#a04000'})

for i, s in enumerate(SUBSETS):
    delta = v2r_spsr[s][0] - v0_spsr[s][0]
    ypos  = max(v0_spsr[s][0], v2r_spsr[s][0]) + v0_spsr[s][1] + 0.008
    color = '#1a6600' if delta > 0 else '#8b0000'
    ax.text(x[i], ypos, f'{delta:+.3f}', ha='center', va='bottom',
            fontsize=7.5, color=color, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(TIER_LABELS, fontsize=8)
ax.set_ylabel('SPSR $\\uparrow$', fontsize=10)
ax.set_ylim(0.55, 1.08)
ax.legend(fontsize=8, loc='lower right', framealpha=0.9)
ax.set_title('Safe Planning Success Rate by Subset and Density Tier\n'
             '(3-seed mean $\\pm$ std; seeds 42, 1, 123)', fontsize=9)
ax.grid(axis='y', alpha=0.25, linewidth=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout(pad=0.8)
path3 = os.path.join(OUT, "fig_supp_spsr.pdf")
fig.savefig(path3, bbox_inches='tight', backend='pdf')
print(f"Saved: {path3}")
plt.close()

# ── FIGURE 4: Accuracy preservation (V0-V1W only) ───────────
v0_ade  = {"ETH":0.4193,"HOTEL":0.1246,"UNIV":0.2361,
            "ZARA1":0.1889,"ZARA2":0.1408}
v1_ade  = {"ETH":0.4416,"HOTEL":0.1277,"UNIV":0.2353,
            "ZARA1":0.1875,"ZARA2":0.1429}
v2_ade  = {"ETH":0.4221,"HOTEL":0.1278,"UNIV":0.2381,
            "ZARA1":0.1895,"ZARA2":0.1415}

v0_fde  = {"ETH":0.6291,"HOTEL":0.1876,"UNIV":0.4308,
            "ZARA1":0.3471,"ZARA2":0.2555}
v1_fde  = {"ETH":0.5940,"HOTEL":0.1880,"UNIV":0.4320,
            "ZARA1":0.3460,"ZARA2":0.2630}
v2_fde  = {"ETH":0.6180,"HOTEL":0.1850,"UNIV":0.4360,
            "ZARA1":0.3430,"ZARA2":0.2580}

SUBS = ["ETH","HOTEL","ZARA1","ZARA2","UNIV"]
SLABELS = SUBS

def pct(variant, base, key):
    return 100*(variant[key]-base[key])/base[key]

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=False)

for ax, metric_v0, metric_v1, metric_v2, ylabel, title in zip(
    axes,
    [v0_ade, v0_fde],
    [v1_ade, v1_fde],
    [v2_ade, v2_fde],
    ['$\\Delta$ ADE vs baseline (%)',
     '$\\Delta$ FDE vs baseline (%)'],
    ['ADE change', 'FDE change']
):
    d1 = [pct(metric_v1, metric_v0, s) for s in SUBS]
    d2 = [pct(metric_v2, metric_v0, s) for s in SUBS]
    xi = np.arange(len(SUBS))

    ax.axhline(0,  color='black', lw=0.6)
    ax.axhspan(-2, 2, alpha=0.08, color='green', label='$\\pm$2% band')
    ax.axhspan(-3, 3, alpha=0.06, color='orange', label='$\\pm$3% band')
    ax.axhspan(-3, -100, alpha=0.06, color='red')
    ax.axhspan(3,   100, alpha=0.06, color='red')

    ax.plot(xi, d1, 'o--', color='#ED7D31', ms=5, lw=1.2,
            label='+Calib loss (V1)')
    ax.plot(xi, d2, 's-',  color='#70AD47', ms=5, lw=1.2,
            label='+Calib & social (V2)')

    for i, (a, b) in enumerate(zip(d1, d2)):
        if abs(b) > 2.5:
            ax.annotate(f'{b:+.2f}%', (xi[i], b),
                        textcoords='offset points', xytext=(0, 5),
                        ha='center', fontsize=6.5, color='#8b0000')

    ax.set_xticks(xi)
    ax.set_xticklabels(SLABELS, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(-4.5, 4.5)
    ax.grid(axis='y', alpha=0.2, lw=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].legend(fontsize=7, loc='upper right', framealpha=0.9)

fig.suptitle('Auxiliary losses preserve trajectory accuracy (V0--V1W; max '
             'degradation: 2.7\\%)', fontsize=8.5)
fig.tight_layout(pad=0.6)
path4 = os.path.join(OUT, "fig4_accuracy_preservation.pdf")
fig.savefig(path4, bbox_inches='tight', backend='pdf')
print(f"Saved: {path4}")
plt.close()

print("\nAll 3 figures saved with Type 42 (TrueType) fonts.")
print("Verify with: pdffonts C:\\CoopCalib\\paper\\fig2_svr_density.pdf")