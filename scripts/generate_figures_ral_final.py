"""
generate_figures_ral_final.py
CoopCalib-TP — Final RAL submission figures
IEEE RAL standards:
  - pdf.fonttype = 42  (TrueType — no Type 3 fonts)
  - Times New Roman serif (matches IEEEtran body font)
  - Single column width: 3.5 in
  - Double column width: 7.16 in
  - Max figure height: 3.5 in (single), 2.8 in (double)
  - 300 DPI minimum for raster elements

All numbers from verified Session 16 results:
  V0/V2R: v0_multiseed_summary.json, v2r_multiseed_summary.json (3-seed)
  V1/V2/V3/V1W: *_metrics_corrected.json (seed 42 only)

DO NOT CHANGE ANY NUMBER without re-running the metrics pipeline.
"""

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42   # IEEE-required: TrueType
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size':          9,
    'axes.labelsize':    10,
    'axes.titlesize':     9,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'legend.fontsize':    8,
    'legend.framealpha':  0.9,
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':     'tight',
    'savefig.pad_inches': 0.04,
    'axes.linewidth':    0.7,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'lines.linewidth':   1.2,
})

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import os

# ── Output directory ────────────────────────────────────────
OUT = r"/home/claude"

# ── RAL column widths ────────────────────────────────────────
COL1 = 3.5    # single column (inches)
COL2 = 7.16   # double column (inches)

# ── Colour scheme: IEEE-friendly, print-safe, colourblind-ok ─
# Blue / Red / Green / Orange / Purple / Brown
C_V0  = '#1f77b4'   # blue   — baseline
C_V1  = '#ff7f0e'   # orange — calibration
C_V2  = '#2ca02c'   # green  — energy+calibration
C_V3  = '#9467bd'   # purple — energy only
C_V1W = '#8c564b'   # brown  — warmup
C_V2R = '#d62728'   # red    — ranking-aware (KEY RESULT)

TIER_BG = {
    'Sparse': '#ddeeff',
    'Medium': '#ddffd8',
    'Dense':  '#ffeedd',
}

# ════════════════════════════════════════════════════════════
# VERIFIED 3-SEED DATA — DO NOT CHANGE
# Source: v0_multiseed_summary.json, v2r_multiseed_summary.json
# ════════════════════════════════════════════════════════════
SUBSETS = ["ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2"]
TIERS   = {
    "ETH":   "Sparse",
    "HOTEL": "Medium",
    "UNIV":  "Dense",
    "ZARA1": "Medium",
    "ZARA2": "Medium",
}
XLABS = [
    "ETH\n(Sparse)",
    "HOTEL\n(Medium)",
    "UNIV\n(Dense)",
    "ZARA1\n(Medium)",
    "ZARA2\n(Medium)",
]

# V0 — 3-seed mean ± std
v0 = {
    "ade":  {"ETH":(0.4193,0.0077), "HOTEL":(0.1246,0.0003),
             "UNIV":(0.2361,0.0006), "ZARA1":(0.1889,0.0002),
             "ZARA2":(0.1408,0.0002)},
    "fde":  {"ETH":(0.6291,0.0048), "HOTEL":(0.1876,0.0032),
             "UNIV":(0.4308,0.0001), "ZARA1":(0.3471,0.0012),
             "ZARA2":(0.2555,0.0029)},
    "ece":  {"ETH":(0.6271,0.0020), "HOTEL":(0.5191,0.0040),
             "UNIV":(0.5581,0.0054), "ZARA1":(0.5149,0.0005),
             "ZARA2":(0.5529,0.0012)},
    "svr":  {"ETH":(0.8384,0.0028), "HOTEL":(0.9437,0.0001),
             "UNIV":(0.9893,0.0001), "ZARA1":(0.8801,0.0006),
             "ZARA2":(0.9613,0.0002)},
    "spsr": {"ETH":(0.8178,0.0091), "HOTEL":(0.9195,0.0008),
             "UNIV":(0.6603,0.0008), "ZARA1":(0.9288,0.0016),
             "ZARA2":(0.7694,0.0002)},
}

# V2R — 3-seed mean ± std
v2r = {
    "ade":  {"ETH":(0.4753,0.0115), "HOTEL":(0.1529,0.0018),
             "UNIV":(0.2797,0.0021), "ZARA1":(0.2401,0.0010),
             "ZARA2":(0.1787,0.0019)},
    "fde":  {"ETH":(0.6245,0.0141), "HOTEL":(0.1977,0.0020),
             "UNIV":(0.4370,0.0030), "ZARA1":(0.3543,0.0020),
             "ZARA2":(0.2641,0.0010)},
    "ece":  {"ETH":(0.6253,0.0052), "HOTEL":(0.5229,0.0011),
             "UNIV":(0.5726,0.0017), "ZARA1":(0.5200,0.0022),
             "ZARA2":(0.5542,0.0015)},
    "svr":  {"ETH":(0.8095,0.0010), "HOTEL":(0.9370,0.0003),
             "UNIV":(0.9885,0.0000), "ZARA1":(0.8699,0.0023),
             "ZARA2":(0.9573,0.0007)},
    "spsr": {"ETH":(0.7894,0.0111), "HOTEL":(0.9632,0.0030),
             "UNIV":(0.6855,0.0151), "ZARA1":(0.9288,0.0016),
             "ZARA2":(0.8599,0.0270)},
}

# Other variants — seed 42 only (from *_metrics_corrected.json)
v1  = {"ade": [0.4416,0.1277,0.2353,0.1875,0.1429],
       "spsr":[0.8130,0.9180,0.6580,0.9330,0.7650]}
v2  = {"ade": [0.4221,0.1278,0.2381,0.1895,0.1415],
       "spsr":[0.7890,0.9220,0.6720,0.9330,0.8100]}
v3  = {"ade": [0.4430,0.1270,0.2370,0.1890,0.1440],
       "spsr":[0.8130,0.9160,0.6700,0.9340,0.8140]}
v1w = {"ade": [0.4290,0.1280,0.2340,0.1860,0.1430],
       "spsr":[0.8050,0.9160,0.6600,0.9340,0.7690]}

def avg(d, key):
    return np.mean([d[key][s][0] for s in SUBSETS])

def shade(ax, alpha=0.10):
    for i, s in enumerate(SUBSETS):
        ax.axvspan(i-0.5, i+0.5, alpha=alpha,
                   color=TIER_BG[TIERS[s]], zorder=0)

# ════════════════════════════════════════════════════════════
# FIGURE 1 — 3-PANEL SUMMARY FIGURE
# Panel A: Pareto front (ADE vs SPSR, all 6 variants)
# Panel B: SVR by subset (V0 vs V2R, 3-seed error bars)
# Panel C: SPSR by subset (V0 vs V2R, 3-seed error bars)
# ════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(COL2, 4.2))
gs  = gridspec.GridSpec(1, 3, figure=fig,
                         width_ratios=[1.1, 1.0, 1.0],
                         wspace=0.38)

# ── Panel A: Pareto Front ────────────────────────────────────
ax_a = fig.add_subplot(gs[0])

# All 6 variants as points
variants_pareto = [
    ('V0',  avg(v0,'ade'),  avg(v0,'spsr'),
             np.std([v0['ade'][s][0]  for s in SUBSETS]),
             np.std([v0['spsr'][s][0] for s in SUBSETS]),
             C_V0, 'o', True),
    ('V1',  np.mean(v1['ade']),  np.mean(v1['spsr']),
             0, 0, C_V1, 's', False),
    ('V2',  np.mean(v2['ade']),  np.mean(v2['spsr']),
             0, 0, C_V2, '^', False),
    ('V3',  np.mean(v3['ade']),  np.mean(v3['spsr']),
             0, 0, C_V3, 'D', False),
    ('V1W', np.mean(v1w['ade']), np.mean(v1w['spsr']),
             0, 0, C_V1W, 'v', False),
    ('V2R', avg(v2r,'ade'),  avg(v2r,'spsr'),
             np.std([v2r['ade'][s][0]  for s in SUBSETS]),
             np.std([v2r['spsr'][s][0] for s in SUBSETS]),
             C_V2R, 'P', True),
]

for name, mx, my, ex, ey, col, mrk, has_err in variants_pareto:
    ax_a.errorbar(mx, my,
                  xerr=ex if has_err else None,
                  yerr=ey if has_err else None,
                  fmt=mrk, color=col, ms=7, capsize=2.5,
                  linewidth=0.8, zorder=5,
                  label=f'{name}{"*" if has_err else ""}')
    offset = (4, 4) if name not in ('V2','V3') else (4, -9)
    ax_a.annotate(name, (mx, my),
                  textcoords='offset points', xytext=offset,
                  fontsize=7, color=col, fontweight='bold')

# Arrow V0 → V2R showing trade-off direction
ax_a.annotate('',
    xy  =(avg(v2r,'ade'),  avg(v2r,'spsr')),
    xytext=(avg(v0,'ade'), avg(v0,'spsr')),
    arrowprops=dict(arrowstyle='->', color='dimgray',
                    lw=1.0, linestyle='dashed'))

# Ideal corner
ax_a.text(0.135, 0.955, 'Ideal', fontsize=6.5,
          color='darkgreen', ha='left', va='top',
          style='italic', transform=ax_a.transAxes)
ax_a.annotate('', xy=(0.04, 0.96), xytext=(0.15, 0.96),
              xycoords='axes fraction',
              arrowprops=dict(arrowstyle='->', color='darkgreen', lw=0.8))
ax_a.annotate('', xy=(0.04, 0.96), xytext=(0.04, 0.84),
              xycoords='axes fraction',
              arrowprops=dict(arrowstyle='->', color='darkgreen', lw=0.8))

ax_a.set_xlabel('minADE $\\downarrow$ (m)', fontsize=9)
ax_a.set_ylabel('SPSR $\\uparrow$', fontsize=9)
ax_a.set_title('(a) Safety--Accuracy\nPareto Front', fontsize=8.5)
ax_a.legend(fontsize=6.5, loc='lower right', framealpha=0.9,
            handlelength=1.2, handletextpad=0.4,
            title='* = 3-seed', title_fontsize=6)
ax_a.grid(alpha=0.18, linewidth=0.5)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)

# ── Panel B: SVR by subset ────────────────────────────────────
ax_b = fig.add_subplot(gs[1])
x = np.arange(len(SUBSETS))
w = 0.38
shade(ax_b)

v0m_svr  = [v0['svr'][s][0]  for s in SUBSETS]
v0e_svr  = [v0['svr'][s][1]  for s in SUBSETS]
v2rm_svr = [v2r['svr'][s][0] for s in SUBSETS]
v2re_svr = [v2r['svr'][s][1] for s in SUBSETS]

ax_b.bar(x-w/2, v0m_svr,  w, yerr=v0e_svr,  color=C_V0,
         alpha=0.88, capsize=2.5, label='V0',
         error_kw={'lw':1.0,'ecolor':'#0a3060'})
ax_b.bar(x+w/2, v2rm_svr, w, yerr=v2re_svr, color=C_V2R,
         alpha=0.88, capsize=2.5, label='V2R',
         error_kw={'lw':1.0,'ecolor':'#7a0000'})

for i, s in enumerate(SUBSETS):
    delta = v2r['svr'][s][0] - v0['svr'][s][0]
    ypos  = max(v0['svr'][s][0]+v0['svr'][s][1],
                v2r['svr'][s][0]+v2r['svr'][s][1]) + 0.004
    col   = '#1a6600' if delta < 0 else '#8b0000'
    ax_b.text(x[i], ypos, f'{delta:+.3f}',
              ha='center', va='bottom', fontsize=6.5,
              color=col, fontweight='bold')

ax_b.set_xticks(x)
ax_b.set_xticklabels(['ETH','HOT','UNV','ZR1','ZR2'], fontsize=7.5)
ax_b.set_ylabel('SVR $\\downarrow$', fontsize=9)
ax_b.set_ylim(0.75, 1.05)
ax_b.set_title('(b) Social Violation Rate\n(3-seed mean $\\pm$ std)', fontsize=8.5)
ax_b.legend(loc='lower right', fontsize=7.5)
ax_b.axhline(1.0, color='gray', lw=0.5, ls='--', alpha=0.4)
ax_b.grid(axis='y', alpha=0.18, lw=0.5)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)

# Tier labels at top
for i, s in enumerate(SUBSETS):
    t = TIERS[s][0]  # S, M, D
    ax_b.text(x[i], 1.048, t, ha='center', fontsize=6,
              color={'S':'#336699','M':'#336633','D':'#994433'}[t])

# ── Panel C: SPSR by subset ───────────────────────────────────
ax_c = fig.add_subplot(gs[2])
shade(ax_c)

v0m_spsr  = [v0['spsr'][s][0]  for s in SUBSETS]
v0e_spsr  = [v0['spsr'][s][1]  for s in SUBSETS]
v2rm_spsr = [v2r['spsr'][s][0] for s in SUBSETS]
v2re_spsr = [v2r['spsr'][s][1] for s in SUBSETS]

ax_c.bar(x-w/2, v0m_spsr,  w, yerr=v0e_spsr,  color=C_V0,
         alpha=0.88, capsize=2.5, label='V0',
         error_kw={'lw':1.0,'ecolor':'#0a3060'})
ax_c.bar(x+w/2, v2rm_spsr, w, yerr=v2re_spsr, color=C_V2R,
         alpha=0.88, capsize=2.5, label='V2R',
         error_kw={'lw':1.0,'ecolor':'#7a0000'})

for i, s in enumerate(SUBSETS):
    delta = v2r['spsr'][s][0] - v0['spsr'][s][0]
    ypos  = max(v0['spsr'][s][0]+v0['spsr'][s][1],
                v2r['spsr'][s][0]+v2r['spsr'][s][1]) + 0.006
    col   = '#1a6600' if delta > 0 else '#8b0000'
    ax_c.text(x[i], ypos, f'{delta:+.3f}',
              ha='center', va='bottom', fontsize=6.5,
              color=col, fontweight='bold')

ax_c.set_xticks(x)
ax_c.set_xticklabels(['ETH','HOT','UNV','ZR1','ZR2'], fontsize=7.5)
ax_c.set_ylabel('SPSR $\\uparrow$', fontsize=9)
ax_c.set_ylim(0.55, 1.12)
ax_c.set_title('(c) Safe Planning Success Rate\n(3-seed mean $\\pm$ std)', fontsize=8.5)
ax_c.legend(loc='lower right', fontsize=7.5)
ax_c.grid(axis='y', alpha=0.18, lw=0.5)
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)

for i, s in enumerate(SUBSETS):
    t = TIERS[s][0]
    ax_c.text(x[i], 1.098, t, ha='center', fontsize=6,
              color={'S':'#336699','M':'#336633','D':'#994433'}[t])

fig.suptitle(
    'CoopCalib-TP: Safety--accuracy trade-off revealed by V2R '
    '(V2R reduces SVR and improves SPSR vs V0, at cost of +19.5\\% ADE)',
    fontsize=8.5, y=1.01)

fig.savefig(os.path.join(OUT, 'fig_main_3panel.pdf'))
print("Saved: fig_main_3panel.pdf")
plt.close()

# ════════════════════════════════════════════════════════════
# FIGURE 2 — ECE + Accuracy preservation (single column, 2 panels)
# Panel A: ECE across subsets, all variants
# Panel B: Delta ADE for V1/V2/V3/V1W (accuracy preservation)
# ════════════════════════════════════════════════════════════
fig, (ax_ece, ax_acc) = plt.subplots(1, 2, figsize=(COL2, 2.6))

# Panel A: ECE
ece_data = {
    'V0':  [v0['ece'][s][0]  for s in SUBSETS],
    'V1':  [0.6299,0.5215,0.5653,0.5150,0.5474],  # v1_metrics_corrected
    'V2':  [0.6411,0.5191,0.5686,0.5167,0.5537],  # v2_metrics_corrected
    'V3':  [0.6304,0.5186,0.5662,0.5152,0.5529],  # v3_metrics_corrected
    'V1W': [0.6403,0.5257,0.5610,0.5140,0.5527],  # v1w_metrics_corrected
    'V2R': [v2r['ece'][s][0] for s in SUBSETS],
}
xi = np.arange(len(SUBSETS))
ece_colors = {'V0':C_V0,'V1':C_V1,'V2':C_V2,
              'V3':C_V3,'V1W':C_V1W,'V2R':C_V2R}
ece_markers = {'V0':'o','V1':'s','V2':'^',
               'V3':'D','V1W':'v','V2R':'P'}

for vname, vals in ece_data.items():
    yerr = [v0['ece'][s][1] for s in SUBSETS] if vname == 'V0' else \
           ([v2r['ece'][s][1] for s in SUBSETS] if vname == 'V2R' else None)
    lw = 1.5 if vname in ('V0','V2R') else 0.9
    ax_ece.errorbar(xi, vals, yerr=yerr,
                    marker=ece_markers[vname], color=ece_colors[vname],
                    ms=4.5, lw=lw, capsize=2, label=vname,
                    linestyle='-' if vname in ('V0','V2R') else '--')

ax_ece.set_xticks(xi)
ax_ece.set_xticklabels(['ETH','HOT','UNV','ZR1','ZR2'], fontsize=7.5)
ax_ece.set_ylabel('ECE $\\downarrow$', fontsize=9)
ax_ece.set_ylim(0.49, 0.68)
ax_ece.set_title('(a) ECE — All Variants\n(no bold: first measurement)', fontsize=8)
ax_ece.legend(fontsize=6.5, ncol=2, loc='upper right', framealpha=0.9)
ax_ece.grid(alpha=0.18, lw=0.5)
ax_ece.spines['top'].set_visible(False)
ax_ece.spines['right'].set_visible(False)

# Panel B: Delta ADE V1/V2/V3/V1W (not V2R — too large)
v0_ade_list = [v0['ade'][s][0] for s in SUBSETS]
variants_acc = {
    'V1':  v1['ade'],
    'V2':  v2['ade'],
    'V3':  v3['ade'],
    'V1W': v1w['ade'],
}
for vname, vals in variants_acc.items():
    deltas = [100*(a-b)/b for a, b in zip(vals, v0_ade_list)]
    ax_acc.plot(xi, deltas,
                marker=ece_markers[vname], color=ece_colors[vname],
                ms=4.5, lw=0.9, ls='--', label=vname)

ax_acc.axhline(0,   color='black', lw=0.6)
ax_acc.axhspan(-2,  2,  alpha=0.07, color='green',  zorder=0)
ax_acc.axhspan(-3,  3,  alpha=0.05, color='orange', zorder=0)
ax_acc.axhspan( 3,  10, alpha=0.07, color='red',    zorder=0)
ax_acc.axhspan(-10,-3,  alpha=0.07, color='red',    zorder=0)

ax_acc.text(4.5, 2.1, '$\\pm$2\\%', ha='right', va='bottom',
            fontsize=6.5, color='darkgreen')
ax_acc.text(4.5, 3.2, '$\\pm$3\\%', ha='right', va='bottom',
            fontsize=6.5, color='darkorange')

ax_acc.set_xticks(xi)
ax_acc.set_xticklabels(['ETH','HOT','UNV','ZR1','ZR2'], fontsize=7.5)
ax_acc.set_ylabel('$\\Delta$ADE vs V0 (\\%)', fontsize=9)
ax_acc.set_ylim(-4.5, 5.5)
ax_acc.set_title('(b) Accuracy Preservation\n(V1--V1W; V2R not shown: +19.5\\%)', fontsize=8)
ax_acc.legend(fontsize=7, loc='upper right', framealpha=0.9)
ax_acc.grid(axis='y', alpha=0.18, lw=0.5)
ax_acc.spines['top'].set_visible(False)
ax_acc.spines['right'].set_visible(False)

fig.tight_layout(pad=0.5)
fig.savefig(os.path.join(OUT, 'fig_ece_accuracy.pdf'))
print("Saved: fig_ece_accuracy.pdf")
plt.close()

print("\nAll figures generated.")
print("Font compliance: pdf.fonttype=42 (TrueType) — no Type 3 fonts.")
print(f"Output directory: {OUT}")
