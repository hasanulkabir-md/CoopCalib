import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Data (locked from JSON files) ─────────────────────────────────────────────
folds = ['ETH', 'HOTEL', 'ZARA1', 'ZARA2', 'UNIV']

ade = {
    'V0': [0.4139, 0.1244, 0.1888, 0.1407, 0.2365],
    'V1': [0.4221, 0.1258, 0.1884, 0.1411, 0.2380],
    'V2': [0.4221, 0.1278, 0.1895, 0.1415, 0.2381],
}

fde = {
    'V0': [0.6291, 0.1854, 0.3462, 0.2535, 0.4309],
    'V1': [0.6178, 0.1834, 0.3441, 0.2566, 0.4345],
    'V2': [0.6180, 0.1849, 0.3428, 0.2578, 0.4358],
}

# ── Relative change calculation ───────────────────────────────────────────────
def rel_change(v_new, v_base):
    return [(n - b) / b * 100 for n, b in zip(v_new, v_base)]

rel_ade_v1 = rel_change(ade['V1'], ade['V0'])
rel_ade_v2 = rel_change(ade['V2'], ade['V0'])
rel_fde_v1 = rel_change(fde['V1'], fde['V0'])
rel_fde_v2 = rel_change(fde['V2'], fde['V0'])

# ── Style Configuration ───────────────────────────────────────────────────────
C_BASE  = '#3266AD'
C_V1    = '#E87722'
C_V2    = '#2A9D5C'
C_BAND2 = '#BBBBBB'
C_BAND3 = '#DDDDDD'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

x       = np.arange(len(folds))
width   = 0.26
gap     = 0.02
offsets = [-width - gap, 0, width + gap]

# ── Figure Layout ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8),
                         gridspec_kw={'height_ratios': [1.4, 1]})

# ── Row 1: Absolute Bar Charts (Cleaned) ──────────────────────────────────────
def draw_bar_panel(ax, data_dict, ylabel, panel_label):
    for key, offset, color in zip(['V0', 'V1', 'V2'], offsets, [C_BASE, C_V1, C_V2]):
        ax.bar(x + offset, data_dict[key], width=width, color=color, zorder=3)

    all_vals = [v for vals in data_dict.values() for v in vals]
    ax.set_ylim(min(all_vals) * 0.86, max(all_vals) * 1.14)
    ax.set_xticks(x)
    ax.set_xticklabels(folds, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.grid(axis='y', linewidth=0.4, color='#e8e8e8', zorder=0)
    ax.text(0.02, 0.97, panel_label, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top')

draw_bar_panel(axes[0, 0], ade, 'minADE (m)', 'a')
draw_bar_panel(axes[0, 1], fde, 'minFDE (m)', 'b')

# ── Row 2: Relative Change Dot Plots ──────────────────────────────────────────
def draw_rel_panel(ax, rel_v1, rel_v2, ylabel, panel_label):
    ax.axhspan(-3, 3, color=C_BAND3, alpha=0.55, zorder=0)
    ax.axhspan(-2, 2, color=C_BAND2, alpha=0.55, zorder=1)
    ax.axhline(0, color='#888888', linewidth=0.8, zorder=2)

    dot_offset = 0.13
    for rel, offset, color, marker in [(rel_v1, -dot_offset, C_V1, 'o'), (rel_v2, dot_offset, C_V2, 's')]:
        ax.scatter(x + offset, rel, color=color, s=48, marker=marker, zorder=5, edgecolors='white', linewidths=0.5)
        ax.plot(x + offset, rel, color=color, linewidth=0.8, alpha=0.5, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(folds, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(-4.5, 4.5)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%+.0f%%'))
    ax.grid(axis='y', linewidth=0.3, color='#e8e8e8', zorder=0)
    ax.text(0.02, 0.97, panel_label, transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')

    # Single exceedance annotation
    max_idx = int(np.argmax(np.abs(rel_v2)))
    if abs(rel_v2[max_idx]) > 2.0:
        ax.annotate(f'{rel_v2[max_idx]:+.2f}%', xy=(x[max_idx] + dot_offset, rel_v2[max_idx]),
                    xytext=(8, 6), textcoords='offset points', fontsize=8, color=C_V2,
                    arrowprops=dict(arrowstyle='->', color=C_V2, lw=0.7))

draw_rel_panel(axes[1, 0], rel_ade_v1, rel_ade_v2, 'Δ ADE vs baseline (%)', 'c')
draw_rel_panel(axes[1, 1], rel_fde_v1, rel_fde_v2, 'Δ FDE vs baseline (%)', 'd')

# ── Legend and Titling ────────────────────────────────────────────────────────
h_base = mpatches.Patch(color=C_BASE,  label='Baseline')
h_v1   = mpatches.Patch(color=C_V1,    label='+Calibration loss')
h_v2   = mpatches.Patch(color=C_V2,    label='+Calibration & social loss')
h_b2   = mpatches.Patch(color=C_BAND2, alpha=0.85, label='±2% band')
h_b3   = mpatches.Patch(color=C_BAND3, alpha=0.85, label='±3% band')

fig.legend(handles=[h_base, h_v1, h_v2, h_b2, h_b3], loc='lower center',
           ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.04))

fig.suptitle('Calibration and social losses preserve trajectory accuracy (max degradation: 2.7%)',
             fontsize=11, y=1.01)

plt.tight_layout(h_pad=2.5)
out_path = os.path.join('experiments', 'figures', 'fig4_accuracy_preservation.pdf')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()