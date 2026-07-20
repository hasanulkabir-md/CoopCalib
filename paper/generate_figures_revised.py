import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "lines.linewidth": 1.2,
    "figure.dpi": 300,
})

SUBSETS = ["ETH", "HOT", "UNV", "ZR1", "ZR2"]
TIERS   = ["Sparse", "Medium", "Dense", "Medium", "Medium"]
TIER_COLORS = {"Sparse": "#4878CF", "Medium": "#6ACC65", "Dense": "#D65F5F"}

V0_ADE_AVG,  V0_ADE_STD  = 0.222, 0.002
V0_SPSR_AVG, V0_SPSR_STD = 0.819, 0.002
V2R_ADE_AVG, V2R_ADE_STD  = 0.265, 0.003
V2R_SPSR_AVG,V2R_SPSR_STD = 0.845, 0.008

CLUSTER_ADE  = np.array([0.227, 0.224, 0.228, 0.224])
CLUSTER_SPSR = np.array([0.817, 0.825, 0.829, 0.817])

V0_SVR  = np.array([0.838, 0.944, 0.989, 0.880, 0.961])
V0_SVR_STD  = np.array([0.003, 0.000, 0.000, 0.001, 0.000])
V2R_SVR = np.array([0.810, 0.937, 0.988, 0.870, 0.957])
V2R_SVR_STD = np.array([0.001, 0.000, 0.000, 0.002, 0.001])

V0_SPSR  = np.array([0.824, 0.919, 0.660, 0.929, 0.769])
V0_SPSR_STD2 = np.array([0.009, 0.001, 0.001, 0.002, 0.000])
V2R_SPSR2= np.array([0.789, 0.963, 0.685, 0.929, 0.853])
V2R_SPSR_STD2= np.array([0.011, 0.003, 0.015, 0.002, 0.027])

# ════════════════════════════════════════════════════════════
# FIGURE 1
# ════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(1, 3, figsize=(6.8, 2.5),
    gridspec_kw={"width_ratios": [1.0, 1.05, 1.05]})
fig1.subplots_adjust(left=0.07, right=0.96, top=0.84,
                     bottom=0.16, wspace=0.40)

# ── (a) Pareto ───────────────────────────────────────────────
ax = axes[0]
# Cluster ellipse — moved lower so it does not overlap V0*
cluster_x = np.mean(CLUSTER_ADE)
cluster_y = np.mean(CLUSTER_SPSR)
ellipse = mpatches.Ellipse(
    (cluster_x, cluster_y), width=0.012, height=0.018,
    fill=True, facecolor="#CCCCCC", edgecolor="#888888",
    linewidth=0.8, alpha=0.7, zorder=2)
ax.add_patch(ellipse)
# Label ellipse BELOW the ellipse, not inside it
ax.text(cluster_x, cluster_y - 0.014,
        "V1–V1W", ha="center", va="top", fontsize=6, color="#555555")

# V0* — label to the LEFT to avoid overlap with ellipse
ax.errorbar(V0_ADE_AVG, V0_SPSR_AVG,
            xerr=V0_ADE_STD*3, yerr=V0_SPSR_STD*3,
            fmt="o", color="#2166AC", markersize=5,
            linewidth=0.9, capsize=2, zorder=5, label="V0* (3-seed)")
ax.annotate("V0*", (V0_ADE_AVG, V0_SPSR_AVG),
            xytext=(-28, 3), textcoords="offset points",
            fontsize=6.5, color="#2166AC",
            arrowprops=dict(arrowstyle="-", color="#2166AC",
                            lw=0.5, alpha=0.5))

# V2R* — label above-right
ax.errorbar(V2R_ADE_AVG, V2R_SPSR_AVG,
            xerr=V2R_ADE_STD*3, yerr=V2R_SPSR_STD*3,
            fmt="s", color="#D6604D", markersize=5,
            linewidth=0.9, capsize=2, zorder=5, label="V2R* (3-seed)")
ax.annotate("V2R*", (V2R_ADE_AVG, V2R_SPSR_AVG),
            xytext=(4, 6), textcoords="offset points",
            fontsize=6.5, color="#D6604D")

# Arrow
ax.annotate("", xy=(V2R_ADE_AVG-0.005, V2R_SPSR_AVG-0.004),
            xytext=(V0_ADE_AVG+0.005, V0_SPSR_AVG+0.004),
            arrowprops=dict(arrowstyle="->", color="#666666",
                            lw=0.9, linestyle="dashed"))

# Ideal — bottom-left shaded box, text inside box not overlapping data
ax.add_patch(mpatches.FancyBboxPatch(
    (0.200, 0.836), 0.018, 0.010,
    boxstyle="round,pad=0.001",
    facecolor="#e8f5e9", edgecolor="#4caf50", linewidth=0.6,
    zorder=1, alpha=0.8))
ax.text(0.209, 0.841, "Ideal", ha="center", va="center",
        fontsize=6, color="#2e7d32")

ax.set_xlabel("minADE $\\downarrow$ (m)", fontsize=8)
ax.set_ylabel("SPSR $\\uparrow$", fontsize=8)
ax.set_title("(a) Safety–Accuracy Trade-off", fontsize=8, pad=3)
ax.set_xlim(0.198, 0.292)
ax.set_ylim(0.800, 0.875)
ax.legend(fontsize=5.5, loc="upper left", framealpha=0.8,
          handlelength=1.0, borderpad=0.4)
ax.tick_params(labelsize=7)

# ── helper: bar panel ────────────────────────────────────────
def bar_panel(ax, v0_vals, v0_std, v2r_vals, v2r_std,
              ylabel, title, ylim, svr_mode=True):
    x = np.arange(len(SUBSETS))
    w = 0.30
    for i, tier in enumerate(TIERS):
        ax.axvspan(i-0.45, i+0.45, alpha=0.07,
                   color=TIER_COLORS[tier], zorder=0)
    ax.bar(x-w/2, v0_vals,  w, color="#2166AC",
           alpha=0.85, zorder=3)
    ax.bar(x+w/2, v2r_vals, w, color="#D6604D",
           alpha=0.85, zorder=3)
    ax.errorbar(x-w/2, v0_vals,  yerr=v0_std*3,
                fmt="none", color="#1a4a7a", capsize=2,
                linewidth=0.8, zorder=4)
    ax.errorbar(x+w/2, v2r_vals, yerr=v2r_std*3,
                fmt="none", color="#8B1A0A", capsize=2,
                linewidth=0.8, zorder=4)

    deltas = v2r_vals - v0_vals
    for i, d in enumerate(deltas):
        bar_top = max(v0_vals[i], v2r_vals[i])
        err_top = max(v0_std[i], v2r_std[i]) * 3
        # ── KEY FIX: place annotation ABOVE error bar, not above bar top ──
        ypos = bar_top + err_top + (ylim[1] - ylim[0]) * 0.03
        suffix = " \u2021" if (SUBSETS[i] == "ETH" and not svr_mode) else ""
        ax.text(i, ypos, f"{d:+.3f}{suffix}",
                ha="center", va="bottom", fontsize=5.5, color="#222222")

    ax.set_xticks(x)
    ax.set_xticklabels(SUBSETS, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_ylim(ylim)
    ax.tick_params(axis="y", labelsize=7)

    tier_handles = [mpatches.Patch(color=TIER_COLORS[t], alpha=0.35, label=t)
                    for t in ["Sparse","Medium","Dense"]]
    v0h = mpatches.Patch(color="#2166AC", alpha=0.85, label="V0*")
    v2h = mpatches.Patch(color="#D6604D", alpha=0.85, label="V2R*")
    ax.legend(handles=tier_handles+[v0h,v2h],
              fontsize=5.5, loc="lower right",
              ncol=1, framealpha=0.75, borderpad=0.4)

# ── (b) SVR ─────────────────────────────────────────────────
# ylim top extended to 1.08 to give annotation room ABOVE bars
bar_panel(axes[1], V0_SVR, V0_SVR_STD, V2R_SVR, V2R_SVR_STD,
          "SVR $\\downarrow$",
          "(b) Social Violation Rate (SVR)",
          ylim=(0.770, 1.08), svr_mode=True)

# ── (c) SPSR ────────────────────────────────────────────────
bar_panel(axes[2], V0_SPSR, V0_SPSR_STD2, V2R_SPSR2, V2R_SPSR_STD2,
          "SPSR $\\uparrow$",
          "(c) Planning Success Rate (SPSR)",
          ylim=(0.530, 1.12), svr_mode=False)

# ETH footnote inside panel (c) — bottom of axes, not clipped
axes[2].text(0.01, 0.01,
    "\u2021 ETH: 61.8% scenes no real neighbours",
    transform=axes[2].transAxes,
    fontsize=5, ha="left", va="bottom", color="#555555")

fig1.savefig("fig_main_3panel_revised.pdf", bbox_inches="tight")
fig1.savefig("fig_main_3panel_revised.png", dpi=200, bbox_inches="tight")
print("[OK] fig_main_3panel_revised saved")

# ════════════════════════════════════════════════════════════
# FIGURE 2
# ════════════════════════════════════════════════════════════
ECE_VARIANTS = {
    "V0*":  ([0.629,0.519,0.558,0.515,0.553],[0.002,0.004,0.005,0.001,0.001]),
    "V1":   ([0.630,0.522,0.565,0.515,0.547],None),
    "V2":   ([0.641,0.519,0.569,0.517,0.554],None),
    "V3":   ([0.630,0.519,0.566,0.515,0.553],None),
    "V1W":  ([0.640,0.526,0.561,0.514,0.553],None),
    "V2R*": ([0.626,0.523,0.574,0.520,0.558],[0.005,0.001,0.002,0.002,0.002]),
}
VCOLS = {"V0*":"#2166AC","V1":"#5aae61","V2":"#9970ab",
         "V3":"#d6604d","V1W":"#f4a582","V2R*":"#a50026"}

V0_s42 = np.array([0.414,0.124,0.236,0.189,0.141])
ADE_DELTA = {
    "V1":  (np.array([0.442,0.128,0.235,0.188,0.143])-V0_s42)/V0_s42*100,
    "V2":  (np.array([0.422,0.128,0.238,0.189,0.142])-V0_s42)/V0_s42*100,
    "V3":  (np.array([0.443,0.127,0.237,0.189,0.144])-V0_s42)/V0_s42*100,
    "V1W": (np.array([0.429,0.128,0.234,0.186,0.143])-V0_s42)/V0_s42*100,
}
DCOLS = {"V1":"#5aae61","V2":"#9970ab","V3":"#d6604d","V1W":"#f4a582"}

x_sub = np.arange(len(SUBSETS))
v0_ece_mean = np.mean(ECE_VARIANTS["V0*"][0])   # 0.5548

fig2, axes2 = plt.subplots(1, 2, figsize=(6.8, 2.4))
fig2.subplots_adjust(left=0.08, right=0.88, top=0.86,
                     bottom=0.16, wspace=0.36)

# ── (a) ECE ─────────────────────────────────────────────────
ax = axes2[0]

# Reference band
ax.axhspan(v0_ece_mean-0.005, v0_ece_mean+0.005,
           alpha=0.15, color="#2166AC")
ax.axhline(v0_ece_mean, color="#2166AC",
           linewidth=1.0, linestyle="--", alpha=0.6)

for v in ["V0*","V1","V2","V3","V1W","V2R*"]:
    vals, errs = ECE_VARIANTS[v]
    c  = VCOLS[v]
    lw = 1.4 if v in ("V0*","V2R*") else 0.8
    ls = "--" if v == "V2R*" else "-"
    al = 1.0 if v in ("V0*","V2R*") else 0.6
    ax.plot(x_sub, vals, marker="o", markersize=3.2, color=c,
            linewidth=lw, linestyle=ls, alpha=al, label=v, zorder=4)
    if errs:
        ax.errorbar(x_sub, vals, yerr=np.array(errs)*3,
                    fmt="none", color=c, capsize=1.5,
                    linewidth=0.7, alpha=0.8)

# ── KEY FIX: annotation moved to BOTTOM-LEFT so legend (top-right) is clear ──
ax.text(0.02, 0.04,
        "All variants within \u00b10.005 of V0",
        transform=ax.transAxes, fontsize=6, va="bottom", ha="left",
        color="#2166AC",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="#2166AC", alpha=0.85))

ax.set_xticks(x_sub)
ax.set_xticklabels(["ETH","HOT","UNV","ZR1","ZR2"], fontsize=7)
ax.set_ylabel("ECE $\\downarrow$", fontsize=8)
ax.set_title("(a) ECE Across Variants", fontsize=8, pad=3)
ax.set_ylim(0.490, 0.695)   # extra headroom at top for legend
# Legend top-right — now unobstructed
ax.legend(fontsize=5.5, ncol=2, loc="upper right",
          framealpha=0.85, handlelength=1.1, borderpad=0.4)
ax.tick_params(axis="y", labelsize=7)

# ── (b) ADE preservation ────────────────────────────────────
ax = axes2[1]
ax.axhspan(-2, 2, alpha=0.12, color="#FF8C00")
ax.axhspan(-3, 3, alpha=0.06, color="#FFC04D")
ax.axhline(0, color="#888888", linewidth=0.8, linestyle=":")

for v, deltas in ADE_DELTA.items():
    ax.plot(x_sub, deltas, marker="o", markersize=3.5,
            color=DCOLS[v], linewidth=0.9, label=v)

ax.set_xticks([0,1,2,3,4])
ax.set_xticklabels(["ETH","HOT","UNV","ZR1","ZR2"], fontsize=7)
ax.set_ylabel("ΔADE vs V0 (%)", fontsize=8)
ax.set_title("(b) ADE Preservation", fontsize=8, pad=3)

# Threshold labels in pure axes coords (transAxes) - never touches spine
for dy, lbl, col in [(2.1,'±2%','#FF8C00'),(3.1,'±3%','#CC8800')]:
    ay  = (dy  + 5.5) / 11.0
    ay2 = (-dy + 5.5) / 11.0
    ax.text(0.97, ay,  lbl, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=6,
            color=col, fontweight='bold')
    ax.text(0.97, ay2, lbl, transform=ax.transAxes,
            ha='right', va='top', fontsize=6,
            color=col, fontweight='bold')
ax.set_xlim(-0.5, 4.5)
ax.legend(fontsize=6, loc="upper left", framealpha=0.8)
ax.tick_params(axis="y", labelsize=7)

fig2.savefig("fig_ece_accuracy_revised.pdf", bbox_inches="tight")
fig2.savefig("fig_ece_accuracy_revised.png", dpi=200, bbox_inches="tight")
print("[OK] fig_ece_accuracy_revised saved")
print("[DONE] Both figures regenerated.")
