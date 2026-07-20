"""
Generate failure case analysis figure for RAL paper — SUPER POLISHED VERSION
Makes trade-off visually undeniable with strong candidates and dynamic scaling
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys

sys.path.insert(0, r'C:\CoopCalib\metrics')
from eval_suite import compute_svr_from_tutr_batch

# Font settings for IEEE
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "font.size": 9,
})

def load_data(subset, variant='v0'):
    """Load predictions for a subset and variant"""
    base = Path(r'C:\CoopCalib\experiments\results')
    preds_dir = base / f'preds_{variant}_fixed'

    samples = np.load(preds_dir / f'{subset}_samples.npy')
    gt = np.load(preds_dir / f'{subset}_gt.npy')
    obs = np.load(preds_dir / f'{subset}_obs.npy')
    neis = np.load(preds_dir / f'{subset}_neis.npy')

    return samples, gt, obs, neis

def get_best_traj(samples, gt):
    """Select best trajectory by minADE"""
    ade = np.mean(np.linalg.norm(samples - gt, axis=2), axis=1)
    return samples[np.argmin(ade)]

def compute_all_neighbor_distances(pred, neighbors, obs_traj):
    """Compute minimum distance from prediction to each neighbor trajectory"""
    distances = []
    for agent_idx in range(1, neighbors.shape[0]):
        traj = neighbors[agent_idx]
        if np.any(traj != 0):
            # Compute min distance between pred and this neighbor at each timestep
            dists = np.linalg.norm(pred - traj, axis=1)
            min_dist = np.min(dists)
            distances.append((agent_idx, min_dist))
    distances.sort(key=lambda x: x[1])
    return distances

def plot_scene(ax, samples_v0, samples_v2r, gt, obs, neis, scene_idx, title, 
               highlight_closest=True, show_avoidance_arrow=False, 
               show_closest_label=False, is_dense=False):
    """Plot a single scene with V0 vs V2R predictions — SUPER POLISHED"""

    # Extract scene data
    scene_samples_v0 = samples_v0[scene_idx]
    scene_samples_v2r = samples_v2r[scene_idx]
    gt_traj = gt[scene_idx]
    obs_traj = obs[scene_idx]

    # Best trajectory selection (minADE)
    pred_v0 = get_best_traj(scene_samples_v0, gt_traj)
    pred_v2r = get_best_traj(scene_samples_v2r, gt_traj)

    neighbors = neis[scene_idx]

    # NORMALIZE (CRITICAL FIX)
    origin = obs_traj[-1]
    gt_traj = gt_traj - origin
    obs_traj = obs_traj - origin
    pred_v0 = pred_v0 - origin
    pred_v2r = pred_v2r - origin
    neighbors = neighbors - origin

    # Compute distances from ego to each neighbor at prediction start
    neighbor_distances = []
    for agent_idx in range(1, neighbors.shape[0]):
        traj = neighbors[agent_idx]
        if np.any(traj != 0):
            dist = np.linalg.norm(traj[0] - obs_traj[-1])
            neighbor_distances.append((agent_idx, dist))

    neighbor_distances.sort(key=lambda x: x[1])
    closest_indices = [x[0] for x in neighbor_distances[:3]]

    # === FIX: Plot ALL non-zero neighbors with higher visibility ===
    for agent_idx in range(1, neighbors.shape[0]):
        traj = neighbors[agent_idx]
        if np.any(traj != 0):
            if highlight_closest and agent_idx in closest_indices:
                # Highlight closest neighbors in bold orange
                ax.plot(traj[:, 0], traj[:, 1], 
                       color='#ff7f0e', alpha=0.7, linewidth=1.5, zorder=2)
            else:
                # All others in visible gray (not too faint)
                alpha_val = 0.5 if is_dense else 0.4
                lw_val = 1.0 if is_dense else 0.9
                ax.plot(traj[:, 0], traj[:, 1], 
                       color='#999999', alpha=alpha_val, linewidth=lw_val, zorder=1)

    # === FIX: Thicker observed trajectory ===
    ax.plot(obs_traj[:, 0], obs_traj[:, 1], 
           'k--', linewidth=2.0, label='Observed', zorder=3)

    # Plot GT (green, thick)
    ax.plot(gt_traj[:, 0], gt_traj[:, 1], 
           color='#2ca02c', linewidth=2.2, label='Ground Truth', zorder=4)

    # Plot V0 prediction (blue)
    ax.plot(pred_v0[:, 0], pred_v0[:, 1], 
           color='#1f77b4', linewidth=2.0, linestyle='-.', 
           label='V0', zorder=5)

    # Plot V2R prediction (red) — slightly thicker for emphasis
    ax.plot(pred_v2r[:, 0], pred_v2r[:, 1], 
           color='#d62728', linewidth=2.2, 
           label='V2R', zorder=6)

    # === FIX: Strong avoidance arrow ===
    if show_avoidance_arrow:
        # Find point of maximum divergence
        diff = np.linalg.norm(pred_v0 - pred_v2r, axis=1)
        max_div_idx = np.argmax(diff)

        if diff[max_div_idx] > 0.05:
            # Draw bold red arrow from V0 to V2R at max divergence
            ax.annotate('', 
                       xy=(pred_v2r[max_div_idx, 0], pred_v2r[max_div_idx, 1]),
                       xytext=(pred_v0[max_div_idx, 0], pred_v0[max_div_idx, 1]),
                       arrowprops=dict(arrowstyle='->', color='#d62728', 
                                      lw=2.0, alpha=0.9),
                       zorder=8)

    # === FIX: Label closest interaction point ===
    if show_closest_label and len(closest_indices) > 0:
        closest_idx = closest_indices[0]
        closest_traj = neighbors[closest_idx]
        # Find point closest to V2R path
        min_dist = float('inf')
        closest_point = None
        for t in range(len(pred_v2r)):
            dists = np.linalg.norm(closest_traj - pred_v2r[t], axis=1)
            if np.min(dists) < min_dist:
                min_dist = np.min(dists)
                closest_point = closest_traj[np.argmin(dists)]

        if closest_point is not None and min_dist < 1.5:
            ax.text(closest_point[0] + 0.15, closest_point[1] + 0.15,
                   'closest\ninteraction',
                   fontsize=6, color='#ff7f0e', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                            edgecolor='#ff7f0e', alpha=0.8),
                   zorder=9)

    # Start/end markers
    ax.scatter(obs_traj[0, 0], obs_traj[0, 1], 
              color='black', s=60, marker='o', zorder=7, edgecolors='white', linewidths=0.5)
    ax.scatter(gt_traj[-1, 0], gt_traj[-1, 1], 
              color='#2ca02c', s=90, marker='*', zorder=7, edgecolors='white', linewidths=0.5)

    # === FIX: Dynamic axis scaling ===
    all_pts = np.vstack([obs_traj, gt_traj, pred_v0, pred_v2r])
    # Include neighbor points too
    for agent_idx in range(1, neighbors.shape[0]):
        traj = neighbors[agent_idx]
        if np.any(traj != 0):
            all_pts = np.vstack([all_pts, traj])

    margin = 0.8
    x_min, x_max = all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin
    y_min, y_max = all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin

    # Ensure minimum range for readability
    x_range = x_max - x_min
    y_range = y_max - y_min
    if x_range < 4.0:
        center = (x_min + x_max) / 2
        x_min, x_max = center - 2.0, center + 2.0
    if y_range < 4.0:
        center = (y_min + y_max) / 2
        y_min, y_max = center - 2.0, center + 2.0

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15, linestyle='--')
    ax.set_xlabel('X (m)', fontsize=9)
    ax.set_ylabel('Y (m)', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    ax.legend(fontsize=7, loc='best', framealpha=0.9)
    ax.ticklabel_format(style='plain')

def generate_failure_analysis_figure(dense_scene_idx=17044, trade_scene_idx=1106, 
                                     trade_subset='hotel'):
    """Main function - generate complete figure"""

    print("="*60)
    print("  Generating Failure Case Analysis Figure")
    print("  SUPER POLISHED VERSION")
    print("="*60)

    # Load data
    d_v0, d_gt, d_obs, d_neis = load_data('univ', 'v0')
    d_v2r, _, _, _ = load_data('univ', 'v2r')

    t_v0, t_gt, t_obs, t_neis = load_data(trade_subset, 'v0')
    t_v2r, _, _, _ = load_data(trade_subset, 'v2r')

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.15, wspace=0.30)

    # (a) Dense failure
    plot_scene(axes[0], d_v0, d_v2r, d_gt, d_obs, d_neis, dense_scene_idx,
               "(a) Dense Failure (UNIV)", 
               highlight_closest=True, show_avoidance_arrow=False, 
               show_closest_label=False, is_dense=True)

    # Add annotation
    axes[0].text(0.05, 0.95, 
                "Both V0 and V2R\nviolate personal space",
                transform=axes[0].transAxes,
                fontsize=7, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.4', 
                         facecolor='white', edgecolor='red', 
                         alpha=0.9, linewidth=1.2))

    # (b) Trade-off — USE STRONGER CANDIDATE + ALL VISUAL CUES
    plot_scene(axes[1], t_v0, t_v2r, t_gt, t_obs, t_neis, trade_scene_idx,
               "(b) Safety–Accuracy Trade-off (HOTEL)", 
               highlight_closest=True, show_avoidance_arrow=True, 
               show_closest_label=True, is_dense=False)

    # === FIX: Stronger annotation tied to geometry ===
    axes[1].text(0.05, 0.95,
                "V2R deviates to\navoid nearby pedestrian",
                transform=axes[1].transAxes,
                fontsize=7, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.4',
                         facecolor='white', edgecolor='orange',
                         alpha=0.9, linewidth=1.2))

    # Save
    output_path = Path(r'C:\CoopCalib\paper\fig_failure_analysis.pdf')
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    fig.savefig(output_path.with_suffix('.png'), bbox_inches='tight', dpi=200)

    print(f"\n[SAVED] {output_path}")
    print(f"[SAVED] {output_path.with_suffix('.png')}")
    print("\n[DONE] Failure analysis figure generated.")
    print("\n  ⚠️  If trade-off still looks subtle, try:")
    print("     trade_scene_idx=1148  (divergence=0.331m, ADEΔ=+0.138m)")

if __name__ == "__main__":
    # === FIX: Use stronger candidate (#1106 or #1148) ===
    # #1106: divergence=0.333m, ADEΔ=+0.063m
    # #1148: divergence=0.331m, ADEΔ=+0.138m  ← strongest overall
    generate_failure_analysis_figure(
        dense_scene_idx=17044,   # UNIV dense failure
        trade_scene_idx=1106,    # STRONG candidate (divergence=0.333m)
        trade_subset='hotel'
    )