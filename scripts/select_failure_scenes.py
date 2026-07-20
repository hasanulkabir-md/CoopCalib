"""
Principled scene selection for failure analysis — FINAL POLISHED VERSION
Uses statistical criteria to find REPRESENTATIVE cases, not outliers
HIGH-IMPACT VERSION: Optimized for visual separability and figure quality
+ Visual preview of top-5 candidates
+ Sanity checks and divergence validation
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import json
from pathlib import Path

sys.path.insert(0, r'C:\CoopCalib\metrics')
from eval_suite import compute_svr_from_tutr_batch

# Font settings for IEEE (for preview plots)
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

def compute_per_scene_metrics(samples, gt, neis):
    """Compute ADE and SVR for each scene individually"""
    N = len(samples)
    ades = np.zeros(N)
    svrs = np.zeros(N)

    neis_t = neis.transpose(0, 2, 1, 3)

    for i in range(N):
        scene_samples = samples[i]
        scene_gt = gt[i]
        ade_per_k = np.mean(np.linalg.norm(scene_samples - scene_gt, axis=2), axis=1)
        ades[i] = np.min(ade_per_k)

        scene_samples_5d = samples[i:i+1]
        scene_neis_5d = neis_t[i:i+1]
        svrs[i] = compute_svr_from_tutr_batch(scene_samples_5d, scene_neis_5d, ped_radius=0.3)

    return ades, svrs

def has_valid_neighbors(neis, scene_idx, min_neighbors=4):
    """Check if scene has enough neighbors for meaningful visualization"""
    scene_neis = neis[scene_idx]
    valid_count = 0
    for agent_idx in range(1, min(scene_neis.shape[0], 10)):
        if np.any(scene_neis[agent_idx] != 0):
            valid_count += 1
    return valid_count >= min_neighbors

def compute_visual_complexity(obs, gt, neis, scene_idx):
    """Score scene visual complexity with CONTINUOUS scoring"""
    obs_traj = obs[scene_idx]
    gt_traj = gt[scene_idx]

    obs_length = np.sum(np.linalg.norm(np.diff(obs_traj, axis=0), axis=1))
    gt_length = np.sum(np.linalg.norm(np.diff(gt_traj, axis=0), axis=1))
    total_length = obs_length + gt_length

    neighbor_count = 0
    scene_neis = neis[scene_idx]
    for i in range(1, scene_neis.shape[0]):
        if np.any(scene_neis[i] != 0):
            neighbor_count += 1

    complexity_score = 2.0 * neighbor_count + 1.0 * total_length

    return complexity_score, neighbor_count, total_length

def compute_trajectory_divergence(samples_v0, samples_v2r, gt, idx):
    """Compute visual difference between V0 and V2R best trajectories"""
    scene_samples_v0 = samples_v0[idx]
    scene_samples_v2r = samples_v2r[idx]
    scene_gt = gt[idx]

    traj_v0 = get_best_traj(scene_samples_v0, scene_gt)
    traj_v2r = get_best_traj(scene_samples_v2r, scene_gt)

    traj_diff = np.mean(np.linalg.norm(traj_v0 - traj_v2r, axis=1))
    return traj_diff

def plot_scene_preview(ax, samples_v0, samples_v2r, gt, obs, neis, scene_idx, title, highlight_closest=True):
    """
    Plot a single scene for visual preview.
    highlight_closest: if True, highlights 2-3 closest neighbors in color.
    """
    scene_samples_v0 = samples_v0[scene_idx]
    scene_samples_v2r = samples_v2r[scene_idx]
    gt_traj = gt[scene_idx]
    obs_traj = obs[scene_idx]

    # Best trajectory selection
    pred_v0 = get_best_traj(scene_samples_v0, gt_traj)
    pred_v2r = get_best_traj(scene_samples_v2r, gt_traj)

    neighbors = neis[scene_idx]

    # NORMALIZE
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

    # Sort by distance and get closest 3
    neighbor_distances.sort(key=lambda x: x[1])
    closest_indices = [x[0] for x in neighbor_distances[:3]]

    # Plot neighbors
    for agent_idx in range(1, neighbors.shape[0]):
        traj = neighbors[agent_idx]
        if np.any(traj != 0):
            if highlight_closest and agent_idx in closest_indices:
                # Highlight closest neighbors in orange
                ax.plot(traj[:, 0], traj[:, 1], 
                       color='#ff7f0e', alpha=0.5, linewidth=1.2, zorder=2)
            else:
                # Others in light gray
                ax.plot(traj[:, 0], traj[:, 1], 
                       color='#cccccc', alpha=0.3, linewidth=0.8, zorder=1)

    # Plot trajectories
    ax.plot(obs_traj[:, 0], obs_traj[:, 1], 'k--', linewidth=1.5, label='Observed', zorder=3)
    ax.plot(gt_traj[:, 0], gt_traj[:, 1], color='#2ca02c', linewidth=2.0, label='GT', zorder=4)
    ax.plot(pred_v0[:, 0], pred_v0[:, 1], color='#1f77b4', linewidth=1.8, linestyle='-.', label='V0', zorder=5)
    ax.plot(pred_v2r[:, 0], pred_v2r[:, 1], color='#d62728', linewidth=1.8, label='V2R', zorder=6)

    # Markers
    ax.scatter(obs_traj[0, 0], obs_traj[0, 1], color='black', s=40, marker='o', zorder=7)
    ax.scatter(gt_traj[-1, 0], gt_traj[-1, 1], color='#2ca02c', s=60, marker='*', zorder=7)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_title(title, fontsize=8, pad=4)
    ax.tick_params(labelsize=7)

def preview_top5(scene_indices, samples_v0, samples_v2r, gt, obs, neis, subset_name, output_dir):
    """Generate preview figure of top-5 candidates for manual selection"""
    print(f"\n  Generating preview of top-5 {subset_name} candidates...")

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.15, wspace=0.25)

    for i, idx in enumerate(scene_indices):
        title = f"#{i+1} Scene {idx}"
        plot_scene_preview(axes[i], samples_v0, samples_v2r, gt, obs, neis, idx, title)
        if i == 0:
            axes[i].legend(fontsize=6, loc='upper right', framealpha=0.8)

    fig.suptitle(f"Top-5 Trade-off Candidates ({subset_name.upper()}) — Pick the clearest visually", 
                 fontsize=10, fontweight='bold', y=0.98)

    preview_path = output_dir / f'preview_top5_{subset_name}.pdf'
    fig.savefig(preview_path, bbox_inches='tight', dpi=300)
    fig.savefig(preview_path.with_suffix('.png'), bbox_inches='tight', dpi=200)

    print(f"  [SAVED] Preview: {preview_path}")
    plt.close(fig)

def find_dense_failure_representative():
    """Find UNIV scene where BOTH V0 and V2R fail — prioritize crowd density"""
    print("\n" + "="*60)
    print("  [1/2] Finding Dense Failure Scene (UNIV)")
    print("="*60)

    samples_v0, gt, obs, neis = load_data('univ', 'v0')
    samples_v2r, _, _, _ = load_data('univ', 'v2r')

    print("  Computing per-scene metrics (this takes ~2 min)...")
    ades_v0, svrs_v0 = compute_per_scene_metrics(samples_v0, gt, neis)
    ades_v2r, svrs_v2r = compute_per_scene_metrics(samples_v2r, gt, neis)

    print(f"\n  V0  SVR: mean={np.mean(svrs_v0):.3f}, median={np.median(svrs_v0):.3f}")
    print(f"  V2R SVR: mean={np.mean(svrs_v2r):.3f}, median={np.median(svrs_v2r):.3f}")

    threshold = 0.95
    both_fail = (svrs_v0 > threshold) & (svrs_v2r > threshold)
    fail_indices = np.where(both_fail)[0]

    print(f"\n  Scenes where both V0 and V2R fail (SVR > {threshold}): {len(fail_indices)}")

    if len(fail_indices) == 0:
        print("  WARNING: No scenes meet criteria, lowering threshold to 0.90")
        threshold = 0.90
        both_fail = (svrs_v0 > threshold) & (svrs_v2r > threshold)
        fail_indices = np.where(both_fail)[0]

    valid_candidates = []
    for idx in fail_indices:
        if has_valid_neighbors(neis, idx, min_neighbors=5):
            complexity, n_neighbors, traj_len = compute_visual_complexity(obs, gt, neis, idx)
            traj_diff = compute_trajectory_divergence(samples_v0, samples_v2r, gt, idx)

            if traj_diff < 0.05:
                continue

            valid_candidates.append((idx, svrs_v0[idx], svrs_v2r[idx], 
                                    complexity, n_neighbors, traj_len, traj_diff))

    print(f"  Candidates with valid neighbors: {len(valid_candidates)}")

    valid_candidates.sort(key=lambda x: (x[4], x[3]), reverse=True)

    top_candidates = valid_candidates[:min(10, len(valid_candidates))]
    median_idx = len(top_candidates) // 2

    scene_idx, svr_v0, svr_v2r, complexity, n_neighbors, traj_len, traj_diff = top_candidates[median_idx]

    print(f"\n  ✓ SELECTED Scene #{scene_idx}")
    print(f"      SVR: V0={svr_v0:.3f}, V2R={svr_v2r:.3f}")
    print(f"      Neighbors: {n_neighbors}, Trajectory length: {traj_len:.2f}m")
    print(f"      Complexity score: {complexity:.1f}")
    print(f"      Trajectory divergence: {traj_diff:.3f}m")

    # SANITY CHECK
    print(f"\n  [SANITY CHECK] Dense divergence: {traj_diff:.3f}m (should be > 0.05)")

    return scene_idx, samples_v0, samples_v2r, gt, obs, neis, traj_diff

def find_tradeoff_representative():
    """Find scene where V2R improves safety but hurts ADE — with top-5 preview"""
    print("\n" + "="*60)
    print("  [2/2] Finding Safety-Accuracy Trade-off Scene")
    print("="*60)

    output_dir = Path(r'C:\CoopCalib\experiments\results')

    for subset in ['hotel', 'zara1', 'zara2']:
        print(f"\n  Checking {subset.upper()}...")

        samples_v0, gt, obs, neis = load_data(subset, 'v0')
        samples_v2r, _, _, _ = load_data(subset, 'v2r')

        print("  Computing per-scene metrics...")
        ades_v0, svrs_v0 = compute_per_scene_metrics(samples_v0, gt, neis)
        ades_v2r, svrs_v2r = compute_per_scene_metrics(samples_v2r, gt, neis)

        svr_improved = svrs_v2r < svrs_v0
        ade_worse = ades_v2r > ades_v0 * 1.10

        tradeoff_scenes = svr_improved & ade_worse
        tradeoff_indices = np.where(tradeoff_scenes)[0]

        print(f"    Trade-off scenes found: {len(tradeoff_indices)}")

        if len(tradeoff_indices) < 5:
            continue

        valid_candidates = []
        for idx in tradeoff_indices:
            if has_valid_neighbors(neis, idx, min_neighbors=4):
                ade_delta = ades_v2r[idx] - ades_v0[idx]
                svr_delta = svrs_v2r[idx] - svrs_v0[idx]
                complexity, n_neighbors, traj_len = compute_visual_complexity(obs, gt, neis, idx)
                traj_diff = compute_trajectory_divergence(samples_v0, samples_v2r, gt, idx)

                if traj_diff < 0.05:
                    continue

                valid_candidates.append((idx, ades_v0[idx], ades_v2r[idx], 
                                        svrs_v0[idx], svrs_v2r[idx],
                                        ade_delta, svr_delta, 
                                        complexity, n_neighbors, traj_len, traj_diff))

        if len(valid_candidates) < 3:
            continue

        valid_candidates.sort(key=lambda x: (x[7], x[10]), reverse=True)

        top_candidates = valid_candidates[:min(10, len(valid_candidates))]
        top5 = top_candidates[:5]

        # PREVIEW TOP 5
        top5_indices = [c[0] for c in top5]
        preview_top5(top5_indices, samples_v0, samples_v2r, gt, obs, neis, subset, output_dir)

        median_idx = len(top_candidates) // 2

        (scene_idx, ade_v0, ade_v2r, svr_v0, svr_v2r, 
         ade_delta, svr_delta, complexity, n_neighbors, traj_len, traj_diff) = top_candidates[median_idx]

        print(f"\n  ✓ AUTO-SELECTED Scene #{scene_idx} from {subset.upper()}")
        print(f"      ADE: V0={ade_v0:.3f}, V2R={ade_v2r:.3f} (Δ={ade_delta:+.3f}m, {ade_delta/ade_v0*100:+.1f}%)")
        print(f"      SVR: V0={svr_v0:.3f}, V2R={svr_v2r:.3f} (Δ={svr_delta:+.3f})")
        print(f"      Neighbors: {n_neighbors}, Trajectory length: {traj_len:.2f}m")
        print(f"      Complexity score: {complexity:.1f}")
        print(f"      Trajectory divergence: {traj_diff:.3f}m")

        # SANITY CHECK
        print(f"\n  [SANITY CHECK] Trade-off divergence: {traj_diff:.3f}m (should be > 0.05)")
        print(f"  [SANITY CHECK] SVR improvement: {abs(svr_delta):.3f} (visually check if clear)")

        print(f"\n  ⚠️  RECOMMENDATION: Review preview_top5_{subset}.pdf")
        print(f"     If auto-selected scene is not clearest, manually pick from top 5 above.")

        return scene_idx, samples_v0, samples_v2r, gt, obs, neis, subset, top5, traj_diff

    raise ValueError("No suitable trade-off scenes found in any subset")

def save_scene_info(dense_idx, trade_idx, trade_subset, dense_div, trade_div, top5_tradeoff=None):
    """Save selected scene indices for reproducibility"""
    output = {
        "dense_failure": {
            "subset": "univ",
            "scene_index": int(dense_idx),
            "divergence_m": float(dense_div),
            "selection_method": "median of top-10 by crowd density + complexity among SVR>0.95 scenes"
        },
        "tradeoff": {
            "subset": trade_subset,
            "scene_index": int(trade_idx),
            "divergence_m": float(trade_div),
            "selection_method": "median of top-10 by complexity + visual divergence among ADE↑10%+SVR↓ scenes"
        },
        "sanity_checks": {
            "dense_divergence_m": float(dense_div),
            "trade_divergence_m": float(trade_div),
            "min_acceptable_divergence_m": 0.05
        }
    }

    if top5_tradeoff:
        output["tradeoff"]["top5_candidates"] = [
            {"rank": i+1, "scene_index": int(c[0]), "divergence_m": float(c[10]), "ade_delta_m": float(c[5])}
            for i, c in enumerate(top5_tradeoff)
        ]

    output_path = Path(r'C:\CoopCalib\experiments\results\selected_failure_scenes.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n[SAVED] Scene indices: {output_path}")

if __name__ == "__main__":
    print("="*60)
    print("  PRINCIPLED SCENE SELECTION — FINAL POLISHED VERSION")
    print("  Visual separability + crowd density + continuous scoring")
    print("  + Top-5 preview + sanity checks")
    print("="*60)

    # Find scenes
    dense_idx, d_v0, d_v2r, d_gt, d_obs, d_neis, dense_div = find_dense_failure_representative()
    trade_idx, t_v0, t_v2r, t_gt, t_obs, t_neis, trade_subset, top5, trade_div = find_tradeoff_representative()

    # Save for reproducibility
    save_scene_info(dense_idx, trade_idx, trade_subset, dense_div, trade_div, top5)

    print("\n" + "="*60)
    print("  SCENE SELECTION COMPLETE")
    print("="*60)
    print(f"  Dense failure:  UNIV scene #{dense_idx} (div={dense_div:.3f}m)")
    print(f"  Trade-off:      {trade_subset.upper()} scene #{trade_idx} (div={trade_div:.3f}m)")
    print("\n  ⚠️  ACTION REQUIRED:")
    print("     1. Check preview_top5_hotel.pdf (or zara1/zara2)")
    print("     2. If auto-selected scene is unclear, manually override")
    print("     3. Then run generate_failure_analysis.py with final indices")