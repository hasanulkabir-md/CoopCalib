"""
verify_spsr_eth.py
Manual SPSR verification for ETH subset, V0 vs V2R.
Loads preds directly and computes SPSR scene-by-scene for first 20 scenes.
Prints per-scene pass/fail so we can see WHY V2R SPSR is low.
"""
import numpy as np
import os

PREDS_V0  = r"C:\CoopCalib\experiments\results\preds_v0_neis"
PREDS_V2R = r"C:\CoopCalib\experiments\results\preds_v2r"

PLANNER_R  = 0.5   # planner collision radius (metres)
GOAL_R     = 1.0   # goal success radius (metres)
N_SCENES   = 20    # inspect first 20 scenes

def compute_spsr_scene(samples, gt, obs, neis, planner_r, goal_r):
    """
    samples : (K, T, 2)
    gt      : (T, 2)
    obs     : (T_obs, 2)   -- not used but loaded for shape check
    neis    : (T, N, 2)    -- neighbour ground truth futures
    Returns: 1 if any candidate reaches goal without collision, else 0
    """
    K, T, _ = samples.shape
    goal = gt[-1]   # final GT position = goal

    for k in range(K):
        traj = samples[k]   # (T, 2)

        # Goal check: final predicted position within goal_r of GT final
        dist_to_goal = np.linalg.norm(traj[-1] - goal)
        if dist_to_goal > goal_r:
            continue   # this candidate doesn't reach goal

        # Collision check: any timestep within planner_r of any neighbour
        collision = False
        N = neis.shape[1]
        for n in range(N):
            nei_traj = neis[:, n, :]   # (T, 2)
            # skip sentinel neighbours
            if np.any(np.abs(nei_traj) > 1e6):
                continue
            dists = np.linalg.norm(traj - nei_traj, axis=-1)  # (T,)
            if np.any(dists < planner_r):
                collision = True
                break

        if not collision:
            return 1   # safe candidate that reaches goal found

    return 0   # no safe goal-reaching candidate


def load_and_check(preds_dir, label):
    samples = np.load(os.path.join(preds_dir, "eth_samples.npy"))
    gt      = np.load(os.path.join(preds_dir, "eth_gt.npy"))
    obs     = np.load(os.path.join(preds_dir, "eth_obs.npy"))
    neis    = np.load(os.path.join(preds_dir, "eth_neis.npy"))

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  samples : {samples.shape}  gt : {gt.shape}  neis : {neis.shape}")
    print(f"{'='*60}")

    # neis shape from inference is (N_scenes, T_obs, MaxN, 2)
    # need to transpose to (N_scenes, T_fut, MaxN, 2) — check what shape we have
    print(f"  neis raw shape: {neis.shape}")

    # Transpose neis to (N_scenes, T_fut, MaxN, 2) if needed
    if neis.shape[1] == 8:   # T_obs dimension — need future neis
        print("  WARNING: neis appears to be observation-time only (T=8)")
        print("  Using gt shape to infer — this subset may have neis=obs only")
        # In this project neis stored as (N, T_obs, MaxN, 2) then transposed
        # to (N, MaxN, T_fut, 2) in compute_metrics. Re-check actual storage.
        neis_t = np.transpose(neis, (0, 2, 1, 3))   # (N, MaxN, T_obs, 2)
    else:
        neis_t = neis  # already (N, T, MaxN, 2)

    results = []
    print(f"\n  Scene-by-scene SPSR (first {N_SCENES} scenes):")
    print(f"  {'Scene':>6}  {'SPSR':>6}  {'Best_goal_dist':>14}  {'Min_nei_dist':>12}")

    for i in range(min(N_SCENES, len(samples))):
        s   = samples[i]                     # (K, T, 2)
        g   = gt[i]                          # (T, 2)
        o   = obs[i]                         # (T_obs, 2)

        # neis_t shape: (N, MaxN, T_obs, 2) or (N, T, MaxN, 2)
        # For per-scene: grab scene i
        n_scene = neis_t[i]                  # (MaxN, T, 2) or (T, MaxN, 2)
        if n_scene.shape[0] != g.shape[0]:
            n_scene = np.transpose(n_scene, (1, 0, 2))  # → (T, MaxN, 2)

        goal = g[-1]
        best_goal_dist = min(np.linalg.norm(s[k][-1] - goal) for k in range(s.shape[0]))

        # Min distance to any neighbour across all candidates and timesteps
        min_nei_dist = float('inf')
        for k in range(s.shape[0]):
            for nn in range(n_scene.shape[1]):
                d = np.linalg.norm(s[k] - n_scene[:, nn, :], axis=-1).min()
                if d < min_nei_dist:
                    min_nei_dist = d

        spsr = compute_spsr_scene(s, g, o, n_scene, PLANNER_R, GOAL_R)
        results.append(spsr)
        print(f"  {i:>6}  {spsr:>6}  {best_goal_dist:>14.4f}  {min_nei_dist:>12.4f}")

    overall = np.mean(results)
    print(f"\n  SPSR over first {N_SCENES} scenes: {overall:.4f}")
    print(f"  (Full dataset SPSR from metrics file for reference)")
    return overall


v0_spsr  = load_and_check(PREDS_V0,  "V0  (baseline)")
v2r_spsr = load_and_check(PREDS_V2R, "V2R (ranking-aware SVR loss)")

print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"  V0  SPSR (20 scenes) : {v0_spsr:.4f}")
print(f"  V2R SPSR (20 scenes) : {v2r_spsr:.4f}")
print(f"  Delta                : {v2r_spsr - v0_spsr:+.4f}")
if v2r_spsr < v0_spsr - 0.1:
    print("  STATUS: LARGE DROP CONFIRMED — checking whether real or artifact")
    print("  Look at 'Best_goal_dist' column:")
    print("  If V2R has large goal distances → trajectories drifted away from GT goal")
    print("  If V2R has small goal distances but SPSR=0 → collision blocking all candidates")
else:
    print("  STATUS: Drop is small — likely noise, not a bug")
print(f"{'='*60}")