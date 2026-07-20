"""
verify_spsr_sanity.py
Compute SPSR for first 20 ETH scenes using preds_v0_fixed neis.
Skip neighbour 0 (ego self) when checking collisions.
"""
import numpy as np

BASE      = r"C:\CoopCalib\experiments\results\preds_v0_fixed"
PLANNER_R = 0.5
GOAL_R    = 1.0
N_CHECK   = 20

samples = np.load(BASE + r"\eth_samples.npy")   # (364, 20, 12, 2)
gt      = np.load(BASE + r"\eth_gt.npy")         # (364, 12, 2)
neis    = np.load(BASE + r"\eth_neis.npy")       # (364, MaxN, 12, 2)

print(f"samples: {samples.shape}")
print(f"gt     : {gt.shape}")
print(f"neis   : {neis.shape}")
print()
print(f"{'Scene':>6}  {'SPSR':>5}  {'BestGoalDist':>13}  {'MinNeiDist':>11}  Notes")
print("-" * 65)

results = []
for i in range(N_CHECK):
    s      = samples[i]       # (K=20, T=12, 2)
    g      = gt[i]            # (T=12, 2)
    goal   = g[-1]            # target = final GT position
    neis_i = neis[i]          # (MaxN, T=12, 2)

    best_goal_dist = min(
        np.linalg.norm(s[k, -1] - goal) for k in range(s.shape[0])
    )

    # Min distance to any NON-SELF neighbour across all K candidates
    # Skip neighbour 0 — it is the ego agent itself in TUTR's format
    min_nei_dist = float('inf')
    n_valid = 0
    for nn in range(1, neis_i.shape[0]):   # start from 1, skip ego
        nei_traj = neis_i[nn]              # (T, 2)
        if np.any(np.abs(nei_traj) > 1e6):
            continue                        # sentinel — absent
        n_valid += 1
        for k in range(s.shape[0]):
            d = np.linalg.norm(s[k] - nei_traj, axis=-1).min()
            if d < min_nei_dist:
                min_nei_dist = d

    # SPSR: does any K candidate reach goal without hitting a real neighbour?
    scene_spsr = 0
    for k in range(s.shape[0]):
        dist_goal = np.linalg.norm(s[k, -1] - goal)
        if dist_goal > GOAL_R:
            continue
        collision = False
        for nn in range(1, neis_i.shape[0]):   # skip ego at index 0
            nei_traj = neis_i[nn]
            if np.any(np.abs(nei_traj) > 1e6):
                continue
            d = np.linalg.norm(s[k] - nei_traj, axis=-1)
            if np.any(d < PLANNER_R):
                collision = True
                break
        if not collision:
            scene_spsr = 1
            break

    results.append(scene_spsr)
    notes = f"n_valid_neis={n_valid}"
    if min_nei_dist == float('inf'):
        notes += " NO_REAL_NEIS"
    print(f"{i:>6}  {scene_spsr:>5}  {best_goal_dist:>13.4f}  {min_nei_dist:>11.4f}  {notes}")

print("-" * 65)
print(f"SPSR over {N_CHECK} scenes: {sum(results)/N_CHECK:.4f}")
print(f"Expected: ~0.30 (full dataset V0 SPSR should be ~0.327)")
print()
if sum(results)/N_CHECK < 0.05:
    print("FAIL: SPSR still near zero — ego-skip or sentinel logic may still be wrong")
elif sum(results)/N_CHECK > 0.60:
    print("WARN: SPSR very high — may be skipping too many neighbours")
else:
    print("PASS: SPSR in plausible range — proceed to full metrics recomputation")