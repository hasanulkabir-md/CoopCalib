"""
verify_neis_values.py
Confirms that neis in preds_v0_fixed contains FUTURE neighbour positions,
not past observation positions.
"""
import numpy as np

BASE = r"C:\CoopCalib\experiments\results"

n_new = np.load(BASE + r"\preds_v0_fixed\eth_neis.npy")
n_old = np.load(BASE + r"\preds_v0_neis\eth_neis.npy")
gt    = np.load(BASE + r"\preds_v0_fixed\eth_gt.npy")
obs   = np.load(BASE + r"\preds_v0_fixed\eth_obs.npy")

print("=" * 60)
print("SHAPE CHECK")
print(f"  NEW neis shape : {n_new.shape}  <- expect (N, MaxN, 12, 2)")
print(f"  OLD neis shape : {n_old.shape}")
print(f"  GT shape       : {gt.shape}     <- (N, 12, 2)")
print(f"  OBS shape      : {obs.shape}    <- (N, 8, 2)")

print()
print("=" * 60)
print("VALUE CHECK — Scene 0, Neighbour 0")
print()
print(f"  OBS scene 0, last 3 steps (ego past):")
print(f"    {obs[0, -3:, :]}")
print()
print(f"  GT scene 0, first 3 steps (ego future):")
print(f"    {gt[0, :3, :]}")
print()
print(f"  NEW neis[0, 0, :3, :] (neighbour future, first 3 steps):")
print(f"    {n_new[0, 0, :3, :]}")
print()
print(f"  OLD neis[0, 0, :3, :] (old file, first 3 steps):")
print(f"    {n_old[0, 0, :3, :]}")

print()
print("=" * 60)
print("DIAGNOSIS")

# The neighbour future should NOT match the ego obs window
# The neighbour future first step should be a small real number (not 1e9)
# The old file first step should match the observation window coordinates

new_first = n_new[0, 0, 0, :]
old_first = n_old[0, 0, 0, :]
obs_last  = obs[0, -1, :]
gt_first  = gt[0, 0, :]

print(f"  new_first : {new_first}")
print(f"  old_first : {old_first}")
print(f"  obs_last  : {obs_last}  (ego last observed = origin after translation)")
print(f"  gt_first  : {gt_first}  (ego first future step)")

if np.any(np.abs(new_first) > 1e6):
    print()
    print("  FAIL: NEW neis first step is sentinel (1e9) — neighbour absent")
    print("        This is OK only if the neighbour truly exits the scene")
elif np.allclose(new_first, old_first, atol=0.01):
    print()
    print("  FAIL: NEW and OLD neis have same values — file not updated")
else:
    print()
    print("  PASS: NEW neis values differ from OLD — file was correctly regenerated")

# Check if new neis values are plausible future positions
# (small magnitude, similar scale to gt)
gt_scale = np.abs(gt[0]).mean()
new_scale = np.abs(new_first).mean() if np.all(np.abs(new_first) < 1e6) else None
print()
if new_scale is not None:
    print(f"  GT magnitude avg (scene 0): {gt_scale:.4f}")
    print(f"  NEW neis magnitude (first step): {new_scale:.4f}")
    if new_scale < 10.0:
        print("  PASS: Neighbour future positions are plausible (< 10m from origin)")
    else:
        print("  WARN: Large magnitude — check coordinate frame")

print()
print("=" * 60)
print("SPSR SANITY — compute for first 10 scenes with corrected neis")
PLANNER_R = 0.5
GOAL_R    = 1.0
samples = np.load(BASE + r"\preds_v0_fixed\eth_samples.npy")
passed = 0
for i in range(10):
    s      = samples[i]         # (K, T, 2)
    g      = gt[i]              # (T, 2)
    goal   = g[-1]
    neis_i = n_new[i]           # (MaxN, T, 2)
    scene_pass = 0
    for k in range(s.shape[0]):
        dist_goal = np.linalg.norm(s[k, -1] - goal)
        if dist_goal > GOAL_R:
            continue
        collision = False