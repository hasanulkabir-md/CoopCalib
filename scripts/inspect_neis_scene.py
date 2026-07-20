"""
inspect_neis_scene.py
Print ALL neighbour slots for first 5 ETH scenes.
Determines which indices are real vs sentinel, and whether ego is at index 0.
"""
import numpy as np

BASE = r"C:\CoopCalib\experiments\results\preds_v0_fixed"

neis = np.load(BASE + r"\eth_neis.npy")   # (364, MaxN=8, T=12, 2)
gt   = np.load(BASE + r"\eth_gt.npy")     # (364, T=12, 2)
obs  = np.load(BASE + r"\eth_obs.npy")    # (364, T=8, 2)

print(f"neis shape: {neis.shape}")
print(f"MaxN = {neis.shape[1]}  T_fut = {neis.shape[2]}")
print()

for scene_i in range(5):
    print(f"{'='*70}")
    print(f"SCENE {scene_i}")
    print(f"  GT first step  : {gt[scene_i, 0, :]}")
    print(f"  OBS last step  : {obs[scene_i, -1, :]}  (should be [0,0] after translation)")
    print()
    print(f"  Neighbour slots:")
    print(f"  {'Slot':>5}  {'Status':>10}  {'First step xy':>25}  {'Last step xy':>25}")

    for nn in range(neis.shape[1]):
        traj = neis[scene_i, nn]   # (T=12, 2)
        first = traj[0]
        last  = traj[-1]

        is_sentinel = np.any(np.abs(traj) > 1e6)
        is_ego      = np.allclose(traj, gt[scene_i], atol=1e-3)
        is_zero     = np.allclose(first, [0.0, 0.0], atol=1e-6)

        if is_sentinel:
            status = "SENTINEL"
        elif is_ego:
            status = "EGO"
        elif is_zero:
            status = "ZERO-START"
        else:
            status = "REAL-NEI"

        print(f"  {nn:>5}  {status:>10}  {str(first.round(4)):>25}  {str(last.round(4)):>25}")

    # Count real neighbours (non-sentinel, non-ego)
    real_count = 0
    ego_idx    = None
    for nn in range(neis.shape[1]):
        traj = neis[scene_i, nn]
        if np.any(np.abs(traj) > 1e6):
            continue
        if np.allclose(traj, gt[scene_i], atol=1e-3):
            ego_idx = nn
            continue
        real_count += 1

    print()
    print(f"  Ego found at slot index : {ego_idx}")
    print(f"  Real neighbours (non-ego, non-sentinel): {real_count}")
    print()