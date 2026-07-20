"""
inspect_neis_all_subsets.py
Check neighbour slot quality across all 5 subsets.
Reports: how many scenes have 0, 1, 2+ real neighbours (non-ego, non-sentinel, non-zero).
"""
import numpy as np
import os

BASE    = r"C:\CoopCalib\experiments\results\preds_v0_fixed"
SUBSETS = ["eth", "hotel", "univ", "zara1", "zara2"]

for ds in SUBSETS:
    neis_path = os.path.join(BASE, f"{ds}_neis.npy")
    gt_path   = os.path.join(BASE, f"{ds}_gt.npy")

    neis = np.load(neis_path)   # (N, MaxN, T=12, 2)
    gt   = np.load(gt_path)     # (N, T=12, 2)

    N, MaxN, T, _ = neis.shape

    count_0_neis  = 0   # scenes with NO real neighbours
    count_1_nei   = 0   # scenes with exactly 1 real neighbour
    count_2plus   = 0   # scenes with 2+ real neighbours
    zero_start_total = 0

    for i in range(N):
        real = 0
        for nn in range(MaxN):
            traj = neis[i, nn]
            is_sentinel   = np.any(np.abs(traj) > 1e6)
            is_ego        = np.allclose(traj, gt[i], atol=1e-3)
            is_zero_start = np.allclose(traj[0], [0.0, 0.0], atol=1e-4) and \
                            np.allclose(traj,
                                        np.zeros_like(traj), atol=1e-4)
            if is_sentinel or is_ego:
                continue
            if is_zero_start:
                zero_start_total += 1
                continue
            real += 1

        if real == 0:
            count_0_neis += 1
        elif real == 1:
            count_1_nei += 1
        else:
            count_2plus += 1

    print(f"\n{'='*55}")
    print(f"  {ds.upper():6s}  N={N}  MaxN={MaxN}")
    print(f"  Scenes with 0 real neighbours : {count_0_neis:6d} ({100*count_0_neis/N:.1f}%)")
    print(f"  Scenes with 1 real neighbour  : {count_1_nei:6d} ({100*count_1_nei/N:.1f}%)")
    print(f"  Scenes with 2+ real neighbours: {count_2plus:6d} ({100*count_2plus/N:.1f}%)")
    print(f"  Total zero-start slots seen   : {zero_start_total:6d}")

    # Print first scene with 2+ real neighbours as example
    printed = False
    for i in range(N):
        real_slots = []
        for nn in range(MaxN):
            traj = neis[i, nn]
            if np.any(np.abs(traj) > 1e6): continue
            if np.allclose(traj, gt[i], atol=1e-3): continue
            if np.allclose(traj, np.zeros_like(traj), atol=1e-4): continue
            real_slots.append(nn)
        if len(real_slots) >= 2 and not printed:
            print(f"\n  Example scene {i} with {len(real_slots)} real neighbours:")
            for nn in real_slots[:3]:
                print(f"    Slot {nn}: first={neis[i,nn,0,:].round(4)}"
                      f"  last={neis[i,nn,-1,:].round(4)}")
            printed = True

print()