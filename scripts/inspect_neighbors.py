"""
inspect_neighbors.py — Understand sample[2] neighbour array layout.
"""
import pickle
import numpy as np
from pathlib import Path

PKL_DIR = Path(r"C:\CoopCalib\TUTR\dataset")

with open(PKL_DIR / "eth_train.pkl", "rb") as f:
    data = pickle.load(f)

print(f"Total samples: {len(data)}")
print(f"Sample tuple length: {len(data[0])}")
for idx in range(len(data[0])):
    el = data[0][idx]
    arr = np.array(el)
    print(f"  sample[{idx}]: shape={arr.shape}  dtype={arr.dtype}")

print()

# Deep dive on sample[2] — neighbours
for s_idx in range(5):
    neigh = np.array(data[s_idx][2])   # (N, T, 6) or similar
    ego   = np.array(data[s_idx][0])   # (T, 6)
    print(f"--- sample[{s_idx}] ---")
    print(f"  ego   shape: {ego.shape}   ego[0]: {ego[0]}")
    print(f"  neigh shape: {neigh.shape}")
    if neigh.ndim == 3 and neigh.shape[0] > 0:
        # Show first non-zero neighbour
        for n in range(neigh.shape[0]):
            row = neigh[n]
            if not np.all(row == 0):
                print(f"  neigh[{n}] first non-zero:")
                print(f"    {row}")
                print(f"    col0 range: {row[:,0].min():.3f}→{row[:,0].max():.3f}")
                print(f"    col1 range: {row[:,1].min():.3f}→{row[:,1].max():.3f}")
                break
    print()

# Check if coords are absolute or relative to ego
print("=== Checking if neighbour coords are relative to ego ===")
sample = data[10]
ego_xy  = np.array(sample[0])[0, :2]   # ego position (repeated)
neigh   = np.array(sample[2])
for n in range(min(3, neigh.shape[0])):
    row = neigh[n]
    if not np.all(row == 0):
        neigh_xy_t0 = row[0, :2]
        print(f"  ego_xy     : {ego_xy}")
        print(f"  neigh[{n}] t=0: {neigh_xy_t0}")
        print(f"  difference : {neigh_xy_t0 - ego_xy}")
        print(f"  abs distance to ego: {np.linalg.norm(neigh_xy_t0 - ego_xy):.3f} m")
        print()
