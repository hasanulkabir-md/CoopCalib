"""
inspect_coords.py — Check coordinate ranges and available homography files.
Run from C:\CoopCalib with venv activated.
"""

import pickle
import numpy as np
from pathlib import Path

PKL_DIR  = Path(r"C:\CoopCalib\TUTR\dataset")
DATA_DIR = Path(r"C:\CoopCalib\TUTR\data")

# ── 1. Print coordinate ranges for first 100 samples of eth ──────────────────
print("=== Coordinate ranges (eth_train.pkl, first 100 samples) ===")
with open(PKL_DIR / "eth_train.pkl", "rb") as f:
    data = pickle.load(f)

for i, sample in enumerate(data[:100]):
    ego = np.array(sample[0])   # (8, 6)
    xy  = ego[:, :2]
    if i == 0:
        print(f"ego shape : {ego.shape}")
        print(f"ego[0]    : {ego[0]}")   # full first row — see all 6 columns
        print()

xy_all = np.array([np.array(s[0])[:, :2] for s in data[:100]])
print(f"X range : {xy_all[...,0].min():.2f} → {xy_all[...,0].max():.2f}")
print(f"Y range : {xy_all[...,1].min():.2f} → {xy_all[...,1].max():.2f}")

# ── 2. Look for homography / calibration files ────────────────────────────────
print("\n=== Searching for homography / H.txt / calib files ===")
for subset_dir in DATA_DIR.iterdir():
    if subset_dir.is_dir():
        files = list(subset_dir.rglob("*"))
        for f in files:
            if any(kw in f.name.lower() for kw in ["homo", "calib", "h.txt", "world"]):
                print(f"  FOUND: {f}")
        # Also list all files so we can see what's there
        all_files = [f.name for f in files if f.is_file()]
        print(f"  {subset_dir.name}: {all_files}")

# ── 3. Check what the 6 columns are (look at get_data_pkl.py output clues) ──
print("\n=== Sample[0] all 6 columns for 3 samples ===")
for i in range(3):
    ego = np.array(data[i][0])
    print(f"sample[{i}] ego:\n{ego}\n")
