"""
check_neis_shape.py - Print raw neis tensor shape from dataloader.
Run from C:\CoopCalib\TUTR\
"""
import sys, os, importlib
sys.path.insert(0, os.path.abspath("../TUTR") if os.path.exists("../TUTR") else ".")

from dataset import TrajectoryDataset
from torch.utils.data import DataLoader

test_dataset = TrajectoryDataset(
    dataset_path="./dataset/",
    dataset_name="eth",
    dataset_type="test",
    translation=True,
    rotation=True,
    scaling=False,
    obs_len=8,
)
loader = DataLoader(test_dataset, collate_fn=test_dataset.coll_fn,
                    batch_size=4, shuffle=False, num_workers=0)

for ped, neis, mask in loader:
    print(f"ped shape  : {ped.shape}")
    print(f"neis shape : {neis.shape}")
    print(f"mask shape : {mask.shape}")
    print()
    print("Interpreting neis shape:")
    print(f"  dim0 = B       = {neis.shape[0]}")
    print(f"  dim1 = ?       = {neis.shape[1]}")
    print(f"  dim2 = ?       = {neis.shape[2]}")
    print(f"  dim3 = 2 (xy)  = {neis.shape[3]}")
    print()
    # ped has shape (B, T_obs+T_fut, 2) = (B, 20, 2)
    T_total = ped.shape[1]
    T_obs   = 8
    T_fut   = 12
    MaxN    = None
    if neis.shape[1] == T_total:
        print(f"  LAYOUT: (B, T_total={T_total}, MaxN, 2)")
        print(f"  neis_obs = neis[:, :T_obs, :, :]  shape={list(neis[:, :T_obs, :, :].shape)}")
        print(f"  neis_fut = neis[:, T_obs:, :, :]  shape={list(neis[:, T_obs:, :, :].shape)}")
    elif neis.shape[2] == T_total:
        print(f"  LAYOUT: (B, MaxN, T_total={T_total}, 2)")
        print(f"  neis_obs = neis[:, :, :T_obs, :]  shape={list(neis[:, :, :T_obs, :].shape)}")
        print(f"  neis_fut = neis[:, :, T_obs:, :]  shape={list(neis[:, :, T_obs:, :].shape)}")
    else:
        print(f"  LAYOUT: UNKNOWN — neither dim matches T_total={T_total}")
    break  # only need first batch