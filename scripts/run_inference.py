"""
CoopCalib-TP — Step 2: TUTR Inference
======================================
File:  C:\\CoopCalib\\scripts\\run_inference.py

Loads the best checkpoint for one dataset fold, runs the full test set,
and saves raw prediction arrays to C:\\CoopCalib\\experiments\\results\\preds\\

Usage (run from C:\\CoopCalib\\TUTR\\):
    python ..\scripts\run_inference.py --dataset_name eth
    python ..\scripts\run_inference.py --dataset_name hotel
    python ..\scripts\run_inference.py --dataset_name univ
    python ..\scripts\run_inference.py --dataset_name zara1
    python ..\scripts\run_inference.py --dataset_name zara2

Outputs (for each dataset):
    C:\\CoopCalib\\experiments\\results\\preds\\{name}_samples.npy   shape (N, 20, 12, 2)
    C:\\CoopCalib\\experiments\\results\\preds\\{name}_gt.npy        shape (N, 12, 2)
    C:\\CoopCalib\\experiments\\results\\preds\\{name}_obs.npy       shape (N,  8, 2)
    C:\\CoopCalib\\experiments\\results\\preds\\{name}_neis.npy      shape (N, MaxN, 12, 2)
"""

import argparse
import os
import sys
import pickle
import numpy as np
import torch
import importlib

# -- Make sure TUTR modules are importable (run from TUTR\ dir) --
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../TUTR")

from dataset import TrajectoryDataset
from torch.utils.data import DataLoader
from model import TrajectoryModel

# ---------------------------------------------------------------------------
# Config map -- mirrors train.py hp_config values we need
# ---------------------------------------------------------------------------
CONFIG_MAP = {
    "eth":   {"config": "config/eth.py",   "data_scaling": [1.9, 0.4]},
    "hotel": {"config": "config/hotel.py",  "data_scaling": [1.0, 1.0]},
    "univ":  {"config": "config/univ.py",   "data_scaling": [1.0, 1.0]},
    "zara1": {"config": "config/zara1.py",  "data_scaling": [1.0, 1.0]},
    "zara2": {"config": "config/zara2.py",  "data_scaling": [1.0, 1.0]},
}

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True,
                    choices=list(CONFIG_MAP.keys()))
parser.add_argument("--dataset_path",  type=str, default="./dataset/")
parser.add_argument("--checkpoint",    type=str, default="./checkpoint/")
parser.add_argument("--out_dir",       type=str,
                    default=r"..\experiments\results\preds")
parser.add_argument("--obs_len",  type=int, default=8)
parser.add_argument("--pred_len", type=int, default=12)
parser.add_argument("--num_k",    type=int, default=20)
parser.add_argument("--gpu",      type=str, default="0")
parser.add_argument("--num_works",type=int, default=0)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# ---------------------------------------------------------------------------
# Load hp_config (need model_hidden_dim, batch_size, dist_threshold)
# ---------------------------------------------------------------------------
cfg_path = CONFIG_MAP[args.dataset_name]["config"]
spec = importlib.util.spec_from_file_location("hp_config", cfg_path)
hp_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hp_config)

data_scaling = CONFIG_MAP[args.dataset_name]["data_scaling"]

# ---------------------------------------------------------------------------
# Dataset + DataLoader  (no augmentation for inference)
# ---------------------------------------------------------------------------
print(f"\n[run_inference] dataset={args.dataset_name}  K={args.num_k}")

test_dataset = TrajectoryDataset(
    dataset_path=args.dataset_path,
    dataset_name=args.dataset_name,
    dataset_type="test",
    translation=True,
    rotation=True,
    scaling=False,
    obs_len=args.obs_len,
)

test_loader = DataLoader(
    test_dataset,
    collate_fn=test_dataset.coll_fn,
    batch_size=hp_config.batch_size,
    shuffle=False,          # keep order deterministic for alignment with gt
    num_workers=args.num_works,
)

# ---------------------------------------------------------------------------
# Load motion modes
# ---------------------------------------------------------------------------
motion_modes_file = args.dataset_path + args.dataset_name + "_motion_modes.pkl"
assert os.path.exists(motion_modes_file), \
    f"Motion modes file not found: {motion_modes_file}\n" \
    f"Re-run train.py once so it generates the pkl."

with open(motion_modes_file, "rb") as f:
    motion_modes = pickle.load(f)
motion_modes = torch.tensor(motion_modes, dtype=torch.float32).cuda()

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
model = TrajectoryModel(
    in_size=2,
    obs_len=args.obs_len,
    pred_len=args.pred_len,
    embed_size=hp_config.model_hidden_dim,
    enc_num_layers=2,
    int_num_layers_list=[1, 1],
    heads=4,
    forward_expansion=2,
)
model = model.cuda()

ckpt_path = os.path.join(args.checkpoint, args.dataset_name, "best.pth")
assert os.path.exists(ckpt_path), \
    f"Checkpoint not found: {ckpt_path}\n" \
    f"Has training finished for '{args.dataset_name}'?"

model.load_state_dict(torch.load(ckpt_path, map_location="cuda"))
model.eval()
print(f"[run_inference] Loaded checkpoint: {ckpt_path}")

# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------
all_samples = []   # list of (B, 20, 12, 2) tensors
all_gt      = []   # list of (B, 12, 2)
all_obs     = []   # list of (B,  8, 2)
all_neis    = []   # list of (B, MaxN, 12, 2) neighbour futures

with torch.no_grad():
    for batch_idx, (ped, neis, mask) in enumerate(test_loader):
        ped  = ped.cuda()
        neis = neis.cuda()
        mask = mask.cuda()

        ped_obs  = ped[:, :args.obs_len]            # (B, 8, 2)
        gt       = ped[:, args.obs_len:]            # (B, 12, 2)
        neis_obs = neis[:, :, :args.obs_len]        # (B, N, 8, 2)

        pred_trajs, scores = model(
            ped_obs, neis_obs, motion_modes, mask,
            None, test=True, num_k=args.num_k
        )
        # pred_trajs: (B, 20, pred_len*2)
        B = ped_obs.shape[0]
        pred_trajs = pred_trajs.reshape(B, args.num_k, args.pred_len, 2)

        # neighbour futures: (B, MaxN, 12, 2)
        neis_fut = neis[:, :, args.obs_len:].cpu().numpy()

        all_samples.append(pred_trajs.cpu().numpy())
        all_gt.append(gt.cpu().numpy())
        all_obs.append(ped_obs.cpu().numpy())
        all_neis.append(neis_fut)

        if (batch_idx + 1) % 10 == 0:
            print(f"  batch {batch_idx+1}/{len(test_loader)}")

# ---------------------------------------------------------------------------
# Concatenate + save
# ---------------------------------------------------------------------------
samples_np = np.concatenate(all_samples, axis=0)   # (N, 20, 12, 2)
gt_np      = np.concatenate(all_gt,      axis=0)   # (N, 12, 2)
obs_np     = np.concatenate(all_obs,     axis=0)   # (N,  8, 2)
max_n = max(a.shape[1] for a in all_neis)
neis_padded = []
for a in all_neis:
    pad = max_n - a.shape[1]
    if pad > 0:
        a = np.concatenate([a, np.full((a.shape[0], pad, a.shape[2], a.shape[3]), 1e9)], axis=1)
    neis_padded.append(a)
neis_np    = np.concatenate(neis_padded, axis=0)   # (N, MaxN, 12, 2)

print(f"\n[run_inference] Total scenes: {samples_np.shape[0]}")
print(f"  samples shape : {samples_np.shape}   (N, K, T, 2)")
print(f"  gt shape      : {gt_np.shape}        (N, T, 2)")
print(f"  obs shape     : {obs_np.shape}        (N, T_obs, 2)")
print(f"  neis shape    : {neis_np.shape}       (N, MaxN, T, 2)")

os.makedirs(args.out_dir, exist_ok=True)
np.save(os.path.join(args.out_dir, f"{args.dataset_name}_samples.npy"), samples_np)
np.save(os.path.join(args.out_dir, f"{args.dataset_name}_gt.npy"),      gt_np)
np.save(os.path.join(args.out_dir, f"{args.dataset_name}_obs.npy"),     obs_np)
np.save(os.path.join(args.out_dir, f"{args.dataset_name}_neis.npy"),    neis_np)

print(f"\n[run_inference] Saved to {args.out_dir}")
print(f"  {args.dataset_name}_samples.npy")
print(f"  {args.dataset_name}_gt.npy")
print(f"  {args.dataset_name}_obs.npy")
print(f"  {args.dataset_name}_neis.npy")
print("[run_inference] Done.\n")
