"""
Runtime profiling for CoopCalib-TP
Measures training and inference time for all components.

FIXES APPLIED (vs original):
  1. Switched from Dataloader (dataloader.py) to TrajectoryDataset (dataset.py)
     - Dataloader returns (x, y, neighbor) with 6D features and NO mask
     - TrajectoryDataset returns (ped, neis, mask) with 2D features — the format
       TrajectoryModel.forward() expects
  2. Added motion_modes generation via get_motion_modes() from utils.py
     (or loads from .pkl cache if available)
  3. Added get_cls_label() to compute closest_mode_indices per batch
  4. Fixed model call:
     - Training: model(ped_obs, neis_obs, motion_modes, mask, closest_mode_indices)
     - Inference: model(ped_obs, neis_obs, motion_modes, mask, None, test=True)
  5. Fixed loss computation — model returns (pred_traj, scores), not a single tensor
  6. Fixed inference loop — model returns (pred_trajs, scores) in test mode
"""
import torch
import torch.nn.functional as F
import time
import numpy as np
import json
import sys
import os
import pickle
import importlib
import concurrent.futures
from pathlib import Path

# ═══════════════════════════════════════════════════════════
# WINDOWS MULTIPROCESSING PATCH
# ═══════════════════════════════════════════════════════════
original_init = concurrent.futures.ProcessPoolExecutor.__init__
def patched_init(self, max_workers=None, *args, **kwargs):
    if max_workers is None or max_workers <= 0:
        max_workers = 1
    original_init(self, max_workers=max_workers, *args, **kwargs)
concurrent.futures.ProcessPoolExecutor.__init__ = patched_init


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cls_label(gt, motion_modes, soft_label=True):
    """Compute soft classification labels and closest mode indices.

    Args:
        gt: [B, pred_len, 2] ground truth future trajectory
        motion_modes: [K, pred_len, 2] K cluster centers
    Returns:
        soft_label: [B, K] softmax over negative distances
        closest_mode_indices: [B] index of closest mode per sample
    """
    gt = gt.reshape(gt.shape[0], -1).unsqueeze(1)                       # [B, 1, pred_len*2]
    motion_modes_flat = motion_modes.reshape(motion_modes.shape[0], -1).unsqueeze(0)  # [1, K, pred_len*2]
    distance = torch.norm(gt - motion_modes_flat, dim=-1)               # [B, K]
    soft_label = F.softmax(-distance, dim=-1)                           # [B, K]
    closest_mode_indices = torch.argmin(distance, dim=-1)               # [B]
    return soft_label, closest_mode_indices


# ── Per-dataset hyper-parameters (from TUTR config/*.py) ──────────────
DATASET_CONFIGS = {
    "eth":   {"n_clusters": 50, "dist_threshold": 2, "smooth_size": 3,
              "random_rotation": True, "traj_seg": False},
    "hotel": {"n_clusters": 50, "dist_threshold": 2, "smooth_size": 3,
              "random_rotation": True, "traj_seg": False},
    "univ":  {"n_clusters": 50, "dist_threshold": 2, "smooth_size": 3,
              "random_rotation": True, "traj_seg": False},
    "zara1": {"n_clusters": 50, "dist_threshold": 2, "smooth_size": 3,
              "random_rotation": True, "traj_seg": False},
    "zara2": {"n_clusters": 50, "dist_threshold": 2, "smooth_size": 3,
              "random_rotation": True, "traj_seg": False},
}

OBS_LEN  = 8
PRED_LEN = 12


def measure_tutr_runtime():
    """Measure TUTR training and inference on actual ETH-UCY data"""

    tutr_path = r'C:\CoopCalib\TUTR'
    if tutr_path not in sys.path:
        sys.path.insert(0, tutr_path)
    if r'C:\CoopCalib' not in sys.path:
        sys.path.append(r'C:\CoopCalib')

    try:
        from model import TrajectoryModel as TUTR
        from dataset import TrajectoryDataset       # ← CORRECT dataloader
        from utils import get_motion_modes
        from torch.utils.data import DataLoader
    except ImportError as e:
        print(f"Critical Import Error: {e}")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Runtime Analysis] Device: {device}")

    results = {
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "training": {}, "inference": {}, "metrics": {}
    }

    subsets = ["eth", "hotel", "univ", "zara1", "zara2"]
    dataset_path = os.path.join(tutr_path, "dataset") + os.sep   # e.g. C:\CoopCalib\TUTR\dataset\

    # ═══════════════════════════════════════════════════════════
    # 1. TRAINING TIME PER EPOCH
    # ═══════════════════════════════════════════════════════════
    for dataset in subsets:
        print(f"\n[Training] {dataset.upper()}")

        cfg = DATASET_CONFIGS[dataset]

        # --- Check that the .pkl data files exist --------------------
        train_pkl = os.path.join(dataset_path, f"{dataset}_train.pkl")
        test_pkl  = os.path.join(dataset_path, f"{dataset}_test.pkl")
        if not os.path.exists(train_pkl):
            print(f"  [!] Missing {train_pkl}")
            print(f"      Run first:  python get_data_pkl.py "
                  f"--train data/{dataset}/train --test data/{dataset}/test "
                  f"--config config/{dataset}.py")
            continue

        # --- Build dataset (TrajectoryDataset from dataset.py) ------
        train_dataset = TrajectoryDataset(
            dataset_path=dataset_path, dataset_name=dataset,
            dataset_type='train', translation=True, rotation=True,
            scaling=True, obs_len=OBS_LEN,
            dist_threshold=cfg["dist_threshold"], smooth=False
        )
        train_loader = DataLoader(
            train_dataset, collate_fn=train_dataset.coll_fn,
            batch_size=128, shuffle=True, num_workers=0   # num_workers=0 for Windows
        )

        # --- Motion modes (generate once or load from cache) --------
        motion_modes_file = os.path.join(dataset_path, f"{dataset}_motion_modes.pkl")
        if os.path.exists(motion_modes_file):
            print(f"  Loading cached motion modes from {motion_modes_file}")
            with open(motion_modes_file, 'rb') as f:
                motion_modes = pickle.load(f)
            motion_modes = torch.tensor(motion_modes, dtype=torch.float32).to(device)
        else:
            print(f"  Generating motion modes ({cfg['n_clusters']} clusters)...")
            motion_modes = get_motion_modes(
                train_dataset, OBS_LEN, PRED_LEN,
                cfg["n_clusters"], dataset_path, dataset,
                smooth_size=cfg["smooth_size"],
                random_rotation=cfg["random_rotation"],
                traj_seg=cfg["traj_seg"]
            )
            motion_modes = torch.tensor(motion_modes, dtype=torch.float32).to(device)

        # --- Initialize Model --------------------------------------
        model = TUTR(
            in_size=2, obs_len=OBS_LEN, pred_len=PRED_LEN,
            embed_size=64, enc_num_layers=1,
            int_num_layers_list=[1, 1], heads=4, forward_expansion=2
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        reg_criterion = torch.nn.SmoothL1Loss().to(device)
        cls_criterion = torch.nn.CrossEntropyLoss().to(device)

        epoch_times = []
        for ep in range(3):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()
            model.train()

            for ped, neis, mask in train_loader:
                ped  = ped.to(device)     # [B, T_total, 2]
                neis = neis.to(device)    # [B, N, T_total, 2]
                mask = mask.to(device)    # [B, N, N]

                # Split obs / future
                ped_obs  = ped[:, :OBS_LEN]            # [B, obs_len, 2]
                gt       = ped[:, OBS_LEN:]            # [B, pred_len, 2]
                neis_obs = neis[:, :, :OBS_LEN]        # [B, N, obs_len, 2]

                # Compute closest mode indices from ground truth
                with torch.no_grad():
                    soft_label, closest_mode_indices = get_cls_label(gt, motion_modes)

                optimizer.zero_grad()
                pred_traj, scores = model(
                    ped_obs, neis_obs, motion_modes, mask, closest_mode_indices
                )

                reg_label = gt.reshape(pred_traj.shape)             # [B, pred_len*2]
                reg_loss  = reg_criterion(pred_traj, reg_label)
                clf_loss  = cls_criterion(scores.squeeze(), soft_label)
                loss = reg_loss + clf_loss

                loss.backward()
                optimizer.step()

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            epoch_times.append(time.perf_counter() - start)

        median_time = np.median(epoch_times)
        results["training"][dataset] = {"seconds_per_epoch": round(median_time, 2)}
        print(f"  → {median_time:.2f}s per epoch")

    # ═══════════════════════════════════════════════════════════
    # 2. INFERENCE TIME (K=20)
    # ═══════════════════════════════════════════════════════════
    for dataset in subsets:
        if dataset not in results["training"]:
            continue
        print(f"\n[Inference] {dataset.upper()}")

        cfg = DATASET_CONFIGS[dataset]

        test_pkl = os.path.join(dataset_path, f"{dataset}_test.pkl")
        if not os.path.exists(test_pkl):
            print(f"  [!] Missing {test_pkl}, skipping inference.")
            continue

        test_dataset = TrajectoryDataset(
            dataset_path=dataset_path, dataset_name=dataset,
            dataset_type='test', translation=True, rotation=True,
            scaling=False, obs_len=OBS_LEN,
            dist_threshold=cfg["dist_threshold"], smooth=False
        )
        test_loader = DataLoader(
            test_dataset, collate_fn=test_dataset.coll_fn,
            batch_size=1, shuffle=False, num_workers=0
        )

        model.eval()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()

        with torch.no_grad():
            for ped, neis, mask in test_loader:
                ped  = ped.to(device)
                neis = neis.to(device)
                mask = mask.to(device)

                ped_obs  = ped[:, :OBS_LEN]
                neis_obs = neis[:, :, :OBS_LEN]

                for _ in range(20):
                    _ = model(ped_obs, neis_obs, motion_modes, mask, None, test=True)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start
        results["inference"][dataset] = {"total_seconds": round(elapsed, 2)}
        print(f"  → {elapsed:.2f}s total inference")

    # ═══════════════════════════════════════════════════════════
    # 3. SAVE RESULTS
    # ═══════════════════════════════════════════════════════════
    output_json = Path(r'C:\CoopCalib\experiments\results\runtime_profile.json')
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    set_seed(42)
    print("="*60)
    print("  CoopCalib-TP Runtime Profiling (Windows-Patched)")
    print("="*60)

    profile_data = measure_tutr_runtime()

    if profile_data:
        print("\n[DONE] Runtime profiling complete.")
        print(f"Results saved to: C:\\CoopCalib\\experiments\\results\\runtime_profile.json")
