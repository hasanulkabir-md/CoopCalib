"""
tutr_trajnet_inference.py  --  CoopCalib-TP
Runs TUTR zero-shot inference on the 50 TrajNet++ scenes.
Handles the scene_NNN.txt format directly without needing
the standard ETH-UCY folder structure.

Run from C:\CoopCalib\TUTR:
    cd C:\CoopCalib\TUTR
    python ..\tutr_trajnet_inference.py --variant v0
    python ..\tutr_trajnet_inference.py --variant v2

Outputs:
    C:\CoopCalib\experiments\results\preds_trajnet_v0\trajnet_samples.npy
    C:\CoopCalib\experiments\results\preds_trajnet_v0\trajnet_gt.npy
    C:\CoopCalib\experiments\results\preds_trajnet_v0\trajnet_obs.npy
    C:\CoopCalib\experiments\results\preds_trajnet_v0\trajnet_neis.npy
    (same for v2)
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch

# -- Add TUTR to path (run from TUTR dir) ------------------------------------
TUTR_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR  = os.path.dirname(TUTR_DIR)
sys.path.insert(0, TUTR_DIR)

from model import TrajectoryModel

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--variant", type=str, default="v0",
                    choices=["v0", "v2"],
                    help="v0=baseline, v2=L_ECE+L_energy")
parser.add_argument("--scene_dir", type=str,
                    default=r"..\data\processed\trajnet_sample")
parser.add_argument("--checkpoint_base", type=str,
                    default=r".\checkpoint")
parser.add_argument("--fold", type=str, default="eth",
                    help="Which fold checkpoint to use for zero-shot")
parser.add_argument("--obs_len",   type=int, default=8)
parser.add_argument("--pred_len",  type=int, default=12)
parser.add_argument("--num_k",     type=int, default=20)
parser.add_argument("--gpu",       type=str, default="0")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))

# Checkpoint selection
ckpt_name = "best_v0.pth" if args.variant == "v0" else "best_v2.pth"
ckpt_path = os.path.join(args.checkpoint_base, args.fold, ckpt_name)
out_dir   = os.path.join(BASE_DIR, "experiments", "results",
                         "preds_trajnet_" + args.variant)
os.makedirs(out_dir, exist_ok=True)

print("Variant   : " + args.variant)
print("Checkpoint: " + ckpt_path)
print("Scene dir : " + args.scene_dir)
print("Output dir: " + out_dir)

# ---------------------------------------------------------------------------
# Step 1: Parse scene_NNN.txt files into trajectory windows
# ---------------------------------------------------------------------------
def parse_scene_file(fpath, obs_len=8, pred_len=12):
    """
    Parses one ETH-UCY format txt file.
    Format: frameID personID x y  (tab or space separated)
    Returns list of (obs, gt, all_agents) tuples for each valid window.
    obs: (obs_len, 2), gt: (pred_len, 2)
    """
    data = {}   # personID -> list of (frame, x, y)
    with open(fpath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                frame = int(float(parts[0]))
                pid   = int(float(parts[1]))
                x     = float(parts[2])
                y     = float(parts[3])
            except ValueError:
                continue
            if pid not in data:
                data[pid] = []
            data[pid].append((frame, x, y))

    # Sort each person's trajectory by frame
    for pid in data:
        data[pid].sort(key=lambda t: t[0])

    # Build frame index
    all_frames = sorted(set(f for traj in data.values() for f, x, y in traj))
    if len(all_frames) < obs_len + pred_len:
        return []

    # Slide window over frames
    window_len = obs_len + pred_len
    windows = []

    for start_idx in range(0, len(all_frames) - window_len + 1, obs_len):
        obs_frames  = all_frames[start_idx : start_idx + obs_len]
        pred_frames = all_frames[start_idx + obs_len : start_idx + window_len]

        # Find agents present in all obs+pred frames
        valid_agents = []
        for pid, traj in data.items():
            traj_frames = {f: (x, y) for f, x, y in traj}
            if all(f in traj_frames for f in obs_frames + pred_frames):
                obs_xy  = np.array([traj_frames[f] for f in obs_frames],  dtype=np.float32)
                pred_xy = np.array([traj_frames[f] for f in pred_frames], dtype=np.float32)
                valid_agents.append((pid, obs_xy, pred_xy))

        if len(valid_agents) < 1:
            continue

        windows.append(valid_agents)

    return windows


def load_all_scenes(scene_dir, obs_len=8, pred_len=12):
    """
    Loads all 50 scene txt files, returns list of agent windows.
    Each window: list of (pid, obs (8,2), gt (12,2)) tuples.
    """
    all_windows = []
    scene_files = sorted([f for f in os.listdir(scene_dir)
                           if f.startswith("scene_") and f.endswith(".txt")])
    print("Found " + str(len(scene_files)) + " scene files.")

    for fname in scene_files:
        fpath = os.path.join(scene_dir, fname)
        windows = parse_scene_file(fpath, obs_len, pred_len)
        all_windows.extend(windows)

    print("Total valid windows: " + str(len(all_windows)))
    return all_windows


# ---------------------------------------------------------------------------
# Step 2: Load TUTR model
# ---------------------------------------------------------------------------
def load_model(ckpt_path, obs_len, pred_len, device):
    assert os.path.exists(ckpt_path), \
        "Checkpoint not found: " + ckpt_path
    model = TrajectoryModel(
        in_size=2,
        obs_len=obs_len,
        pred_len=pred_len,
        embed_size=256,
        enc_num_layers=2,
        int_num_layers_list=[1, 1],
        heads=4,
        forward_expansion=2,
    )
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    print("Loaded model from: " + ckpt_path)
    return model


# ---------------------------------------------------------------------------
# Step 3: Load motion modes (needed by TUTR forward pass)
# ---------------------------------------------------------------------------
def load_motion_modes(fold="eth"):
    """
    Loads pre-computed motion modes pkl for the given fold.
    These were generated during training and stored in dataset_path.
    """
    candidates = [
        os.path.join(BASE_DIR, "TUTR", "dataset", fold + "_motion_modes.pkl"),
        os.path.join(BASE_DIR, "data", "splits", fold, "split_full",
                     fold + "_motion_modes.pkl"),
        os.path.join(BASE_DIR, "data", "raw", "ethucy",
                     fold + "_motion_modes.pkl"),
    ]
    # Also search recursively under TUTR dir
    for root, dirs, files in os.walk(os.path.join(BASE_DIR, "TUTR")):
        for f in files:
            if f == fold + "_motion_modes.pkl":
                candidates.insert(0, os.path.join(root, f))
        if len(candidates) > 10:
            break

    for path in candidates:
        if os.path.exists(path):
            with open(path, "rb") as f:
                modes = pickle.load(f)
            print("Loaded motion modes: " + path +
                  "  shape=" + str(np.array(modes).shape))
            return torch.tensor(modes, dtype=torch.float32).to(device)

    # If not found, generate simple linear motion modes as fallback
    print("[WARN] Motion modes pkl not found. Using linear motion fallback.")
    print("       Searched: " + str(candidates[:4]))
    # Generate 20 linear motion templates (straight lines at various angles)
    pred_len_local = args.pred_len
    modes = []
    for i in range(20):
        angle = 2 * np.pi * i / 20
        traj = np.array([[np.cos(angle) * t * 0.3,
                          np.sin(angle) * t * 0.3]
                         for t in range(1, pred_len_local + 1)],
                        dtype=np.float32)
        modes.append(traj)
    modes = np.stack(modes)   # (20, pred_len, 2)
    print("  Generated fallback motion modes: " + str(modes.shape))
    return torch.tensor(modes, dtype=torch.float32).to(device)


# ---------------------------------------------------------------------------
# Step 4: Run inference on all windows
# ---------------------------------------------------------------------------
def run_inference(model, windows, motion_modes, obs_len, pred_len, num_k, device):
    """
    Runs TUTR forward pass on each window individually.
    Returns arrays: samples (N,K,T,2), gt (N,T,2), obs (N,T_obs,2), neis (N,MaxN,T,2)
    """
    all_samples = []
    all_gt      = []
    all_obs     = []
    all_neis    = []

    max_neis = 10   # pad neighbours to this size

    for wi, window in enumerate(windows):
        if len(window) == 0:
            continue

        # Use first agent as ego; rest as neighbours
        ego_pid, ego_obs, ego_gt = window[0]
        neis_data = window[1:] if len(window) > 1 else []

        # Translate to ego-relative coordinates (last obs point = origin)
        origin = ego_obs[-1:].copy()   # (1, 2)
        ego_obs_rel = ego_obs - origin
        ego_gt_rel  = ego_gt  - origin

        # Build neighbour tensor (max_neis, obs_len, 2)
        nei_obs_rel = np.zeros((max_neis, obs_len, 2), dtype=np.float32)
        for ni, (_, n_obs, _) in enumerate(neis_data[:max_neis]):
            nei_obs_rel[ni] = n_obs - origin

        # To tensors: add batch dim
        ped_tensor  = torch.tensor(ego_obs_rel, dtype=torch.float32).unsqueeze(0).to(device)
        neis_tensor = torch.tensor(nei_obs_rel, dtype=torch.float32).unsqueeze(0).to(device)
        mask_tensor = torch.ones(1, max_neis, dtype=torch.float32).to(device)

        # Zero out absent neighbours in mask
        n_real = min(len(neis_data), max_neis)
        if n_real < max_neis:
            mask_tensor[0, n_real:] = 0.0

        with torch.no_grad():
            try:
                pred_trajs, scores = model(
                    ped_tensor, neis_tensor, motion_modes, mask_tensor,
                    None, test=True, num_k=num_k
                )
                # pred_trajs: (1, num_k, pred_len*2) or (1, num_k, pred_len, 2)
                if pred_trajs.ndim == 3:
                    pred_trajs = pred_trajs.reshape(1, num_k, pred_len, 2)
                pred_np = pred_trajs[0].cpu().numpy()   # (K, T, 2)
            except Exception as e:
                if wi < 3:
                    print("  [WARN] window " + str(wi) + " failed: " + str(e))
                continue

        all_samples.append(pred_np)
        all_gt.append(ego_gt_rel)
        all_obs.append(ego_obs_rel)

        # Neighbour futures (relative)
        nei_fut = np.zeros((max_neis, pred_len, 2), dtype=np.float32)
        for ni, (_, _, n_gt) in enumerate(neis_data[:max_neis]):
            nei_fut[ni] = n_gt - origin
        all_neis.append(nei_fut)

        if (wi + 1) % 50 == 0:
            print("  Processed " + str(wi + 1) + "/" + str(len(windows)) + " windows")

    if not all_samples:
        print("[ERROR] No successful inference windows.")
        return None, None, None, None

    samples_np = np.stack(all_samples)   # (N, K, T, 2)
    gt_np      = np.stack(all_gt)        # (N, T, 2)
    obs_np     = np.stack(all_obs)       # (N, T_obs, 2)
    neis_np    = np.stack(all_neis)      # (N, MaxN, T, 2)

    return samples_np, gt_np, obs_np, neis_np


# ---------------------------------------------------------------------------
# Step 5: Compute metrics
# ---------------------------------------------------------------------------
def compute_metrics(samples, gt, obs):
    """
    Computes minADE, minFDE, FPR from raw arrays.
    samples: (N, K, T, 2)
    gt:      (N, T, 2)
    """
    N, K, T, _ = samples.shape

    # Distance of each sample from GT at each timestep
    dists = np.linalg.norm(
        samples - gt[:, np.newaxis, :, :], axis=-1)   # (N, K, T)

    # minADE: min over K of mean displacement
    ade_per_k = dists.mean(axis=2)          # (N, K)
    min_ade   = ade_per_k.min(axis=1)       # (N,)
    mean_ade  = float(min_ade.mean())

    # minFDE: min over K of final displacement
    fde_per_k = dists[:, :, -1]             # (N, K)
    min_fde   = fde_per_k.min(axis=1)       # (N,)
    mean_fde  = float(min_fde.mean())

    # FPR: fraction of agents where ALL K samples have mean disp < threshold
    threshold = 0.5
    all_frozen = (ade_per_k < threshold).all(axis=1)   # (N,) -- all K frozen
    fpr = float(all_frozen.mean())

    # SPSR: fraction where best sample stays within 0.5m of GT at all timesteps
    within = (dists < 0.5).all(axis=2)     # (N, K) -- within at ALL timesteps
    any_safe = within.any(axis=1)          # (N,)
    spsr = float(any_safe.mean())

    # ECE approximation
    conf_per_agent = (dists < 1.5).mean(axis=(1, 2))   # (N,)
    acc_per_agent  = (dists[:, :, -1].min(axis=1) < 1.5).astype(float)
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for i in range(10):
        mask = (conf_per_agent >= bins[i]) & (conf_per_agent < bins[i+1])
        if mask.sum() > 0:
            w = mask.sum() / N
            ece += w * abs(conf_per_agent[mask].mean() - acc_per_agent[mask].mean())

    return {
        "minADE": round(mean_ade, 4),
        "minFDE": round(mean_fde, 4),
        "FPR":    round(fpr, 4),
        "SPSR":   round(spsr, 4),
        "ECE":    round(ece, 4),
        "N":      int(N),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("CoopCalib-TP  --  TrajNet++ Zero-Shot Inference")
    print("=" * 60)

    # Load scenes
    scene_dir = os.path.join(BASE_DIR,
                             "data", "processed", "trajnet_sample")
    if not os.path.isdir(scene_dir):
        scene_dir = args.scene_dir
    windows = load_all_scenes(scene_dir, args.obs_len, args.pred_len)

    if not windows:
        print("[ERROR] No valid trajectory windows found.")
        print("Check scene_dir: " + scene_dir)
        return

    # Load model
    model = load_model(ckpt_path, args.obs_len, args.pred_len, device)

    # Load motion modes
    motion_modes = load_motion_modes(args.fold)

    # Run inference
    print("\nRunning inference on " + str(len(windows)) + " windows ...")
    samples, gt, obs, neis = run_inference(
        model, windows, motion_modes,
        args.obs_len, args.pred_len, args.num_k, device
    )

    if samples is None:
        print("[ERROR] Inference failed.")
        return

    print("\nResults:")
    print("  samples: " + str(samples.shape))
    print("  gt:      " + str(gt.shape))
    print("  obs:     " + str(obs.shape))
    print("  neis:    " + str(neis.shape))

    # Compute metrics
    metrics = compute_metrics(samples, gt, obs)
    print("\nMetrics (TrajNet++ zero-shot, " + args.variant + "):")
    for k, v in metrics.items():
        print("  " + k + ": " + str(v))

    # Save arrays
    np.save(os.path.join(out_dir, "trajnet_samples.npy"), samples)
    np.save(os.path.join(out_dir, "trajnet_gt.npy"),      gt)
    np.save(os.path.join(out_dir, "trajnet_obs.npy"),     obs)
    np.save(os.path.join(out_dir, "trajnet_neis.npy"),    neis)

    # Save metrics JSON
    import json
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved to: " + out_dir)
    print("Done. Run again with --variant v2 for V2 checkpoint.")
    print("Then run generate_figures.py to update Fig 4.")


if __name__ == "__main__":
    main()
