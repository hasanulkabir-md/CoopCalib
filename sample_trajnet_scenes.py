"""
sample_trajnet_scenes.py  (fixed)
Samples 50 pedestrian-dense interaction scenes from existing ETH-UCY splits
for Day 9 zero-shot TrajNet++-style evaluation.

Output: C:\CoopCalib\data\processed\trajnet_sample\
  scene_000.txt ... scene_049.txt  (each = one SEQ_LEN window, 2+ peds)
  scene_index.json
"""

import os, random, json
import numpy as np
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
SPLITS_ROOT = r"C:\CoopCalib\data\splits"
OUT_DIR     = r"C:\CoopCalib\data\processed\trajnet_sample"
OBS_LEN     = 8
PRED_LEN    = 12
SEQ_LEN     = OBS_LEN + PRED_LEN   # 20 frames
MIN_PEDS    = 2                     # interaction scenes only
N_SCENES    = 50
SEED        = 42

random.seed(SEED)
np.random.seed(SEED)

# All split_full/train folders across all folds
FOLDS = ["eth", "hotel", "univ", "zara1", "zara2"]

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_txt(filepath):
    """Load ETH-UCY txt (tab or space sep) -> dict: frameID(int) -> {pid(int): (x,y)}"""
    frames = defaultdict(dict)
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            fid = int(float(parts[0]))
            pid = int(float(parts[1]))
            x, y = float(parts[2]), float(parts[3])
            frames[fid][pid] = (x, y)
    return frames

def extract_scenes(frames, source_name):
    """
    Slide SEQ_LEN-length windows over sorted frame IDs.
    No contiguity requirement — just take sequential index windows.
    Keep windows where MIN_PEDS are present in ALL frames.
    """
    frame_ids = sorted(frames.keys())
    scenes = []
    step = 4  # stride to avoid massive overlap
    for i in range(0, len(frame_ids) - SEQ_LEN + 1, step):
        window = frame_ids[i:i + SEQ_LEN]
        if len(window) < SEQ_LEN:
            continue
        # peds present in every frame of the window
        present = set(frames[window[0]].keys())
        for fid in window[1:]:
            present &= set(frames[fid].keys())
        if len(present) >= MIN_PEDS:
            scenes.append((source_name, window, sorted(present)))
    return scenes

# ── Collect candidates ────────────────────────────────────────────────────────
all_scenes = []

for fold in FOLDS:
    train_dir = os.path.join(SPLITS_ROOT, fold, "split_full", "train")
    if not os.path.exists(train_dir):
        print(f"  [SKIP] {train_dir} not found")
        continue
    for fname in sorted(os.listdir(train_dir)):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(train_dir, fname)
        frames = load_txt(fpath)
        scenes = extract_scenes(frames, fname)
        all_scenes.extend([(fpath, w, p) for (_, w, p) in scenes])
        print(f"  {fold}/{fname}: {len(frames)} frames, {len(scenes)} candidate scenes")

print(f"\nTotal candidate scenes: {len(all_scenes)}")

if len(all_scenes) == 0:
    print("ERROR: No scenes found. Check SPLITS_ROOT path.")
    exit(1)

if len(all_scenes) < N_SCENES:
    print(f"WARNING: Only {len(all_scenes)} candidates, sampling all of them.")
    N_SCENES = len(all_scenes)

# ── Weighted sample (more peds = more likely) ─────────────────────────────────
weights = np.array([len(p) for (_, _, p) in all_scenes], dtype=float)
weights /= weights.sum()

chosen_idx = np.random.choice(len(all_scenes), size=N_SCENES, replace=False, p=weights)
print(f"Sampling {N_SCENES} scenes weighted by pedestrian count...\n")

# ── Write output ──────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
scene_log = []

for out_i, idx in enumerate(chosen_idx):
    fpath, window, peds = all_scenes[idx]
    frames = load_txt(fpath)
    rows = []
    for fid in window:
        for pid in peds:
            if pid in frames[fid]:
                x, y = frames[fid][pid]
                rows.append(f"{fid}\t{pid}\t{x:.4f}\t{y:.4f}")

    out_file = os.path.join(OUT_DIR, f"scene_{out_i:03d}.txt")
    with open(out_file, "w") as f:
        f.write("\n".join(rows))

    source = os.path.basename(fpath)
    entry = {"scene": out_i, "source": source, "n_peds": len(peds),
             "frame_start": window[0], "frame_end": window[-1]}
    scene_log.append(entry)
    print(f"  scene_{out_i:03d} — {source}, {len(peds)} peds, "
          f"frames {window[0]}–{window[-1]}")

# ── Save index ────────────────────────────────────────────────────────────────
index_path = os.path.join(OUT_DIR, "scene_index.json")
with open(index_path, "w") as f:
    json.dump(scene_log, f, indent=2)

ped_counts = [e["n_peds"] for e in scene_log]
print(f"\n{'='*50}")
print(f"DONE — {N_SCENES} scenes -> {OUT_DIR}")
print(f"Avg peds per scene : {np.mean(ped_counts):.1f}")
print(f"Max peds per scene : {max(ped_counts)}")
print(f"Scene index        : {index_path}")
