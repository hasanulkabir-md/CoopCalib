"""
preprocess_sdd.py
Converts SDD raw annotations -> ETH-UCY format .txt files (pedestrian only).
Output: C:\CoopCalib\data\processed\sdd\  with train/ and test/ splits.

SDD annotation format (space-separated):
  trackID x1 y1 x2 y2 frameID lost occluded generated label

Output ETH-UCY format (tab-separated):
  frameID personID x y

Usage:
  python preprocess_sdd.py
"""

import os
import numpy as np
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
SDD_ROOT   = r"C:\CoopCalib\data\raw\sdd"
OUT_ROOT   = r"C:\CoopCalib\data\processed\sdd"
FRAME_SKIP = 12          # subsample: ~2.5 fps matching ETH-UCY
MIN_LEN    = 20          # minimum trajectory length (frames after subsampling)
SCALE      = 0.05        # pixel -> meter approx (SDD ~20px per meter)

# PECNet standard split — scenes used for test (held out), rest for train
# This matches the split used in most trajectory prediction papers
TEST_SCENES = {
    ("deathCircle", "video0"),
    ("deathCircle", "video1"),
    ("hyang",       "video3"),
    ("little",      "video1"),
    ("nexus",       "video5"),
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_annotations(filepath):
    """Parse one SDD annotations.txt -> list of pedestrian trajectory dicts."""
    tracks = defaultdict(list)  # trackID -> [(frameID, cx, cy)]
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            label = parts[9].strip('"')
            if label != "Pedestrian":
                continue
            track_id = int(parts[0])
            x1, y1, x2, y2 = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            frame_id = int(parts[5])
            lost = int(parts[6])
            if lost == 1:
                continue  # skip lost detections
            cx = (x1 + x2) / 2.0 * SCALE
            cy = (y1 + y2) / 2.0 * SCALE
            tracks[track_id].append((frame_id, cx, cy))
    return tracks


def subsample_track(points, skip):
    """Keep every `skip`-th frame; return sorted list."""
    points_sorted = sorted(points, key=lambda p: p[0])
    return points_sorted[::skip]


def tracks_to_ethucy(tracks, global_pid_offset=0):
    """
    Convert tracks dict to ETH-UCY rows: (frameID, personID, x, y).
    Renumber frameIDs to be contiguous starting at 0.
    """
    rows = []
    pid = global_pid_offset
    all_frames = set()

    filtered = {}
    for tid, pts in tracks.items():
        sub = subsample_track(pts, FRAME_SKIP)
        if len(sub) >= MIN_LEN:
            filtered[tid] = sub
            for (f, x, y) in sub:
                all_frames.add(f)

    if not filtered:
        return rows, pid

    frame_list = sorted(all_frames)
    frame_map  = {f: i for i, f in enumerate(frame_list)}

    for tid, pts in filtered.items():
        for (f, x, y) in pts:
            rows.append((frame_map[f], pid, x, y))
        pid += 1

    return rows, pid


def write_ethucy(rows, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    rows_sorted = sorted(rows, key=lambda r: (r[0], r[1]))
    with open(filepath, "w") as f:
        for (fid, pid, x, y) in rows_sorted:
            f.write(f"{fid}\t{pid}\t{x:.4f}\t{y:.4f}\n")
    return len(rows_sorted)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    scenes = []
    for scene in sorted(os.listdir(SDD_ROOT)):
        scene_path = os.path.join(SDD_ROOT, scene)
        if not os.path.isdir(scene_path):
            continue
        for video in sorted(os.listdir(scene_path)):
            ann_path = os.path.join(scene_path, video, "annotations.txt")
            if os.path.exists(ann_path):
                scenes.append((scene, video, ann_path))

    print(f"Found {len(scenes)} scene/video pairs\n")

    train_scenes, test_scenes = [], []
    for (scene, video, ann_path) in scenes:
        if (scene, video) in TEST_SCENES:
            test_scenes.append((scene, video, ann_path))
        else:
            train_scenes.append((scene, video, ann_path))

    print(f"Train scenes: {len(train_scenes)}  |  Test scenes: {len(test_scenes)}\n")

    pid_counter = 0

    # ── Write train ──────────────────────────────────────────────────────────
    print("=== Processing TRAIN ===")
    for (scene, video, ann_path) in train_scenes:
        tracks = parse_annotations(ann_path)
        rows, pid_counter = tracks_to_ethucy(tracks, pid_counter)
        if not rows:
            print(f"  [{scene}/{video}] — no valid pedestrian tracks, skipping")
            continue
        out_file = os.path.join(OUT_ROOT, "train", f"{scene}_{video}.txt")
        n = write_ethucy(rows, out_file)
        print(f"  [{scene}/{video}] — {n} rows written -> {out_file}")

    # ── Write test ───────────────────────────────────────────────────────────
    print("\n=== Processing TEST ===")
    for (scene, video, ann_path) in test_scenes:
        tracks = parse_annotations(ann_path)
        rows, pid_counter = tracks_to_ethucy(tracks, pid_counter)
        if not rows:
            print(f"  [{scene}/{video}] — no valid pedestrian tracks, skipping")
            continue
        out_file = os.path.join(OUT_ROOT, "test", f"{scene}_{video}.txt")
        n = write_ethucy(rows, out_file)
        print(f"  [{scene}/{video}] — {n} rows written -> {out_file}")

    # ── Summary ──────────────────────────────────────────────────────────────
    train_dir = os.path.join(OUT_ROOT, "train")
    test_dir  = os.path.join(OUT_ROOT, "test")
    train_files = os.listdir(train_dir) if os.path.exists(train_dir) else []
    test_files  = os.listdir(test_dir)  if os.path.exists(test_dir)  else []

    print(f"\n{'='*50}")
    print(f"DONE — Output: {OUT_ROOT}")
    print(f"  Train files : {len(train_files)}")
    print(f"  Test files  : {len(test_files)}")
    print(f"  Total persons: {pid_counter}")
    print(f"\nVerify with:")
    print(f"  python -c \"import numpy as np; d=open(r'{train_dir}\\\\bookstore_video1.txt').readlines(); print(d[:3])\"")


if __name__ == "__main__":
    main()
