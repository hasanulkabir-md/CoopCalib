"""
make_splits.py — Ultra-low-data split generator for RQ3 KL collapse experiment.

Reads raw ETH-UCY .txt files from SocialVAE's data/ folder and creates
subsampled train splits at N=500, 1000, 1500, and full trajectory counts.

Output structure:
    C:\CoopCalib\data\splits\
        eth\
            split_500\train\    <- 500 trajectories sampled from eth train
            split_1000\train\
            split_1500\train\
            split_full\train\   <- symlink/copy of original (all trajectories)
        hotel\  ...
        univ\   ...
        zara1\  ...
        zara2\  ...

Each split_N\train\ contains the same .txt filenames as the original but with
rows subsampled so that the total trajectory count is approximately N.
A trajectory = one unique pedestrian ID in a contiguous block.

Usage:
    cd C:\CoopCalib
    python make_splits.py

Then train SocialVAE on a split:
    cd models\socialvae
    python main.py --config config/eth.py \
        --train C:\CoopCalib\data\splits\eth\split_500\train \
        --test data/eth/test --device cuda --lambda3 0.0
"""

import os
import shutil
import random
import numpy as np
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
SOCIALVAE_DATA = r"C:\CoopCalib\models\socialvae\data"
OUTPUT_ROOT    = r"C:\CoopCalib\data\splits"
FOLDS          = ["eth", "hotel", "univ", "zara1", "zara2"]
SPLIT_SIZES    = [500, 1000, 1500]   # "full" is always added automatically
SEED           = 42

random.seed(SEED)
np.random.seed(SEED)

# ── Helpers ───────────────────────────────────────────────────────────────────

def read_txt(path):
    """Read a standard ETH-UCY txt file.
    Format: frame_id  ped_id  x  y  (whitespace separated)
    Returns list of lines (strings, no newline stripping beyond rstrip).
    """
    with open(path, "r") as f:
        return [l.rstrip("\n") for l in f if l.strip()]


def parse_trajectories(lines):
    """Group lines by ped_id.
    Returns dict: ped_id -> list of lines belonging to that pedestrian.
    """
    traj = defaultdict(list)
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        ped_id = parts[1]
        traj[ped_id].append(line)
    return traj


def count_trajectories(train_dir):
    """Count total unique pedestrian trajectories across all txt files."""
    total = set()
    for fname in os.listdir(train_dir):
        if not fname.endswith(".txt"):
            continue
        lines = read_txt(os.path.join(train_dir, fname))
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                total.add((fname, parts[1]))
    return len(total)


def subsample_files(train_dir, target_n, out_dir):
    """
    Subsample trajectories from train_dir so total count ≈ target_n.
    Writes subsampled txt files to out_dir.
    Preserves original file structure (same filenames).
    """
    os.makedirs(out_dir, exist_ok=True)

    # Collect all (filename, ped_id) pairs
    file_trajs = {}   # fname -> {ped_id: [lines]}
    all_keys   = []   # list of (fname, ped_id)

    for fname in sorted(os.listdir(train_dir)):
        if not fname.endswith(".txt"):
            continue
        path  = os.path.join(train_dir, fname)
        lines = read_txt(path)
        traj  = parse_trajectories(lines)
        file_trajs[fname] = traj
        for pid in traj:
            all_keys.append((fname, pid))

    total = len(all_keys)
    if target_n >= total:
        # Just copy everything
        for fname in file_trajs:
            src = os.path.join(train_dir, fname)
            shutil.copy2(src, os.path.join(out_dir, fname))
        return total

    # Randomly sample target_n trajectories
    random.shuffle(all_keys)
    selected = set(all_keys[:target_n])

    # Write out only the selected trajectories, preserving file grouping
    actual = 0
    for fname, traj in file_trajs.items():
        kept_lines = []
        for pid, lines in traj.items():
            if (fname, pid) in selected:
                kept_lines.extend(lines)
                actual += 1

        if not kept_lines:
            # Write empty placeholder so SocialVAE doesn't error on missing file
            # (actually skip — SocialVAE scans dir, empty files cause issues)
            continue

        # Sort by frame_id to maintain temporal order
        kept_lines.sort(key=lambda l: float(l.split()[0]))

        with open(os.path.join(out_dir, fname), "w") as f:
            f.write("\n".join(kept_lines) + "\n")

    return actual


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("CoopCalib-TP — make_splits.py")
    print(f"Source: {SOCIALVAE_DATA}")
    print(f"Output: {OUTPUT_ROOT}")
    print()

    summary = {}

    for fold in FOLDS:
        train_dir = os.path.join(SOCIALVAE_DATA, fold, "train")
        if not os.path.isdir(train_dir):
            print(f"[SKIP] {fold}: train dir not found at {train_dir}")
            continue

        full_count = count_trajectories(train_dir)
        print(f"Fold: {fold}  (full = {full_count} trajectories)")
        summary[fold] = {}

        # Create splits
        for n in SPLIT_SIZES:
            out_dir = os.path.join(OUTPUT_ROOT, fold, f"split_{n}", "train")
            actual  = subsample_files(train_dir, n, out_dir)
            summary[fold][n] = actual
            print(f"  split_{n:>5}: {actual:>5} trajectories -> {out_dir}")

        # Full split — just copy
        out_full = os.path.join(OUTPUT_ROOT, fold, "split_full", "train")
        os.makedirs(out_full, exist_ok=True)
        for fname in os.listdir(train_dir):
            if fname.endswith(".txt"):
                shutil.copy2(
                    os.path.join(train_dir, fname),
                    os.path.join(out_full, fname)
                )
        summary[fold]["full"] = full_count
        print(f"  split_full : {full_count:>5} trajectories -> {out_full}")
        print()

    # Print summary table
    print("=" * 60)
    print("SPLIT SUMMARY")
    print("=" * 60)
    header = f"{'Fold':<8}" + "".join(f"{'N='+str(n):>8}" for n in SPLIT_SIZES) + f"{'Full':>8}"
    print(header)
    print("-" * 60)
    for fold in FOLDS:
        if fold not in summary:
            continue
        row = f"{fold:<8}"
        for n in SPLIT_SIZES:
            row += f"{summary[fold].get(n, 0):>8}"
        row += f"{summary[fold].get('full', 0):>8}"
        print(row)
    print("=" * 60)
    print()
    print("Done. To train SocialVAE on a split (from models\\socialvae\\):")
    print()
    print("  python main.py --config config/eth.py \\")
    print(r"      --train C:\CoopCalib\data\splits\eth\split_500\train \\")
    print("      --test data/eth/test --device cuda --lambda3 0.0 \\")
    print(r"      --ckpt C:\CoopCalib\experiments\results\socialvae\eth_500_v0")
    print()
    print("Run all 4 sizes x 2 versions (lambda3=0.0 and 0.1) = 8 runs per fold.")


if __name__ == "__main__":
    main()
