"""
compute_mnd.py — Mean Nearest-Neighbour Distance (MND) from raw ETH-UCY .txt files
CoopCalib-TP | Day 1 Afternoon

WHY raw txt instead of pkl:
  TUTR pkl stores sliding-window samples, each centred on one ego pedestrian
  with the 20 spatially-closest neighbours packed in. Every sample therefore
  looks dense (close neighbours by construction). True scene-level crowd
  density must be measured from raw data: all pedestrians present in a frame.

Raw ETH-UCY .txt format (space-separated):
  frame_id  ped_id  x  y     (world-metre coordinates, already calibrated)

MND per frame:
  For each frame that has >= 2 pedestrians, compute every pedestrian's
  distance to its nearest neighbour, average across all peds.
Scene MND = mean of per-frame MNDs across all frames in that file.
Subset MND = mean across all scene files in that subset.

Usage (from C:\\CoopCalib, venv activated):
    python scripts\\compute_mnd.py
"""

import json
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(r"C:\CoopCalib")
DATA_DIR = BASE_DIR / "TUTR" / "data"
OUT_DIR  = BASE_DIR / "data" / "processed"
OUT_FILE = OUT_DIR / "density_manifest.json"

# Each ETH-UCY subset folder contains ALL txt files (cross-val setup),
# but only specific files are the "home" scenes for that subset.
SUBSET_SCENE_FILES = {
    "eth":    ("eth",    ["biwi_eth.txt"]),
    "hotel":  ("hotel",  ["biwi_hotel.txt"]),
    "univ":   ("univ",   ["students001.txt", "students003.txt"]),
    "zara01": ("zara01", ["crowds_zara01.txt"]),
    "zara02": ("zara02", ["crowds_zara02.txt"]),
}

SPARSE_THR = 2.0
DENSE_THR  = 1.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def classify(mnd):
    if mnd > SPARSE_THR:  return "sparse"
    if mnd < DENSE_THR:   return "dense"
    return "medium"


def load_txt(path):
    """Load raw ETH-UCY txt -> dict: frame_id -> list of (x, y)."""
    frames = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                frame = int(float(parts[0]))
                x, y  = float(parts[2]), float(parts[3])
            except ValueError:
                continue
            frames.setdefault(frame, []).append((x, y))
    return frames


def frame_mnd(positions):
    """MND for one frame. Returns NaN if < 2 peds."""
    if len(positions) < 2:
        return float("nan")
    pos   = np.array(positions)
    diff  = pos[:, None] - pos[None]
    dists = np.sqrt((diff**2).sum(-1))
    np.fill_diagonal(dists, np.inf)
    return float(dists.min(axis=1).mean())


def scene_mnd(txt_path):
    """Compute per-frame MNDs for one scene file."""
    frames    = load_txt(txt_path)
    per_frame = [frame_mnd(frames[fid]) for fid in sorted(frames)]
    per_frame = [v for v in per_frame if not np.isnan(v)]
    if not per_frame:
        return float("nan"), []
    return float(np.mean(per_frame)), per_frame


# ── Per-subset processing ─────────────────────────────────────────────────────

def process_subset(name):
    folder_name, scene_files = SUBSET_SCENE_FILES[name]
    all_frame_mnds = []
    scene_records  = []

    for fname in scene_files:
        txt_path = None
        for split in ("train", "test"):
            candidate = DATA_DIR / folder_name / split / fname
            if candidate.exists():
                txt_path = candidate
                break
        if txt_path is None:
            print(f"  WARNING: {fname} not found")
            continue

        mnd, frame_mnds = scene_mnd(txt_path)
        print(f"  {fname:30s}  frames={len(frame_mnds):4d}  mean_mnd={mnd:.4f} m")
        all_frame_mnds.extend(frame_mnds)
        scene_records.append({"file": fname, "mean_mnd_m": round(mnd, 4),
                               "n_frames": len(frame_mnds)})

    if not all_frame_mnds:
        raise RuntimeError(f"No valid frames for {name}")

    arr      = np.array(all_frame_mnds)
    mean_mnd = float(arr.mean())
    tiers    = [classify(v) for v in all_frame_mnds]
    counts   = {t: tiers.count(t) for t in ("sparse", "medium", "dense")}

    print(f"  --> mean MND: {mean_mnd:.4f} m  |  "
          f"S/M/D frames: {counts['sparse']}/{counts['medium']}/{counts['dense']}")

    return {
        "mean_mnd_m":   round(mean_mnd, 4),
        "sparse":       counts["sparse"],
        "medium":       counts["medium"],
        "dense":        counts["dense"],
        "total_frames": len(all_frame_mnds),
        "scenes":       scene_records,
        "mnd_percentiles": {
            "p10": round(float(np.percentile(arr, 10)), 4),
            "p25": round(float(np.percentile(arr, 25)), 4),
            "p50": round(float(np.percentile(arr, 50)), 4),
            "p75": round(float(np.percentile(arr, 75)), 4),
            "p90": round(float(np.percentile(arr, 90)), 4),
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    totals   = {"sparse": 0, "medium": 0, "dense": 0, "total_frames": 0}

    for name in SUBSET_SCENE_FILES:
        print(f"\n[{name.upper()}]")
        result = process_subset(name)
        manifest[name] = result
        for tier in ("sparse", "medium", "dense"):
            totals[tier] += result[tier]
        totals["total_frames"] += result["total_frames"]

    manifest["summary"] = totals

    with open(OUT_FILE, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅  density_manifest.json  ->  {OUT_FILE}")
    print(f"   Total frames : {totals['total_frames']}")
    print(f"   Sparse / Medium / Dense : {totals['sparse']} / {totals['medium']} / {totals['dense']}")

    expected_dominant = {
        "eth":    "sparse",
        "hotel":  "dense",
        "univ":   "dense",
        "zara01": "medium",
        "zara02": "medium",
    }
    print("\n── Density tier validation (proposal §4.2) ──────────────────────────")
    all_ok = True
    for name, exp in expected_dominant.items():
        r        = manifest[name]
        dominant = max(("sparse", "medium", "dense"), key=lambda t: r[t])
        ok       = dominant == exp
        if not ok: all_ok = False
        flag     = "OK" if ok else "UNEXPECTED"
        print(f"  {name:8s}  mean_mnd={r['mean_mnd_m']:.3f}m  "
              f"dominant={dominant:6s}  expected={exp:6s}  {flag}")
    print()
    if all_ok:
        print("All subsets match expected density tiers -- Day 1 COMPLETE.")
    else:
        print("Mismatch found. Actual MNDs printed above.")
        print("Proposal thresholds (sparse>2m, dense<1m) may need adjusting")
        print("to match the real ETH-UCY MND distribution.")


if __name__ == "__main__":
    main()
