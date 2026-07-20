"""
CoopCalib-TP — Regenerate V0 predictions WITH neis
===================================================
File:  C:\\CoopCalib\\scripts\\regen_v0_neis.py

run_inference.py always loads best.pth.
This script safely swaps best_v0.pth into best.pth,
runs inference for all 5 subsets, saves to preds_v0_neis/,
then restores best.pth exactly as it was.

Usage (from C:\\CoopCalib\\TUTR\\):
    python ..\scripts\regen_v0_neis.py

Output:
    C:\\CoopCalib\\experiments\\results\\preds_v0_neis\\
        {name}_samples.npy
        {name}_gt.npy
        {name}_obs.npy
        {name}_neis.npy     <-- this is what was missing
"""

import os
import sys
import shutil
import subprocess

TUTR_DIR    = os.getcwd()   # must be run from TUTR\
SCRIPTS_DIR = os.path.join(TUTR_DIR, "..", "scripts")
ROOT        = os.path.join(TUTR_DIR, "..")
CKPT_DIR    = os.path.join(TUTR_DIR, "checkpoint")
OUT_DIR     = os.path.join(ROOT, "experiments", "results", "preds_v0_neis")
INFERENCE   = os.path.join(SCRIPTS_DIR, "run_inference.py")

DATASETS = ["eth", "hotel", "univ", "zara1", "zara2"]

# Verify running from TUTR dir
if not os.path.exists(os.path.join(TUTR_DIR, "train.py")):
    print("ERROR: Run from C:\\CoopCalib\\TUTR\\")
    sys.exit(1)

# Verify all V0 checkpoints exist
for ds in DATASETS:
    v0_path = os.path.join(CKPT_DIR, ds, "best_v0.pth")
    if not os.path.exists(v0_path):
        print(f"ERROR: Missing {v0_path}")
        sys.exit(1)

os.makedirs(OUT_DIR, exist_ok=True)

print(f"\n{'='*60}")
print(f"  Regenerating V0 predictions with neis")
print(f"  Output: {OUT_DIR}")
print(f"{'='*60}\n")

for ds in DATASETS:
    best_pth   = os.path.join(CKPT_DIR, ds, "best.pth")
    v0_pth     = os.path.join(CKPT_DIR, ds, "best_v0.pth")
    backup_pth = os.path.join(CKPT_DIR, ds, "best_REGEN_BACKUP.pth")

    print(f"  [{ds.upper()}]")

    # Back up current best.pth
    if os.path.exists(best_pth):
        shutil.copy2(best_pth, backup_pth)

    # Swap V0 into place
    shutil.copy2(v0_pth, best_pth)

    # Run inference
    cmd = [
        sys.executable, INFERENCE,
        "--dataset_name", ds,
        "--out_dir", OUT_DIR,
        "--num_works", "0",
    ]
    result = subprocess.run(cmd, cwd=TUTR_DIR)

    # Restore original best.pth always
    if os.path.exists(backup_pth):
        shutil.copy2(backup_pth, best_pth)
        os.remove(backup_pth)

    if result.returncode != 0:
        print(f"  ERROR: inference failed for {ds}")
        sys.exit(1)

    # Verify neis saved
    neis_path = os.path.join(OUT_DIR, f"{ds}_neis.npy")
    if os.path.exists(neis_path):
        print(f"  ✓ {ds}_neis.npy saved\n")
    else:
        print(f"  WARNING: {ds}_neis.npy not found\n")

print(f"{'='*60}")
print(f"  V0 neis regeneration complete.")
print(f"  Next: python scripts\\compute_metrics_with_svr.py")
print(f"        --preds_dir experiments\\results\\preds_v0_neis")
print(f"        --out_file  experiments\\results\\v0_metrics_full.json")
print(f"{'='*60}\n")
