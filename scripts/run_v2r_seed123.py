"""
run_v2r_seed123.py
Trains V2R on all 5 ETH-UCY subsets with seed=123, backs up each checkpoint
immediately after training, then runs inference and saves metrics.
Run from: C:\CoopCalib\TUTR\
"""
import subprocess
import shutil
import os
import sys

PYTHON   = sys.executable
TUTR_DIR = r"C:\CoopCalib\TUTR"
OUT_DIR  = r"C:\CoopCalib\experiments\results\preds_v2r_seed123"

SUBSETS  = ["eth", "hotel", "univ", "zara1", "zara2"]

TRAIN_ARGS = {
    "seed":         "123",
    "lambda1":      "0.0",
    "lambda2":      "0.1",
    "lambda3":      "0.5",
    "warmup_epoch": "0",
    "gpu":          "0",
    "num_works":    "0",
}

os.makedirs(OUT_DIR, exist_ok=True)

# ── STAGE 1: Train all subsets ──────────────────────────────────────────────
print("\n" + "="*60)
print("  STAGE 1: Training V2R seed=123")
print("="*60)

for ds in SUBSETS:
    ckpt_dir    = os.path.join(TUTR_DIR, "checkpoint", ds)
    best_pth    = os.path.join(ckpt_dir, "best.pth")
    backup_pth  = os.path.join(ckpt_dir, "best_v2r_seed123.pth")

    print(f"\n  [{ds.upper()}] Training...")
    cmd = [
        PYTHON, "train.py",
        "--dataset_name", ds,
        "--hp_config",    f"config\\{ds}.py",
        "--gpu",          TRAIN_ARGS["gpu"],
        "--num_works",    TRAIN_ARGS["num_works"],
        "--seed",         TRAIN_ARGS["seed"],
        "--lambda1",      TRAIN_ARGS["lambda1"],
        "--lambda2",      TRAIN_ARGS["lambda2"],
        "--lambda3",      TRAIN_ARGS["lambda3"],
        "--warmup_epoch", TRAIN_ARGS["warmup_epoch"],
    ]
    result = subprocess.run(cmd, cwd=TUTR_DIR)
    if result.returncode != 0:
        print(f"  ERROR: Training failed for {ds} — stopping.")
        sys.exit(1)

    if not os.path.exists(best_pth):
        print(f"  ERROR: best.pth not found after training {ds} — stopping.")
        sys.exit(1)

    shutil.copy2(best_pth, backup_pth)
    print(f"  ✓ Backed up: {backup_pth}")

print("\n  [STAGE 1 COMPLETE] All V2R seed123 checkpoints saved.")

# ── STAGE 2: Inference ──────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"  STAGE 2: Inference → {OUT_DIR}")
print("="*60)

for ds in SUBSETS:
    ckpt_dir   = os.path.join(TUTR_DIR, "checkpoint", ds)
    backup_pth = os.path.join(ckpt_dir, "best_v2r_seed123.pth")
    best_pth   = os.path.join(ckpt_dir, "best.pth")

    shutil.copy2(backup_pth, best_pth)
    print(f"\n  [{ds.upper()}] Inference...")

    cmd = [
        PYTHON, r"..\scripts\run_inference.py",
        "--dataset_name", ds,
        "--out_dir",      OUT_DIR,
        "--num_works",    "0",
    ]
    result = subprocess.run(cmd, cwd=TUTR_DIR)
    if result.returncode != 0:
        print(f"  ERROR: Inference failed for {ds}")
        sys.exit(1)
    print(f"  ✓ Done")

print("\n  [STAGE 2 COMPLETE]")

# ── STAGE 3: Metrics ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STAGE 3: Computing metrics...")
print("="*60)

metrics_script = r"C:\CoopCalib\scripts\compute_metrics_with_svr.py"
out_json       = r"C:\CoopCalib\experiments\results\v2r_seed123_metrics.json"

cmd = [
    PYTHON, metrics_script,
    "--preds_dir", OUT_DIR,
    "--out_file",  out_json,
]
result = subprocess.run(cmd, cwd=r"C:\CoopCalib")
if result.returncode != 0:
    print("  ERROR: Metrics computation failed")
    sys.exit(1)

print(f"\n  [STAGE 3 COMPLETE] Saved: {out_json}")
print("\n  V2R SEED 123 PIPELINE COMPLETE")