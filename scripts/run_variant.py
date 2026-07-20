"""
CoopCalib-TP — Safe Variant Runner
===================================
File:  C:\\CoopCalib\\scripts\\run_variant.py

PURPOSE
-------
Runs the full pipeline for ONE variant (train → backup → inference → metrics)
without touching other variants' checkpoints or prediction files.

SAFE GUARANTEES
---------------
1. Never overwrites best_v0.pth, best_v2.pth
2. Saves predictions to preds_vX/ not preds/ (V0 baseline stays untouched)
3. Copies best.pth → best_vX.pth immediately after each subset trains
4. compute_metrics.py is called with a patched PREDS_DIR per variant

USAGE (run from C:\\CoopCalib\\TUTR\\)
--------------------------------------
  # Train + evaluate V1 (L_ECE only):
  python ..\scripts\run_variant.py --variant V1 --lambda1 0.1

  # Train + evaluate V3 (L_energy only):
  python ..\scripts\run_variant.py --variant V3 --lambda2 0.1

  # Inference + metrics only (skip training, use existing checkpoint):
  python ..\scripts\run_variant.py --variant V1 --lambda1 0.1 --skip_train

VARIANTS
--------
  V1 : --lambda1 0.1              (L_ECE only)
  V3 : --lambda2 0.1              (L_energy only)
  V2 already done — do not re-run unless explicitly needed

ESTIMATED TIME
--------------
  Training 5 subsets : ~40 min per variant
  Inference 5 subsets: ~10 min per variant
  Metrics            : ~2  min per variant
  Total per variant  : ~52 min
"""

import argparse
import os
import sys
import shutil
import subprocess
import json
import numpy as np

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="CoopCalib-TP safe variant runner")
parser.add_argument("--variant",     type=str, required=True,
                    choices=["V1", "V3"],
                    help="Which variant to run. V1=ECE only, V3=energy only.")
parser.add_argument("--lambda1",     type=float, default=0.0,
                    help="Weight for L_ECE (use 0.1 for V1)")
parser.add_argument("--lambda2",     type=float, default=0.0,
                    help="Weight for L_energy (use 0.1 for V3)")
parser.add_argument("--seed",        type=int, default=42,
                    help="Random seed (default 42)")
parser.add_argument("--gpu",         type=str, default="0")
parser.add_argument("--skip_train",  action="store_true",
                    help="Skip training, go straight to inference+metrics")
parser.add_argument("--skip_inference", action="store_true",
                    help="Skip inference, recompute metrics only")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Paths — all absolute, Windows-safe
# ---------------------------------------------------------------------------
TUTR_DIR     = os.path.dirname(os.path.abspath(__file__)) + "\\..\\TUTR"
SCRIPTS_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT         = os.path.dirname(os.path.abspath(__file__)) + "\\.."
CKPT_DIR     = os.path.join(TUTR_DIR, "checkpoint")
RESULTS_DIR  = os.path.join(ROOT, "experiments", "results")

VARIANT      = args.variant           # e.g. "V1"
VARIANT_LOW  = VARIANT.lower()        # e.g. "v1"
PREDS_OUT    = os.path.join(RESULTS_DIR, f"preds_{VARIANT_LOW}")
METRICS_OUT  = os.path.join(RESULTS_DIR, f"{VARIANT_LOW}_metrics.json")

DATASETS = ["eth", "hotel", "univ", "zara1", "zara2"]

CONFIG_MAP = {
    "eth":   "config\\eth.py",
    "hotel": "config\\hotel.py",
    "univ":  "config\\univ.py",
    "zara1": "config\\zara1.py",
    "zara2": "config\\zara2.py",
}

# ---------------------------------------------------------------------------
# Safety checks
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  CoopCalib-TP — Variant Runner: {VARIANT}")
print(f"  lambda1={args.lambda1}  lambda2={args.lambda2}  seed={args.seed}")
print(f"  skip_train={args.skip_train}  skip_inference={args.skip_inference}")
print(f"{'='*60}\n")

# Verify lambda settings match variant
if VARIANT == "V1" and args.lambda1 == 0.0:
    print("ERROR: V1 requires --lambda1 > 0.0 (suggest --lambda1 0.1)")
    sys.exit(1)
if VARIANT == "V1" and args.lambda2 != 0.0:
    print("ERROR: V1 should have lambda2=0.0 (ECE only). "
          "Use V2 for ECE+energy combined.")
    sys.exit(1)
if VARIANT == "V3" and args.lambda2 == 0.0:
    print("ERROR: V3 requires --lambda2 > 0.0 (suggest --lambda2 0.1)")
    sys.exit(1)
if VARIANT == "V3" and args.lambda1 != 0.0:
    print("ERROR: V3 should have lambda1=0.0 (energy only).")
    sys.exit(1)

# Verify we are being run from TUTR directory
if not os.path.exists(os.path.join(os.getcwd(), "train.py")):
    print("ERROR: Run this script from C:\\CoopCalib\\TUTR\\")
    print(f"  Current dir: {os.getcwd()}")
    print("  Expected:    C:\\CoopCalib\\TUTR\\")
    sys.exit(1)

# Verify existing V0 and V2 checkpoints are safe
for ds in DATASETS:
    v0_path = os.path.join(CKPT_DIR, ds, "best_v0.pth")
    v2_path = os.path.join(CKPT_DIR, ds, "best_v2.pth")
    if not os.path.exists(v0_path):
        print(f"WARNING: best_v0.pth missing for {ds} — V0 baseline may be at risk")
    if not os.path.exists(v2_path):
        print(f"WARNING: best_v2.pth missing for {ds} — V2 results may be at risk")

print(f"Safety checks passed. Output predictions → {PREDS_OUT}")
print(f"Output metrics      → {METRICS_OUT}\n")
os.makedirs(PREDS_OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# STAGE 1 — Training
# ---------------------------------------------------------------------------
if not args.skip_train:
    print(f"{'─'*60}")
    print(f"  STAGE 1: Training {VARIANT} — all 5 subsets")
    print(f"{'─'*60}\n")

    for ds in DATASETS:
        cfg = CONFIG_MAP[ds]
        ckpt_out = os.path.join(CKPT_DIR, ds, f"best_{VARIANT_LOW}.pth")
        best_pth  = os.path.join(CKPT_DIR, ds, "best.pth")

        print(f"  [{ds.upper()}] Training {VARIANT}...")

        cmd = [
            sys.executable, "train.py",
            "--dataset_name", ds,
            "--hp_config", cfg,
            "--gpu", args.gpu,
            "--num_works", "0",
            "--seed", str(args.seed),
            "--lambda1", str(args.lambda1),
            "--lambda2", str(args.lambda2),
        ]

        print(f"  CMD: {' '.join(cmd)}\n")
        result = subprocess.run(cmd, cwd=os.getcwd())

        if result.returncode != 0:
            print(f"\nERROR: Training failed for {ds} with return code "
                  f"{result.returncode}")
            print("Stopping. Fix the error before continuing.")
            sys.exit(1)

        # Immediately copy best.pth → best_vX.pth
        if os.path.exists(best_pth):
            shutil.copy2(best_pth, ckpt_out)
            print(f"  ✓ Saved: {ckpt_out}")
        else:
            print(f"  ERROR: best.pth not found after training {ds}. "
                  f"Training may have failed silently.")
            sys.exit(1)

        print()

    print(f"  STAGE 1 complete. All {VARIANT} checkpoints saved.\n")

else:
    print("  STAGE 1 skipped (--skip_train)\n")
    # Verify checkpoints exist before proceeding
    for ds in DATASETS:
        ckpt_path = os.path.join(CKPT_DIR, ds, f"best_{VARIANT_LOW}.pth")
        if not os.path.exists(ckpt_path):
            print(f"ERROR: --skip_train set but checkpoint missing: {ckpt_path}")
            print("Run without --skip_train to train first.")
            sys.exit(1)
    print(f"  Verified: all best_{VARIANT_LOW}.pth checkpoints exist.\n")

# ---------------------------------------------------------------------------
# STAGE 2 — Inference
# Copies best_vX.pth → best.pth temporarily, runs inference, restores
# ---------------------------------------------------------------------------
if not args.skip_inference:
    print(f"{'─'*60}")
    print(f"  STAGE 2: Inference {VARIANT} — all 5 subsets")
    print(f"  Output → {PREDS_OUT}")
    print(f"{'─'*60}\n")

    INFERENCE_SCRIPT = os.path.join(SCRIPTS_DIR, "run_inference.py")

    for ds in DATASETS:
        best_pth     = os.path.join(CKPT_DIR, ds, "best.pth")
        variant_pth  = os.path.join(CKPT_DIR, ds, f"best_{VARIANT_LOW}.pth")
        backup_pth   = os.path.join(CKPT_DIR, ds, "best_BACKUP_DO_NOT_DELETE.pth")

        print(f"  [{ds.upper()}] Running inference for {VARIANT}...")

        # Back up current best.pth before overwriting
        if os.path.exists(best_pth):
            shutil.copy2(best_pth, backup_pth)

        # Put variant checkpoint in place
        shutil.copy2(variant_pth, best_pth)

        cmd = [
            sys.executable, INFERENCE_SCRIPT,
            "--dataset_name", ds,
            "--out_dir", PREDS_OUT,
            "--gpu", args.gpu,
            "--num_works", "0",
        ]

        print(f"  CMD: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=os.getcwd())

        # Always restore best.pth from backup regardless of success/failure
        if os.path.exists(backup_pth):
            shutil.copy2(backup_pth, best_pth)
            os.remove(backup_pth)

        if result.returncode != 0:
            print(f"\nERROR: Inference failed for {ds}. best.pth restored.")
            sys.exit(1)

        # Verify neis file was saved
        neis_path = os.path.join(PREDS_OUT, f"{ds}_neis.npy")
        if not os.path.exists(neis_path):
            print(f"  WARNING: {ds}_neis.npy not found in output. "
                  f"SVR cannot be computed for this fold.")
        else:
            print(f"  ✓ {ds}_neis.npy saved — SVR computable")

        print()

    print(f"  STAGE 2 complete. All predictions saved to {PREDS_OUT}\n")

else:
    print("  STAGE 2 skipped (--skip_inference)\n")

# ---------------------------------------------------------------------------
# STAGE 3 — Metrics
# Runs compute_metrics.py with PREDS_DIR patched to preds_vX/
# ---------------------------------------------------------------------------
print(f"{'─'*60}")
print(f"  STAGE 3: Computing metrics for {VARIANT}")
print(f"{'─'*60}\n")

# We run compute_metrics.py but patch its PREDS_DIR by passing env variable
# The script uses os.path hardcoded PREDS_DIR — we monkey-patch via a wrapper

METRICS_SCRIPT = os.path.join(SCRIPTS_DIR, "compute_metrics.py")

# Read the metrics script and patch PREDS_DIR inline
with open(METRICS_SCRIPT, "r", encoding="utf-8") as f:
    metrics_src = f.read()

# Write a patched version to a temp file
temp_metrics = os.path.join(SCRIPTS_DIR, "_temp_compute_metrics.py")
patched_src = metrics_src.replace(
    'PREDS_DIR = os.path.join(ROOT, "experiments", "results", "preds")',
    f'PREDS_DIR = r"{PREDS_OUT}"'
).replace(
    'out_path = os.path.join(OUT_DIR, "metrics_per_fold.json")',
    f'out_path = r"{METRICS_OUT}"'
)

with open(temp_metrics, "w", encoding="utf-8") as f:
    f.write(patched_src)

print(f"  Running metrics on: {PREDS_OUT}")
result = subprocess.run([sys.executable, temp_metrics], cwd=ROOT)

# Clean up temp file
if os.path.exists(temp_metrics):
    os.remove(temp_metrics)

if result.returncode != 0:
    print(f"\nERROR: Metric computation failed for {VARIANT}.")
    sys.exit(1)

print(f"\n  STAGE 3 complete. Metrics saved to {METRICS_OUT}")

# ---------------------------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  {VARIANT} Pipeline Complete")
print(f"{'='*60}")
print(f"  Checkpoints : checkpoint/{{subset}}/best_{VARIANT_LOW}.pth")
print(f"  Predictions : {PREDS_OUT}")
print(f"  Metrics     : {METRICS_OUT}")

# Print quick summary if metrics file exists
if os.path.exists(METRICS_OUT):
    try:
        with open(METRICS_OUT) as f:
            m = json.load(f)
        if "per_fold" in m:
            folds = m["per_fold"]
        else:
            folds = m  # v1/v2 format
        print(f"\n  Quick results summary:")
        print(f"  {'Subset':<8} {'ECE':>7} {'FPR':>7} {'SPSR':>7} "
              f"{'SVR':>7} {'ADE':>7} {'FDE':>7}")
        print(f"  {'─'*55}")
        ade_vals, fde_vals, ece_vals = [], [], []
        for ds in DATASETS:
            if ds not in folds:
                continue
            fold = folds[ds]
            ece  = fold.get("ece",     fold.get("ECE",  float("nan")))
            fpr  = fold.get("fpr_50cm", fold.get("FPR", float("nan")))
            spsr = fold.get("spsr",    fold.get("SPSR", float("nan")))
            svr  = fold.get("svr",     fold.get("SVR",  0.0))
            ade  = fold.get("min_ade", float("nan"))
            fde  = fold.get("min_fde", float("nan"))
            print(f"  {ds:<8} {ece:>7.4f} {fpr:>7.4f} {spsr:>7.4f} "
                  f"{svr:>7.4f} {ade:>7.4f} {fde:>7.4f}")
            ade_vals.append(ade); fde_vals.append(fde); ece_vals.append(ece)
        if ade_vals:
            print(f"  {'─'*55}")
            print(f"  {'AVG':<8} {np.mean(ece_vals):>7.4f} {'':>7} {'':>7} "
                  f"{'':>7} {np.mean(ade_vals):>7.4f} {np.mean(fde_vals):>7.4f}")
    except Exception as e:
        print(f"  (Could not parse metrics file: {e})")

print(f"\n  Next step:")
if VARIANT == "V1":
    print("  Run V3:  python ..\\scripts\\run_variant.py "
          "--variant V3 --lambda2 0.1")
else:
    print("  All variants complete. "
          "Run scripts\\aggregate_results.py to build final table.")
print()
