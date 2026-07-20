"""
CoopCalib-TP — Aggressive Variant Runner
=========================================
File:  C:\\CoopCalib\\scripts\\run_aggressive.py

Runs two new experimental variants back-to-back:

  V1W — ECE loss with warm-up (activates at epoch 50)
        Hypothesis: winner-takes-all prevents ECE improvement early
        because candidates have not yet diverged. Warm-up gives the
        model time to learn diverse hypotheses before calibrating.
        Expected: ECE improvement of 0.02-0.08 over V0.

  V2R — Energy loss + Ranking-aware SVR loss
        Hypothesis: SVR did not improve because L_energy penalises
        all K candidates equally. LossRankingSVR directly penalises
        the top-ranked candidate for social violations, putting
        gradient into the CLF-FC selection mechanism.
        Expected: SVR reduction of 0.05-0.15 over V0.

PREREQUISITE
------------
Add LossRankingSVR to C:\\CoopCalib\\metrics\\loss_functions.py
(see LossRankingSVR.py output file for the class to paste in)

Add these two arguments to train.py BEFORE running:
    parser.add_argument('--lambda3', type=float, default=0.0,
                        help='Weight for L_RankingSVR loss (V2R)')
    parser.add_argument('--warmup_epoch', type=int, default=0,
                        help='Epoch after which L_ECE activates (V1W)')

And add to the train() function in train.py, replacing the lambda1 block:

    # CoopCalib V1W — L_ECE with warm-up
    if args.lambda1 > 0.0 and epoch >= args.warmup_epoch:
        pred_traj_k, scores_k = model(ped_obs, neis_obs, motion_modes,
                                       mask, None, test=True)
        pred_traj_2d = pred_traj_k.reshape(
            pred_traj_k.shape[0], pred_traj_k.shape[1], -1, 2)
        gt_2d     = gt.reshape(gt.shape[0], -1, 2)
        scores_2d = scores_k.squeeze(-1) if scores_k.dim()==3 else scores_k
        loss = loss + args.lambda1 * ece_criterion(pred_traj_2d, scores_2d, gt_2d)

    # CoopCalib V2R — L_RankingSVR (ranking-aware social loss)
    if args.lambda3 > 0.0:
        if args.lambda1 <= 0.0 and args.lambda2 <= 0.0:
            pred_traj_k, scores_k = model(ped_obs, neis_obs, motion_modes,
                                           mask, None, test=True)
        pred_traj_2d = pred_traj_k.reshape(
            pred_traj_k.shape[0], pred_traj_k.shape[1], -1, 2)
        scores_2d    = scores_k.squeeze(-1) if scores_k.dim()==3 else scores_k
        # nei_futures: (B, N, T, 2) — neighbour GT futures
        nei_futures  = neis[:, :, args.obs_len:, :]  # (B, N, T, 2)
        loss = loss + args.lambda3 * ranking_svr_criterion(
            pred_traj_2d, scores_2d, nei_futures)

And add after the existing loss instantiations in train.py:

    from metrics.loss_functions import LossRankingSVR
    ranking_svr_criterion = LossRankingSVR(r_ped=0.3, temperature=5.0).cuda()

USAGE (from C:\\CoopCalib\\TUTR\\)
----------------------------------
    python ..\\scripts\\run_aggressive.py

    # Or run variants separately:
    python ..\\scripts\\run_aggressive.py --variant V1W
    python ..\\scripts\\run_aggressive.py --variant V2R

OUTPUT
------
    checkpoint/{subset}/best_v1w.pth     — V1W checkpoints
    checkpoint/{subset}/best_v2r.pth     — V2R checkpoints
    experiments/results/preds_v1w/       — V1W predictions
    experiments/results/preds_v2r/       — V2R predictions
    experiments/results/v1w_metrics_full.json
    experiments/results/v2r_metrics_full.json
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
parser = argparse.ArgumentParser()
parser.add_argument("--variant", type=str, default="both",
                    choices=["V1W", "V2R", "both"],
                    help="Which variant to run (default: both)")
parser.add_argument("--gpu",  type=str, default="0")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TUTR_DIR    = os.getcwd()
SCRIPTS_DIR = os.path.join(TUTR_DIR, "..", "scripts")
ROOT        = os.path.join(TUTR_DIR, "..")
CKPT_DIR    = os.path.join(TUTR_DIR, "checkpoint")
RESULTS_DIR = os.path.join(ROOT, "experiments", "results")
INFERENCE   = os.path.join(SCRIPTS_DIR, "run_inference.py")
METRICS     = os.path.join(SCRIPTS_DIR, "compute_metrics_with_svr.py")
SPSR_SCRIPT = os.path.join(SCRIPTS_DIR, "compute_simple_spsr.py")

DATASETS = ["eth", "hotel", "univ", "zara1", "zara2"]
CONFIG_MAP = {
    "eth":   "config\\eth.py",
    "hotel": "config\\hotel.py",
    "univ":  "config\\univ.py",
    "zara1": "config\\zara1.py",
    "zara2": "config\\zara2.py",
}

# ---------------------------------------------------------------------------
# Verify prerequisites
# ---------------------------------------------------------------------------
if not os.path.exists(os.path.join(TUTR_DIR, "train.py")):
    print("ERROR: Run from C:\\CoopCalib\\TUTR\\")
    sys.exit(1)

# Check train.py has the new arguments
with open(os.path.join(TUTR_DIR, "train.py")) as f:
    train_src = f.read()

missing_args = []
if "--lambda3" not in train_src:
    missing_args.append("--lambda3")
if "--warmup_epoch" not in train_src:
    missing_args.append("--warmup_epoch")
if "LossRankingSVR" not in train_src:
    missing_args.append("LossRankingSVR integration")

if missing_args:
    print("ERROR: train.py is missing required additions:")
    for m in missing_args:
        print(f"  - {m}")
    print("\nSee the PREREQUISITE section in this script's docstring.")
    print("Add the required lines to train.py first, then re-run.")
    sys.exit(1)

print(f"\n{'='*60}")
print(f"  CoopCalib-TP — Aggressive Variant Runner")
print(f"  Running: {args.variant}  seed={args.seed}  gpu={args.gpu}")
print(f"{'='*60}\n")

# ---------------------------------------------------------------------------
# Helper: safe train → backup → inference → metrics pipeline
# ---------------------------------------------------------------------------
def run_pipeline(variant_name, train_kwargs, lambda_str):
    """
    Full pipeline for one variant.
    variant_name: e.g. 'V1W' or 'V2R'
    train_kwargs: dict of extra args for train.py
    lambda_str:   human-readable description of losses
    """
    vl = variant_name.lower()
    preds_dir  = os.path.join(RESULTS_DIR, f"preds_{vl}")
    metrics_out = os.path.join(RESULTS_DIR, f"{vl}_metrics_full.json")

    os.makedirs(preds_dir, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"  VARIANT: {variant_name} — {lambda_str}")
    print(f"{'─'*60}")

    # STAGE 1 — Training
    print(f"\n  [STAGE 1] Training {variant_name}")
    for ds in DATASETS:
        best_pth    = os.path.join(CKPT_DIR, ds, "best.pth")
        variant_pth = os.path.join(CKPT_DIR, ds, f"best_{vl}.pth")

        print(f"\n  [{ds.upper()}] Training {variant_name}...")
        cmd = [
            sys.executable, "train.py",
            "--dataset_name", ds,
            "--hp_config", CONFIG_MAP[ds],
            "--gpu", args.gpu,
            "--num_works", "0",
            "--seed", str(args.seed),
        ]
        for k, v in train_kwargs.items():
            cmd += [k, str(v)]

        print(f"  CMD: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=TUTR_DIR)

        if result.returncode != 0:
            print(f"\n  ERROR: Training failed for {ds}")
            sys.exit(1)

        if os.path.exists(best_pth):
            shutil.copy2(best_pth, variant_pth)
            print(f"  ✓ Saved: {variant_pth}")
        else:
            print(f"  ERROR: best.pth not found after training {ds}")
            sys.exit(1)

    print(f"\n  [STAGE 1 COMPLETE] All {variant_name} checkpoints saved.")

    # STAGE 2 — Inference
    print(f"\n  [STAGE 2] Inference {variant_name} → {preds_dir}")
    for ds in DATASETS:
        best_pth    = os.path.join(CKPT_DIR, ds, "best.pth")
        variant_pth = os.path.join(CKPT_DIR, ds, f"best_{vl}.pth")
        backup_pth  = os.path.join(CKPT_DIR, ds, f"best_AGGRESSIVE_BACKUP.pth")

        print(f"  [{ds.upper()}] Inference...")
        if os.path.exists(best_pth):
            shutil.copy2(best_pth, backup_pth)
        shutil.copy2(variant_pth, best_pth)

        cmd = [
            sys.executable, INFERENCE,
            "--dataset_name", ds,
            "--out_dir", preds_dir,
            "--gpu", args.gpu,
            "--num_works", "0",
        ]
        result = subprocess.run(cmd, cwd=TUTR_DIR)

        if os.path.exists(backup_pth):
            shutil.copy2(backup_pth, best_pth)
            os.remove(backup_pth)

        if result.returncode != 0:
            print(f"  ERROR: Inference failed for {ds}")
            sys.exit(1)

        neis_path = os.path.join(preds_dir, f"{ds}_neis.npy")
        status = "✓ neis saved" if os.path.exists(neis_path) else "WARNING: no neis"
        print(f"  {status}")

    print(f"\n  [STAGE 2 COMPLETE]")

    # STAGE 3 — Full metrics with SVR
    print(f"\n  [STAGE 3] Computing metrics (with SVR)...")
    cmd = [
        sys.executable, METRICS,
        "--preds_dir", preds_dir,
        "--out_file", metrics_out,
    ]
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"  ERROR: Metrics failed for {variant_name}")
        sys.exit(1)

    # STAGE 4 — Simple SPSR
    print(f"\n  [STAGE 4] Computing simple SPSR...")
    spsr_cmd = [sys.executable, SPSR_SCRIPT]
    subprocess.run(spsr_cmd, cwd=ROOT)

    print(f"\n  {'='*50}")
    print(f"  {variant_name} COMPLETE")
    print(f"  Checkpoints : checkpoint/{{subset}}/best_{vl}.pth")
    print(f"  Predictions : {preds_dir}")
    print(f"  Metrics     : {metrics_out}")

    # Quick summary
    if os.path.exists(metrics_out):
        with open(metrics_out) as f:
            m = json.load(f)
        overall = m.get("overall", {})
        print(f"\n  RESULTS SUMMARY:")
        print(f"  ECE  avg: {overall.get('ece',  {}).get('mean', 'N/A'):.4f}")
        print(f"  SVR  avg: {overall.get('svr',  {}).get('mean', 'N/A'):.4f}")
        print(f"  ADE  avg: {overall.get('min_ade', {}).get('mean', 'N/A'):.4f}")

    return metrics_out

# ---------------------------------------------------------------------------
# V1W — ECE with warm-up at epoch 50, lambda1=0.3 (stronger signal)
# ---------------------------------------------------------------------------
V1W_CONFIG = {
    "--lambda1":      0.3,    # stronger ECE signal than V1's 0.1
    "--lambda2":      0.0,
    "--lambda3":      0.0,
    "--warmup_epoch": 50,     # activate ECE only after model has diversified
}

# ---------------------------------------------------------------------------
# V2R — Energy loss + Ranking-aware SVR loss
# ---------------------------------------------------------------------------
V2R_CONFIG = {
    "--lambda1":      0.0,
    "--lambda2":      0.1,    # keep original energy loss
    "--lambda3":      0.5,    # ranking SVR loss — stronger signal
    "--warmup_epoch": 0,
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
results = {}

if args.variant in ("V1W", "both"):
    print("\n" + "="*60)
    print("  RUNNING V1W — ECE with warm-up (epoch 50, lambda1=0.3)")
    print("  Expected: ECE improvement over V0 baseline (0.5547)")
    print("="*60)
    out = run_pipeline(
        "V1W",
        V1W_CONFIG,
        "L_ECE (warmup=50, lambda=0.3)"
    )
    results["V1W"] = out

if args.variant in ("V2R", "both"):
    print("\n" + "="*60)
    print("  RUNNING V2R — Ranking-aware SVR loss (lambda3=0.5)")
    print("  Expected: SVR reduction below V0 baseline (0.9229)")
    print("="*60)
    out = run_pipeline(
        "V2R",
        V2R_CONFIG,
        "L_energy (0.1) + L_RankingSVR (0.5)"
    )
    results["V2R"] = out

# ---------------------------------------------------------------------------
# Final comparison vs baselines
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("  AGGRESSIVE PATH — RESULTS VS BASELINES")
print(f"{'='*60}")
print(f"\n  Target improvements needed for methods paper:")
print(f"  ECE:  V0=0.5547 → need V1W < 0.535 (improvement ≥0.02)")
print(f"  SVR:  V0=0.9229 → need V2R < 0.873 (improvement ≥0.05)")
print(f"\n  Checking results files...")

for vname, metrics_path in results.items():
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        overall = m.get("overall", {})
        ece_mean = overall.get("ece", {}).get("mean", None)
        svr_mean = overall.get("svr", {}).get("mean", None)

        print(f"\n  {vname}:")
        if ece_mean is not None:
            delta = 0.5547 - ece_mean
            result_str = "✓ TARGET MET" if delta >= 0.02 else "✗ below target"
            print(f"    ECE:  {ece_mean:.4f}  (Δ={delta:+.4f}) {result_str}")
        if svr_mean is not None:
            delta = 0.9229 - svr_mean
            result_str = "✓ TARGET MET" if delta >= 0.05 else "✗ below target"
            print(f"    SVR:  {svr_mean:.4f}  (Δ={delta:+.4f}) {result_str}")

print(f"\n{'='*60}")
print("  DECISION GUIDE:")
print("  Both targets met   → Methods paper. Keep all 4 contributions.")
print("  One target met     → Mixed paper. 2 method + 2 measurement.")
print("  No targets met     → Audit paper. Submit honest null results.")
print(f"{'='*60}\n")
