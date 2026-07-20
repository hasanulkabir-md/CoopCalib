"""
eval_single_rank.py  --  CoopCalib-TP
======================================
Computes effective rank for ONE specified SocialVAE checkpoint.
Handles two checkpoint structures:
  - Flat:    ckpt["q_z.mu.weight"]          (original seed-1 checkpoints)
  - Wrapped: ckpt["model"]["q_z.mu.weight"] (seed-2+ checkpoints)

Usage:
    python eval_single_rank.py --checkpoint PATH_TO_DIR --label NAME

Examples:
    python eval_single_rank.py ^
        --checkpoint experiments\results\socialvae\eth_500_v0_s2 ^
        --label eth_500_v0_s2

    python eval_single_rank.py ^
        --checkpoint experiments\results\socialvae\eth_500_v3_s2 ^
        --label eth_500_v3_s2

Run from C:\\CoopCalib\\ with venv active.
"""

import os
import sys
import argparse
import numpy as np
import torch

# ---------------------------------------------------------------------------
# ARGS
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True,
                    help="Path to checkpoint directory containing ckpt-best")
parser.add_argument("--label", default="",
                    help="Label for this run (e.g. eth_500_v0_s2)")
args = parser.parse_args()

ckpt_dir  = args.checkpoint
label     = args.label or os.path.basename(ckpt_dir)
ckpt_path = os.path.join(ckpt_dir, "ckpt-best")

# ---------------------------------------------------------------------------
# VALIDATE
# ---------------------------------------------------------------------------
if not os.path.exists(ckpt_dir):
    sys.exit(f"[ERROR] Directory not found:\n  {ckpt_dir}")
if not os.path.exists(ckpt_path):
    contents = os.listdir(ckpt_dir) if os.path.isdir(ckpt_dir) else []
    sys.exit(f"[ERROR] ckpt-best not found in:\n  {ckpt_dir}\n"
             f"Contents: {contents}")

# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------
print(f"\n{'='*56}")
print(f"eval_single_rank.py  --  CoopCalib-TP")
print(f"{'='*56}")
print(f"Label     : {label}")
print(f"Checkpoint: {ckpt_path}")
print(f"File size : {os.path.getsize(ckpt_path) // 1024} KB")

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

print(f"Top-level keys: {list(ckpt.keys())}")

# ---------------------------------------------------------------------------
# RESOLVE WEIGHT DICT
# Two structures observed:
#   Flat    : ckpt IS the state dict  (original runs, keys like "q_z.mu.weight")
#   Wrapped : ckpt["model"] is the state dict (seed-2+ runs)
# ---------------------------------------------------------------------------
MU_KEY = "q_z.mu.weight"

if MU_KEY in ckpt:
    # Flat structure
    state_dict = ckpt
    print("Checkpoint structure: FLAT (weights at top level)")

elif "model" in ckpt and isinstance(ckpt["model"], dict):
    state_dict = ckpt["model"]
    print("Checkpoint structure: WRAPPED (weights under 'model' key)")
    print(f"  model keys (first 8): {list(state_dict.keys())[:8]}")

    # Also print ADE/FDE stored in checkpoint for cross-check
    if "ade" in ckpt:
        print(f"  Stored ADE: {ckpt['ade']:.4f}  "
              f"FDE: {ckpt['fde']:.4f}  "
              f"Epoch: {ckpt.get('epoch', 'n/a')}")

else:
    # Unknown structure — print all keys to help debug
    print("[WARN] Unknown checkpoint structure.")
    print(f"All keys: {list(ckpt.keys())}")
    # Try to find any mu-related key anywhere
    all_keys = list(ckpt.keys())
    mu_candidates = [k for k in all_keys if "mu" in str(k).lower()]
    print(f"mu-related keys: {mu_candidates}")
    sys.exit("[ERROR] Cannot locate weight dict. Paste output above for diagnosis.")

# ---------------------------------------------------------------------------
# FIND MU LAYER IN STATE DICT
# ---------------------------------------------------------------------------
if MU_KEY not in state_dict:
    candidates = [k for k in state_dict if "mu" in k and "weight" in k]
    print(f"[WARN] '{MU_KEY}' not found.")
    print(f"mu candidates: {candidates}")
    if not candidates:
        print(f"All state_dict keys: {list(state_dict.keys())[:20]}")
        sys.exit("[ERROR] No mu weight layer found.")
    MU_KEY = candidates[0]
    print(f"[INFO] Using: {MU_KEY}")

mu_weight = state_dict[MU_KEY]
print(f"\nMu layer  : '{MU_KEY}'  shape={tuple(mu_weight.shape)}")

# ---------------------------------------------------------------------------
# COMPUTE EFFECTIVE RANK
# Roy & Vetterli (2007): eff_rank = exp(H(eigenvalue spectrum))
# ---------------------------------------------------------------------------
z_proxy = mu_weight.detach().cpu().numpy()   # (latent_dim, hidden_dim)
cov     = np.cov(z_proxy.T)
evals   = np.linalg.eigvalsh(cov)
evals   = evals[evals > 1e-10]
p       = evals / evals.sum()
eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))

# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------
print(f"Eigenvalues used: {len(evals)}")
print(f"\n{'='*56}")
print(f"EFFECTIVE RANK: {eff_rank:.4f}")
print(f"{'='*56}")

# ---------------------------------------------------------------------------
# VARIANCE VERDICT
# Compare against seed-1 reference values
# ---------------------------------------------------------------------------
SEED1_REF = {
    "eth_500_v0":  22.0195,
    "eth_500_v3":  16.5178,
    "eth_full_v0": 19.6184,
    "eth_full_v3": 19.7182,
}

# Strip seed suffix to get base name
base = label
for suffix in ["_s1", "_s2", "_s3", "_s4"]:
    base = base.replace(suffix, "")

if base in SEED1_REF:
    ref  = SEED1_REF[base]
    diff = abs(eff_rank - ref)
    pct  = diff / ref * 100
    print(f"\nSeed-1 reference ({base}): {ref:.4f}")
    print(f"This seed ({label})       : {eff_rank:.4f}")
    print(f"Absolute difference       : {diff:.4f}  ({pct:.1f}%)")
    print()
    if diff < 0.5:
        verdict = "VERY LOW variance. Bands will be tight and convincing."
        action  = "PROCEED — run seed 3 overnight, then update Fig 3."
    elif diff < 2.0:
        verdict = "MODERATE variance. Bands visible but acceptable."
        action  = "Run seed 3 to confirm, then decide."
    else:
        verdict = "HIGH variance. Crossover gap (0.10 at N=1845) is fragile."
        action  = "Reframe RQ3 as proof-of-absence. Do not spend more GPU time."
    print(f"VERDICT : {verdict}")
    print(f"ACTION  : {action}")
else:
    print(f"\n[INFO] No seed-1 reference for '{base}'. "
          f"Record this value manually for comparison.")

print(f"\n{'='*56}")
print(f"Next command (if proceeding):")
print(f"  python eval_single_rank.py "
      f"--checkpoint experiments\\results\\socialvae\\eth_500_v3_s2 "
      f"--label eth_500_v3_s2")
print(f"{'='*56}\n")
