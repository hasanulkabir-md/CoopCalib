# CoopCalib-TP

**Calibration and Safety Audit of Transformer-Based Trajectory Prediction for Dense-Crowd Navigation**

Target venue: IEEE Robotics and Automation Letters (RAL)

---

## Overview

This repository contains the code and experiments for a systematic calibration and safety audit of transformer-based pedestrian trajectory prediction. We build on **TUTR** (Trajectory Unified Transformer, ICCV 2023) and evaluate on the **ETH-UCY** benchmark (ETH, HOTEL, UNIV, ZARA1, ZARA2), introducing four metrics not previously reported together for this task:

- **ECE** — Expected Calibration Error (M=15 bins)
- **FPR** — Freezing Predictor Rate
- **SVR** — Social Violation Rate (multi-agent, collision-radius based)
- **SPSR** — Safe Planning Success Rate (virtual-planner based)

The core finding is a statistically significant **safety–accuracy trade-off**: our ranking-aware social loss variant (**V2R**) reduces SVR and improves SPSR relative to the vanilla baseline (**V0**), at the cost of increased ADE.

## Key Results (3-seed, seeds 42/1/123)

| Metric | V0 (baseline) | V2R | Δ | Significance (3σ) |
|---|---|---|---|---|
| SVR ↓  | 0.923 ± 0.001 | 0.912 ± 0.000 | −0.010 | Significant |
| SPSR ↑ | 0.819 ± 0.002 | 0.845 ± 0.008 | +0.026 | Significant |
| ADE ↓  | 0.222 ± 0.002 | 0.265 ± 0.003 | +0.043 | Significant (cost) |

FPR = 0.000 across all variants, all subsets, all thresholds tested.

Full per-subset breakdown (ADE/ECE/SVR/SPSR by tier) is in the paper (Section 3, `paper/main_ral.tex`) and in `experiments/results/*_multiseed_summary.json`.

## Repository Structure

```
CoopCalib-TP/
├── TUTR/                     # Base model (patched with L_ranking, warmup)
│   ├── train.py
│   └── checkpoint/{eth,hotel,univ,zara1,zara2}/
├── metrics/
│   ├── eval_suite.py         # ECE, FPR, SVR, SPSR implementations
│   └── loss_functions.py     # LossRankingSVR and other custom losses
├── scripts/
│   ├── run_variant.py        # Train → backup → infer → metrics pipeline
│   ├── compute_metrics_with_svr.py
│   └── generate_figures_ral_final.py
├── experiments/results/      # Predictions and metrics JSONs (multiseed)
└── paper/
    ├── main_ral.tex          # IEEE RAL submission (IEEEtran)
    └── ref.bib
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # or venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
```

Requires Python 3.11, PyTorch 2.5.1, CUDA 12.1 (tested on RTX 3050 Ti, 4 GB VRAM; CPU-only fallback supported for metric computation).

## Usage

**Train a variant (all 5 ETH-UCY subsets, single seed):**
```bash
python scripts/run_variant.py --variant v2r --seed 42
```

**Compute metrics on saved predictions:**
```bash
python scripts/compute_metrics_with_svr.py --preds_dir experiments/results/preds_v2r_fixed
```

**Regenerate paper figures:**
```bash
python scripts/generate_figures_ral_final.py
```

## Ablation Variants

| ID | Loss terms | Purpose |
|---|---|---|
| V0 | — | TUTR baseline |
| V1 | L_ECE | Calibration only |
| V2 | L_ECE + L_energy | Combined |
| V3 | L_energy | Social margin only |
| V1W | L_ECE (warmup) | Delayed calibration pressure |
| **V2R** | L_energy + L_ranking | Ranking-aware social loss (main contribution) |

## Citation

```bibtex
@article{coopcalibtp2026,
  title   = {Calibration and Safety Audit of Transformer-Based Trajectory Prediction for Dense-Crowd Navigation},
  author  = {[Author names pending]},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2026},
  note    = {Under submission}
}
```

## Status

Paper in preparation for IEEE RAL submission. Author names, institution, and funding statement are placeholders pending finalization — see `paper/main_ral.tex`.

## License

[To be determined prior to public release]
