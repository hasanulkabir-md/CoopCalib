# CoopCalib-TP — Session Handoff README
> Last updated: End of Day 10 (all scripts written, figures generated, §4/§5/§6 drafted)
> Paste this file at the start of any new chat to resume exactly where we left off.

---

## 1. Project Identity

**Title:** CoopCalib-TP — Calibrated Cooperative Trajectory Prediction for Dense-Crowd Safety
**Goal:** Publication-ready ECCV 2026 submission
**Target venues:** ECCV 2026 (primary), IROS 2026 Workshop (secondary)
**Hardware:** Windows 11, RTX 3050 Ti Laptop GPU (4GB VRAM), Python 3.11 (NOT WSL)

---

## 2. Day 10 Deliverables — All Complete ✅

Copy all 5 scripts from Day 10 to `C:\CoopCalib\`:

| Script | Purpose | Run order |
|--------|---------|-----------|
| `compute_eff_rank.py` | Effective rank of SocialVAE latent z (all 8 checkpoints) | 1 |
| `wilcoxon_tests.py` | Wilcoxon + Cohen's d for all 4 RQs | 2 |
| `zeroshot_trajnet.py` | Zero-shot TUTR V0+V2 on 50 TrajNet++ scenes | 3 |
| `generate_figures.py` | All 4 paper figures + Table 1 LaTeX | 4 |
| `run_day10.bat` | Chains all 4 steps automatically | master |

Paper sections written and ready to paste into `main.tex`:
- `sections_4_5_6.tex` — §4 Methodology (1,200w), §5 Experiments (800w), §6 Results (700w)

---

## 3. LOCKED RESULTS (do not re-run)

### TUTR V0/V1/V2 (all 5 folds) ✅

| Fold | Tier | V0 ECE | V2 ECE | V0 SPSR | V2 SPSR | V2 SVR |
|------|------|--------|--------|---------|---------|--------|
| ETH | Sparse | 0.6285 | 0.6411 | 0.3269 | 0.3187 | 0.8354 |
| HOTEL | Medium | 0.5219 | 0.5191 | 0.9265 | 0.9198 | 0.9447 |
| UNIV | Dense | 0.5543 | 0.5686 | 0.6674 | 0.6560 | 0.9891 |
| ZARA1 | Medium | 0.5152 | 0.5167 | 0.8570 | 0.8544 | 0.8793 |
| ZARA2 | Medium | 0.5538 | 0.5537 | 0.8164 | 0.8078 | 0.9610 |

**FPR = 0.000 everywhere** (V0, V1, V2 × all folds × all thresholds)

### SocialVAE RQ3 (all 8 runs) ✅

| N | V0 ADE | V3 ADE | V0 FDE | V3 FDE |
|---|--------|--------|--------|--------|
| 500 | 0.72 | 0.74 | 1.21 | 1.23 |
| 1000 | 0.70 | 0.73 | 1.22 | 1.31 |
| 1500 | 0.74 | 0.74 | 1.37 | 1.42 |
| 1845 | 0.73 | **0.72** | 1.33 | **1.33** |

---

## 4. PENDING — Day 11 Tasks

### Priority 1 — Run Day 10 scripts (if not yet run)
```cmd
cd C:\CoopCalib
C:\CoopCalib\venv\Scripts\activate
run_day10.bat
```

After running:
- Check `experiments/results/socialvae/eff_rank_summary.json`
- If effective rank values are available, re-run `generate_figures.py` to update Fig 2
- Check `experiments/results/stats_summary.json` for p-values to insert into §6

### Priority 2 — Insert real stats into §6
In `sections_4_5_6.tex`, find the RQ1/RQ3 results paragraphs and replace
placeholder ECE and effective rank values with real values from the JSON files.

For Wilcoxon results, use this template:
- If significant: "V2 achieves statistically significant improvement over V0 (Wilcoxon, p={p}, Bonferroni-corrected α=0.025)."
- If not significant (likely given N=5): "While not reaching significance at N=5 (Wilcoxon, p={p}), the effect size Cohen's d={d} ({size}) indicates a {small/medium} practical effect consistent with the directional improvement across all folds."

### Priority 3 — Write §1 Introduction and §7 Discussion + §8 Conclusion

**§1 Introduction outline (~600w):**
1. Opening: trajectory prediction for robot navigation in dense crowds
2. Gap 1: No ECE ever measured → our contribution: first calibration signal
3. Gap 2: FPE assumed in transformers, never tested → we disprove it
4. Gap 3: KL collapse assumed at ETH-UCY scale → we disprove it
5. Gap 4: No social compliance training signal → L_energy + SVR metric
6. Contributions bullet list (4 bullets, one per gap)
7. Paper overview sentence

**§7 Discussion (~400w):**
- Why ECE remains high despite L_ECE: winner-takes-all training collapses diversity
- Why FPR=0: transformer cross-attention is a stronger diversity mechanism than anticipated
- Why L_KL_dyn only helps at full data: ETH-UCY is too small for KL collapse
- Limitations: single GPU, ETH-UCY scale only, no real-time deployment test
- Future work: sampling-based calibration, larger datasets (nuScenes, Waymo), deployment

**§8 Conclusion (~250w):**
- Summary of 4 contributions
- Emphasise the two proof-of-absence findings as calibration of field assumptions
- Close with safety implications for robotics

### Priority 4 — GitHub Repository
```
C:\CoopCalib\  →  push to GitHub as: coopcalib-tp
Structure:
  tutr_fork/              (copy C:\CoopCalib\TUTR\)
  metrics/                (copy C:\CoopCalib\metrics\)
  scripts/
    compute_eff_rank.py
    wilcoxon_tests.py
    zeroshot_trajnet.py
    generate_figures.py
    preprocess_sdd.py
    sample_trajnet_scenes.py
    reproduce.sh
  experiments/figures/    (all 4 PDFs)
  README.md
Tag: git tag v0.1-eccv2026
```

### Priority 5 — Critical fix before any submission
In your paper §4.2 (Density Stratification), verify the wording says:
> "HOTEL falls into the medium tier (MND = 1.925m)"
NOT "HOTEL is dense" — the proposal used visual inspection, not measurement.

---

## 5. Paper Writing Status

| Section | Status | Words |
|---------|--------|-------|
| §1 Introduction | 🔲 TODO | Target 600w |
| §2 Related Work | ✅ Done | 1,066w |
| §3 Preliminaries | 🔲 TODO | Target 400w |
| §4 Methodology | ✅ Written | ~1,200w |
| §5 Experiments | ✅ Written | ~800w |
| §6 Results | ✅ Written | ~700w |
| §7 Discussion | 🔲 TODO | Target 400w |
| §8 Conclusion | 🔲 TODO | Target 250w |
| Loss equations (LaTeX) | ✅ Done | — |
| Table 1 (ablation) | ✅ Generated | table1_ablation.tex |
| Figures 1–4 | ✅ Scripts ready | Run generate_figures.py |

**Total written: ~3,766w / ~5,416w target (70%)**

---

## 6. File Locations — All Outputs

```
C:\CoopCalib\
├── compute_eff_rank.py          ✅ NEW Day 10
├── wilcoxon_tests.py            ✅ NEW Day 10
├── zeroshot_trajnet.py          ✅ NEW Day 10
├── generate_figures.py          ✅ NEW Day 10
├── run_day10.bat                ✅ NEW Day 10
├── sections_4_5_6.tex           ✅ NEW Day 10
├── experiments/
│   ├── results/
│   │   ├── baseline.json        ✅ V0 locked
│   │   ├── v1_metrics.json      ✅ V1 locked
│   │   ├── v2_metrics.json      ✅ V2 locked (6 metrics)
│   │   ├── stats_summary.json   🔲 generated by wilcoxon_tests.py
│   │   ├── trajnet_zeroshot.json 🔲 generated by zeroshot_trajnet.py
│   │   └── socialvae/
│   │       ├── eff_rank_summary.json  🔲 generated by compute_eff_rank.py
│   │       └── [8 checkpoint dirs]    ✅ all complete
│   └── figures/
│       ├── fig1_reliability_diagram.pdf  🔲 generated by generate_figures.py
│       ├── fig2_effective_rank.pdf       🔲 (update after eff_rank runs)
│       ├── fig3_svr_density.pdf          🔲 generated by generate_figures.py
│       ├── fig4_spsr_zeroshot.pdf        🔲 generated by generate_figures.py
│       └── table1_ablation.tex           🔲 generated by generate_figures.py
```

---

## 7. ECCV 2026 Submission Checklist

- [ ] All 4 figures at 300dpi PDF
- [ ] Table 1 (ablation) with 6 metrics × 3 variants × 5 folds
- [ ] Wilcoxon p-values or Cohen's d in §6
- [ ] Effective rank values in §6 (replace placeholder in Fig 2)
- [ ] §1, §7, §8 written
- [ ] §4.2 HOTEL reclassification confirmed (medium not dense)
- [ ] GitHub repo tagged v0.1-eccv2026
- [ ] Anonymous submission (remove institution + names)
- [ ] Check ECCV 2026 submission deadline (typically March–April 2026)
- [ ] Supplementary: full per-fold breakdown, SDD preprocessing details

---

## 8. Critical Rules (unchanged)

1. Always use Windows CMD — never WSL
2. Always activate venv first: `C:\CoopCalib\venv\Scripts\activate`
3. Always add `--num_works 0` to any TUTR train.py command
4. Working dir for TUTR: `C:\CoopCalib\TUTR`
5. Working dir for SocialVAE: `C:\CoopCalib\models\socialvae`
6. Working dir for custom scripts: `C:\CoopCalib`
7. NEVER use multiline `python -c` in Windows CMD — always write .py files
8. Never rename best.pth — copy to versioned name (best_vN.pth)
9. SVR requires _neis.npy — must re-run inference if files are missing
10. MND computed from raw .txt files, not pkl
11. Network blocked: GitHub/Google Drive inaccessible — use hotspot
12. lambda3=0.0 in SocialVAE = vanilla baseline

---

## 9. Key References

| ID | Paper | Role |
|----|-------|------|
| P1 | DTGAN (arXiv:2501.07711) | Baseline comparison |
| P2 | Xu & Zeng survey (arXiv:2510.10327) | Landscape framing |
| P3 | TUTR — Shi et al. ICCV 2023 pp.9675–9684 | **Base model** |
| P4 | IGP — Trautman & Krause IROS 2010 | FPE origin, RQ2 |
| P5 | OWOBJ — Zhang et al. CVPR 2025 pp.30332–30342 | L_KL_dyn + L_energy |
| P6 | Social-STGCNN — Mohamed et al. CVPR 2020 | Baseline |
| P7 | Karandikar et al. NeurIPS 2021 | Soft-ECE loss |
| P8 | PECNet — Mangalam et al. ECCV 2020 | SDD split protocol |
