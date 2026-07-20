@echo off
REM ============================================================
REM CoopCalib-TP  —  Day 10 Master Run Script
REM Run from: C:\CoopCalib
REM Usage: C:\CoopCalib\venv\Scripts\activate  then  run_day10.bat
REM ============================================================

echo.
echo ============================================================
echo CoopCalib-TP Day 10  --  Full Pipeline
echo ============================================================
echo.

REM Step 1: Effective Rank
echo [STEP 1/4] Computing effective rank of SocialVAE latent spaces...
python compute_eff_rank.py
if errorlevel 1 (
    echo [ERROR] compute_eff_rank.py failed. Check checkpoint paths.
    pause
    exit /b 1
)
echo.

REM Step 2: Statistical tests
echo [STEP 2/4] Running Wilcoxon signed-rank tests...
python wilcoxon_tests.py
if errorlevel 1 (
    echo [ERROR] wilcoxon_tests.py failed.
    pause
    exit /b 1
)
echo.

REM Step 3: Zero-shot TrajNet++
echo [STEP 3/4] Running zero-shot TrajNet++ inference...
python zeroshot_trajnet.py
if errorlevel 1 (
    echo [WARN] zeroshot_trajnet.py had errors. Check TUTR checkpoint paths.
    echo        Continuing to figures...
)
echo.

REM Step 4: Generate all figures
echo [STEP 4/4] Generating paper figures...
python generate_figures.py
if errorlevel 1 (
    echo [ERROR] generate_figures.py failed. Check matplotlib installation.
    pause
    exit /b 1
)
echo.

echo ============================================================
echo Day 10 Complete!
echo ============================================================
echo.
echo Outputs:
echo   experiments/results/socialvae/eff_rank_summary.json
echo   experiments/results/stats_summary.json
echo   experiments/results/trajnet_zeroshot.json
echo   experiments/figures/fig1_reliability_diagram.pdf
echo   experiments/figures/fig2_effective_rank.pdf
echo   experiments/figures/fig3_svr_density.pdf
echo   experiments/figures/fig4_spsr_zeroshot.pdf
echo   experiments/figures/table1_ablation.tex
echo.
echo Next: copy sections_4_5_6.tex into your main paper.tex
echo       and update fig2 after effective rank runs.
echo.
pause
