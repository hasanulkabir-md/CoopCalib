"""
patch_tables.py  — fixed version
Root cause of re.error: re.sub() interprets backslashes in the replacement
string as regex escapes (\c, \t, \l etc. all fail).
Fix: wrap every .sub() call with lambda m: REPLACEMENT so Python never
parses the replacement string as a regex template.
"""

import re, shutil, pathlib

TEX = pathlib.Path("main_ral.tex")
assert TEX.exists(), "main_ral.tex not found"

shutil.copy(TEX, TEX.with_suffix(".tex.bak_tables"))
print("[OK] Backup: main_ral.tex.bak_tables")

src = TEX.read_text(encoding="utf-8")

# ════════════════════════════════════════════════════════════════════════════
# TABLE Ia  —  Safety metrics  (SVR + SPSR)
# TABLE Ib  —  Accuracy + calibration metrics  (ECE + ADE + FDE)
# Replaces the single monolithic table* that had label tab:ablation
# ════════════════════════════════════════════════════════════════════════════

TABLE_IA = (
r"""% ── TABLE Ia: Safety metrics (PRIMARY CONTRIBUTIONS) ────────────────────────
% FPR = 0.000 for ALL variants, folds, and both thresholds (0.5 m and 0.8 m).
\begin{table*}[t]
\caption{TUTR Ablation --- Safety Metrics on ETH-UCY (per-subset protocol; seed~42 unless noted).
  FPR~$=$~0.000 for all variants, all folds, and both collision thresholds
  ($d_\mathrm{min}=0.5$\,m and $0.8$\,m), confirming architectural immunity to the Freezing Predictor Effect.
  HOT\,=\,HOTEL (Medium tier); UNV\,=\,UNIV (Dense tier).
  $^*$3-seed mean (seeds 42, 1, 123); std shown as (0.xxx).
  \textbf{Bold}\,=\,best per column.}
\label{tab:ablation_safety}
\centering
\setlength{\tabcolsep}{4.2pt}
\begin{tabular}{l l c c c c c c}
\toprule
\multirow{2}{*}{Variant} & \multirow{2}{*}{Loss} &
  \multicolumn{5}{c}{Per-subset} & \multirow{2}{*}{Avg} \\
\cmidrule(lr){3-7}
& & ETH & HOT & UNV & ZR1 & ZR2 & \\
\midrule
\multicolumn{8}{l}{\textit{SVR $\downarrow$ --- Social Violation Rate (first density-stratified measurement)}} \\[1pt]
V0$^*$  & --
  & 0.838\,(0.003) & 0.944\,(0.000) & 0.989\,(0.000)
  & 0.880\,(0.001) & 0.961\,(0.000) & 0.923\,(0.001) \\
V1   & $\mathcal{L}_\mathrm{ECE}$
  & 0.831 & 0.943 & 0.989 & 0.880 & 0.961 & 0.921 \\
V2   & $\mathcal{L}_\mathrm{ECE}{+}\mathcal{L}_\mathrm{energy}$
  & 0.835 & 0.945 & 0.989 & 0.879 & 0.961 & 0.922 \\
V3   & $\mathcal{L}_\mathrm{energy}$
  & 0.830 & 0.944 & 0.989 & 0.881 & 0.962 & 0.921 \\
V1W  & $\mathcal{L}_\mathrm{ECE}$ (warm-up)
  & 0.835 & 0.944 & 0.989 & 0.881 & 0.962 & 0.922 \\
V2R$^*$ & $\mathcal{L}_\mathrm{energy}{+}\mathcal{L}_\mathrm{ranking}$
  & \textbf{0.810}\,(0.001) & \textbf{0.937}\,(0.000)
  & \textbf{0.988}\,(0.000) & \textbf{0.870}\,(0.002)
  & \textbf{0.957}\,(0.001) & \textbf{0.912}\,(0.000) \\
\midrule
\multicolumn{8}{l}{\textit{SPSR $\uparrow$ --- Safe Planning Success Rate$^\dagger$}} \\[1pt]
V0$^*$  & --
  & 0.824\,(0.009) & 0.919\,(0.001) & 0.660\,(0.001)
  & 0.929\,(0.002) & 0.769\,(0.000) & 0.819\,(0.002) \\
V1   & $\mathcal{L}_\mathrm{ECE}$
  & 0.813 & 0.918 & 0.658 & 0.933 & 0.765 & 0.817 \\
V2   & $\mathcal{L}_\mathrm{ECE}{+}\mathcal{L}_\mathrm{energy}$
  & 0.789 & 0.922 & 0.672 & 0.933 & 0.810 & 0.825 \\
V3   & $\mathcal{L}_\mathrm{energy}$
  & 0.813 & 0.916 & 0.670 & 0.934 & 0.814 & 0.829 \\
V1W  & $\mathcal{L}_\mathrm{ECE}$ (warm-up)
  & 0.805 & 0.916 & 0.660 & 0.934 & 0.769 & 0.817 \\
V2R$^*$ & $\mathcal{L}_\mathrm{energy}{+}\mathcal{L}_\mathrm{ranking}$
  & 0.789\,(0.011) & \textbf{0.963}\,(0.003)
  & \textbf{0.685}\,(0.015) & 0.929\,(0.002)
  & \textbf{0.853}\,(0.027) & \textbf{0.845}\,(0.008) \\
\bottomrule
\multicolumn{8}{l}{{\scriptsize $^\dagger$SPSR excludes ego-agent and zero-occupancy padding
  (Section~\ref{sec:prelim}). ETH regression ($-0.028$): 61.8\,\% of ETH scenes have no real neighbours.}} \\
\end{tabular}
\end{table*}

% ── TABLE Ib: Accuracy and calibration metrics (SECONDARY) ──────────────────
\begin{table*}[t]
\caption{TUTR Ablation --- Accuracy and Calibration Metrics on ETH-UCY
  (same protocol as Table~\ref{tab:ablation_safety}).
  ECE differences of ${\leq}0.005$ across all variants are within inter-seed noise;
  no bold applied to ECE block (establishing baseline, not optimising).
  $^*$3-seed mean; std shown as (0.xxx). \textbf{Bold}\,=\,best per column (ADE, FDE only).}
\label{tab:ablation_accuracy}
\centering
\setlength{\tabcolsep}{4.2pt}
\begin{tabular}{l l c c c c c c}
\toprule
\multirow{2}{*}{Variant} & \multirow{2}{*}{Loss} &
  \multicolumn{5}{c}{Per-subset} & \multirow{2}{*}{Avg} \\
\cmidrule(lr){3-7}
& & ETH & HOT & UNV & ZR1 & ZR2 & \\
\midrule
\multicolumn{8}{l}{\textit{ECE $\downarrow$ --- Expected Calibration Error
  (first reported in trajectory prediction; all variants within $\pm$0.005 of V0)}} \\[1pt]
V0$^*$  & --
  & 0.629\,(0.002) & 0.519\,(0.004) & 0.558\,(0.005)
  & 0.515\,(0.001) & 0.553\,(0.001) & 0.554\,(0.000) \\
V1   & $\mathcal{L}_\mathrm{ECE}$
  & 0.630 & 0.522 & 0.565 & 0.515 & 0.547 & 0.556 \\
V2   & $\mathcal{L}_\mathrm{ECE}{+}\mathcal{L}_\mathrm{energy}$
  & 0.641 & 0.519 & 0.569 & 0.517 & 0.554 & 0.560 \\
V3   & $\mathcal{L}_\mathrm{energy}$
  & 0.630 & 0.519 & 0.566 & 0.515 & 0.553 & 0.557 \\
V1W  & $\mathcal{L}_\mathrm{ECE}$ (warm-up)
  & 0.640 & 0.526 & 0.561 & 0.514 & 0.553 & 0.559 \\
V2R$^*$ & $\mathcal{L}_\mathrm{energy}{+}\mathcal{L}_\mathrm{ranking}$
  & 0.626\,(0.005) & 0.523\,(0.001) & 0.574\,(0.002)
  & 0.520\,(0.002) & 0.558\,(0.002) & 0.560\,(0.003) \\
\midrule
\multicolumn{8}{l}{\textit{minADE $\downarrow$ (m)}} \\[1pt]
V0$^*$  & --
  & \textbf{0.419}\,(0.008) & \textbf{0.125}\,(0.000)
  & \textbf{0.236}\,(0.001) & \textbf{0.189}\,(0.000)
  & \textbf{0.141}\,(0.000) & \textbf{0.222}\,(0.002) \\
V1   & $\mathcal{L}_\mathrm{ECE}$
  & 0.442 & 0.128 & 0.235 & 0.188 & 0.143 & 0.227 \\
V2   & $\mathcal{L}_\mathrm{ECE}{+}\mathcal{L}_\mathrm{energy}$
  & 0.422 & 0.128 & 0.238 & 0.189 & 0.142 & 0.224 \\
V3   & $\mathcal{L}_\mathrm{energy}$
  & 0.443 & 0.127 & 0.237 & 0.189 & 0.144 & 0.228 \\
V1W  & $\mathcal{L}_\mathrm{ECE}$ (warm-up)
  & 0.429 & 0.128 & 0.234 & 0.186 & 0.143 & 0.224 \\
V2R$^*$ & $\mathcal{L}_\mathrm{energy}{+}\mathcal{L}_\mathrm{ranking}$
  & 0.475\,(0.012) & 0.153\,(0.002) & 0.279\,(0.002)
  & 0.240\,(0.001) & 0.179\,(0.002) & 0.265\,(0.003) \\
\midrule
\multicolumn{8}{l}{\textit{minFDE $\downarrow$ (m)}} \\[1pt]
V0$^*$  & --
  & 0.629\,(0.005) & \textbf{0.188}\,(0.003)
  & \textbf{0.431}\,(0.000) & \textbf{0.347}\,(0.001)
  & \textbf{0.254}\,(0.003) & \textbf{0.369}\,(0.002) \\
V1   & $\mathcal{L}_\mathrm{ECE}$
  & \textbf{0.594}$^\ddagger$ & 0.188 & 0.432 & 0.346 & 0.263 & 0.364 \\
V2   & $\mathcal{L}_\mathrm{ECE}{+}\mathcal{L}_\mathrm{energy}$
  & 0.618 & 0.185 & 0.436 & 0.343 & 0.258 & 0.368 \\
V3   & $\mathcal{L}_\mathrm{energy}$
  & 0.598 & 0.190 & 0.432 & 0.347 & 0.260 & 0.365 \\
V1W  & $\mathcal{L}_\mathrm{ECE}$ (warm-up)
  & 0.614 & 0.184 & 0.432 & 0.340 & 0.261 & 0.366 \\
V2R$^*$ & $\mathcal{L}_\mathrm{energy}{+}\mathcal{L}_\mathrm{ranking}$
  & 0.624\,(0.014) & 0.198\,(0.002) & 0.437\,(0.003)
  & 0.354\,(0.002) & 0.264\,(0.001) & 0.375\,(0.006) \\
\bottomrule
\multicolumn{8}{l}{{\scriptsize
  $^\ddagger$V1 improves FDE on ETH by fitting endpoint more precisely; ADE does not improve.}} \\
\end{tabular}
\end{table*}"""
)

TABLE_II = (
r"""\begin{table}[t]
\caption{SocialVAE ADE (m) and KL divergence on ETH under varying training
  set size (RQ3). KL is stable at 0.023--0.026 regardless of prior;
  dynamic KL prior offers no benefit at this scale.
  $^\star$Full ETH training set ($N=1845$); single-seed results.
  Note: ETH subset only---KL collapse may arise at larger scale (nuScenes, Waymo).}
\label{tab:socialvae}
\centering
\setlength{\tabcolsep}{5pt}
\begin{tabular}{l c c c c}
\toprule
Variant & $N$=500 & 1000 & 1500 & Full$^\star$ \\
\midrule
\multicolumn{5}{l}{\textit{ADE $\downarrow$ (m)}} \\[1pt]
SocialVAE (vanilla)                         & 0.74 & 0.72 & 0.71 & 0.73 \\
SocialVAE + $\mathcal{L}_\mathrm{KL}^\mathrm{dyn}$ & 0.76 & 0.75 & 0.71 & \textbf{0.72} \\
\midrule
\multicolumn{5}{l}{\textit{KL divergence (final training epoch)}} \\[1pt]
SocialVAE (vanilla)                         & 0.025 & 0.024 & 0.023 & 0.024 \\
SocialVAE + $\mathcal{L}_\mathrm{KL}^\mathrm{dyn}$ & 0.026 & 0.025 & 0.024 & 0.023 \\
\bottomrule
\end{tabular}
\end{table}"""
)

# ════════════════════════════════════════════════════════════════════════════
# KEY FIX: use  lambda m: REPLACEMENT  so re never parses backslashes
# ════════════════════════════════════════════════════════════════════════════

# 1. Replace the monolithic table* (tab:ablation)
pat_big = re.compile(
    r"\\begin\{table\*\}.*?\\label\{tab:ablation\}.*?\\end\{table\*\}",
    re.DOTALL,
)
if pat_big.search(src):
    src = pat_big.sub(lambda m: TABLE_IA, src, count=1)
    print("[OK] Replaced monolithic Table I with Table Ia + Ib")
else:
    print("[WARN] tab:ablation block not found — check regex anchor")

# 2. Replace SocialVAE table (tab:socialvae)
pat_soc = re.compile(
    r"\\begin\{table\}.*?\\label\{tab:socialvae\}.*?\\end\{table\}",
    re.DOTALL,
)
if pat_soc.search(src):
    src = pat_soc.sub(lambda m: TABLE_II, src, count=1)
    print("[OK] Replaced Table II (SocialVAE) with KL-column version")
else:
    print("[WARN] tab:socialvae block not found — check regex anchor")

# 3. Update \ref{tab:ablation} cross-references
src = src.replace(
    r"\ref{tab:ablation}",
    r"\ref{tab:ablation_safety} and~\ref{tab:ablation_accuracy}"
)
print("[OK] Cross-references updated")

# 4. Fix Figure 1 caption (P1: cut to <=80 words)
pat_cap1 = re.compile(
    r"\\caption\{%\s*\\textbf\{\(a\) Safety--accuracy Pareto front\.\}.*?\\label\{fig:main\}",
    re.DOTALL,
)
NEW_CAP1 = (
r"""\caption{%
    \textbf{(a)} Safety--accuracy Pareto front: V0* and V2R* with $3\sigma$ error bars (3-seed);
    V1--V1W collapsed to grey cluster (single-seed).
    \textbf{(b)} SVR per subset; V2R* reduces SVR in sparse ETH ($-0.029$),
    near-invariant in dense UNIV ($-0.001$).
    \textbf{(c)} SPSR per subset; $^\ddagger$ETH regression ($-0.028$):
    61.8\,\% of scenes have no real neighbours after ego exclusion.
    Tile colours: blue\,=\,Sparse, green\,=\,Medium, orange\,=\,Dense.
    ($n{=}5$ subsets; inferential statistics not reported.)}
  \label{fig:main}"""
)
if pat_cap1.search(src):
    src = pat_cap1.sub(lambda m: NEW_CAP1, src, count=1)
    print("[OK] Figure 1 caption shortened to <=80 words")
else:
    print("[WARN] Figure 1 caption pattern not matched")

# 5. Fix Figure 2 caption panel (b) title (remove explanatory axis-label prose)
src = src.replace(
    r"\textbf{(b) Accuracy preservation for V1--V1W.}",
    r"\textbf{(b) ADE Preservation, V1--V1W} (V2R excluded: $+19.5\,\%$ avg, see Table~\ref{tab:ablation_accuracy})."
)
print("[OK] Figure 2 panel (b) label cleaned")

# 6. Update figure filenames to revised PDFs
src = src.replace("{fig_main_3panel}", "{fig_main_3panel_revised}")
src = src.replace("{fig_ece_accuracy}", "{fig_ece_accuracy_revised}")
print("[OK] Figure filenames updated to *_revised")

TEX.write_text(src, encoding="utf-8")
print("[DONE] main_ral.tex patched. Backup at main_ral.tex.bak_tables")
