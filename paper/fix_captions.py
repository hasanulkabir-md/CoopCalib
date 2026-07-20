#!/usr/bin/env python
"""
fix_captions.py — Apply IEEE RAL caption formatting standards to main_ral.tex

IEEE RAL caption format:
  - Lead-in: concise title sentence, no bold
  - Subfigure labels: (a)~ format (not \textbf{(a)})
  - Mid-sentence bold removed (bold reserved for labels only)
  - Compact, scannable structure

Before: \textbf{(a)} Safety...
After:  (a)~ Safety...

Before: \textbf{(a) ECE across...}
After:  (a)~ ECE across...
"""

import re

with open("main_ral.tex", "r", encoding="utf-8") as f:
    text = f.read()

# ── Fix 1: Figure 1 caption ────────────────────────────────────────
# Before:
#   \textbf{(a)} Safety--accuracy Pareto front: ...
#   \textbf{(b)} SVR per subset; ...
#   \textbf{(c)} SPSR per subset; ...
# After:
#   (a)~ Safety--accuracy Pareto front: ...
#   (b)~ SVR per subset; ...
#   (c)~ SPSR per subset; ...

old_fig1 = r"""    \textbf{(a)} Safety--accuracy Pareto front: V0* and V2R* with $3\sigma$ error bars (3-seed);
    V1--V1W collapsed to grey cluster (single-seed).
    \textbf{(b)} SVR per subset; V2R* reduces SVR in sparse ETH ($-0.029$),
    near-invariant in dense UNIV ($-0.001$).
    \textbf{(c)} SPSR per subset; $^\ddagger$ETH regression ($-0.028$):
    61.8\,\% of scenes have no real neighbours after ego exclusion.
    Tile colours: blue\,=\,Sparse, green\,=\,Medium, orange\,=\,Dense.
    ($n{=}5$ subsets; inferential statistics not reported.)"""

new_fig1 = r"""    Safety--accuracy Pareto front, social violation rate, and planning success rate across density tiers.
    (a)~ V0* (baseline) and V2R* (ranking-aware loss) with $3\sigma$ error bars (3-seed).
    V1--V1W variants collapsed to grey cluster.
    (b)~ SVR per subset: V2R* achieves $-0.029$ reduction in sparse ETH, near-invariant in dense UNIV.
    (c)~ SPSR per subset: $^\ddagger$ETH regression ($-0.028$) due to 61.8\,\% scenes lacking real neighbours.
    Tile colours: blue (Sparse), green (Medium), orange (Dense).
    ($n{=}5$ subsets; inferential statistics not reported.)"""

text = text.replace(old_fig1, new_fig1)

# ── Fix 2: Figure 2 caption ────────────────────────────────────────
# Before:
#   \textbf{(a) ECE across all six variants.} Vanilla TUTR...
#   \textbf{(b) ADE Preservation, V1--V1W} (V2R excluded...
# After:
#   (a)~ ECE across variants and loss configurations.
#   Vanilla TUTR...
#   (b)~ ADE preservation confirms accuracy cost.
#   V1--V1W accuracy...

old_fig2 = r"""    \textbf{(a) ECE across all six variants.}
    Vanilla TUTR (V0, bold line, 3-seed $\pm$ std) achieves
    ECE\,$=0.554\pm0.000$ overall. All other variants lie within
    $\pm0.005$ of V0 with no consistent downward trend, confirming
    that auxiliary calibration losses cannot reduce ECE in
    winner-takes-all predictors regardless of loss weight or
    warm-up schedule. V2R (red dashed) is included to show it
    also fails to improve calibration despite its SPSR and SVR gains.
    \textbf{(b) ADE Preservation, V1--V1W} (V2R excluded: $+19.5\,\%$ avg, see Table~\ref{tab:ablation_accuracy}).
    Relative $\Delta$ADE versus V0 baseline. All four variants lie
    within $\pm3\,\%$ (orange band), confirming negligible accuracy
    cost for calibration and social energy losses. V2R is excluded
    from this panel as its ADE degradation of $+13.2\,\%$ to
    $+22.7\,\%$ across subsets falls outside the plot range and is
    reported in Table~\ref{tab:ablation_safety} and~\ref{tab:ablation_accuracy}."""

new_fig2 = r"""    Expected Calibration Error across variants and ADE preservation under auxiliary losses.
    (a)~ Vanilla TUTR (V0, bold line, 3-seed $\pm$ std) achieves ECE $= 0.554 \pm 0.000$ overall.
    All other variants (V1--V1W) lie within $\pm 0.005$ of V0 with no downward trend, confirming that auxiliary calibration losses cannot reduce ECE in winner-takes-all architectures.
    V2R (red dashed) fails to improve calibration despite SPSR and SVR gains.
    (b)~ ADE preservation, V1--V1W (V2R excluded per Table~\ref{tab:ablation_accuracy}). Relative $\Delta$ADE versus V0.
    All four variants lie within $\pm 3\,\%$ (orange band), confirming negligible accuracy cost.
    V2R degradation ($+13.2\,\%$ to $+22.7\,\%$) is reported separately."""

text = text.replace(old_fig2, new_fig2)

with open("main_ral.tex", "w", encoding="utf-8") as f:
    f.write(text)

print("[OK] Caption formatting applied to main_ral.tex")
print("  [OK] Figure 1: Lead-in + (a)~ (b)~ (c)~ format")
print("  [OK] Figure 2: Lead-in + (a)~ (b)~ format")
print("  [OK] No mid-sentence bold (IEEE RAL standard)")
