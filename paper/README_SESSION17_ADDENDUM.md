# CoopCalib-TP Session 17 Addendum — Figure/Table/Margin Work Log

## What Was Done This Session (Figures, Tables, Margins)

### Figures Revised
- `generate_figures_revised.py` — canonical figure script
- Fig 1: 3-panel (Pareto + SVR + SPSR)
  - V1-V1W collapsed to grey ellipse cluster (P3 fix)
  - Delta annotations moved above error bars with dynamic ypos (P5 fix)
  - ETH footnote added with dagger symbol
  - Density tier background tiles added
- Fig 2: ECE dot-plot + ADE preservation
  - Annotation "All variants within ±0.005" moved to bottom-left (P7 fix)
  - ±2%/±3% threshold labels moved inside axes via transAxes (P8 fix)
  - Panel (b) x-axis labels fixed to explicit ETH/HOT/UNV/ZR1/ZR2
  - Panel (b) title added
  - Fig 2(a) title clarified: "V0 line bold, no table bold"

### Tables Revised (patch_tables.py)
- Table I split into Table Ia (safety: SVR+SPSR) and Table Ib (accuracy: ECE+ADE+FDE)
- Table II (SocialVAE) — KL divergence column added, variant renamed from V3 to SocialVAE+L_KL^dyn
- FPR=0.000 stated explicitly in Table Ia caption
- ECE editorial header moved to caption
- All std formats changed from (±.xxx) to (0.xxx)
- Figure 1 caption shortened from ~230 words to ~80 words

### LaTeX Fixes Applied
- Removed conflicting \usepackage[pass,letterpaper]{geometry} — was setting oddsidemargin=-23pt
- Removed \pdfpagewidth, \pdfpageheight, \special{papersize} lines
- Removed \IEEEoverridecommandlockouts (was in wrong position — preamble not supported)
- Removed \IEEEtriggeratref (caused margin impositions)
- Removed \flushbottom (conflicts with IEEEtran two-column mode)
- Added \raggedbottom after \begin{document}
- Float specifiers changed from [!t] to [tp]
- Float counters: topnumber=4, dbltopnumber=4, totalnumber=6
- Removed hyperref (caused invisible link boxes flagged by IEEE checker)
- Float tuning moved to preamble as \renewcommand (not \setlength after \maketitle)

### Margin Issue Status — UNRESOLVED
- IEEE PDF eXpress reports margin impositions on ALL pages (1-7)
- No Overfull \hbox in log — so no line exceeds column width
- Underfull \vbox (badness 10000) present — float placement issue
- Geometry package confirmed removed
- hyperref confirmed removed
- All fixes above did NOT resolve the margin checker flags
- Root cause NOT yet confirmed — need exact element+margin identification

### NEXT STEP REQUIRED
Download the "overlaid with border templates" PDF from IEEE PDF eXpress.
For each page identify:
  - Which specific element crosses which margin (top/bottom/left/right)
  - Is it text, a figure, a table, or a caption
Then report here for targeted fix.

### Backup Files Present
- main_ral.tex.bak_tables      — before table split
- main_ral.tex.bak_margins     — before geometry removal
- main_ral.tex.bak_margins2    — before IEEEoverridecommandlockouts removal
- main_ral.tex.bak_geometry    — before geometry package removal
- main_ral.tex.bak_href        — before hyperref removal
- main_ral.tex.bak_vspace      — before flushbottom addition
- main_ral.tex.bak_vbox        — before raggedbottom/float fixes

### Current File State
- main_ral.tex        — active working file
- main_ral.pdf        — 7 pages (grew from 6 due to float reflow)
- fig_main_3panel_revised.pdf   — Figure 1 (vector PDF, IEEE fonts)
- fig_ece_accuracy_revised.pdf  — Figure 2 (vector PDF, IEEE fonts)
- generate_figures_revised.py   — canonical figure script
- patch_tables.py               — table replacement script
- fix_geometry.py               — geometry removal script
- fix_vbox.py                   — float/vbox fix script

### IEEE PDF eXpress Results So Far
| Attempt | Change | Result |
|---------|--------|--------|
| 1 | Original submission | All pages margin fail |
| 2 | Added geometry pass | Still fail |
| 3 | Removed geometry | Still fail |
| 4 | Removed hyperref | Still fail |
| 5 | Added raggedbottom + float [tp] | Still fail — now 7 pages |

### Critical Rule Added
Never use \usepackage{geometry} with IEEEtran — even with [pass] option
it overwrites oddsidemargin to -23pt which IEEE checker reads from PDF metadata.