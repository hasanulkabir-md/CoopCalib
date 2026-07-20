import pathlib, shutil, re

f = pathlib.Path("main_ral.tex")
src = f.read_text(encoding="utf-8")
shutil.copy(f, f.with_suffix(".tex.bak_vbox"))

# FIX 1: Remove \flushbottom - it CAUSES underfull vbox in two-column mode
# IEEEtran uses \flushbottom internally already; adding it again conflicts
src = src.replace("% Strict bottom margin enforcement\n\\flushbottom\n", "")
src = src.replace("\\flushbottom\n", "")
print("[OK] Removed conflicting flushbottom")

# FIX 2: Add \raggedbottom after \begin{document}
# This stops LaTeX stretching content to fill pages, eliminating vbox badness
if "\\raggedbottom" not in src:
    src = src.replace(
        "\\begin{document}\n",
        "\\begin{document}\n\\raggedbottom\n"
    )
    print("[OK] Added raggedbottom")

# FIX 3: Change all figure* and table* to use [tp] instead of [!t]
# t = top of page, p = float page — gives LaTeX two valid placements
# ! alone is too restrictive and causes cascading float buildup
src = src.replace(r"\begin{figure*}[!t]", r"\begin{figure*}[tp]")
src = src.replace(r"\begin{figure*}[t]",  r"\begin{figure*}[tp]")
src = src.replace(r"\begin{table*}[!t]",  r"\begin{table*}[tp]")
src = src.replace(r"\begin{table*}[t]",   r"\begin{table*}[tp]")
print("[OK] Float specifiers changed to [tp]")

# FIX 4: Increase LaTeX float patience parameters in preamble
FLOAT_PARAMS = (
    "% Float patience — prevents float buildup causing vbox badness\n"
    "\\setcounter{topnumber}{4}\n"
    "\\setcounter{dbltopnumber}{4}\n"
    "\\setcounter{totalnumber}{6}\n"
)
if "\\setcounter{topnumber}" not in src:
    src = src.replace(
        "\\begin{document}\n\\raggedbottom\n",
        "\\begin{document}\n\\raggedbottom\n" + FLOAT_PARAMS
    )
    print("[OK] Float counters added")

f.write_text(src, encoding="utf-8")
print("[DONE]")