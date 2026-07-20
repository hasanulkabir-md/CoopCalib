import pathlib, shutil

f = pathlib.Path("main_ral.tex")
src = f.read_text(encoding="utf-8")
shutil.copy(f, f.with_suffix(".tex.bak_floatfix"))

# FIX 1: Remove the setlength block from after \maketitle
BAD = (
    "\n% Float tuning for IEEEtran double-column\n"
    "\\setlength{\\dbltopfraction}{0.9}\n"
    "\\setlength{\\topfraction}{0.9}\n"
    "\\setlength{\\dblfloatpagefraction}{0.7}\n"
    "\\setlength{\\floatpagefraction}{0.7}\n"
)
if BAD in src:
    src = src.replace(BAD, "\n")
    print("[OK] Removed setlength block from after maketitle")
else:
    # Try without the comment line
    BAD2 = (
        "\\setlength{\\dbltopfraction}{0.9}\n"
        "\\setlength{\\topfraction}{0.9}\n"
        "\\setlength{\\dblfloatpagefraction}{0.7}\n"
        "\\setlength{\\floatpagefraction}{0.7}\n"
    )
    if BAD2 in src:
        src = src.replace(BAD2, "")
        print("[OK] Removed setlength block (no comment line)")
    else:
        print("[WARN] setlength block not found as string - check manually")

# FIX 2: Remove IEEEtriggeratref entirely - causes margin impositions
src = src.replace("\\IEEEtriggeratref{8}\n", "")
src = src.replace("\\IEEEtriggeratref{4}\n", "")
print("[OK] Removed IEEEtriggeratref")

# FIX 3: Insert float tuning BEFORE \begin{document} in preamble
# This is the correct IEEEtran location - preamble setlength does not echo
PREAMBLE_FLOATS = (
    "% Float placement tuning — must be in preamble for IEEEtran\n"
    "\\renewcommand{\\topfraction}{0.9}\n"
    "\\renewcommand{\\dbltopfraction}{0.9}\n"
    "\\renewcommand{\\floatpagefraction}{0.7}\n"
    "\\renewcommand{\\dblfloatpagefraction}{0.7}\n"
)
if "\\renewcommand{\\topfraction}" not in src:
    src = src.replace(
        "\\begin{document}",
        PREAMBLE_FLOATS + "\\begin{document}"
    )
    print("[OK] Float tuning inserted in preamble")
else:
    print("[SKIP] Float tuning already in preamble")

f.write_text(src, encoding="utf-8")
print("[DONE] Saved. Backup at main_ral.tex.bak_floatfix")