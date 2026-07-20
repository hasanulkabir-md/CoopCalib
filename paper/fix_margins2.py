import pathlib, re, shutil

TEX = pathlib.Path("main_ral.tex")
src = TEX.read_text(encoding="utf-8")
shutil.copy(TEX, TEX.with_suffix(".tex.bak_margins2"))

# FIX 1: Remove both occurrences of \IEEEoverridecommandlockouts from preamble
src = src.replace("\\IEEEoverridecommandlockouts\n", "")
print("[OK] Removed all IEEEoverridecommandlockouts instances")

# FIX 2: Remove the bad float tuning block entirely (has renewcommand + wrong placement)
bad_block = re.compile(
    r"% ── Float placement tuning.*?\\renewcommand\{\\topfraction\}\{0\.9\}\s*\n",
    re.DOTALL
)
if bad_block.search(src):
    src = bad_block.sub("", src)
    print("[OK] Removed bad float tuning block")
else:
    print("[WARN] Float block pattern not found - checking alternate")
    bad_block2 = re.compile(
        r"\\setlength\{\\dbltopfraction\}.*?\\renewcommand\{\\topfraction\}\{[0-9.]+\}\s*\n",
        re.DOTALL
    )
    if bad_block2.search(src):
        src = bad_block2.sub("", src)
        print("[OK] Removed bad float tuning block (alternate match)")

# FIX 3: Insert correct float tuning AFTER \maketitle (not before)
GOOD_FLOAT = (
    "\\maketitle\n\n"
    "% Float tuning for IEEEtran double-column\n"
    "\\setlength{\\dbltopfraction}{0.9}\n"
    "\\setlength{\\topfraction}{0.9}\n"
    "\\setlength{\\dblfloatpagefraction}{0.7}\n"
    "\\setlength{\\floatpagefraction}{0.7}\n"
)
if "\\dbltopfraction" not in src:
    src = src.replace("\\maketitle\n", GOOD_FLOAT, 1)
    print("[OK] Inserted correct float tuning after maketitle")
else:
    print("[SKIP] Float tuning already present")

# FIX 4: IEEEtriggeratref - bump from 4 to 8 so it fires later
src = src.replace("\\IEEEtriggeratref{4}", "\\IEEEtriggeratref{8}")
print("[OK] IEEEtriggeratref updated to {8}")

TEX.write_text(src, encoding="utf-8")
print("[DONE] Saved. Backup at main_ral.tex.bak_margins2")