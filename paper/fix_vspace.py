import pathlib, shutil

f = pathlib.Path("main_ral.tex")
src = f.read_text(encoding="utf-8")
shutil.copy(f, f.with_suffix(".tex.bak_vspace"))

# FIX: Add \vfill and proper IEEEtran spacing controls
# Place after \begin{document} before \maketitle
# These tell IEEEtran to strictly respect bottom margin

CONTROLS = (
    "% Strict bottom margin enforcement\n"
    "\\flushbottom\n"
)

if "\\flushbottom" not in src:
    src = src.replace(
        "\\begin{document}\n",
        "\\begin{document}\n" + CONTROLS
    )
    print("[OK] Added flushbottom")
else:
    print("[SKIP] flushbottom already present")

# FIX 2: The footnote on page 2 is too long - shorten it in tex
OLD_FN = (
    "Real neighbours exclude: (i) the ego agent (slot 0, whose future matches\n"
    "ground truth); (ii) zero-occupancy padding slots (all-zero future trajectories).\n"
    "Only non-sentinel, non-ego, non-zero slots are checked."
)
NEW_FN = (
    "Real neighbours exclude the ego agent (slot~0) and zero-occupancy padding slots.\n"
    "Only non-sentinel, non-ego, non-zero slots are checked."
)
if OLD_FN in src:
    src = src.replace(OLD_FN, NEW_FN)
    print("[OK] Footnote shortened")
else:
    print("[SKIP] Footnote not matched - trying partial")
    OLD_FN2 = "Real neighbours exclude: (i) the ego agent (slot 0, whose future matches"
    if OLD_FN2 in src:
        # find and replace the whole footnote block
        import re
        src = re.sub(
            r"Real neighbours exclude:.*?are checked\.",
            "Real neighbours exclude the ego agent (slot~0) and zero-occupancy padding slots. Only non-sentinel, non-ego, non-zero slots are checked.",
            src, flags=re.DOTALL, count=1
        )
        print("[OK] Footnote shortened via regex")

f.write_text(src, encoding="utf-8")
print("[DONE]")