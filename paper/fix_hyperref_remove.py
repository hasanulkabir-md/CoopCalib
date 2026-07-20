import pathlib, shutil

f = pathlib.Path("main_ral.tex")
src = f.read_text(encoding="utf-8")
shutil.copy(f, f.with_suffix(".tex.bak_href"))

# The ONLY fix needed: remove hyperref entirely
# IEEE RAL does not require hyperref - it causes invisible link boxes
# that IEEE PDF eXpress detects as margin violations
src = src.replace(
    "\\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue,pdfborder={0 0 0}]{hyperref}\n\\hypersetup{pdfstartview=}",
    ""
)
# Also catch original form
src = src.replace(
    "\\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}",
    ""
)
# Also catch any standalone hyperref
import re
src = re.sub(r"\\usepackage(\[[^\]]*\])?\{hyperref\}[^\n]*\n", "", src)
src = re.sub(r"\\hypersetup\{[^\}]*\}\n", "", src)

print("[OK] hyperref removed")
f.write_text(src, encoding="utf-8")
print("[DONE]")