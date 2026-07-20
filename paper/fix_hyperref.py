import pathlib
f = pathlib.Path("main_ral.tex")
src = f.read_text(encoding="utf-8")
src = src.replace(
    "\\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}",
    "\\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue,pdfborder={0 0 0}]{hyperref}\n\\hypersetup{pdfstartview=}"
)
f.write_text(src, encoding="utf-8")
print("[OK] hyperref border suppressed")