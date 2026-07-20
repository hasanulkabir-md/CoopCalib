import pathlib, shutil

f = pathlib.Path("main_ral.tex")
src = f.read_text(encoding="utf-8")
shutil.copy(f, f.with_suffix(".tex.bak_geometry"))

# Remove the entire geometry block (lines 33-39)
REMOVE = [
    "% Force US Letter dimensions for MiKTeX/PDFLaTeX\n\\usepackage[pass, letterpaper]{geometry}\n\\pdfpagewidth=8.5in\n\\pdfpageheight=11in\n\n% Optional: Ensure the PDF driver knows it's Letter\n\\special{papersize=8.5in,11in}\n",
    "% Force US Letter dimensions for MiKTeX/PDFLaTeX\r\n\\usepackage[pass, letterpaper]{geometry}\r\n\\pdfpagewidth=8.5in\r\n\\pdfpageheight=11in\r\n\r\n% Optional: Ensure the PDF driver knows it's Letter\r\n\\special{papersize=8.5in,11in}\r\n",
]

removed = False
for block in REMOVE:
    if block in src:
        src = src.replace(block, "")
        removed = True
        print("[OK] Removed geometry block (exact match)")
        break

if not removed:
    # Line-by-line removal fallback
    lines = src.splitlines()
    filtered = []
    skip_patterns = [
        "usepackage[pass, letterpaper]{geometry}",
        "usepackage[pass,letterpaper]{geometry}",
        "pdfpagewidth=8.5in",
        "pdfpageheight=11in",
        "special{papersize=8.5in,11in}",
        "Force US Letter dimensions",
        "Optional: Ensure the PDF driver",
    ]
    removed_count = 0
    for line in lines:
        if any(p in line for p in skip_patterns):
            print(f"  [REMOVED] {line}")
            removed_count += 1
        else:
            filtered.append(line)
    src = "\n".join(filtered) + "\n"
    print(f"[OK] Removed {removed_count} lines via fallback")

f.write_text(src, encoding="utf-8")
print("[DONE] geometry removed. Backup at main_ral.tex.bak_geometry")