import pathlib

f = pathlib.Path("main_ral.tex")
src = f.read_text(encoding="utf-8")
lines = src.splitlines()

print("=== PREAMBLE (lines 1-70) ===")
for i, line in enumerate(lines[:70], 1):
    print(f"L{i:3d}: {line}")