import pathlib

f = pathlib.Path("generate_figures_revised.py")
lines = f.read_text(encoding="utf-8").splitlines()

# Print lines 228-260 so we can see exactly what is there
for i in range(227, min(260, len(lines))):
    print(f"L{i+1}: {lines[i]}")