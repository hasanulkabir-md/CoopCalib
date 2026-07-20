import pathlib, re

# Read the log and find ALL warning lines with context
log = pathlib.Path("main_ral.log").read_text(encoding="utf-8", errors="ignore")
lines = log.splitlines()

print("=== ALL OVERFULL/UNDERFULL WITH CONTEXT ===")
for i, line in enumerate(lines):
    if any(x in line for x in ["Overfull", "Underfull", "vbox", "hbox"]):
        start = max(0, i-1)
        end   = min(len(lines), i+3)
        for j in range(start, end):
            print(f"L{j+1}: {lines[j]}")
        print("---")

print("\n=== PAGE BREAKS ===")
for i, line in enumerate(lines):
    if "[" in line and "]" in line and any(
        c.isdigit() for c in line) and "page" in line.lower():
        print(f"L{i+1}: {line}")

print("\n=== TEXTHEIGHT AND LAYOUT ===")
for i, line in enumerate(lines):
    if any(x in line for x in
           ["textheight","textwidth","paperheight","paperwidth",
            "hoffset","voffset","topmargin","marginpar"]):
        print(f"L{i+1}: {line}")