import pathlib

f = pathlib.Path("generate_figures_revised.py")
lines = f.read_text(encoding="utf-8").splitlines()

# Insert after line 250 (index 249) — after the for v, deltas loop
# Lines to insert for panel (b)
insert = [
    "",
    'ax.set_xticks([0,1,2,3,4])',
    'ax.set_xticklabels(["ETH","HOT","UNV","ZR1","ZR2"], fontsize=7)',
    'ax.set_ylabel("\u0394ADE vs V0 (%)", fontsize=8)',
    'ax.set_title("(b) ADE Preservation: V1\u2013V1W (seed 42)\\nAll within \u00b13% of V0; V2R excluded (avg +19.5%)", fontsize=8, pad=3)',
]

# Insert after index 250 (line 251)
lines = lines[:250] + insert + lines[250:]

# Verify
for i in range(248, 262):
    print(f"L{i+1}: {lines[i]}")

f.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("[DONE]")