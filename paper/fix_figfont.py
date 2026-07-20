#!/usr/bin/env python
"""
fix_figfont.py — Verify that figure fonts are set correctly

This script checks that:
  1. generate_figures_revised.py has Times New Roman configured
  2. pdf.fonttype = 42 (TrueType subset, not Type 3)
  3. font.serif includes Times New Roman

The actual figure regeneration happens via generate_figures_revised.py
"""

import os

# ── Check that generate_figures_revised.py has Times New Roman ────
with open("generate_figures_revised.py", "r", encoding="utf-8") as f:
    content = f.read()

checks = [
    ("pdf.fonttype", '"pdf.fonttype": 42'),
    ("Times New Roman", '"font.serif": ["Times New Roman"]'),
    ("figure dpi", '"figure.dpi": 300'),
]

print("[CHECK] generate_figures_revised.py font configuration:")
all_ok = True
for name, pattern in checks:
    if pattern in content:
        print(f"  [OK] {name:25s} found")
    else:
        print(f"  [FAIL] {name:25s} NOT FOUND")
        all_ok = False

if all_ok:
    print("\n[OK] All font configurations present in generate_figures_revised.py")
    print("\nNext step: Run 'python generate_figures_revised.py' to regenerate figures with correct fonts")
else:
    print("\n[FAIL] Missing font configurations")
