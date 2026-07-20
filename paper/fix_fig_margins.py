import pathlib

f = pathlib.Path("generate_figures_revised.py")
src = f.read_text(encoding="utf-8")

# FIX 1: Figure 1 width — reduce from 7.2 to 6.8 inches to stay within textwidth
src = src.replace(
    'fig1, axes = plt.subplots(1, 3, figsize=(7.2, 2.6),',
    'fig1, axes = plt.subplots(1, 3, figsize=(6.8, 2.5),'
)

# FIX 2: Figure 1 margins — tighten right margin
src = src.replace(
    'fig1.subplots_adjust(left=0.07, right=0.97, top=0.84,\n                     bottom=0.16, wspace=0.42)',
    'fig1.subplots_adjust(left=0.07, right=0.96, top=0.84,\n                     bottom=0.16, wspace=0.40)'
)

# FIX 3: Figure 2 width — reduce from 7.2 to 6.8 inches
src = src.replace(
    'fig2, axes2 = plt.subplots(1, 2, figsize=(7.2, 2.4))',
    'fig2, axes2 = plt.subplots(1, 2, figsize=(6.8, 2.4))'
)

# FIX 4: Figure 2 margins — tighten right so annotations dont bleed
src = src.replace(
    'fig2.subplots_adjust(left=0.08, right=0.97, top=0.86,\n                     bottom=0.16, wspace=0.36)',
    'fig2.subplots_adjust(left=0.08, right=0.88, top=0.86,\n                     bottom=0.16, wspace=0.36)'
)

# FIX 5: The ±% annotations use annotation_clip=False which draws outside axes
# Change xytext offset from 6 to 3 points so they stay closer to spine
src = src.replace(
    "                xytext=(6, 0), textcoords='offset points',\n                ha='left', va='bottom', fontsize=6,",
    "                xytext=(3, 0), textcoords='offset points',\n                ha='left', va='bottom', fontsize=6,"
)
src = src.replace(
    "                xytext=(6, 0), textcoords='offset points',\n                ha='left', va='top', fontsize=6,",
    "                xytext=(3, 0), textcoords='offset points',\n                ha='left', va='top', fontsize=6,"
)

# FIX 6: Move ±% labels fully inside axes using axes fraction for both coords
# Replace annotation_clip=False approach with text inside at x=0.96
src = src.replace(
    "for dy, lbl, col in [(2.1,'\u00b12%','#FF8C00'),(3.1,'\u00b13%','#CC8800')]:\n"
    "    ay = (dy + 5.5) / 11.0\n"
    "    ax.annotate(lbl, xy=(1.0, ay), xycoords='axes fraction',\n"
    "                xytext=(3, 0), textcoords='offset points',\n"
    "                ha='left', va='bottom', fontsize=6,\n"
    "                color=col, fontweight='bold',\n"
    "                annotation_clip=False)\n"
    "    ay2 = (-dy + 5.5) / 11.0\n"
    "    ax.annotate(lbl, xy=(1.0, ay2), xycoords='axes fraction',\n"
    "                xytext=(3, 0), textcoords='offset points',\n"
    "                ha='left', va='top', fontsize=6,\n"
    "                color=col, fontweight='bold',\n"
    "                annotation_clip=False)",
    "for dy, lbl, col in [(2.1,'\u00b12%','#FF8C00'),(3.1,'\u00b13%','#CC8800')]:\n"
    "    ay  = (dy  + 5.5) / 11.0\n"
    "    ay2 = (-dy + 5.5) / 11.0\n"
    "    ax.text(0.97, ay,  lbl, transform=ax.transAxes,\n"
    "            ha='right', va='bottom', fontsize=6,\n"
    "            color=col, fontweight='bold')\n"
    "    ax.text(0.97, ay2, lbl, transform=ax.transAxes,\n"
    "            ha='right', va='top', fontsize=6,\n"
    "            color=col, fontweight='bold')"
)

print("[OK] All size/margin/annotation fixes applied")

# Verify key values
for check in ["6.8, 2.5", "6.8, 2.4", "right=0.96", "right=0.88", "transAxes"]:
    status = "OK" if check in src else "MISS"
    print(f"  [{status}] {check}")

f.write_text(src, encoding="utf-8")
print("[DONE]")