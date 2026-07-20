import pathlib, re

f = pathlib.Path("generate_figures_revised.py")
src = f.read_text(encoding="utf-8")

# Remove ALL existing threshold label lines regardless of form
old = re.compile(
    r"# Threshold labels.*?ax\.set_xlim\([^)]+\)\n",
    re.DOTALL
)

# Use transAxes for BOTH x and y - fully decoupled from data coords
# ylim is -5.5 to 5.5, range=11. Convert data y to axes fraction:
# axes_y = (data_y - ymin) / (ymax - ymin) = (data_y + 5.5) / 11
NEW = (
    "# Threshold labels in pure axes coords (transAxes) - never touches spine\n"
    "for dy, lbl, col in [(2.1,'\u00b12%','#FF8C00'),(3.1,'\u00b13%','#CC8800')]:\n"
    "    ay = (dy + 5.5) / 11.0\n"
    "    ax.annotate(lbl, xy=(1.0, ay), xycoords='axes fraction',\n"
    "                xytext=(6, 0), textcoords='offset points',\n"
    "                ha='left', va='bottom', fontsize=6,\n"
    "                color=col, fontweight='bold',\n"
    "                annotation_clip=False)\n"
    "    ay2 = (-dy + 5.5) / 11.0\n"
    "    ax.annotate(lbl, xy=(1.0, ay2), xycoords='axes fraction',\n"
    "                xytext=(6, 0), textcoords='offset points',\n"
    "                ha='left', va='top', fontsize=6,\n"
    "                color=col, fontweight='bold',\n"
    "                annotation_clip=False)\n"
    "ax.set_xlim(-0.5, 4.5)\n"
)

if old.search(src):
    src = old.sub(lambda m: NEW, src, count=1)
    print("[OK] Replaced threshold block")
else:
    # Show context so we can find it manually
    for i, line in enumerate(src.splitlines(), 1):
        if any(x in line for x in ["xlim","4.55","threshold","00b1","±","get_yaxis"]):
            print(f"  L{i}: {line}")
    import sys; sys.exit(1)

f.write_text(src, encoding="utf-8")
print("[DONE]")