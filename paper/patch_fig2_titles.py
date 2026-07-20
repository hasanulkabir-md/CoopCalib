import pathlib, re

f = pathlib.Path("generate_figures_revised.py")
src = f.read_text(encoding="utf-8")

# FIX 1: Fig2(b) — restore x-axis subset labels and title
# The xticks are set but xticklabels line may be missing after patches
OLD_ADE_XTICKS = 'ax.set_xticks(x_sub)\nax.set_xticklabels(SUBSETS, fontsize=7)\nax.set_ylabel("\u0394ADE vs V0 (%)", fontsize=8)\nax.set_title("(b) ADE Preservation (V1\u2013V1W)\nV2R excluded (+19.5% avg)",'
NEW_ADE_XTICKS = 'ax.set_xticks(x_sub)\nax.set_xticklabels(["ETH","HOT","UNV","ZR1","ZR2"], fontsize=7)\nax.set_ylabel("\u0394ADE vs V0 (%)", fontsize=8)\nax.set_title("(b) ADE Preservation: V1\u2013V1W (seed 42)\\nV2R excluded (\u00b1ref: +19.5\\% avg, Table~II)",'

if OLD_ADE_XTICKS in src:
    src = src.replace(OLD_ADE_XTICKS, NEW_ADE_XTICKS)
    print("[OK] Fig2(b) xticklabels and title fixed")
else:
    # Fallback: patch just the set_title for panel b
    old2 = re.compile(
        r'ax\.set_title\("\\(b\\) ADE Preservation[^"]*"[^)]*\)',
        re.DOTALL
    )
    NEW_TITLE = ('ax.set_title("(b) ADE Preservation: V1\u2013V1W (seed\u00a042)\\n'
                 'V2R excluded (avg +19.5\\%, see Table II)", fontsize=8, pad=3)')
    if old2.search(src):
        src = old2.sub(lambda m: NEW_TITLE, src, count=1)
        print("[OK] Fig2(b) title patched via regex")

    # Also ensure xticklabels are explicit strings not SUBSETS variable
    src = re.sub(
        r"(# ── \(b\) ADE preservation.*?ax\.set_xticks\(x_sub\)\n)"
        r"(\s*ax\.set_xticklabels\(SUBSETS)",
        r'\1    ax.set_xticklabels(["ETH","HOT","UNV","ZR1","ZR2"]',
        src, flags=re.DOTALL, count=1
    )
    print("[OK] Fig2(b) xticklabels set to explicit strings")

# FIX 2: Fig2(a) title — clarify "no bold" means table bold not line weight
src = src.replace(
    '"(a) ECE \u2014 All Variants\\n(first measurement; no bold)"',
    '"(a) ECE \u2014 All Variants\\n(first measurement; V0 line bold, no table bold)"'
)
src = src.replace(
    '"(a) ECE \\u2014 All Variants\\n(first measurement; no bold)"',
    '"(a) ECE \u2014 All Variants\\n(first measurement; V0 line bold, no table bold)"'
)
print("[OK] Fig2(a) title ambiguity fixed")

f.write_text(src, encoding="utf-8")
print("[DONE] Patches written")