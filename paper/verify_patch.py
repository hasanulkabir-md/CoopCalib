import pathlib, sys

src = pathlib.Path("main_ral.tex").read_text(encoding="utf-8")
checks = [
    ("tab:ablation_safety",   True,  "Table Ia label present"),
    ("tab:ablation_accuracy", True,  "Table Ib label present"),
    ("KL divergence (final training epoch)", True, "KL row in Table II"),
    ("SocialVAE (vanilla)",   True,  "Renamed SocialVAE variant"),
    ("fig_main_3panel_revised",  True,  "Revised Figure 1 filename"),
    ("fig_ece_accuracy_revised", True,  "Revised Figure 2 filename"),
    ("no bold---establishing baseline", False, "Editorial header removed"),
    ("\\label{tab:ablation}",  False, "Old monolithic table label gone"),
]
fail = 0
for pattern, should_exist, desc in checks:
    found = pattern in src
    ok = found == should_exist
    if not ok:
        fail += 1
    print(("[OK  ]" if ok else "[FAIL]") + " " + desc)
print()
if fail == 0:
    print("ALL CHECKS PASSED")
else:
    print(str(fail) + " CHECK(S) FAILED -- do not compile")
    sys.exit(1)
