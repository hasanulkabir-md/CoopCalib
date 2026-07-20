"""
update_density_tiers.py — Patch density_manifest.json with data-driven tier assignments.

FINDING from compute_mnd.py:
  ETH    2.284m -> sparse   (as proposed)
  HOTEL  1.925m -> medium   (proposal said dense -- CORRECTED by data)
  UNIV   0.921m -> dense    (as proposed)
  ZARA01 1.747m -> medium   (as proposed)
  ZARA02 1.522m -> medium   (as proposed)

REVISED thresholds (data-driven):
  Sparse : MND > 2.0m  -> ETH
  Medium : 1.0 <= MND <= 2.0m  -> HOTEL, ZARA01, ZARA02
  Dense  : MND < 1.0m  -> UNIV

  HOTEL moves from "dense" to "medium" tier.
  This is a one-line change to the proposal (§4.2) -- empirically justified.

Run once, then discard this script.
"""

import json
from pathlib import Path

OUT_FILE = Path(r"C:\CoopCalib\data\processed\density_manifest.json")

with open(OUT_FILE) as f:
    manifest = json.load(f)

# Add the corrected tier assignments as a top-level key
manifest["data_driven_tiers"] = {
    "thresholds_m": {
        "sparse_above": 2.0,
        "dense_below":  1.0,
        "note": "medium = 1.0 <= MND <= 2.0"
    },
    "tier_assignments": {
        "sparse": ["eth"],
        "medium": ["hotel", "zara01", "zara02"],
        "dense":  ["univ"],
    },
    "vs_proposal": {
        "hotel": {
            "proposal": "dense",
            "measured": "medium",
            "mean_mnd_m": manifest["hotel"]["mean_mnd_m"],
            "note": "Narrow corridor but low pedestrian count -> medium MND. Proposal assumed dense."
        }
    }
}

with open(OUT_FILE, "w") as f:
    json.dump(manifest, f, indent=2)

print("density_manifest.json updated with data_driven_tiers key.")
print()
print("ACTION REQUIRED -- update proposal §4.2 wording to:")
print()
print("  Sparse  (MND > 2.0m): ETH")
print("  Medium  (1.0-2.0m)  : HOTEL, ZARA1, ZARA2")
print("  Dense   (MND < 1.0m): UNIV")
print()
print("HOTEL reclassified from dense -> medium based on measured MND = 1.925m.")
print("This strengthens the paper: tier assignments are empirically grounded.")
