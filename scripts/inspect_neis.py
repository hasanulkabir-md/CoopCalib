"""
inspect_neis.py — Diagnose what is stored in _neis.npy files.
"""
import numpy as np

for variant, preds_dir in [
    ("V0",  r"C:\CoopCalib\experiments\results\preds_v0_neis"),
    ("V2R", r"C:\CoopCalib\experiments\results\preds_v2r"),
]:
    print(f"\n{'='*60}")
    print(f"  {variant}")
    samples = np.load(preds_dir + r"\eth_samples.npy")
    gt      = np.load(preds_dir + r"\eth_gt.npy")
    obs     = np.load(preds_dir + r"\eth_obs.npy")
    neis    = np.load(preds_dir + r"\eth_neis.npy")

    print(f"  samples shape : {samples.shape}  → (N, K, T_fut=12, 2)")
    print(f"  gt shape      : {gt.shape}       → (N, T_fut=12, 2)")
    print(f"  obs shape     : {obs.shape}      → (N, T_obs=8, 2)")
    print(f"  neis shape    : {neis.shape}")

    # Print what T dimension is in neis
    if neis.shape[1] == 8:
        print(f"  DIAGNOSIS: neis dim1=8 → storing T_OBS neighbour trajectories")
        print(f"             SPSR is WRONG — needs T_FUT=12 neighbour futures")
    elif neis.shape[1] == 12:
        print(f"  DIAGNOSIS: neis dim1=12 → storing T_FUT neighbour trajectories")
        print(f"             SPSR should be correct")
    else:
        print(f"  DIAGNOSIS: neis dim1={neis.shape[1]} — unexpected shape")

    # Show first scene, first neighbour, all timesteps
    print(f"\n  Scene 0, Neighbour 0, all timesteps (neis[0, :, 0, :]):")
    print(f"  {neis[0, :, 0, :]}")
    print(f"\n  Scene 0 GT (ground truth future):")
    print(f"  {gt[0, :5, :]}  ... (first 5 of 12 steps)")
    print(f"\n  Scene 0 OBS (observed past):")
    print(f"  {obs[0, :, :]}")