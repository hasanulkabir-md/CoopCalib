"""
patch_inference.py  --  adds neis saving to run_inference.py
Run from C:\CoopCalib\
"""
import pathlib

src = pathlib.Path('scripts/run_inference.py').read_text(encoding='utf-8')

# 1. Add all_neis list after all_obs
old_init = "all_obs     = []   # list of (B,  8, 2)"
new_init  = "all_obs     = []   # list of (B,  8, 2)\n    all_neis    = []   # list of (B, N, 12, 2) neighbour futures"
src = src.replace(old_init, new_init)

# 2. Collect neis futures inside loop
old_append = "        all_obs.append(ped_obs.cpu().numpy())"
new_append  = ("        all_obs.append(ped_obs.cpu().numpy())\n"
               "        neis_fut = neis[:, :, args.obs_len:].cpu().numpy()\n"
               "        all_neis.append(neis_fut)")
src = src.replace(old_append, new_append)

# 3. Concatenate neis
old_concat = "obs_np     = np.concatenate(all_obs,      axis=0)   # (N,  8, 2)"
new_concat  = ("obs_np     = np.concatenate(all_obs,      axis=0)   # (N,  8, 2)\n"
               "neis_np    = np.concatenate(all_neis,     axis=0)   # (N, MaxN, 12, 2)")
src = src.replace(old_concat, new_concat)

# 4. Save neis
old_save = 'np.save(os.path.join(args.out_dir, f"{args.dataset_name}_obs.npy"),     obs_np)'
new_save  = (old_save + "\n"
             'np.save(os.path.join(args.out_dir, f"{args.dataset_name}_neis.npy"),    neis_np)')
src = src.replace(old_save, new_save)

pathlib.Path('scripts/run_inference.py').write_text(src, encoding='utf-8')
print("Patched scripts/run_inference.py — neis futures will now be saved.")
print("Re-run inference for all 5 folds to get _neis.npy files.")
