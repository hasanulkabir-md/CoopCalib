import pathlib

src = pathlib.Path('scripts/run_inference.py').read_text()

old = 'neis_np    = np.concatenate(all_neis,    axis=0)   # (N, MaxN, 12, 2)'

new = '''max_n = max(a.shape[1] for a in all_neis)
neis_padded = []
for a in all_neis:
    pad = max_n - a.shape[1]
    if pad > 0:
        a = np.concatenate([a, np.full((a.shape[0], pad, a.shape[2], a.shape[3]), 1e9)], axis=1)
    neis_padded.append(a)
neis_np    = np.concatenate(neis_padded, axis=0)   # (N, MaxN, 12, 2)'''

assert old in src, "String not found — check run_inference.py"
src = src.replace(old, new)
pathlib.Path('scripts/run_inference.py').write_text(src)
print("Patched OK")
