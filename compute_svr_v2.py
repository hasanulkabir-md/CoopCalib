"""
compute_svr_v2.py  --  CoopCalib-TP  (final version using neis arrays)
"""
import numpy as np
import sys, os, json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metrics'))
from eval_suite import compute_svr_from_tutr_batch

PREDS_DIR  = 'experiments/results/preds_v2'
PED_RADIUS = 0.3

folds = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
tiers = {'eth': 'Sparse', 'hotel': 'Medium', 'zara1': 'Medium',
         'zara2': 'Medium', 'univ': 'Dense'}

svr_results = {}

for fold in folds:
    print(f'\n[{fold}]')
    samples = np.load(f'{PREDS_DIR}/{fold}_samples.npy')  # (N, 20, 12, 2)
    neis    = np.load(f'{PREDS_DIR}/{fold}_neis.npy')     # (N, MaxN, 12, 2)
    neis_t  = neis.transpose(0, 2, 1, 3)                 # (N, 12, MaxN, 2)
    print(f'  samples: {samples.shape}, neis_t: {neis_t.shape}')
    svr = compute_svr_from_tutr_batch(ego_samples=samples, neighbour_obs=neis_t, ped_radius=PED_RADIUS)
    svr_results[fold] = round(svr, 4)
    print(f'  SVR={svr:.4f}')

print('\n=== SVR Summary ===')
for fold, svr in svr_results.items():
    print(f'  {fold:6s} ({tiers[fold]:6s})  SVR={svr:.4f}')

metrics_path = 'experiments/results/v2_metrics.json'
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        metrics = json.load(f)
    for fold in folds:
        if fold in metrics:
            metrics[fold]['SVR'] = svr_results[fold]
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'\nMerged SVR into {metrics_path}')
else:
    with open('experiments/results/v2_svr.json', 'w') as f:
        json.dump(svr_results, f, indent=2)
    print('\nSaved experiments/results/v2_svr.json')
