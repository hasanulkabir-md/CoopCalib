import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metrics'))
from eval_suite import compute_ece, compute_fpr, compute_spsr

folds = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
results = {}
for f in folds:
    s=np.load(f'experiments/results/preds_v2/{f}_samples.npy')
    g=np.load(f'experiments/results/preds_v2/{f}_gt.npy')
    ece  = round(compute_ece(s, g), 4)
    fpr  = round(compute_fpr(s, g), 4)
    spsr = round(compute_spsr(s, g), 4)
    results[f] = {'ECE': ece, 'FPR': fpr, 'SPSR': spsr}
    print(f, results[f])

import json
with open('experiments/results/v2_metrics.json', 'w') as fh:
    json.dump(results, fh, indent=2)
print('saved v2_metrics.json')