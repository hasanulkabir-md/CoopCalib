import sys 
sys.path.insert(0,'C:/CoopCalib') 
import numpy as np, json 
from metrics.eval_suite import compute_ece, compute_fpr, compute_spsr 
folds = ['eth','hotel','univ','zara1','zara2'] 
results = {} 
for f in folds: 
    s=np.load(f'experiments/results/preds_v1/{f}_samples.npy') 
    g=np.load(f'experiments/results/preds_v1/{f}_gt.npy') 
    ece=compute_ece(s,g); fpr=compute_fpr(s,g); spsr=compute_spsr(s,g) 
    results[f]={'ECE':round(ece,4),'FPR':round(fpr,4),'SPSR':round(spsr,4)} 
    print(f,results[f]) 
json.dump(results,open('experiments/results/v1_metrics.json','w'),indent=2) 
print('saved v1_metrics.json') 
