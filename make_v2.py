t = pathlib.Path('run_v1_metrics.py').read_text() 
t = t.replace('preds_v1', 'preds_v2').replace('v1_metrics', 'v2_metrics') 
pathlib.Path('run_v2_metrics.py').write_text(t) 
