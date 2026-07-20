import json

p = r'C:\CoopCalib\data\processed\density_manifest.json'

with open(p) as f:
    m = json.load(f)

m['sdd'] = {
    'status': 'preprocessed',
    'format': 'ethucy_txt',
    'scale': 0.05,
    'frame_skip': 12,
    'train_files': 55,
    'test_files': 5,
    'total_persons': 4117,
    'split': 'PECNet_standard',
    'path': r'C:\CoopCalib\data\processed\sdd'
}

m['trajnet_sample'] = {
    'status': 'ready',
    'source': 'ETH-UCY split_full',
    'n_scenes': 50,
    'avg_peds': 19.0,
    'max_peds': 57,
    'min_peds': 2,
    'seed': 42,
    'path': r'C:\CoopCalib\data\processed\trajnet_sample'
}

with open(p, 'w') as f:
    json.dump(m, f, indent=2)

print('Manifest updated successfully.')
print(f"Keys now in manifest: {list(m.keys())}")
