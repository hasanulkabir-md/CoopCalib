"""
inspect_pkl.py — Print the structure of the first few scenes in a pkl file.
Run this once to understand the data layout, then we fix compute_mnd.py.

Usage:
    python scripts\inspect_pkl.py
"""

import pickle
import numpy as np
from pathlib import Path

PKL_PATH = Path(r"C:\CoopCalib\TUTR\dataset\eth_train.pkl")

with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

print(f"Top-level type : {type(data)}")

if isinstance(data, (list, tuple)):
    print(f"Number of items: {len(data)}")
    for i, item in enumerate(data[:3]):
        print(f"\n--- item[{i}] ---")
        print(f"  type : {type(item)}")
        if isinstance(item, dict):
            for k, v in item.items():
                if isinstance(v, np.ndarray):
                    print(f"  key={k!r:20s}  ndarray shape={v.shape}  dtype={v.dtype}")
                elif isinstance(v, (list, tuple)):
                    print(f"  key={k!r:20s}  {type(v).__name__} len={len(v)}", end="")
                    if len(v) > 0:
                        el = v[0]
                        if isinstance(el, np.ndarray):
                            print(f"  → ndarray shape={el.shape}", end="")
                        elif isinstance(el, (list, tuple)) and len(el) > 0:
                            el2 = el[0]
                            if isinstance(el2, np.ndarray):
                                print(f"  → list of ndarray shape={el2.shape}", end="")
                    print()
                else:
                    print(f"  key={k!r:20s}  {type(v).__name__}  val={v!r:.80}")
        elif isinstance(item, np.ndarray):
            print(f"  ndarray shape={item.shape}  dtype={item.dtype}")
        elif isinstance(item, (list, tuple)):
            print(f"  {type(item).__name__} len={len(item)}")
            for j, sub in enumerate(item[:3]):
                if isinstance(sub, np.ndarray):
                    print(f"    [{j}] ndarray shape={sub.shape}  dtype={sub.dtype}")
                else:
                    print(f"    [{j}] {type(sub).__name__}")
elif isinstance(data, dict):
    print("Top-level dict keys:", list(data.keys())[:10])
elif isinstance(data, np.ndarray):
    print(f"ndarray shape={data.shape}  dtype={data.dtype}")
