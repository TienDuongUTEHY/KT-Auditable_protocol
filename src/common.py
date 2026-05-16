"""
Ý NGHĨA TIẾN TRÌNH:
Chứa các hàm tiện ích dùng chung như load YAML, băm file, và sinh thư mục.
"""

import os
import yaml
import hashlib
import pandas as pd
import random
from datetime import datetime

def load_yaml(path):
    with open(path, 'r') as f: return yaml.safe_load(f)

def load_config(path):
    global_cfg = load_yaml("configs/global.yaml")
    cfg = load_yaml(path)
    for k, v in cfg.items():
        if k in global_cfg and isinstance(global_cfg[k], dict) and isinstance(v, dict):
            global_cfg[k].update(v)
        else:
            global_cfg[k] = v
    return global_cfg

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
def now_iso():
    return datetime.now().isoformat()
def stable_hash_file(path):
    return "mock_file_hash"
def stable_hash_dataframe(df):
    return "mock_df_hash"
def set_random_seed(seed):
    random.seed(seed)
def write_markdown(path, text):
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f: f.write(text)
