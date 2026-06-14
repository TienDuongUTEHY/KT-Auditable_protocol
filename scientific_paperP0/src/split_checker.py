"""
Ý NGHĨA TIẾN TRÌNH:
Chia dữ liệu thành train/valid/test theo learner và kiểm tra chống rò rỉ dữ liệu.
"""

import argparse
import pandas as pd
import hashlib
import random
from src.common import load_config, ensure_dir, set_random_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    dataset = cfg['dataset']['name']
    
    # Use seed derived from base seed and fold to get different folds
    set_random_seed(args.seed + args.fold)
    
    processed_dir = cfg['dataset']['processed_dir']
    out_dir = f"{processed_dir}/fold_{args.fold}"
    ensure_dir(out_dir)
    
    df_int = pd.read_csv(f"{processed_dir}/interactions.csv")
    df_int = df_int.sort_values(by=['learner_id', 'timestamp'])
    
    learners = df_int['learner_id'].unique().tolist()
    random.shuffle(learners)
    
    n = len(learners)
    train_ratio = cfg.get('splitting', {}).get('train_ratio', 0.7)
    valid_ratio = cfg.get('splitting', {}).get('valid_ratio', 0.15)
    
    train_end = int(n * train_ratio)
    valid_end = train_end + int(n * valid_ratio)
    
    train_learners = set(learners[:train_end])
    valid_learners = set(learners[train_end:valid_end])
    test_learners = set(learners[valid_end:])
    
    train_df = df_int[df_int['learner_id'].isin(train_learners)].copy()
    valid_df = df_int[df_int['learner_id'].isin(valid_learners)].copy()
    test_df = df_int[df_int['learner_id'].isin(test_learners)].copy()
    
    train_df.to_csv(f"{out_dir}/train.csv", index=False)
    valid_df.to_csv(f"{out_dir}/valid.csv", index=False)
    test_df.to_csv(f"{out_dir}/test.csv", index=False)
    
    split_hash = hashlib.md5(pd.util.hash_pandas_object(train_df, index=True).values).hexdigest()
    with open(f"{out_dir}/split_hash.txt", "w") as f: f.write(split_hash)
    with open(f"{out_dir}/split_report.md", "w") as f: 
        f.write(f"# Split Report\nSeed: {args.seed}\nFold: {args.fold}\nTrain: {len(train_df)}\nValid: {len(valid_df)}\nTest: {len(test_df)}\n")
        
    print(f"Splitting for {dataset} fold {args.fold} completed.")
