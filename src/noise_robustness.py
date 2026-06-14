import argparse
import pandas as pd
import numpy as np
import os
from src.common import load_config, ensure_dir, set_random_seed

def inject_noise(df, noise_ratio):
    """Flips the 'correct' label for a given percentage of the dataset."""
    df_noisy = df.copy()
    num_to_flip = int(len(df_noisy) * noise_ratio)
    if num_to_flip > 0:
        indices_to_flip = np.random.choice(df_noisy.index, num_to_flip, replace=False)
        df_noisy.loc[indices_to_flip, 'correct'] = 1 - df_noisy.loc[indices_to_flip, 'correct']
    return df_noisy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise_ratio", type=float, default=0.10)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    dataset = cfg['dataset']['name']
    set_random_seed(args.seed)
    
    data_dir = f"{cfg['dataset']['processed_dir']}/fold_{args.fold}"
    out_dir = f"{data_dir}/noisy_{args.noise_ratio}"
    ensure_dir(out_dir)
    
    print(f"Injecting {args.noise_ratio*100}% noise into {dataset}...")
    
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    train_noisy = inject_noise(train_df, args.noise_ratio)
    train_noisy.to_csv(f"{out_dir}/train.csv", index=False)
    
    # Copy valid/test as is
    pd.read_csv(f"{data_dir}/test.csv").to_csv(f"{out_dir}/test.csv", index=False)
    try:
        pd.read_csv(f"{data_dir}/valid.csv").to_csv(f"{out_dir}/valid.csv", index=False)
    except:
        pass
        
    print(f"Noisy dataset created at {out_dir}")
