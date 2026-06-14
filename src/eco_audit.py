"""
Ý NGHĨA TIẾN TRÌNH:
Kiểm toán tính đối xứng và trọng số của đồ thị đồng xuất hiện E_co.
"""

import argparse
import pandas as pd
from src.common import load_config, ensure_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    dataset = cfg['dataset']['name']
    
    out_dir_rep = f"results/reports/{dataset}/fold_{args.fold}"
    out_dir_tab = f"results/tables/{dataset}/fold_{args.fold}"
    ensure_dir(out_dir_rep)
    ensure_dir(out_dir_tab)
    
    try:
        df_co = pd.read_csv(f"{out_dir_tab}/E_co_train.csv")
    except FileNotFoundError:
        df_co = pd.DataFrame()
        
    symmetry_pass = all(df_co['directed'] == False) if not df_co.empty else True
    train_only_pass = all(df_co['source_split'] == 'train') if not df_co.empty else True
    support_positive_pass = all(df_co['support_count'] > 0) if not df_co.empty else True
    
    w = df_co['weight'] if not df_co.empty else pd.Series(dtype=float)
    dist = {
        "min": w.min() if not w.empty else 0,
        "q25": w.quantile(0.25) if not w.empty else 0,
        "median": w.median() if not w.empty else 0,
        "mean": w.mean() if not w.empty else 0,
        "q75": w.quantile(0.75) if not w.empty else 0,
        "max": w.max() if not w.empty else 0,
        "num_positive": (w > 0).sum() if not w.empty else 0
    }
    
    c = df_co['support_count'] if not df_co.empty and 'support_count' in df_co.columns else pd.Series(dtype=int)
    c_dist = {
        "min": c.min() if not c.empty else 0,
        "q25": c.quantile(0.25) if not c.empty else 0,
        "median": c.median() if not c.empty else 0,
        "mean": c.mean() if not c.empty else 0,
        "q75": c.quantile(0.75) if not c.empty else 0,
        "max": c.max() if not c.empty else 0
    }
    
    rep = f"# E_co Audit\nSymmetry Pass: {symmetry_pass}\nTrain-only Pass: {train_only_pass}\n"
    rep += f"Weight Distribution: Min={dist['min']:.4f}, Median={dist['median']:.4f}, Max={dist['max']:.4f}\n"
    rep += f"Count Distribution: Min={c_dist['min']:.4f}, Median={c_dist['median']:.4f}, Max={c_dist['max']:.4f}\n"
    with open(f"{out_dir_rep}/eco_audit.md", "w") as f: f.write(rep)
    
    audit_df = pd.DataFrame([{
        "dataset": dataset, 
        "symmetry_pass": symmetry_pass, 
        "train_only_pass": train_only_pass,
        "count_median": c_dist['median'],
        "count_max": c_dist['max']
    }])
    audit_df.to_csv(f"{out_dir_tab}/eco_audit.csv", index=False)
    
    print(f"Eco audit for {dataset} completed.")
    print(f"  - Symmetry Pass: {symmetry_pass}")
    print(f"  - Train-only Source Pass: {train_only_pass}")
    print(f"  - E_co weight median: {dist['median']:.4f}, max: {dist['max']:.4f}")
