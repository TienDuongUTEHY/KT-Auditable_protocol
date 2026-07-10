"""
PROCESS SIGNIFICANCE:
Audits Q-matrix provenance to ensure scientific validity.
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
    
    in_dir = f"data/processed/{dataset}"
    out_dir_rep = f"results/reports/{dataset}/fold_{args.fold}"
    out_dir_tab = f"results/tables/{dataset}/fold_{args.fold}"
    ensure_dir(out_dir_rep)
    ensure_dir(out_dir_tab)
    
    df_qm = pd.read_csv(f"{in_dir}/q_matrix.csv")
    source = df_qm['source'].iloc[0] if len(df_qm) > 0 else "unknown"
    
    is_valid = source in ['expert_static', 'provided_static', 'train_only_derived', 'synthetic']
    status = "PASS" if is_valid else "FAIL"
    
    report = f"# Q-Matrix Provenance\\nDataset: {dataset}\\nSource: {source}\\nStatus: {status}\\n"
    with open(f"{out_dir_rep}/qmatrix_provenance.md", "w") as f: f.write(report)
    
    audit_df = pd.DataFrame([{"dataset": dataset, "source": source, "status": status}])
    audit_df.to_csv(f"{out_dir_tab}/qmatrix_audit.csv", index=False)
    
    print(f"QMatrix provenance for {dataset} completed. Status: {status}")