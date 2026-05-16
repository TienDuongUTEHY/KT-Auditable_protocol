"""
Ý NGHĨA TIẾN TRÌNH:
Tiền xử lý (Preprocess) dữ liệu thô thành chuẩn bảng interactions thống nhất.
"""

import argparse
from src.common import load_config, ensure_dir
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    dataset = cfg['dataset']['name']
    
    raw_dir = cfg['dataset']['raw_dir']
    out_dir = cfg['dataset']['processed_dir']
    ensure_dir(out_dir)
    
    try:
        df_int = pd.read_csv(f"{raw_dir}/interactions.csv")
        df_qm = pd.read_csv(f"{raw_dir}/q_matrix.csv")
    except Exception as e:
        print(f"Error loading {dataset} raw files: {e}")
        exit(1)
        
    df_int['interaction_id'] = df_int['interaction_id'].astype(str)
    df_int['learner_id'] = df_int['learner_id'].astype(str)
    df_int['question_id'] = df_int['question_id'].astype(str)
    df_int['skill_id'] = df_int['skill_id'].astype(str)
    df_int['correct'] = df_int['correct'].astype(int)
    
    df_qm['question_id'] = df_qm['question_id'].astype(str)
    df_qm['skill_id'] = df_qm['skill_id'].astype(str)
    
    df_int.to_csv(f"{out_dir}/interactions.csv", index=False)
    df_qm.to_csv(f"{out_dir}/q_matrix.csv", index=False)
    
    with open(f"{out_dir}/preprocess_report.md", "w") as f: 
        f.write("# Preprocess Report\\nData successfully normalized.")
        
    stats_dir = f"results/tables/{dataset}"
    ensure_dir(stats_dir)
    stats = {
        "Dataset": dataset,
        "Learners": df_int['learner_id'].nunique(),
        "Questions": df_int['question_id'].nunique(),
        "Skills": df_int['skill_id'].nunique(),
        "Interactions": len(df_int),
        "AvgSeqLen": len(df_int) / df_int['learner_id'].nunique() if df_int['learner_id'].nunique() > 0 else 0,
        "QMatrixSource": df_qm['source'].iloc[0] if len(df_qm) > 0 else "unknown"
    }
    pd.DataFrame([stats]).to_csv(f"{stats_dir}/dataset_stats.csv", index=False)
    print(f"Preprocess for {dataset} completed.")
