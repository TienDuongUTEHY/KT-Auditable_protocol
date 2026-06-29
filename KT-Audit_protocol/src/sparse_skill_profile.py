"""
Ý NGHĨA TIẾN TRÌNH:
Phân tầng và đánh giá hồ sơ cho các kỹ năng thưa thớt (sparse skills).
"""

import argparse
import pandas as pd
import numpy as np
from src.common import load_config, ensure_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    dataset = cfg['dataset']['name']

    rep_dir = f"results/reports/{dataset}/fold_{args.fold}"
    tab_dir = f"results/tables/{dataset}/fold_{args.fold}"
    ensure_dir(rep_dir)
    ensure_dir(tab_dir)

    # Load train split
    train_path = f"data/processed/{dataset}/fold_{args.fold}/train.csv"
    df_train = pd.read_csv(train_path)

    # Per-skill interaction counts
    skill_counts = df_train.groupby('skill_id').size().rename('n_interactions')
    skill_learners = df_train.groupby('skill_id')['learner_id'].nunique().rename('n_learners')
    skill_correct = df_train.groupby('skill_id')['correct'].mean().rename('avg_correct')
    skill_questions = df_train.groupby('skill_id')['question_id'].nunique().rename('n_questions')

    df_profile = pd.concat([skill_counts, skill_learners, skill_correct, skill_questions], axis=1).reset_index()

    # Sparsity strata based on interaction count percentiles
    p33 = df_profile['n_interactions'].quantile(0.33)
    p66 = df_profile['n_interactions'].quantile(0.66)

    def strata_label(n):
        if n <= p33:
            return 'sparse'
        elif n <= p66:
            return 'medium'
        else:
            return 'dense'

    df_profile['strata'] = df_profile['n_interactions'].apply(strata_label)

    strata_counts = df_profile['strata'].value_counts().reset_index()
    strata_counts.columns = ['strata', 'num_skills']

    df_profile.to_csv(f"{tab_dir}/sparse_skill_profile.csv", index=False)
    strata_counts.to_csv(f"{tab_dir}/sparse_skill_strata_summary.csv", index=False)

    # Summary stats
    summary = {
        "dataset": dataset, "fold_id": args.fold,
        "total_skills": len(df_profile),
        "sparse_skills": int((df_profile['strata'] == 'sparse').sum()),
        "medium_skills": int((df_profile['strata'] == 'medium').sum()),
        "dense_skills": int((df_profile['strata'] == 'dense').sum()),
        "p33_threshold": round(p33, 1),
        "p66_threshold": round(p66, 1),
        "avg_interactions_per_skill": round(df_profile['n_interactions'].mean(), 2),
        "min_interactions": int(df_profile['n_interactions'].min()),
        "max_interactions": int(df_profile['n_interactions'].max()),
    }
    pd.DataFrame([summary]).to_csv(f"{tab_dir}/sparse_skill_summary.csv", index=False)

    md = f"# Sparse-Skill Profile — {dataset} fold {args.fold}\n\n"
    md += f"- Total skills: {summary['total_skills']}\n"
    md += f"- Sparse (≤p33={summary['p33_threshold']}): {summary['sparse_skills']}\n"
    md += f"- Medium: {summary['medium_skills']}\n"
    md += f"- Dense (>p66={summary['p66_threshold']}): {summary['dense_skills']}\n\n"
    md += df_profile.to_markdown(index=False)
    with open(f"{rep_dir}/sparse_skill_profile.md", "w", encoding='utf-8') as f:
        f.write(md)

    print(f"Sparse-skill profile for {dataset}: sparse={summary['sparse_skills']}, medium={summary['medium_skills']}, dense={summary['dense_skills']}")
