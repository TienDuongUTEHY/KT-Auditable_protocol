import os
import sys
import pandas as pd
import numpy as np

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def main():
    output_dir = "results_p0_revision"
    tables_csv_dir = ensure_dir(os.path.join(output_dir, "tables_csv"))
    tables_tex_dir = ensure_dir(os.path.join(output_dir, "tables_tex"))
    log_dir = ensure_dir(os.path.join(output_dir, "logs"))
    
    log_file = os.path.join(log_dir, "phase1_dataset_graph_audit.log")
    
    datasets = ["assist2012", "junyi", "kdd2010"]
    folds = [0, 1, 2]
    
    stats_rows = []
    
    with open(log_file, "w", encoding="utf-8") as lf:
        lf.write("PHASE_START dataset_statistics\n")
        print("PHASE_START dataset_statistics")
        
        for ds in datasets:
            for fold in folds:
                data_dir = f"data/processed/{ds}/fold_{fold}"
                train_path = f"{data_dir}/train.csv"
                valid_path = f"{data_dir}/valid.csv"
                test_path = f"{data_dir}/test.csv"
                
                if not os.path.exists(train_path):
                    continue
                
                df_train = pd.read_csv(train_path)
                df_valid = pd.read_csv(valid_path) if os.path.exists(valid_path) else pd.DataFrame()
                df_test = pd.read_csv(test_path) if os.path.exists(test_path) else pd.DataFrame()
                
                # Combine to get total statistics
                df_all = pd.concat([df_train, df_valid, df_test], ignore_index=True)
                
                n_users_total = df_all['learner_id'].nunique()
                n_users_train = df_train['learner_id'].nunique()
                n_users_valid = df_valid['learner_id'].nunique() if not df_valid.empty else 0
                n_users_test = df_test['learner_id'].nunique() if not df_test.empty else 0
                
                n_questions_total = df_all['question_id'].nunique()
                n_questions_train = df_train['question_id'].nunique()
                n_questions_valid = df_valid['question_id'].nunique() if not df_valid.empty else 0
                n_questions_test = df_test['question_id'].nunique() if not df_test.empty else 0
                
                n_skills_total = df_all['skill_id'].nunique()
                n_skills_train = df_train['skill_id'].nunique()
                n_skills_valid = df_valid['skill_id'].nunique() if not df_valid.empty else 0
                n_skills_test = df_test['skill_id'].nunique() if not df_test.empty else 0
                
                n_interactions_total = len(df_all)
                n_interactions_train = len(df_train)
                n_interactions_valid = len(df_valid)
                n_interactions_test = len(df_test)
                
                # Densities & metrics on train split
                user_skill_density_train = n_interactions_train / (n_users_train * n_skills_train) if (n_users_train * n_skills_train) > 0 else 0
                
                user_counts = df_train['learner_id'].value_counts()
                mean_interactions_per_user_train = user_counts.mean()
                median_interactions_per_user_train = user_counts.median()
                
                skill_counts = df_train['skill_id'].value_counts()
                mean_interactions_per_skill_train = skill_counts.mean()
                median_interactions_per_skill_train = skill_counts.median()
                
                # Sparse bins
                sparse_50_count = sum(skill_counts <= 50)
                sparse_100_count = sum(skill_counts <= 100)
                sparse_200_count = sum(skill_counts <= 200)
                frequent_500_count = sum(skill_counts > 500)
                
                # Suspected subsampling check
                suspected_subsampling = "no"
                subsampling_reason = "not_applicable"
                if ds == "assist2012" and n_interactions_total < 2500000:
                    suspected_subsampling = "yes"
                    subsampling_reason = "compute_budget"
                elif ds == "junyi" and n_interactions_total < 15000000:
                    suspected_subsampling = "yes"
                    subsampling_reason = "compute_budget"
                
                stats_rows.append({
                    "dataset": ds,
                    "fold": fold,
                    "split_type": "user_stratified",
                    "n_users_total": n_users_total,
                    "n_users_train": n_users_train,
                    "n_users_valid": n_users_valid,
                    "n_users_test": n_users_test,
                    "n_questions_total": n_questions_total,
                    "n_questions_train": n_questions_train,
                    "n_questions_valid": n_questions_valid,
                    "n_questions_test": n_questions_test,
                    "n_skills_total": n_skills_total,
                    "n_skills_train": n_skills_train,
                    "n_skills_valid": n_skills_valid,
                    "n_skills_test": n_skills_test,
                    "n_interactions_total": n_interactions_total,
                    "n_interactions_train": n_interactions_train,
                    "n_interactions_valid": n_interactions_valid,
                    "n_interactions_test": n_interactions_test,
                    "user_skill_density_train": round(user_skill_density_train, 6),
                    "mean_interactions_per_user_train": round(mean_interactions_per_user_train, 2),
                    "median_interactions_per_user_train": int(median_interactions_per_user_train),
                    "mean_interactions_per_skill_train": round(mean_interactions_per_skill_train, 2),
                    "median_interactions_per_skill_train": int(median_interactions_per_skill_train),
                    "sparse_50_count": sparse_50_count,
                    "sparse_100_count": sparse_100_count,
                    "sparse_200_count": sparse_200_count,
                    "frequent_500_count": frequent_500_count,
                    "suspected_subsampling": suspected_subsampling,
                    "subsampling_reason": subsampling_reason
                })
                
                log_msg = f"DATASET_STATS dataset={ds} fold={fold} users_total={n_users_total} interactions_total={n_interactions_total} suspected_subsampling={suspected_subsampling}"
                lf.write(log_msg + "\n")
                print(log_msg)
                
        # Save CSV
        df_stats = pd.DataFrame(stats_rows)
        csv_path = os.path.join(tables_csv_dir, "table_dataset_statistics.csv")
        df_stats.to_csv(csv_path, index=False)
        
        # Save LaTeX (only selected key columns for brevity and beauty)
        tex_path = os.path.join(tables_tex_dir, "table_dataset_statistics.tex")
        
        # We format a clean latex table body
        latex_lines = [
            "% Standalone Table Body for Dataset Statistics",
            "\\begin{tabular}{llcccccccc}",
            "\\hline",
            "Dataset & Fold & Train Users & Test Users & Skills & Total Int. & Train Int. & Density & Sparse $\\le$ 100 & Subsampled \\\\",
            "\\hline"
        ]
        for _, r in df_stats.iterrows():
            latex_lines.append(
                f"{r['dataset'].upper()} & {r['fold']} & {r['n_users_train']:,} & {r['n_users_test']:,} & {r['n_skills_total']} & {r['n_interactions_total']:,} & {r['n_interactions_train']:,} & {r['user_skill_density_train']:.5f} & {r['sparse_100_count']} & {r['suspected_subsampling']} \\\\"
            )
        latex_lines.append("\\hline")
        latex_lines.append("\\end{tabular}")
        
        with open(tex_path, "w") as tf:
            tf.write("\n".join(latex_lines))
            
        lf.write(f"DATASET_STATS_OUTPUT={csv_path}\n")
        lf.write("PHASE_PASS dataset_statistics\n")
        print("PHASE_PASS dataset_statistics")

if __name__ == "__main__":
    main()
