import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def main():
    output_dir = "results_p0_revision"
    tables_csv_dir = ensure_dir(os.path.join(output_dir, "tables_csv"))
    tables_tex_dir = ensure_dir(os.path.join(output_dir, "tables_tex"))
    log_dir = ensure_dir(os.path.join(output_dir, "logs"))
    
    log_file = os.path.join(log_dir, "phase1_esim_trace.log")
    
    datasets = ["assist2012", "junyi", "kdd2010"]
    folds = [0, 1, 2]
    
    trace_rows = []
    empty_datasets_count = 0
    
    with open(log_file, "w", encoding="utf-8") as lf:
        lf.write("PHASE_START esim_trace\n")
        print("PHASE_START esim_trace")
        
        for ds in datasets:
            # We want to check if it's empty across folds
            is_ds_empty = True
            
            for fold in folds:
                train_path = f"data/processed/{ds}/fold_{fold}/train.csv"
                
                if not os.path.exists(train_path):
                    continue
                
                df_train = pd.read_csv(train_path)
                
                # Trace logic
                # 1. n_skills
                skills = df_train['skill_id'].unique()
                n_skills = len(skills)
                
                # 2. Check if questions map to multiple skills
                # Group by question and check number of unique skills
                q_skills = df_train.groupby('question_id')['skill_id'].nunique()
                multi_skill_qs = sum(q_skills > 1)
                
                # Calculate candidate pairs that share a question
                skill_q_map = defaultdict(set)
                q_to_skills = defaultdict(set)
                for sid, qid in zip(df_train['skill_id'], df_train['question_id']):
                    skill_q_map[sid].add(qid)
                    q_to_skills[qid].add(sid)
                    
                candidates = set()
                for q, sks in q_to_skills.items():
                    sks_list = list(sks)
                    for i in range(len(sks_list)):
                        for j in range(i + 1, len(sks_list)):
                            s1, s2 = sks_list[i], sks_list[j]
                            candidates.add((min(s1, s2), max(s1, s2)))
                            
                candidate_pairs = len(candidates)
                threshold_theta_sim = 0.10
                
                # Calculate Jaccard similarity for candidate pairs
                pairs_after_threshold = 0
                edges = []
                for s1, s2 in candidates:
                    q1, q2 = skill_q_map[s1], skill_q_map[s2]
                    intersection = len(q1.intersection(q2))
                    union = len(q1.union(q2))
                    jaccard = intersection / union if union > 0 else 0
                    if jaccard >= threshold_theta_sim:
                        pairs_after_threshold += 1
                        edges.append({
                            'src': s1,
                            'dst': s2,
                            'weight': jaccard
                        })
                
                top_k = 20
                # Top k filtering
                if edges:
                    df_edges = pd.DataFrame(edges)
                    # For each src, keep top-k dst
                    # Group by src and head(top_k) after sorting
                    df_edges = df_edges.sort_values(by='weight', ascending=False)
                    df_edges_topk = df_edges.groupby('src').head(top_k)
                    final_edges = len(df_edges_topk)
                else:
                    final_edges = 0
                    
                if final_edges > 0:
                    is_ds_empty = False
                    
                reason = "NA"
                if final_edges == 0:
                    if multi_skill_qs == 0:
                        reason = "Single-skill Q-matrix (questions map to 1 skill only, Jaccard similarity is 0)"
                    else:
                        reason = "All similarity scores below threshold 0.10"
                        
                trace_rows.append({
                    "dataset": ds,
                    "fold": fold,
                    "n_skills": n_skills,
                    "embedding_file_found": "no",  # In this repo, similarity is graph Jaccard-based, not embedding-based
                    "similarity_matrix_shape": f"{n_skills}x{n_skills}",
                    "candidate_pairs_before_threshold": candidate_pairs,
                    "threshold_theta_sim": threshold_theta_sim,
                    "pairs_after_threshold": pairs_after_threshold,
                    "top_k": top_k,
                    "pairs_after_top_k": final_edges,
                    "final_E_sim_edges": final_edges,
                    "reason_if_zero": reason
                })
                
                log_msg = f"ESIM_TRACE dataset={ds} fold={fold} before_threshold={candidate_pairs} after_threshold={pairs_after_threshold} after_topk={final_edges} final_edges={final_edges} reason_if_zero=\"{reason}\""
                lf.write(log_msg + "\n")
                print(log_msg)
                
            if is_ds_empty:
                empty_datasets_count += 1
                
        # Make ESIM decision
        # E_sim_active: active on at least 2/3 datasets.
        # Otherwise, E_sim_empty_effective (meaning empty on >= 2 datasets).
        esim_decision = "E_sim_empty_effective" if empty_datasets_count >= 2 else "E_sim_active"
        
        lf.write(f"ESIM_DECISION={esim_decision}\n")
        print(f"ESIM_DECISION={esim_decision}")
        
        # Save CSV
        df_trace = pd.DataFrame(trace_rows)
        csv_path = os.path.join(tables_csv_dir, "table_esim_trace.csv")
        df_trace.to_csv(csv_path, index=False)
        
        # Save LaTeX table
        tex_path = os.path.join(tables_tex_dir, "table_esim_trace.tex")
        latex_lines = [
            "% Standalone Table Body for E_sim Pipeline Trace",
            "\\begin{tabular}{llcccccl}",
            "\\hline",
            "Dataset & Fold & Skills & Candidate Pairs & Theta & Pairs Post-Thresh & Final Edges & Reason if Zero \\\\",
            "\\hline"
        ]
        for _, r in df_trace.iterrows():
            reason_clean = r['reason_if_zero']
            if reason_clean == "NA":
                reason_clean = "-"
            elif "Single-skill Q-matrix" in reason_clean:
                reason_clean = "Single-skill Q-matrix"
            latex_lines.append(
                f"{r['dataset'].upper()} & {r['fold']} & {r['n_skills']} & {r['candidate_pairs_before_threshold']:,} & {r['threshold_theta_sim']} & {r['pairs_after_threshold']} & {r['final_E_sim_edges']} & {reason_clean} \\\\"
            )
        latex_lines.append("\\hline")
        latex_lines.append("\\end{tabular}")
        
        with open(tex_path, "w") as tf:
            tf.write("\n".join(latex_lines))
            
        lf.write("PHASE_PASS esim_trace\n")
        print("PHASE_PASS esim_trace")

if __name__ == "__main__":
    main()
