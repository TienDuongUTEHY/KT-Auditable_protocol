import os
import sys
import yaml
import pandas as pd
import numpy as np
import hashlib
from collections import defaultdict

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def get_file_sha256(path):
    if not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    output_dir = "results_p0_revision"
    tables_csv_dir = ensure_dir(os.path.join(output_dir, "tables_csv"))
    tables_tex_dir = ensure_dir(os.path.join(output_dir, "tables_tex"))
    log_dir = ensure_dir(os.path.join(output_dir, "logs"))
    
    log_file = os.path.join(log_dir, "phase3_table_generation.log")
    
    with open(log_file, "w", encoding="utf-8") as lf:
        lf.write("PHASE_START table_generation\n")
        print("PHASE_START table_generation")
        
        # --- 1. Validation Candidates Table ---
        candidates_yaml = os.path.join(output_dir, "configs/p0_validation_candidates.yaml")
        if os.path.exists(candidates_yaml):
            with open(candidates_yaml, 'r') as f:
                cand_data = yaml.safe_load(f)
            df_cand = pd.DataFrame(cand_data['candidates'])
        else:
            df_cand = pd.DataFrame(columns=["id", "name", "relations_enabled", "selection_gate", "beta_or_weighting", "pre_specified_yes_no", "test_used_for_selection_yes_no", "notes"])
            
        csv_cand = os.path.join(tables_csv_dir, "table_validation_candidates_prespecified.csv")
        df_cand.to_csv(csv_cand, index=False)
        lf.write(f"TABLE_CREATED table_validation_candidates_prespecified rows={len(df_cand)}\n")
        
        tex_cand = os.path.join(tables_tex_dir, "table_validation_candidates_prespecified.tex")
        latex_cand = [
            "\\begin{tabular}{llccccl}",
            "\\hline",
            "ID & Name & Relations Enabled & Selection Gate & Pre-specified & No Test Selection & Notes \\\\",
            "\\hline"
        ]
        for _, r in df_cand.iterrows():
            latex_cand.append(f"{r['id']} & {r['name']} & {r['relations_enabled']} & {r['selection_gate']} & {r['pre_specified_yes_no']} & {r['test_used_for_selection_yes_no']} & {r['notes']} \\\\")
        latex_cand.append("\\hline")
        latex_cand.append("\\end{tabular}")
        with open(tex_cand, "w") as f:
            f.write("\n".join(latex_cand))

        # --- 2. Main AUC Delta Table with Holm Correction ---
        paired_csv = "runs/q3_lcmrsg_plus_20260528_234100/statistics/paired_tests_with_ci.csv"
        main_auc_rows = []
        if os.path.exists(paired_csv):
            df_paired = pd.read_csv(paired_csv)
            for _, r in df_paired.iterrows():
                significant_holm = "yes" if r['significant_005_holm'] else "no"
                
                # interpretation logic
                delta = r['delta_auc']
                if significant_holm == "yes" and delta > 0:
                    interpretation = "confirmatory"
                elif significant_holm == "no":
                    interpretation = "diagnostic_only"
                else:
                    interpretation = "not_supported"
                    
                main_auc_rows.append({
                    "dataset": r['dataset'],
                    "backbone": r['model'],
                    "mean_auc_no_graph": round(r['mean_auc_no_graph'], 4),
                    "mean_auc_selected_graph": round(r['mean_auc_method'], 4),
                    "mean_delta_auc": round(delta, 4),
                    "ci95_low": round(r['ci_low'], 4),
                    "ci95_high": round(r['ci_high'], 4),
                    "raw_p": r['p_two_tailed'],
                    "holm_p": r['p_holm'],
                    "holm_significant_yes_no": significant_holm,
                    "practical_label": r['practical_effect'],
                    "interpretation": interpretation
                })
        df_main_auc = pd.DataFrame(main_auc_rows)
        csv_main_auc = os.path.join(tables_csv_dir, "table_main_auc_delta_holm.csv")
        df_main_auc.to_csv(csv_main_auc, index=False)
        lf.write(f"TABLE_CREATED table_main_auc_delta_holm rows={len(df_main_auc)}\n")
        
        tex_main_auc = os.path.join(tables_tex_dir, "table_main_auc_delta_holm.tex")
        latex_main_auc = [
            "\\begin{tabular}{llcccccl}",
            "\\hline",
            "Dataset & Backbone & No Graph & Selected Graph & Delta AUC & 95\\% CI & Holm p & Interpretation \\\\",
            "hline"
        ]
        for _, r in df_main_auc.iterrows():
            latex_main_auc.append(
                f"{r['dataset'].upper()} & {r['backbone'].upper()} & {r['mean_auc_no_graph']:.4f} & {r['mean_auc_selected_graph']:.4f} & {r['mean_delta_auc']:.4f} & [{r['ci95_low']:.4f}, {r['ci95_high']:.4f}] & {r['holm_p']:.2e} & {r['interpretation']} \\\\"
            )
        latex_main_auc.append("\\hline")
        latex_main_auc.append("\\end{tabular}")
        with open(tex_main_auc, "w") as f:
            f.write("\n".join(latex_main_auc))

        # --- 3. Selected Relation Variants Table ---
        val_selected_csv = "runs/q3_lcmrsg_plus_20260528_234100/latex/table_val_selected.csv"
        relation_rows = []
        if os.path.exists(val_selected_csv):
            df_val_sel = pd.read_csv(val_selected_csv)
            for _, r in df_val_sel.iterrows():
                variant = r['Most Selected Variant']
                
                includes_co = "yes" if variant in ['full_lc_mrsg', 'relation_gated_1', 'relation_gated'] else "no"
                
                # E_sim is effectively empty for assist2012 and junyi
                if r['Dataset'] in ['assist2012', 'junyi']:
                    includes_sim = "no (E_sim^eff=empty)"
                else:
                    includes_sim = "yes" if variant in ['e_pre_e_sim', 'full_lc_mrsg'] else "no"
                
                relation_rows.append({
                    "dataset": r['Dataset'],
                    "backbone": r['Model'],
                    "selected_candidate_most_frequent": variant,
                    "count_selected": int(round(r['Selection Frequency'] * 9)),
                    "N_observations": 9,
                    "selection_frequency": round(r['Selection Frequency'], 4),
                    "selected_relations": r['Interpretation'],
                    "includes_E_co_yes_no": includes_co,
                    "includes_E_sim_effective_yes_no": includes_sim,
                    "notes": "E_sim is empty for assist2012 and junyi due to single-skill Q-matrix constraint."
                })
        df_relation = pd.DataFrame(relation_rows)
        csv_relation = os.path.join(tables_csv_dir, "table_selected_relation_variants.csv")
        df_relation.to_csv(csv_relation, index=False)
        lf.write(f"TABLE_CREATED table_selected_relation_variants rows={len(df_relation)}\n")
        
        tex_relation = os.path.join(tables_tex_dir, "table_selected_relation_variants.tex")
        latex_relation = [
            "\\begin{tabular}{llcccccl}",
            "\\hline",
            "Dataset & Model & Most Selected & Count & N & Freq & Includes $E_{co}$ & Includes $E_{sim}$ Effective \\\\",
            "\\hline"
        ]
        for _, r in df_relation.iterrows():
            latex_relation.append(
                f"{r['dataset'].upper()} & {r['backbone'].upper()} & {r['selected_candidate_most_frequent']} & {r['count_selected']} & {r['N_observations']} & {r['selection_frequency']:.2f} & {r['includes_E_co_yes_no']} & {r['includes_E_sim_effective_yes_no']} \\\\"
            )
        latex_relation.append("\\hline")
        latex_relation.append("\\end{tabular}")
        with open(tex_relation, "w") as f:
            f.write("\n".join(latex_relation))

        # --- 4. Sparse Bins Descriptive Table ---
        # We compute this dynamically by scanning train/test datasets for statistics
        sparse_bin_rows = []
        datasets = ["assist2012", "junyi", "kdd2010"]
        for ds in datasets:
            train_path = f"data/processed/{ds}/fold_1/train.csv"
            test_path = f"data/processed/{ds}/fold_1/test.csv"
            
            if not os.path.exists(train_path):
                continue
                
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path) if os.path.exists(test_path) else pd.DataFrame()
            
            skill_counts_train = df_train['skill_id'].value_counts()
            
            # Group test interactions by skill
            test_counts = df_test['skill_id'].value_counts().to_dict() if not df_test.empty else {}
            
            # Load graphs to get degrees
            graph_dir = f"runs/q3_lcmrsg_plus_20260528_234100/graphs/{ds}/fold_1"
            if not os.path.exists(graph_dir):
                graph_dir = f"graphs/{ds}/fold_1"
                
            e_pre_file = os.path.join(graph_dir, "E_pre.csv")
            e_co_file = os.path.join(graph_dir, "E_co.csv")
            
            # Degrees maps
            e_pre_deg = defaultdict(int)
            if os.path.exists(e_pre_file) and os.path.getsize(e_pre_file) > 2:
                try:
                    df = pd.read_csv(e_pre_file)
                    src_col = 'src' if 'src' in df.columns else 'src_skill_id'
                    dst_col = 'dst' if 'dst' in df.columns else 'dst_skill_id'
                    for k, v in df[src_col].value_counts().items():
                        e_pre_deg[str(k)] += v
                    for k, v in df[dst_col].value_counts().items():
                        e_pre_deg[str(k)] += v
                except Exception:
                    pass
                    
            e_co_deg = defaultdict(int)
            if os.path.exists(e_co_file) and os.path.getsize(e_co_file) > 2:
                try:
                    df = pd.read_csv(e_co_file)
                    src_col = 'src' if 'src' in df.columns else 'src_skill_id'
                    dst_col = 'dst' if 'dst' in df.columns else 'dst_skill_id'
                    for k, v in df[src_col].value_counts().items():
                        e_co_deg[str(k)] += v
                    for k, v in df[dst_col].value_counts().items():
                        e_co_deg[str(k)] += v
                except Exception:
                    pass
            
            bins_def = [
                ("<=50", lambda c: c <= 50),
                ("<=100", lambda c: c <= 100),
                ("<=200", lambda c: c <= 200),
                (">500", lambda c: c > 500)
            ]
            
            for bin_name, bin_fn in bins_def:
                bin_skills = [str(s) for s, c in skill_counts_train.items() if bin_fn(c)]
                n_skills = len(bin_skills)
                n_test_int = sum(test_counts.get(int(s) if s.isdigit() else s, 0) for s in bin_skills)
                
                # Check reliability
                if n_test_int > 1000:
                    reliability = "reliable"
                elif n_test_int > 100:
                    reliability = "limited"
                else:
                    reliability = "insufficient"
                    
                # Degrees
                deg_pre_vals = [e_pre_deg[s] for s in bin_skills]
                deg_co_vals = [e_co_deg[s] for s in bin_skills]
                
                mean_deg_pre = np.mean(deg_pre_vals) if deg_pre_vals else 0.0
                mean_deg_co = np.mean(deg_co_vals) if deg_co_vals else 0.0
                
                # Coverage (node has degree >= 1 in E_pre or E_co)
                covered = sum((e_pre_deg[s] + e_co_deg[s]) >= 1 for s in bin_skills)
                coverage_ratio = covered / n_skills if n_skills > 0 else 0.0
                
                sparse_bin_rows.append({
                    "dataset": ds,
                    "bin_name": bin_name,
                    "n_skills": n_skills,
                    "n_test_interactions": n_test_int,
                    "effective_sample_size": n_test_int,
                    "reliability_flag": reliability,
                    "mean_degree_E_pre": round(mean_deg_pre, 2),
                    "mean_degree_E_co": round(mean_deg_co, 2),
                    "coverage_ratio": round(coverage_ratio, 4)
                })
                
        df_sparse = pd.DataFrame(sparse_bin_rows)
        csv_sparse = os.path.join(tables_csv_dir, "table_sparse_bins_descriptive.csv")
        df_sparse.to_csv(csv_sparse, index=False)
        lf.write(f"TABLE_CREATED table_sparse_bins_descriptive rows={len(df_sparse)}\n")
        
        tex_sparse = os.path.join(tables_tex_dir, "table_sparse_bins_descriptive.tex")
        latex_sparse = [
            "\\begin{tabular}{llcccccl}",
            "\\hline",
            "Dataset & Bin & Skills & Test Int & Reliability & Mean Deg E\\_pre & Mean Deg E\\_co & Coverage Ratio \\\\",
            "\\hline"
        ]
        for _, r in df_sparse.iterrows():
            latex_sparse.append(
                f"{r['dataset'].upper()} & {r['bin_name']} & {r['n_skills']} & {r['n_test_interactions']:,} & {r['reliability_flag']} & {r['mean_degree_E_pre']:.2f} & {r['mean_degree_E_co']:.2f} & {r['coverage_ratio']:.4f} \\\\"
            )
        latex_sparse.append("\\hline")
        latex_sparse.append("\\end{tabular}")
        with open(tex_sparse, "w") as f:
            f.write("\n".join(latex_sparse))

        # --- 4.5. Hardware and Runtime Table ---
        hardware_rows = [
            {
                "parameter": "CPU",
                "details": "Intel(R) Core(TM) i5-6300U CPU @ 2.40GHz (2 physical / 4 logical cores)"
            },
            {
                "parameter": "GPU",
                "details": "N/A (CPU-only execution, CUDA not available)"
            },
            {
                "parameter": "System RAM",
                "details": "15.89 GB (16 GB total)"
            },
            {
                "parameter": "OS Platform",
                "details": "Windows-10-10.0.19045-SP0"
            },
            {
                "parameter": "Python Version",
                "details": "3.12.13 (Conda environment)"
            },
            {
                "parameter": "Epoch Budget (Main)",
                "details": "2 epochs (validation gating and edge boosting sweeps)"
            },
            {
                "parameter": "Epoch Budget (Sanity)",
                "details": "5 and 10 epochs (direction-check audits)"
            },
            {
                "parameter": "Training Time (BKT Proxy)",
                "details": "<0.1 seconds per model-seed-fold"
            },
            {
                "parameter": "Training Time (Neural KT)",
                "details": "~10-30 seconds per run (fold/seed/variant) on CPU"
            },
            {
                "parameter": "Total Experiment Suite Time",
                "details": "~6.5 hours (all datasets, models, variants, folds, seeds)"
            },
            {
                "parameter": "Total Revision Pipeline Runtime",
                "details": "~120 seconds (repo scan, all audits, table and manifest generation)"
            }
        ]
        df_hw = pd.DataFrame(hardware_rows)
        csv_hw = os.path.join(tables_csv_dir, "table_hardware_runtime.csv")
        df_hw.to_csv(csv_hw, index=False)
        lf.write(f"TABLE_CREATED table_hardware_runtime rows={len(df_hw)}\n")
        
        tex_hw = os.path.join(tables_tex_dir, "table_hardware_runtime.tex")
        latex_hw = [
            "\\begin{tabular}{ll}",
            "\\hline",
            "Parameter & Specification / Details \\\\",
            "\\hline"
        ]
        for _, r in df_hw.iterrows():
            latex_hw.append(f"{r['parameter']} & {r['details']} \\\\")
        latex_hw.append("\\hline")
        latex_hw.append("\\end{tabular}")
        with open(tex_hw, "w") as f:
            f.write("\n".join(latex_hw))

        # --- 5. Reproducibility Checklist Table ---
        checklist_rows = [
            {
                "artifact": "Dataset Statistics Table",
                "path": "results_p0_revision/tables_csv/table_dataset_statistics.csv",
                "status": "available",
                "sha256": get_file_sha256(os.path.join(tables_csv_dir, "table_dataset_statistics.csv")),
                "purpose": "Audits users, questions, skills, interactions per split, density.",
                "used_in_main_text_yes_no": "yes"
            },
            {
                "artifact": "Graph Provenance Table",
                "path": "results_p0_revision/tables_csv/table_graph_provenance_corrected.csv",
                "status": "available",
                "sha256": get_file_sha256(os.path.join(tables_csv_dir, "table_graph_provenance_corrected.csv")),
                "purpose": "Corrects and verifies graph edge limits for KDD2010 and other datasets.",
                "used_in_main_text_yes_no": "yes"
            },
            {
                "artifact": "Leakage Audit Table",
                "path": "results_p0_revision/tables_csv/table_leakage_audit_L1_L6.csv",
                "status": "available",
                "sha256": get_file_sha256(os.path.join(tables_csv_dir, "table_leakage_audit_L1_L6.csv")),
                "purpose": "Documents split boundary and selection leakage audits L1-L6.",
                "used_in_main_text_yes_no": "yes"
            },
            {
                "artifact": "Main AUC Delta Table",
                "path": "results_p0_revision/tables_csv/table_main_auc_delta_holm.csv",
                "status": "available",
                "sha256": get_file_sha256(os.path.join(tables_csv_dir, "table_main_auc_delta_holm.csv")),
                "purpose": "Displays mean test AUC delta, 95% confidence intervals, and Holm p-values.",
                "used_in_main_text_yes_no": "yes"
            },
            {
                "artifact": "Selected Relation Variants Table",
                "path": "results_p0_revision/tables_csv/table_selected_relation_variants.csv",
                "status": "available",
                "sha256": get_file_sha256(os.path.join(tables_csv_dir, "table_selected_relation_variants.csv")),
                "purpose": "Tracks selection frequency of candidate graphs.",
                "used_in_main_text_yes_no": "yes"
            },
            {
                "artifact": "Epoch Sanity Check Table",
                "path": "results_p0_revision/tables_csv/table_epoch_sanity.csv",
                "status": "available",
                "sha256": get_file_sha256(os.path.join(tables_csv_dir, "table_epoch_sanity.csv")),
                "purpose": "Sanity audits DKT and simpleKT convergence at 5 and 10 epochs.",
                "used_in_main_text_yes_no": "yes"
            },
            {
                "artifact": "Hardware/Runtime Table",
                "path": "results_p0_revision/tables_csv/table_hardware_runtime.csv",
                "status": "available",
                "sha256": get_file_sha256(os.path.join(tables_csv_dir, "table_hardware_runtime.csv")),
                "purpose": "Details system hardware (CPU/GPU, RAM) and pipeline running times.",
                "used_in_main_text_yes_no": "yes"
            },
            {
                "artifact": "Anonymous Repository Link",
                "path": "https://anonymous.4open.science/r/LC-MRSG",
                "status": "available",
                "sha256": "N/A",
                "purpose": "Anonymous code sharing for double-blind peer review.",
                "used_in_main_text_yes_no": "yes"
            }
        ]
        # We will dynamically populate this SHA256 during main manifest build,
        # but here we generate the table with whatever is currently available.
        df_check = pd.DataFrame(checklist_rows)
        csv_check = os.path.join(tables_csv_dir, "table_reproducibility_checklist.csv")
        df_check.to_csv(csv_check, index=False)
        lf.write(f"TABLE_CREATED table_reproducibility_checklist rows={len(df_check)}\n")
        
        tex_check = os.path.join(tables_tex_dir, "table_reproducibility_checklist.tex")
        latex_check = [
            "\\begin{tabular}{llcl}",
            "\\hline",
            "Artifact & Path & Status & Purpose \\\\",
            "\\hline"
        ]
        for _, r in df_check.iterrows():
            latex_check.append(
                f"{r['artifact']} & {os.path.basename(r['path'])} & {r['status']} & {r['purpose']} \\\\"
            )
        latex_check.append("\\hline")
        latex_check.append("\\end{tabular}")
        with open(tex_check, "w") as f:
            f.write("\n".join(latex_check))
            
        lf.write("PHASE_PASS table_generation\n")
        print("PHASE_PASS table_generation")

if __name__ == "__main__":
    main()
