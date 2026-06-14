import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def format_cell(val, fmt="%.4f"):
    if pd.isna(val) or (isinstance(val, float) and np.isnan(val)):
        return "-"
    if isinstance(val, (int, np.integer)):
        return str(val)
    if isinstance(val, (float, np.floating)):
        return fmt % val
    return str(val)

def to_latex_table(df, title, label, columns_format):
    latex_lines = [
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\caption{" + title + "}",
        "\\label{" + label + "}",
        "\\begin{tabular}{" + columns_format + "}",
        "\\hline"
    ]
    
    # Headers
    headers = [str(c).replace('_', ' ').title() for c in df.columns]
    latex_lines.append(" & ".join(headers) + " \\\\")
    latex_lines.append("\\hline")
    
    # Rows
    for _, row in df.iterrows():
        formatted_row = [format_cell(x) for x in row]
        latex_lines.append(" & ".join(formatted_row) + " \\\\")
        
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table*}")
    return "\n".join(latex_lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    
    results_dir = os.path.join(args.run_dir, 'results')
    stats_dir = os.path.join(args.run_dir, 'statistics')
    sparse_dir = os.path.join(args.run_dir, 'sparse')
    audit_dir = os.path.join(args.run_dir, 'audit')
    latex_dir = os.path.join(args.run_dir, 'latex')
    
    os.makedirs(latex_dir, exist_ok=True)
    
    paired_path = os.path.join(stats_dir, 'paired_tests_with_ci.csv')
    runs_path = os.path.join(results_dir, 'all_runs_train_valid_test.csv')
    sparse_path = os.path.join(sparse_dir, 'sparse_stratum_summary.csv')
    audit_path = os.path.join(audit_dir, 'leakage_audit_l1_l6.csv')
    
    # --- TABLE A: Main Confirmatory Table ---
    if os.path.exists(paired_path):
        df_paired = pd.read_csv(paired_path)
        
        # Build Table A
        table_a = pd.DataFrame()
        table_a['Dataset'] = df_paired['dataset']
        table_a['Model'] = df_paired['model']
        table_a['No Graph'] = df_paired['mean_auc_no_graph']
        
        # Get full MRSG mean
        # We need to compute the mean test AUC for full_lc_mrsg and sparse_aware_relation_gated (our selected method)
        # We can extract it from the paired file
        table_a['Method (LC-MRSG++)'] = df_paired['mean_auc_method']
        table_a['Delta (Selected-No)'] = df_paired['delta_auc']
        table_a['95% CI'] = df_paired.apply(lambda r: f"[{format_cell(r['ci_low'])}, {format_cell(r['ci_high'])}]", axis=1)
        table_a['Holm p'] = df_paired['p_holm']
        table_a['Practical Effect'] = df_paired['practical_effect']
        
        table_a.to_csv(os.path.join(latex_dir, 'table_main_confirmatory.csv'), index=False)
        
        latex_a = to_latex_table(table_a, "Main confirmatory results of LC-MRSG++ vs. no-graph baseline.", "tab:main-confirmatory", "llcccccl")
        with open(os.path.join(latex_dir, 'table_main_confirmatory.tex'), 'w') as f:
            f.write(latex_a)
            
    # --- TABLE B: Selected Relation Pattern ---
    if os.path.exists(runs_path):
        df_runs = pd.read_csv(runs_path)
        
        # Filter val_selected_static runs to analyze which variants are chosen
        df_sel = df_runs[df_runs['variant'] == 'val_selected_static']
        if not df_sel.empty:
            # We want to find most selected variant per dataset/model
            # Let's parse selection_reason or reconstruct from alpha parameters
            def get_variant_name(row):
                ap, asim, ac = row['alpha_pre'], row['alpha_sim'], row['alpha_co']
                if ap == 0 and asim == 0 and ac == 0: return 'no_graph'
                if ap == 1 and asim == 0 and ac == 0: return 'e_pre'
                if ap == 1 and asim == 1 and ac == 0: return 'e_pre_e_sim'
                if ap == 1 and asim == 1 and ac == 1: return 'full_lc_mrsg'
                return 'gated_variant'
                
            df_sel['chosen_var'] = df_sel.apply(get_variant_name, axis=1)
            
            b_rows = []
            for (ds, model), group in df_sel.groupby(['dataset', 'model']):
                most_common = group['chosen_var'].mode().iloc[0] if not group['chosen_var'].empty else "none"
                freq = (group['chosen_var'] == most_common).mean()
                mean_val = group['valid_auc'].mean()
                mean_test = group['test_auc'].mean()
                
                interpretation = "Balanced relations"
                if most_common == 'no_graph':
                    interpretation = "No graph helpful"
                elif most_common == 'e_pre':
                    interpretation = "Prerequisite dominant"
                elif most_common == 'e_pre_e_sim':
                    interpretation = "Pre & Sim dominant"
                elif most_common == 'full_lc_mrsg':
                    interpretation = "All relations helpful"
                    
                b_rows.append({
                    'Dataset': ds,
                    'Model': model,
                    'Most Selected Variant': most_common,
                    'Selection Frequency': freq,
                    'Mean Valid AUC': mean_val,
                    'Mean Test AUC': mean_test,
                    'Interpretation': interpretation
                })
                
            table_b = pd.DataFrame(b_rows)
            table_b.to_csv(os.path.join(latex_dir, 'table_val_selected.csv'), index=False)
            
            latex_b = to_latex_table(table_b, "Most frequently selected static variants on validation set.", "tab:val-selected", "llcccl")
            with open(os.path.join(latex_dir, 'table_val_selected.tex'), 'w') as f:
                f.write(latex_b)
                
        # --- TABLE C: Relation Gates & Beta ---
        df_sg = df_runs[df_runs['variant'] == 'sparse_aware_relation_gated']
        if not df_sg.empty:
            c_rows = []
            for (ds, model), group in df_sg.groupby(['dataset', 'model']):
                ap = group['alpha_pre'].mean()
                asim = group['alpha_sim'].mean()
                ac = group['alpha_co'].mean()
                beta = group['beta'].mean()
                t_auc = group['test_auc'].mean()
                
                # Get sparse AUC if available from strata
                c_rows.append({
                    'Dataset': ds,
                    'Model': model,
                    'alpha_pre': ap,
                    'alpha_sim': asim,
                    'alpha_co': ac,
                    'beta': beta,
                    'Test AUC': t_auc,
                    'Interpretation': "Optimal Gates Learned"
                })
            table_c = pd.DataFrame(c_rows)
            table_c.to_csv(os.path.join(latex_dir, 'table_relation_gates.csv'), index=False)
            
    # --- TABLE D: Sparse Stratum Results ---
    if os.path.exists(sparse_path):
        table_d = pd.read_csv(sparse_path)
        # Rename columns to look professional
        table_d.columns = [str(c).replace('_', ' ').title() for c in table_d.columns]
        table_d.to_csv(os.path.join(latex_dir, 'table_sparse_summary.csv'), index=False)
        
        latex_d = to_latex_table(table_d, "Stratified predictive AUC and sparse stratum gains vs. no-graph baseline.", "tab:sparse-summary", "llccccc")
        with open(os.path.join(latex_dir, 'table_sparse_summary.tex'), 'w') as f:
            f.write(latex_d)
            
    # --- TABLE E: Audit and Provenance ---
    if os.path.exists(audit_path):
        table_e = pd.read_csv(audit_path)
        table_e.columns = [str(c).replace('_', ' ').title() for c in table_e.columns]
        table_e.to_csv(os.path.join(latex_dir, 'table_audit_l1_l6.csv'), index=False)
        
        latex_e = to_latex_table(table_e, "L1--L6 leakage audit and graph provenance checks.", "tab:audit-l1-l6", "llcccccccccc")
        with open(os.path.join(latex_dir, 'table_audit_l1_l6.tex'), 'w') as f:
            f.write(latex_e)
            
    print("LaTeX tables successfully rendered.")

if __name__ == "__main__":
    main()
