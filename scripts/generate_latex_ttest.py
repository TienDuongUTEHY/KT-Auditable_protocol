import pandas as pd
import os
from scipy import stats

def format_cell(mean, std, is_best=False):
    cell = f"{mean:.4f} $\\pm$ {std:.4f}"
    if is_best:
        return f"\\textbf{{{cell}}}"
    return cell

def generate_latex():
    csv_path = "ResultBS/confirmatory/multifold_summary.csv"
    raw_csv_path = "ResultBS/confirmatory/confirmatory_results.csv"
    
    if not os.path.exists(csv_path) or not os.path.exists(raw_csv_path):
        print("Required CSV files not found.")
        return

    df_summ = pd.read_csv(csv_path)
    df_raw = pd.read_csv(raw_csv_path)
    
    datasets = ["assist2012", "junyi", "kdd2010"]
    dataset_names = {"assist2012": "ASSISTments 2012", "junyi": "Junyi", "kdd2010": "KDD Cup 2010"}
    
    models = ["BKT", "DKT", "simpleKT", "GIKT", "SKT"]
    variants = ["no_graph", "E_pre", "E_pre_E_sim", "E_pre_E_sim_E_co"]
    variant_names = {
        "no_graph": "Base (No Graph)",
        "E_pre": "$E_{pre}$",
        "E_pre_E_sim": "$E_{pre} + E_{sim}$",
        "E_pre_E_sim_E_co": "$E_{pre} + E_{sim} + E_{co}$ (Ours)"
    }

    # Find best values for bolding
    best_auc = {ds: df_summ[df_summ['dataset'] == ds]['auc_mean'].max() for ds in datasets}
    best_acc = {ds: df_summ[df_summ['dataset'] == ds]['acc_mean'].max() for ds in datasets}

    latex_code = []
    latex_code.append("\\begin{table*}[htbp]")
    latex_code.append("\\centering")
    latex_code.append("\\caption{Knowledge Tracing Performance across 5 Seeds (Mean $\\pm$ Std) with Paired t-test $p$-values (Ours vs. Base)}")
    latex_code.append("\\label{tab:main_results_ttest}")
    latex_code.append("\\resizebox{\\textwidth}{!}{")
    
    col_setup = "ll" + "ccc"*len(datasets)
    latex_code.append("\\begin{tabular}{" + col_setup + "}")
    latex_code.append("\\toprule")
    
    # Header 1
    header1 = "& "
    for ds in datasets:
        header1 += f"& \\multicolumn{{3}}{{c}}{{\\textbf{{{dataset_names[ds]}}}}}"
    header1 += " \\\\"
    latex_code.append(header1)
    
    # Header 2
    header2 = "\\textbf{Model} & \\textbf{Graph Variant} "
    for _ in datasets:
        header2 += "& \\textbf{AUC} & \\textbf{ACC} & \\textbf{$p$-value} "
    header2 += " \\\\"
    latex_code.append(header2)
    latex_code.append("\\midrule")

    # Body
    for i, model in enumerate(models):
        for j, variant in enumerate(variants):
            row_str = ""
            if j == 0:
                row_str += f"\\multirow{{{len(variants)}}}{{*}}{{\\textbf{{{model}}}}}"
            row_str += f" & {variant_names[variant]} "

            for ds in datasets:
                row_data = df_summ[(df_summ['dataset'] == ds) & (df_summ['model'] == model) & (df_summ['graph_variant'] == variant)]
                
                if row_data.empty:
                    row_str += "& - & - & - "
                else:
                    row_data = row_data.iloc[0]
                    auc_mean, auc_std = row_data['auc_mean'], row_data['auc_std']
                    acc_mean, acc_std = row_data['acc_mean'], row_data['acc_std']
                    
                    is_best_auc = (auc_mean >= best_auc[ds] - 1e-5)
                    is_best_acc = (acc_mean >= best_acc[ds] - 1e-5)
                    
                    auc_str = format_cell(auc_mean, auc_std, is_best_auc)
                    acc_str = format_cell(acc_mean, acc_std, is_best_acc)
                    
                    # Calculate p-value (comparing current variant vs Base)
                    pval_str = "-"
                    if variant != "no_graph":
                        base_aucs = df_raw[(df_raw['dataset'] == ds) & (df_raw['model'] == model) & (df_raw['graph_variant'] == 'no_graph')].sort_values(by=['seed', 'fold_id'])['auc'].values
                        curr_aucs = df_raw[(df_raw['dataset'] == ds) & (df_raw['model'] == model) & (df_raw['graph_variant'] == variant)].sort_values(by=['seed', 'fold_id'])['auc'].values
                        
                        if len(base_aucs) == len(curr_aucs) and len(base_aucs) > 0:
                            import numpy as np
                            t_stat, p_val = stats.ttest_rel(curr_aucs, base_aucs)
                            if np.isnan(p_val):
                                pval_str = "-"
                            elif p_val < 0.001:
                                pval_str = "$<$0.001"
                                pval_str = f"\\textbf{{{pval_str}}}"
                            else:
                                pval_str = f"{p_val:.3f}"
                                if p_val < 0.05:
                                    pval_str = f"\\textbf{{{pval_str}}}"

                    row_str += f"& {auc_str} & {acc_str} & {pval_str} "
                    
            row_str += " \\\\"
            latex_code.append(row_str)
        
        if i < len(models) - 1:
            latex_code.append("\\midrule")
            
    latex_code.append("\\bottomrule")
    latex_code.append("\\end{tabular}}")
    latex_code.append("\\end{table*}")

    out_file = "ResultBS/confirmatory/q3_latex_table_ttest.tex"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_code))
    
    print(f"Saved to: {out_file}")

if __name__ == "__main__":
    generate_latex()
