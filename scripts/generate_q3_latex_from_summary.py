import pandas as pd
import os

def format_cell(mean, std, is_best=False):
    cell = f"{mean:.4f} $\\pm$ {std:.4f}"
    if is_best:
        return f"\\textbf{{{cell}}}"
    return cell

def generate_latex():
    csv_path = "ResultBS/confirmatory/multifold_summary.csv"
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
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

    # Find best values for each dataset (to bold them)
    best_auc = {ds: df[df['dataset'] == ds]['auc_mean'].max() for ds in datasets}
    best_acc = {ds: df[df['dataset'] == ds]['acc_mean'].max() for ds in datasets}

    latex_code = []
    latex_code.append("\\begin{table*}[htbp]")
    latex_code.append("\\centering")
    latex_code.append("\\caption{Knowledge Tracing Performance across 5 Independent Seeds (Mean $\\pm$ Std)}")
    latex_code.append("\\label{tab:main_results}")
    latex_code.append("\\resizebox{\\textwidth}{!}{")
    
    # Table format setup
    col_setup = "ll" + "cc"*len(datasets)
    latex_code.append("\\begin{tabular}{" + col_setup + "}")
    latex_code.append("\\toprule")
    
    # Header row 1: Datasets
    header1 = "& "
    for ds in datasets:
        header1 += f"& \\multicolumn{{2}}{{c}}{{\\textbf{{{dataset_names[ds]}}}}}"
    header1 += " \\\\"
    latex_code.append(header1)
    
    # Header row 2: Metrics
    header2 = "\\textbf{Model} & \\textbf{Graph Variant} "
    for _ in datasets:
        header2 += "& \\textbf{AUC} & \\textbf{ACC} "
    header2 += " \\\\"
    latex_code.append(header2)
    latex_code.append("\\midrule")

    # Body
    for i, model in enumerate(models):
        for j, variant in enumerate(variants):
            row_str = ""
            # Print model name only on the first variant
            if j == 0:
                row_str += f"\\multirow{{{len(variants)}}}{{*}}{{\\textbf{{{model}}}}}"
            row_str += f" & {variant_names[variant]} "

            for ds in datasets:
                # Filter row
                row_data = df[(df['dataset'] == ds) & (df['model'] == model) & (df['graph_variant'] == variant)]
                
                if row_data.empty:
                    row_str += "& - & - "
                else:
                    row_data = row_data.iloc[0]
                    auc_mean, auc_std = row_data['auc_mean'], row_data['auc_std']
                    acc_mean, acc_std = row_data['acc_mean'], row_data['acc_std']
                    
                    # Highlight if best
                    is_best_auc = (auc_mean >= best_auc[ds] - 1e-5)
                    is_best_acc = (acc_mean >= best_acc[ds] - 1e-5)
                    
                    auc_str = format_cell(auc_mean, auc_std, is_best_auc)
                    acc_str = format_cell(acc_mean, acc_std, is_best_acc)
                    
                    row_str += f"& {auc_str} & {acc_str} "
                    
            row_str += " \\\\"
            latex_code.append(row_str)
        
        # Add midrule between models (except the last one)
        if i < len(models) - 1:
            latex_code.append("\\midrule")
            
    latex_code.append("\\bottomrule")
    latex_code.append("\\end{tabular}}")
    latex_code.append("\\end{table*}")

    out_file = "ResultBS/confirmatory/q3_latex_table.tex"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_code))
    
    print(f"LaTeX table successfully generated at: {out_file}")
    print("\n" + "="*50)
    print("LATEX CODE PREVIEW:")
    print("="*50)
    print("\n".join(latex_code[:25]))
    print("... (truncated) ...")
    print("\n".join(latex_code[-5:]))

if __name__ == "__main__":
    generate_latex()
