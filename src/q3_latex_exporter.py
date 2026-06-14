import os
import pandas as pd
from pathlib import Path
import glob

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def csv_to_latex():
    in_dir = "ResultBS/tables"
    out_dir = "ResultBS/paper/tables_q3"
    ensure_dir(out_dir)
    
    csv_files = glob.glob(f"{in_dir}/*.csv")
    for file in csv_files:
        try:
            df = pd.read_csv(file)
        except pd.errors.EmptyDataError:
            print(f"Skipping {file} because it is empty.")
            continue
            
        base_name = os.path.basename(file).replace(".csv", ".tex")
        
        # Some tables like multifold need mean +- std formatting
        if "multifold" in base_name and 'auc_mean' in df.columns and 'auc_std' in df.columns:
            df['AUC'] = df.apply(lambda r: f"{r['auc_mean']:.4f} {r'$\pm$'} {r['auc_std']:.4f}", axis=1)
            df['ACC'] = df.apply(lambda r: f"{r['acc_mean']:.4f} {r'$\pm$'} {r['acc_std']:.4f}", axis=1)
            df = df.drop(columns=['auc_mean', 'auc_std', 'acc_mean', 'acc_std'])
            
        latex_str = df.to_latex(index=False, escape=False)
        
        # Add booktabs styling
        latex_str = latex_str.replace("\\begin{tabular}", "\\begin{table*}[htbp]\n\\centering\n\\begin{tabular}")
        latex_str = latex_str.replace("\\end{tabular}", "\\end{tabular}\n\\end{table*}")
        
        with open(f"{out_dir}/{base_name}", "w", encoding='utf-8') as f:
            f.write(latex_str)
            
    print(f"Exported {len(csv_files)} LaTeX tables to {out_dir}")

if __name__ == "__main__":
    csv_to_latex()
