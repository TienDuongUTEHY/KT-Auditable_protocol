import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, log_loss

def expected_calibration_error(y_true, y_pred, n_bins=10):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (y_pred >= bin_lower) & (y_pred < bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_pred[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
    return ece

def get_mean_ci(values, ci_level=0.95):
    arr = np.array(values)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return np.mean(arr) if len(arr) else 0.0, 0.0
    mean = np.mean(arr)
    sem = stats.sem(arr)
    h = sem * stats.t.ppf((1 + ci_level) / 2.0, len(arr) - 1)
    return mean, h

def main():
    datasets = ["assist2012", "junyi", "kdd2010"]
    folds = [0, 1, 2]
    seeds = [42, 43, 44, 2025, 2026]
    models = ["bkt", "dkt", "simplekt", "gikt", "skt"]
    variants = ["no_graph", "full_lc_mrsg"]
    
    pred_root = "results/predictions"
    results = []
    
    for ds in datasets:
        for model in models:
            for variant in variants:
                aucs = []
                eces = []
                briers = []
                
                for fold in folds:
                    for seed in seeds:
                        path = os.path.join(pred_root, ds, f"fold_{fold}", f"seed_{seed}", model, f"{variant}.csv")
                        if not os.path.exists(path):
                            continue
                        
                        try:
                            df = pd.read_csv(path)
                            y_true = df["y_true"].to_numpy()
                            y_pred = df["y_pred"].clip(1e-7, 1.0 - 1e-7).to_numpy()
                            
                            auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan
                            ece = expected_calibration_error(y_true, y_pred)
                            brier = np.mean((y_pred - y_true) ** 2)
                            
                            aucs.append(auc)
                            eces.append(ece)
                            briers.append(brier)
                        except Exception as e:
                            print(f"Error reading {path}: {e}")
                
                if len(aucs) > 0:
                    mean_auc, h_auc = get_mean_ci(aucs)
                    mean_ece, h_ece = get_mean_ci(eces)
                    mean_brier, h_brier = get_mean_ci(briers)
                    
                    results.append({
                        "dataset": ds,
                        "model": model,
                        "variant": variant,
                        "auc_mean": mean_auc,
                        "auc_ci": h_auc,
                        "ece_mean": mean_ece,
                        "ece_ci": h_ece,
                        "brier_mean": mean_brier,
                        "brier_ci": h_brier,
                        "n_runs": len(aucs)
                    })
                    
    df_res = pd.DataFrame(results)
    
    # Save CSV
    out_dir = "results_p0_revision/tables_csv"
    os.makedirs(out_dir, exist_ok=True)
    df_res.to_csv(os.path.join(out_dir, "table_overall_performance_auc_ece_brier.csv"), index=False)
    print(f"Saved performance metrics to {os.path.join(out_dir, 'table_overall_performance_auc_ece_brier.csv')}")
    
    # Create Latex table
    latex_lines = [
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\caption{Overall KT model performance comparison (No Graph vs. LC-MRSG++) across 15 runs.}",
        "\\label{tab:overall-perf-calibration}",
        "\\begin{tabular}{lllccc}",
        "\\hline",
        "Dataset & Model & Variant & AUC (Mean $\\pm$ 95\\% CI) & ECE (Mean $\\pm$ 95\\% CI) & Brier Score (Mean $\\pm$ 95\\% CI) \\\\",
        "\\hline"
    ]
    
    for ds in datasets:
        for model in models:
            for variant in variants:
                r = df_res[(df_res["dataset"] == ds) & (df_res["model"] == model) & (df_res["variant"] == variant)]
                if r.empty:
                    continue
                r = r.iloc[0]
                var_label = "No Graph (Baseline)" if variant == "no_graph" else "LC-MRSG++ (Proposed)"
                latex_lines.append(
                    f"{ds.upper()} & {model.upper()} & {var_label} & {r['auc_mean']:.4f} $\\pm$ {r['auc_ci']:.4f} & {r['ece_mean']:.4f} $\\pm$ {r['ece_ci']:.4f} & {r['brier_mean']:.4f} $\\pm$ {r['brier_ci']:.4f} \\\\"
                )
        latex_lines.append("\\hline")
        
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table*}")
    
    tex_dir = "results_p0_revision/tables_tex"
    os.makedirs(tex_dir, exist_ok=True)
    with open(os.path.join(tex_dir, "table_overall_performance_auc_ece_brier.tex"), "w") as f:
        f.write("\n".join(latex_lines))
    print(f"Saved performance LaTeX table to {os.path.join(tex_dir, 'table_overall_performance_auc_ece_brier.tex')}")

if __name__ == "__main__":
    main()
