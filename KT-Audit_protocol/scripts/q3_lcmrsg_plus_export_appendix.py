import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def generate_heatmap(runs_path, out_path):
    if not os.path.exists(runs_path):
        return
    df_runs = pd.read_csv(runs_path)
    df_sel = df_runs[df_runs['variant'] == 'val_selected_static']
    if df_sel.empty:
        return
        
    def get_variant_name(row):
        ap, asim, ac = row['alpha_pre'], row['alpha_sim'], row['alpha_co']
        if ap == 0 and asim == 0 and ac == 0: return 'no_graph'
        if ap == 1 and asim == 0 and ac == 0: return 'e_pre'
        if ap == 1 and asim == 1 and ac == 0: return 'e_pre_e_sim'
        if ap == 1 and asim == 1 and ac == 1: return 'full_lc_mrsg'
        return 'gated'
        
    df_sel['chosen_var'] = df_sel.apply(get_variant_name, axis=1)
    
    # Pivot to count variant selections by dataset/model
    counts = df_sel.groupby(['dataset', 'model', 'chosen_var']).size().unstack(fill_value=0)
    # Normalize by row sums to get frequencies
    freq = counts.div(counts.sum(axis=1), axis=0)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(freq, annot=True, cmap="YlGnBu", fmt=".2f", cbar_kws={'label': 'Selection Frequency'})
    plt.title("Validation-Guided Graph Relation Selection Pattern (LC-MRSG++)")
    plt.ylabel("Dataset, Model")
    plt.xlabel("Selected Graph Variant")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def generate_forest_plot(paired_path, out_path):
    if not os.path.exists(paired_path):
        return
    df_paired = pd.read_csv(paired_path)
    if df_paired.empty:
        return
        
    df_paired['label'] = df_paired['dataset'] + " - " + df_paired['model']
    df_paired = df_paired.sort_values(by='delta_auc', ascending=True)
    
    plt.figure(figsize=(8, 8))
    
    # Vertical line at 0 (no effect)
    plt.axvline(x=0.0, color='red', linestyle='--', alpha=0.7, label='No Difference')
    
    # Plot mean and CIs
    y_pos = np.arange(len(df_paired))
    plt.errorbar(
        df_paired['delta_auc'], y_pos, 
        xerr=[df_paired['delta_auc'] - df_paired['ci_low'], df_paired['ci_high'] - df_paired['delta_auc']],
        fmt='o', color='blue', ecolor='gray', capsize=4, elinewidth=1.5, markeredgewidth=1.5,
        label='Delta AUC (95% CI)'
    )
    
    plt.yticks(y_pos, df_paired['label'])
    plt.xlabel("Delta AUC (LC-MRSG++ vs. No Graph)")
    plt.title("Effect Sizes and 95% Bootstrap Confidence Intervals")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def generate_sparse_delta_plot(runs_path, strata_path, out_path):
    if not os.path.exists(runs_path) or not os.path.exists(strata_path):
        return
    df_str = pd.read_csv(strata_path)
    
    # Compute mean test AUC per dataset, model, variant, stratum
    df_mean = df_str.groupby(['dataset', 'model', 'variant', 'stratum'])['auc'].mean().reset_index()
    
    # Compare sparse_aware_relation_gated vs no_graph
    no_graph = df_mean[df_mean['variant'] == 'no_graph'].rename(columns={'auc': 'auc_no'})
    method = df_mean[df_mean['variant'] == 'sparse_aware_relation_gated'].rename(columns={'auc': 'auc_method'})
    
    merged = pd.merge(
        method[['dataset', 'model', 'stratum', 'auc_method']],
        no_graph[['dataset', 'model', 'stratum', 'auc_no']],
        on=['dataset', 'model', 'stratum']
    )
    merged['delta_auc'] = merged['auc_method'] - merged['auc_no']
    
    # Plot stratum-wise delta AUC across models
    plt.figure(figsize=(10, 6))
    order = ['very_sparse', 'sparse', 'medium', 'frequent']
    sns.barplot(
        data=merged, x='stratum', y='delta_auc', hue='dataset', 
        order=order, errorbar=None, palette='Set2'
    )
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title("AUC Improvement by Skill Frequency Stratum")
    plt.xlabel("Strata (Train Frequency)")
    plt.ylabel("Delta AUC (LC-MRSG++ vs. No Graph)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    
    results_dir = os.path.join(args.run_dir, 'results')
    stats_dir = os.path.join(args.run_dir, 'statistics')
    figures_dir = os.path.join(args.run_dir, 'figures')
    
    os.makedirs(figures_dir, exist_ok=True)
    
    runs_path = os.path.join(results_dir, 'all_runs_train_valid_test.csv')
    strata_runs_path = os.path.join(results_dir, 'strata_runs_test_auc.csv')
    paired_path = os.path.join(stats_dir, 'paired_tests_with_ci.csv')
    
    print("Generating publication-ready figures...")
    
    # Generate figures
    generate_heatmap(runs_path, os.path.join(figures_dir, 'fig_relation_selection_heatmap.pdf'))
    generate_forest_plot(paired_path, os.path.join(figures_dir, 'fig_delta_auc_forestplot.pdf'))
    generate_sparse_delta_plot(runs_path, strata_runs_path, os.path.join(figures_dir, 'fig_sparse_delta_auc.pdf'))
    
    # Also save as png for easy inspection
    generate_heatmap(runs_path, os.path.join(figures_dir, 'fig_relation_selection_heatmap.png'))
    generate_forest_plot(paired_path, os.path.join(figures_dir, 'fig_delta_auc_forestplot.png'))
    generate_sparse_delta_plot(runs_path, strata_runs_path, os.path.join(figures_dir, 'fig_sparse_delta_auc.png'))
    
    print("Figures completed successfully.")

if __name__ == "__main__":
    main()
