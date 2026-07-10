import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

def get_bootstrap_ci(diffs, num_resamples=10000, ci=0.95):
    if len(diffs) == 0:
        return 0.0, 0.0
    boot_deltas = []
    n = len(diffs)
    # Use fixed seed for bootstrap reproducibility
    rng = np.random.default_rng(42)
    for _ in range(num_resamples):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_deltas.append(np.mean(sample))
    ci_low = np.percentile(boot_deltas, (1.0 - ci) / 2.0 * 100.0)
    ci_high = np.percentile(boot_deltas, (1.0 + ci) / 2.0 * 100.0)
    return ci_low, ci_high

def get_practical_effect(delta):
    ad = abs(delta)
    if ad < 0.001:
        return 'negligible'
    elif ad < 0.003:
        return 'small'
    elif ad < 0.007:
        return 'moderate'
    else:
        return 'practically meaningful'

def holm_correction(p_values):
    m = len(p_values)
    # Keep track of indices
    indexed_p = [(p, i) for i, p in enumerate(p_values)]
    indexed_p.sort(key=lambda x: x[0])
    
    corrected_p = [0.0] * m
    max_val = 0.0
    for rank, (p, orig_idx) in enumerate(indexed_p):
        multiplier = m - rank
        corrected = min(p * multiplier, 1.0)
        max_val = max(max_val, corrected)
        corrected_p[orig_idx] = max_val
        
    return corrected_p

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    
    results_dir = os.path.join(args.run_dir, 'results')
    stats_dir = os.path.join(args.run_dir, 'statistics')
    sparse_dir = os.path.join(args.run_dir, 'sparse')
    
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)
    
    runs_path = os.path.join(results_dir, 'all_runs_train_valid_test.csv')
    strata_path = os.path.join(results_dir, 'strata_runs_test_auc.csv')
    
    if not os.path.exists(runs_path):
        print(f"Error: {runs_path} does not exist.")
        sys.exit(1)
        
    df_runs = pd.read_csv(runs_path)
    
    # 1. Compute Paired Difference Metrics
    # Filter methods to compare
    method = 'sparse_aware_relation_gated'
    
    datasets = df_runs['dataset'].unique()
    models = df_runs['model'].unique()
    
    paired_rows = []
    
    for ds in datasets:
        for model in models:
            sub = df_runs[(df_runs['dataset'] == ds) & (df_runs['model'] == model)]
            if sub.empty:
                continue
                
            no_graph = sub[sub['variant'] == 'no_graph'].set_index(['fold', 'seed'])
            full_graph = sub[sub['variant'] == 'full_lc_mrsg'].set_index(['fold', 'seed'])
            target = sub[sub['variant'] == method].set_index(['fold', 'seed'])
            
            # Find best static baseline (excluding no_graph)
            static_baselines = ['e_pre', 'e_pre_e_sim', 'full_lc_mrsg']
            best_static_name = 'e_pre'
            best_static_auc = -1.0
            
            for sb in static_baselines:
                mean_auc = sub[sub['variant'] == sb]['test_auc'].mean()
                if mean_auc > best_static_auc:
                    best_static_auc = mean_auc
                    best_static_name = sb
            
            best_static = sub[sub['variant'] == best_static_name].set_index(['fold', 'seed'])
            
            # Align indices
            idx = target.index.intersection(no_graph.index).intersection(full_graph.index).intersection(best_static.index)
            if len(idx) < 2:
                continue
                
            y_target = target.loc[idx, 'test_auc'].to_numpy()
            y_no = no_graph.loc[idx, 'test_auc'].to_numpy()
            y_full = full_graph.loc[idx, 'test_auc'].to_numpy()
            y_best_stat = best_static.loc[idx, 'test_auc'].to_numpy()
            
            diff_no = y_target - y_no
            diff_full = y_target - y_full
            diff_best = y_target - y_best_stat
            
            # Paired t-tests vs no_graph
            t_stat, p_two = stats.ttest_rel(y_target, y_no, nan_policy='omit')
            # One-tailed p-value (assuming method improves)
            p_one = p_two / 2.0 if t_stat > 0 else 1.0 - p_two / 2.0
            
            # Wilcoxon p-value
            try:
                _, p_wilcoxon = stats.wilcoxon(diff_no)
            except Exception:
                p_wilcoxon = float('nan')
                
            # Bootstrap CI
            ci_low, ci_high = get_bootstrap_ci(diff_no)
            
            # Cohen's d
            std_diff = diff_no.std(ddof=1)
            cohen_d = diff_no.mean() / std_diff if std_diff != 0 and not np.isnan(std_diff) else 0.0
            
            paired_rows.append({
                'dataset': ds,
                'model': model,
                'n_pairs': len(idx),
                'mean_auc_no_graph': y_no.mean(),
                'mean_auc_method': y_target.mean(),
                'delta_auc': diff_no.mean(),
                'ci_low': ci_low,
                'ci_high': ci_high,
                't_stat': t_stat,
                'p_two_tailed': p_two,
                'p_one_tailed': p_one,
                'p_wilcoxon': p_wilcoxon,
                'cohens_d': cohen_d,
                'delta_auc_vs_full': diff_full.mean(),
                'delta_auc_vs_best_static': diff_best.mean()
            })
            
    df_paired = pd.DataFrame(paired_rows)
    
    # Holm correction
    df_paired['p_holm'] = holm_correction(df_paired['p_two_tailed'].fillna(1.0).tolist())
    df_paired['significant_005_uncorrected'] = df_paired['p_two_tailed'] < 0.05
    df_paired['significant_005_holm'] = df_paired['p_holm'] < 0.05
    df_paired['practical_effect'] = df_paired['delta_auc'].apply(get_practical_effect)
    
    df_paired.to_csv(os.path.join(stats_dir, 'paired_tests_with_ci.csv'), index=False)
    
    # Save specific tables
    df_paired[['dataset', 'model', 'delta_auc', 'ci_low', 'ci_high', 'p_two_tailed', 'p_holm', 'cohens_d']].to_csv(
        os.path.join(stats_dir, 'holm_corrected_tests.csv'), index=False
    )
    df_paired[['dataset', 'model', 'delta_auc', 'practical_effect']].to_csv(
        os.path.join(stats_dir, 'practical_significance_table.csv'), index=False
    )
    
    # 2. Sparse Stratum Analysis
    if os.path.exists(strata_path):
        df_str = pd.read_csv(strata_path)
        
        # Mean stratum test AUCs by dataset, model, variant, stratum
        df_str_summary = df_str.groupby(['dataset', 'model', 'variant', 'stratum'])['auc'].mean().reset_index()
        
        # Pivot strata as columns: very_sparse, sparse, medium, frequent
        pivoted = df_str_summary.pivot(
            index=['dataset', 'model', 'variant'], 
            columns='stratum', 
            values='auc'
        ).reset_index()
        
        # Calculate sparse gain vs no_graph
        no_graph_strata = pivoted[pivoted['variant'] == 'no_graph'].set_index(['dataset', 'model'])
        method_strata = pivoted[pivoted['variant'] == method].set_index(['dataset', 'model'])
        
        common_idx = method_strata.index.intersection(no_graph_strata.index)
        
        strata_rows = []
        for ds, model in common_idx:
            m_row = method_strata.loc[(ds, model)]
            n_row = no_graph_strata.loc[(ds, model)]
            
            # Sparse gain vs no_graph on sparse strata (very_sparse + sparse)
            m_sparse_avg = np.nanmean([m_row.get('very_sparse', np.nan), m_row.get('sparse', np.nan)])
            n_sparse_avg = np.nanmean([n_row.get('very_sparse', np.nan), n_row.get('sparse', np.nan)])
            gain = m_sparse_avg - n_sparse_avg
            
            strata_rows.append({
                'dataset': ds,
                'model': model,
                'variant': method,
                'very_sparse': m_row.get('very_sparse', np.nan),
                'sparse': m_row.get('sparse', np.nan),
                'medium': m_row.get('medium', np.nan),
                'frequent': m_row.get('frequent', np.nan),
                'sparse_gain_vs_no_graph': gain
            })
            
        pd.DataFrame(strata_rows).to_csv(os.path.join(sparse_dir, 'sparse_stratum_summary.csv'), index=False)
        
    print("Statistical and sparse stratum analyses completed.")

if __name__ == "__main__":
    main()
