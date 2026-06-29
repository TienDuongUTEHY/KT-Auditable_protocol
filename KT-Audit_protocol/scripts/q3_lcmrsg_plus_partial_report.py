import os
import sys
import re
import pandas as pd
import numpy as np

log_file = "full_q3_run.log"

def parse_log():
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found.")
        return []
        
    records = []
    # Pattern: [timestamp]   [dataset fold seed] Model model Variant var -> Val AUC: val, Test AUC: test
    pattern = re.compile(
        r"\[.*?\]\s+\[(?P<dataset>\w+)\s+fold\s+(?P<fold>\d+)\s+seed\s+(?P<seed>\d+)\]\s+Model\s+(?P<model>\w+)\s+Variant\s+(?P<variant>\w+)\s+->\s+Val\s+AUC:\s+(?P<val_auc>[\d\.\-]+),\s+Test\s+AUC:\s+(?P<test_auc>[\d\.\-]+)"
    )
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                d = m.groupdict()
                records.append({
                    'dataset': d['dataset'],
                    'fold': int(d['fold']),
                    'seed': int(d['seed']),
                    'model': d['model'],
                    'variant': d['variant'],
                    'valid_auc': float(d['val_auc']) if d['val_auc'] != 'nan' else np.nan,
                    'test_auc': float(d['test_auc']) if d['test_auc'] != 'nan' else np.nan
                })
    return records

def main():
    records = parse_log()
    if not records:
        print("No completed experiment runs found in the log yet.")
        return
        
    df = pd.DataFrame(records)
    print(f"Parsed {len(df)} total runs from {log_file}.")
    
    # Analyze completed datasets
    completed_datasets = []
    for ds, group in df.groupby('dataset'):
        # Check if we have 3 folds x 3 seeds x 5 models x 6 variants = 270 runs
        # Since we use 3 seeds: 42, 2024, 3407
        expected_runs = 3 * 3 * 5 * 6
        if len(group) >= expected_runs:
            completed_datasets.append(ds)
            
    print(f"Datasets fully completed: {completed_datasets}")
    
    # Generate a partial summary table for completed datasets
    for ds in completed_datasets:
        print(f"\n==================== RESULTS FOR COMPLETED DATASET: {ds.upper()} ====================")
        sub = df[df['dataset'] == ds]
        
        # Simulate val_selected_static
        # For each fold, seed, model, find the static variant with the highest valid_auc
        static_vars = ['no_graph', 'e_pre', 'e_pre_e_sim', 'full_lc_mrsg']
        
        selection_rows = []
        for (fold, seed, model), g in sub.groupby(['fold', 'seed', 'model']):
            # Static variants selection
            best_static_var = None
            best_static_val = -1.0
            
            for v in static_vars:
                var_row = g[g['variant'] == v]
                if not var_row.empty:
                    val = var_row['valid_auc'].values[0]
                    if val > best_static_val:
                        best_static_val = val
                        best_static_var = v
            
            # Gated variant selection (all 6)
            best_gate_var = None
            best_gate_val = -1.0
            for v in g['variant'].unique():
                var_row = g[g['variant'] == v]
                if not var_row.empty:
                    val = var_row['valid_auc'].values[0]
                    if val > best_gate_val:
                        best_gate_val = val
                        best_gate_var = v
                        
            # Get test AUCs
            t_no = g[g['variant'] == 'no_graph']['test_auc'].values[0] if not g[g['variant'] == 'no_graph'].empty else np.nan
            t_full = g[g['variant'] == 'full_lc_mrsg']['test_auc'].values[0] if not g[g['variant'] == 'full_lc_mrsg'].empty else np.nan
            t_sel = g[g['variant'] == best_static_var]['test_auc'].values[0] if best_static_var else np.nan
            t_gated = g[g['variant'] == best_gate_var]['test_auc'].values[0] if best_gate_var else np.nan
            
            selection_rows.append({
                'fold': fold, 'seed': seed, 'model': model,
                'no_graph': t_no,
                'full_lc_mrsg': t_full,
                'val_selected_static': t_sel,
                'relation_gated': t_gated
            })
            
        df_sel = pd.DataFrame(selection_rows)
        summary = df_sel.groupby('model')[['no_graph', 'full_lc_mrsg', 'val_selected_static', 'relation_gated']].mean()
        print(summary.round(4))

if __name__ == "__main__":
    main()
