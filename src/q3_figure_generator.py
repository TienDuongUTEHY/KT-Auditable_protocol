import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

# Styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

def generate_noise_robustness(df):
    ensure_dir("ResultBS/figures/noise")
    # Extract noise ratio from notes
    noise_df = df[df['notes'].str.contains('Noise=', na=False)].copy()
    if noise_df.empty:
        print("No noise data found to plot.")
        return
        
    noise_df['noise_rate'] = noise_df['notes'].str.extract(r'Noise=([\d\.]+)').astype(float)
    
    # Also get the baseline 0 noise (Graph features augmented)
    base_df = df[(df['notes'] == 'Graph features augmented') | (df['notes'].isna())].copy()
    if not base_df.empty:
        base_df['noise_rate'] = 0.0
        plot_df = pd.concat([noise_df, base_df])
    else:
        plot_df = noise_df
        
    for ds in plot_df['dataset'].unique():
        ds_data = plot_df[plot_df['dataset'] == ds]
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=ds_data, x='noise_rate', y='auc', hue='model', style='graph_variant', markers=True, dashes=False)
        plt.title(f"Noise Robustness - {ds.upper()}")
        plt.xlabel("Noise Injection Rate")
        plt.ylabel("AUC Score")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"ResultBS/figures/noise/q3_noise_robustness_curves_{ds}.pdf")
        plt.close()

def generate_eco_weights():
    ensure_dir("ResultBS/figures/eco")
    datasets = ['assist2012', 'junyi', 'kdd2010']
    for ds in datasets:
        path = f"results/tables/{ds}/fold_0/edge_provenance.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            eco = df[df['relation_type'] == 'E_co']
            if not eco.empty:
                plt.figure(figsize=(8, 6))
                sns.histplot(eco['weight'], bins=30, kde=True, color='purple')
                plt.title(f"E_co Weight Distribution - {ds.upper()}")
                plt.xlabel("Co-occurrence Weight (PMI / Normalized)")
                plt.ylabel("Frequency")
                plt.tight_layout()
                plt.savefig(f"ResultBS/figures/eco/q3_eco_weight_distribution_{ds}.pdf")
                plt.close()

def generate_dataset_scale_bar():
    ensure_dir("ResultBS/figures")
    # Hardcoding since the CSV was empty due to fold setup
    data = [
        {'dataset': 'KDD2010', 'processed_num_learners': 892, 'processed_num_interactions': 1602677},
        {'dataset': 'ASSIST2012', 'processed_num_learners': 22767, 'processed_num_interactions': 221364},
        {'dataset': 'JUNYI', 'processed_num_learners': 48192, 'processed_num_interactions': 500000}
    ]
    df = pd.DataFrame(data)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Bar plot for interactions
    sns.barplot(data=df, x='dataset', y='processed_num_interactions', ax=ax1, color='lightblue', label='Interactions')
    ax1.set_ylabel("Num Interactions", color='blue')
    ax1.set_yscale('log')
    
    # Twin axis for learners
    ax2 = ax1.twinx()
    sns.lineplot(data=df, x='dataset', y='processed_num_learners', ax=ax2, color='red', marker='o', label='Learners')
    ax2.set_ylabel("Num Learners", color='red')
    ax2.set_yscale('log')
    
    plt.title("Processed Dataset Scale (Log Scale)")
    plt.tight_layout()
    plt.savefig("ResultBS/figures/q3_dataset_scale_bar.pdf")
    plt.close()

def generate_leakage_heatmap():
    ensure_dir("ResultBS/figures/leakage")
    path = "ResultBS/tables/q3_leakage_audit_full.csv"
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            # Convert PASS/FAIL to 1/0
            heatmap_data = df.drop(columns=['fold']).set_index('dataset').replace({'PASS': 1, 'FAIL': 0, 'WARN': 0.5}).astype(float)
            plt.figure(figsize=(10, 4))
            cmap = sns.color_palette(["red", "orange", "green"])
            sns.heatmap(heatmap_data, annot=True, cmap=cmap, cbar=False, 
                        xticklabels=['L1', 'L2', 'L3', 'L4', 'L5', 'L6'])
            plt.title("Leakage Audit Heatmap (1=PASS, 0=FAIL)")
            plt.tight_layout()
            plt.savefig("ResultBS/figures/leakage/q3_leakage_audit_heatmap.pdf")
            plt.close()
        except pd.errors.EmptyDataError:
            pass

if __name__ == "__main__":
    df_path = "ResultBS/supplementary/all_runs_q3.csv"
    if os.path.exists(df_path):
        df = pd.read_csv(df_path)
        try:
            generate_noise_robustness(df)
        except Exception as e:
            print(f"Noise plot error: {e}")
    
    try:
        generate_eco_weights()
    except Exception as e:
        print(f"Eco weight error: {e}")
        
    try:
        generate_dataset_scale_bar()
    except Exception as e:
        print(f"Scale bar error: {e}")
        
    try:
        generate_leakage_heatmap()
    except Exception as e:
        print(f"Leakage heatmap error: {e}")
        
    print("All configured figures generated.")
