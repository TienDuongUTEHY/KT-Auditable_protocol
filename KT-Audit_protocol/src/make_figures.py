"""
Ý NGHĨA TIẾN TRÌNH:
Vẽ và xuất các biểu đồ trực quan dạng PDF cho bài báo P0.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from src.common import load_config, ensure_dir

COLORS = {'E_pre': '#4C72B0', 'E_sim': '#DD8452', 'E_co': '#55A868'}

def save_pdf(path, fig):
    fig.savefig(path, format='pdf', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

def fig1_pipeline_overview(out_dir, dataset, fold, tab_dir):
    """Fig 1: Dataset stats + graph sizes summary bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Fig 1: Pipeline Overview — {dataset} fold {fold}", fontsize=13, fontweight='bold')

    # Left: Dataset statistics
    try:
        df_stats = pd.read_csv(f"results/tables/{dataset}/dataset_stats.csv")
        row = df_stats.iloc[0]
        labels = ['Learners', 'Questions', 'Skills', 'Interactions']
        vals = [row.get('Learners', 0), row.get('Questions', 0), row.get('Skills', 0), row.get('Interactions', 0)]
        bars = axes[0].bar(labels, vals, color=['#4C72B0','#DD8452','#55A868','#C44E52'], edgecolor='black')
        for bar, val in zip(bars, vals):
            axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                        f'{int(val)}', ha='center', va='bottom', fontsize=9)
        axes[0].set_title('Dataset Statistics', fontsize=11)
        axes[0].set_ylabel('Count')
    except Exception as e:
        axes[0].text(0.5, 0.5, f'No dataset stats\n{e}', ha='center', va='center')

    # Right: Edge counts per graph type
    try:
        df_gs = pd.read_csv(f"{tab_dir}/graph_stats.csv")
        types = df_gs['edge_type'].tolist()
        edges = df_gs['num_edges'].tolist()
        colors = [COLORS.get(t, '#888') for t in types]
        bars = axes[1].bar(types, edges, color=colors, edgecolor='black')
        for bar, val in zip(bars, edges):
            axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                        f'{int(val)}', ha='center', va='bottom', fontsize=9)
        axes[1].set_title('Edge Counts per Graph Type', fontsize=11)
        axes[1].set_ylabel('Number of Edges')
    except Exception as e:
        axes[1].text(0.5, 0.5, f'No graph stats\n{e}', ha='center', va='center')

    plt.tight_layout()
    save_pdf(f"{out_dir}/fig1_pipeline.pdf", fig)


def fig2_eco_weight_distribution(out_dir, dataset, fold, tab_dir):
    """Fig 2: Histogram + CDF of E_co PMI weights."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Fig 2: E_co Weight Distribution — {dataset} fold {fold}", fontsize=13, fontweight='bold')
    try:
        df_co = pd.read_csv(f"{tab_dir}/E_co_train.csv")
        weights = df_co['weight'].dropna()
        # Histogram
        axes[0].hist(weights, bins=20, color='#55A868', edgecolor='black', alpha=0.85)
        axes[0].axvline(weights.median(), color='red', linestyle='--', label=f'Median={weights.median():.3f}')
        axes[0].set_title('PMI Weight Histogram')
        axes[0].set_xlabel('PMI Weight')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        # CDF
        sorted_w = np.sort(weights)
        cdf = np.arange(1, len(sorted_w) + 1) / len(sorted_w)
        axes[1].plot(sorted_w, cdf, color='#55A868', linewidth=2)
        axes[1].axvline(weights.median(), color='red', linestyle='--', label=f'Median={weights.median():.3f}')
        axes[1].set_title('PMI Weight CDF')
        axes[1].set_xlabel('PMI Weight')
        axes[1].set_ylabel('Cumulative Proportion')
        axes[1].legend()
    except Exception as e:
        for ax in axes:
            ax.text(0.5, 0.5, f'No E_co data\n{e}', ha='center', va='center', transform=ax.transAxes)
    plt.tight_layout()
    save_pdf(f"{out_dir}/fig2_eco_weight_distribution.pdf", fig)


def fig3_sparse_skill_strata(out_dir, dataset, fold, tab_dir):
    """Fig 3: Skill strata bar chart + skill interaction count distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Fig 3: Sparse-Skill Strata — {dataset} fold {fold}", fontsize=13, fontweight='bold')
    try:
        df_profile = pd.read_csv(f"{tab_dir}/sparse_skill_profile.csv")
        strata_counts = df_profile['strata'].value_counts()
        strata_order = ['sparse', 'medium', 'dense']
        strata_colors = {'sparse': '#C44E52', 'medium': '#DD8452', 'dense': '#55A868'}
        vals = [strata_counts.get(s, 0) for s in strata_order]
        colors = [strata_colors[s] for s in strata_order]
        bars = axes[0].bar(strata_order, vals, color=colors, edgecolor='black')
        for bar, val in zip(bars, vals):
            axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                        f'{int(val)}', ha='center', va='bottom', fontsize=10)
        axes[0].set_title('Number of Skills per Strata')
        axes[0].set_xlabel('Strata')
        axes[0].set_ylabel('# Skills')

        # Interaction count boxplot per strata
        strata_groups = [df_profile[df_profile['strata'] == s]['n_interactions'].values for s in strata_order]
        bp = axes[1].boxplot(strata_groups, labels=strata_order, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1].set_title('Interaction Count Distribution by Strata')
        axes[1].set_xlabel('Strata')
        axes[1].set_ylabel('# Interactions per Skill')
    except Exception as e:
        for ax in axes:
            ax.text(0.5, 0.5, f'No sparse profile data\n{e}', ha='center', va='center', transform=ax.transAxes)
    plt.tight_layout()
    save_pdf(f"{out_dir}/fig3_sparse_skill_strata.pdf", fig)


def fig4_relation_ablation(out_dir, dataset, fold, tab_dir):
    """Fig 4: Grouped bar chart of AUC/ACC by graph variant."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Fig 4: Relation Ablation — {dataset} fold {fold}", fontsize=13, fontweight='bold')
    try:
        df_base = pd.read_csv(f"{tab_dir}/baseline_results.csv")
        df_base = df_base.groupby(['model', 'graph_variant'], as_index=False).last()
        
        models = df_base['model'].unique()
        variants = ['no_graph', 'E_pre', 'E_pre_E_sim', 'E_pre_E_sim_E_co']
        x = np.arange(len(variants))
        width = 0.8 / len(models) if len(models) > 0 else 0.4
        
        for idx, metric in enumerate(['auc', 'acc']):
            ax = axes[idx]
            for m_idx, model in enumerate(models):
                m_df = df_base[df_base['model'] == model]
                vals = []
                for v in variants:
                    v_df = m_df[m_df['graph_variant'] == v]
                    vals.append(v_df[metric].values[0] if not v_df.empty else 0)
                offset = width * m_idx - (width * len(models) / 2) + width/2
                bars = ax.bar(x + offset, vals, width, label=model, edgecolor='black')
                for bar in bars:
                    val = bar.get_height()
                    if val > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., val + 0.005,
                                f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
            ax.set_xticks(x)
            ax.set_xticklabels(variants, rotation=15)
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{metric.upper()} Comparison')
            if len(models) > 0:
                ax.legend()
            ax.set_ylim(0.4, 1.05)
    except Exception as e:
        for ax in axes:
            ax.text(0.5, 0.5, f'Relation ablation not available for this dataset/fold.\n{e}', ha='center', va='center', transform=ax.transAxes)
    plt.tight_layout()
    save_pdf(f"{out_dir}/fig4_relation_ablation.pdf", fig)


def fig5_eco_threshold_sensitivity(out_dir, dataset, fold, tab_dir):
    """Fig 5: How edge count varies with different PMI thresholds."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Fig 5: E_co Threshold Sensitivity — {dataset} fold {fold}", fontsize=13, fontweight='bold')
    try:
        df_co = pd.read_csv(f"{tab_dir}/E_co_train.csv")
        weights = df_co['weight'].dropna().values
        thresholds = np.linspace(0, weights.max(), 30)
        edge_counts = [int((weights >= t).sum()) for t in thresholds]
        axes[0].plot(thresholds, edge_counts, color='#4C72B0', linewidth=2, marker='o', markersize=3)
        axes[0].set_title('Edge Count vs PMI Threshold')
        axes[0].set_xlabel('PMI Threshold')
        axes[0].set_ylabel('# Edges Retained')
        axes[0].grid(True, alpha=0.3)

        # Percentage retained
        pct = [100 * c / len(weights) if len(weights) > 0 else 0 for c in edge_counts]
        axes[1].plot(thresholds, pct, color='#DD8452', linewidth=2, marker='s', markersize=3)
        axes[1].axhline(50, color='gray', linestyle='--', alpha=0.5, label='50% line')
        axes[1].set_title('% Edges Retained vs PMI Threshold')
        axes[1].set_xlabel('PMI Threshold')
        axes[1].set_ylabel('% Edges Retained')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    except Exception as e:
        for ax in axes:
            ax.text(0.5, 0.5, f'No E_co data\n{e}', ha='center', va='center', transform=ax.transAxes)
    plt.tight_layout()
    save_pdf(f"{out_dir}/fig5_eco_threshold_sensitivity_optional.pdf", fig)


def fig6_leakage_audit_summary(out_dir, dataset, fold, tab_dir):
    """Fig 6 (bonus): Leakage audit pass/fail heatmap across checks."""
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle(f"Fig 6: Leakage Audit Summary — {dataset} fold {fold}", fontsize=13, fontweight='bold')
    try:
        df_audit = pd.read_csv(f"{tab_dir}/leakage_audit_log.csv")
        checks = df_audit['check_name'].tolist()
        statuses = df_audit['status'].tolist()
        colors = ['#55A868' if s == 'PASS' else '#C44E52' for s in statuses]
        bars = ax.barh(checks, [1] * len(checks), color=colors, edgecolor='black', height=0.5)
        for bar, status in zip(bars, statuses):
            ax.text(0.5, bar.get_y() + bar.get_height()/2., status,
                   ha='center', va='center', fontweight='bold', color='white', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_title('L1–L5 Leakage Checks')
        ax.invert_yaxis()
    except Exception as e:
        ax.text(0.5, 0.5, f'No audit data\n{e}', ha='center', va='center', transform=ax.transAxes)
    plt.tight_layout()
    save_pdf(f"{out_dir}/fig6_leakage_audit_summary.pdf", fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = cfg['dataset']['name']

    out_dir = f"results/figures/{dataset}/fold_{args.fold}"
    tab_dir = f"results/tables/{dataset}/fold_{args.fold}"
    ensure_dir(out_dir)

    print(f"\n[make_figures] Generating all figures for {dataset} fold {args.fold}...")
    fig1_pipeline_overview(out_dir, dataset, args.fold, tab_dir)
    fig2_eco_weight_distribution(out_dir, dataset, args.fold, tab_dir)
    fig3_sparse_skill_strata(out_dir, dataset, args.fold, tab_dir)
    fig4_relation_ablation(out_dir, dataset, args.fold, tab_dir)
    fig5_eco_threshold_sensitivity(out_dir, dataset, args.fold, tab_dir)
    fig6_leakage_audit_summary(out_dir, dataset, args.fold, tab_dir)
    print(f"[make_figures] All 6 figures for {dataset} generated successfully.\n")
