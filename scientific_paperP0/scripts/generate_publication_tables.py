import pandas as pd
import os
import numpy as np

RESULTS_DIR = "results/tables"
OUT_DIR = "results/paper_ready/final_tables"
os.makedirs(OUT_DIR, exist_ok=True)

def load_csv_safe(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

print("=== GENERATING FINAL PUBLICATION TABLES ===")

datasets = ["kdd2010", "assist2012", "junyi"]

# --- 1. COMPILE DATASET STATISTICS TABLE ---
stats_rows = []
for d in datasets:
    # Load basic stats or sparse skill summaries
    sparse_sum = load_csv_safe(f"{RESULTS_DIR}/{d}/fold_0/sparse_skill_summary.csv")
    if sparse_sum is not None:
        row = sparse_sum.iloc[0].to_dict()
        stats_rows.append({
            'Dataset': d.upper(),
            'Skills': row.get('total_skills'),
            'Interactions': int(row.get('total_skills') * row.get('avg_interactions_per_skill', 0)), # reconstruct roughly or use raw
            'Sparse Skills': row.get('sparse_skills'),
            'Medium Skills': row.get('medium_skills'),
            'Dense Skills': row.get('dense_skills'),
        })

df_t1 = pd.DataFrame(stats_rows)
df_t1.to_csv(f"{OUT_DIR}/table1_dataset_stats.csv", index=False)
print("Generated Table 1: Dataset Statistics")


# --- 2. COMPILE GRAPH ARCHITECTURE TABLE (E_pre, E_sim, E_co) ---
graph_rows = []
for d in datasets:
    gs = load_csv_safe(f"{RESULTS_DIR}/{d}/fold_0/graph_stats.csv")
    if gs is not None:
        for _, r in gs.iterrows():
            graph_rows.append({
                'Dataset': d.upper(),
                'Edge Type': r['edge_type'],
                'Nodes': r['num_nodes'],
                'Edges': r['num_edges'],
                'Density': round(r['density'], 4),
                'Avg Degree': round(r['avg_degree'], 2)
            })

df_t2 = pd.DataFrame(graph_rows)
df_t2.to_csv(f"{OUT_DIR}/table2_graph_comparison.csv", index=False)
print("Generated Table 2: Graph Comparison")


# --- 3. COMPILE PERFORMANCE BASELINES (MEAN +/- STD OVER 5 SEEDS) ---
perf_rows = []
for d in datasets:
    base_res = load_csv_safe(f"{RESULTS_DIR}/{d}/fold_0/baseline_results.csv")
    if base_res is not None:
        # Group by Model and calc mean, std
        stats = base_res.groupby('model')[['auc', 'acc']].agg(['mean', 'std']).reset_index()
        for idx, r in stats.iterrows():
            m_auc = r[('auc', 'mean')]
            s_auc = r[('auc', 'std')]
            m_acc = r[('acc', 'mean')]
            s_acc = r[('acc', 'std')]
            perf_rows.append({
                'Dataset': d.upper(),
                'Model': r['model'].iloc[0] if isinstance(r['model'], pd.Series) else r['model'],
                'AUC': f"{m_auc:.4f} ± {s_auc:.4f}",
                'ACC': f"{m_acc:.4f} ± {s_acc:.4f}"
            })

df_t3 = pd.DataFrame(perf_rows)
df_t3.to_csv(f"{OUT_DIR}/table3_baseline_metrics.csv", index=False)
print("Generated Table 3: Baseline Metrics Summary")

# --- GENERATE FINAL MARKDOWN DIGEST ---
md_report = "# Final Publication Ready Digest\n\n"
md_report += "## Table 1: Datasets Summary\n\n" + df_t1.to_markdown(index=False) + "\n\n"
md_report += "## Table 2: Graph Topology Snapshot\n\n" + df_t2.to_markdown(index=False) + "\n\n"
md_report += "## Table 3: Model Performance (5-Seed Summary)\n\n" + df_t3.to_markdown(index=False) + "\n\n"

with open(f"results/paper_ready/final_publication_tables.md", "w", encoding='utf-8') as f:
    f.write(md_report)

print("\nAll compilation successfully written to results/paper_ready/")
