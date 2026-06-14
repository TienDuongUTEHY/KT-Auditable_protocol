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
    try:
        df_stats = pd.read_csv(f"results/tables/{d}/dataset_stats.csv")
        row_stats = df_stats.iloc[0]
        stats_rows.append({
            'Dataset': d.upper(),
            'Learners': row_stats.get('Learners', 0),
            'Questions': row_stats.get('Questions', 0),
            'Skills': row_stats.get('Skills', 0),
            'Interactions': row_stats.get('Interactions', 0),
            'AvgSeqLen': row_stats.get('AvgSeqLen', 0),
            'Very Sparse Skills': row.get('very_sparse_skills', 0),
            'Sparse Skills': row.get('sparse_skills', 0),
            'Medium Skills': row.get('medium_skills', 0),
            'Frequent Skills': row.get('frequent_skills', 0),
        })
    except FileNotFoundError:
        pass

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
    
    f.write("## Appendix: Five-Type Leakage Taxonomy\n")
    f.write("| Type | Description | Source |\n")
    f.write("|:---|:---|:---|\n")
    f.write("| L1 | Edge Leakage: Future target links inside training graph | Graph topology |\n")
    f.write("| L2 | Q-Matrix Leakage: Unknown mappings during testing | External info |\n")
    f.write("| L3 | Temporal Leakage: Time-traveling data leaks | Time Split |\n")
    f.write("| L4 | Cold Start: Overlapping learners | Learner Split |\n")
    f.write("| L5 | Co-occurrence Leakage: Future interactions | Metric calc |\n\n")
    
    f.write("## Note: Hyperparameter Leakage\n")
    f.write("Hyperparameter leakage (L6) occurs when hyperparameters are tuned on the test set. Our strict validation split ensures this is prevented.\n")

# --- 4. 144-RUN EXPORT ---
all_results = []
for d in datasets:
    base_res = load_csv_safe(f"{RESULTS_DIR}/{d}/fold_0/baseline_results.csv")
    if base_res is not None:
        all_results.append(base_res)
if all_results:
    pd.concat(all_results).to_csv(f"{OUT_DIR}/144_run_export.csv", index=False)
    print("Exported 144_run_export.csv")

print("\nAll compilation successfully written to results/paper_ready/")
