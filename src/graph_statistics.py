"""
Ý NGHĨA TIẾN TRÌNH:
Tính toán và lưu trữ các thống kê, thuộc tính của các đồ thị đã xây dựng.
"""

import argparse
import pandas as pd
import networkx as nx
from src.common import load_config, ensure_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    dataset = cfg['dataset']['name']

    rep_dir = f"results/reports/{dataset}/fold_{args.fold}"
    tab_dir = f"results/tables/{dataset}/fold_{args.fold}"
    ensure_dir(rep_dir)
    ensure_dir(tab_dir)

    rows = []
    for edge_type, directed in [("E_pre_train_pruned", True), ("E_sim_train", False), ("E_co_train", False)]:
        csv_path = f"{tab_dir}/{edge_type}.csv"
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                raise ValueError("empty")
            if directed:
                G = nx.DiGraph()
            else:
                G = nx.Graph()
            for _, row in df.iterrows():
                G.add_edge(row['src_skill_id'], row['dst_skill_id'],
                           weight=row.get('weight', 1.0))
            nodes = G.number_of_nodes()
            edges = G.number_of_edges()
            density = nx.density(G)
            avg_deg = sum(dict(G.degree()).values()) / nodes if nodes > 0 else 0
            
            if directed:
                components = nx.number_weakly_connected_components(G)
            else:
                components = nx.number_connected_components(G)
                
            weights = df['weight'].dropna() if 'weight' in df.columns else pd.Series([1.0] * edges)
            rows.append({
                "dataset": dataset,
                "fold_id": args.fold,
                "edge_type": edge_type.split("_train")[0],
                "directed": directed,
                "num_nodes": nodes,
                "num_edges": edges,
                "density": round(density, 6),
                "avg_degree": round(avg_deg, 4),
                "components": components,
                "weight_min": round(weights.min(), 4),
                "weight_mean": round(weights.mean(), 4),
                "weight_max": round(weights.max(), 4),
                "weight_std": round(weights.std(), 4),
            })
            print(f"  [graph_stats] {edge_type}: nodes={nodes}, edges={edges}, density={density:.4f}")
        except Exception as e:
            print(f"  [graph_stats] {edge_type}: skipped ({e})")
            rows.append({"dataset": dataset, "fold_id": args.fold, "edge_type": edge_type, "num_edges": 0})

    df_out = pd.DataFrame(rows)
    df_out.to_csv(f"{tab_dir}/graph_stats.csv", index=False)

    # Markdown report
    md = f"# Graph Statistics — {dataset} fold {args.fold}\n\n"
    md += df_out.to_markdown(index=False) if not df_out.empty else "No data.\n"
    with open(f"{rep_dir}/graph_stats.md", "w") as f:
        f.write(md)
    print(f"Graph statistics for {dataset} saved.")
