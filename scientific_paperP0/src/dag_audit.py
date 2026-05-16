"""
Ý NGHĨA TIẾN TRÌNH:
Kiểm toán và cắt tỉa (prune) các chu trình trong đồ thị E_pre để đảm bảo tính DAG.
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
    
    out_dir_rep = f"results/reports/{dataset}/fold_{args.fold}"
    out_dir_tab = f"results/tables/{dataset}/fold_{args.fold}"
    ensure_dir(out_dir_rep)
    ensure_dir(out_dir_tab)
    
    df_pre = pd.read_csv(f"{out_dir_tab}/E_pre_train.csv")
    
    G = nx.DiGraph()
    for _, row in df_pre.iterrows():
        G.add_edge(row['src_skill_id'], row['dst_skill_id'], weight=row['weight'])
        
    cycles_before = list(nx.simple_cycles(G))
    num_cycles_before = len(cycles_before)
    
    removed_edges = []
    while True:
        try:
            cycle = nx.find_cycle(G, orientation="original")
            min_edge = min(cycle, key=lambda e: G[e[0]][e[1]]['weight'])
            G.remove_edge(min_edge[0], min_edge[1])
            removed_edges.append(min_edge)
        except nx.NetworkXNoCycle:
            break
            
    cycles_after = list(nx.simple_cycles(G))
    num_cycles_after = len(cycles_after)
    
    pruned_df = nx.to_pandas_edgelist(G, source='src_skill_id', target='dst_skill_id')
    merged_df = pd.merge(df_pre, pruned_df, on=['src_skill_id', 'dst_skill_id']) if not pruned_df.empty else pd.DataFrame(columns=df_pre.columns)
    merged_df.to_csv(f"{out_dir_tab}/E_pre_train_pruned.csv", index=False)
    
    rep = f"# DAG Audit\\nCycles before: {num_cycles_before}\\nCycles after: {num_cycles_after}\\nRemoved edges: {len(removed_edges)}\\n"
    with open(f"{out_dir_rep}/dag_report.md", "w") as f: f.write(rep)
    
    audit_df = pd.DataFrame([{"dataset": dataset, "cycles_before": num_cycles_before, "cycles_after": num_cycles_after}])
    audit_df.to_csv(f"{out_dir_tab}/dag_audit.csv", index=False)
    
    print(f"DAG audit for {dataset} completed.")
    print(f"  - Cycles before pruning: {num_cycles_before}")
    print(f"  - Edges removed: {len(removed_edges)}")
    print(f"  - Cycles after pruning: {num_cycles_after}")
