"""
Ý NGHĨA TIẾN TRÌNH:
Đánh giá độ suy thoái của đồ thị E_pre (DAG Disruption Rate - DDR) thông qua các phép biến đổi đồ thị.
"""

import argparse
import logging
import random
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common import load_config, ensure_dir

logger = logging.getLogger(__name__)

def _rng(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)

def apply_node_drop(edges: pd.DataFrame, p: float, seed: int) -> pd.DataFrame:
    if edges.empty or p <= 0:
        return edges.copy()
    rng = _rng(seed)
    nodes = np.array(sorted(set(edges["src_skill_id"]) | set(edges["dst_skill_id"])))
    drop_nodes = set(nodes[rng.random(len(nodes)) < p])
    result = edges[~edges["src_skill_id"].isin(drop_nodes) & ~edges["dst_skill_id"].isin(drop_nodes)].reset_index(drop=True)
    return result

def apply_edge_drop(edges: pd.DataFrame, p: float, seed: int) -> pd.DataFrame:
    if edges.empty or p <= 0:
        return edges.copy()
    rng = _rng(seed)
    keep = rng.random(len(edges)) >= p
    result = edges.loc[keep].reset_index(drop=True)
    return result

def apply_attribute_mask(edges: pd.DataFrame, p: float, seed: int) -> pd.DataFrame:
    result = edges.copy()
    if result.empty or p <= 0:
        return result
    rng = _rng(seed)
    mask = rng.random(len(result)) < p
    for col in [c for c in result.columns if c not in {"src_skill_id", "dst_skill_id"}]:
        is_num = pd.api.types.is_numeric_dtype(result[col])
        result[col] = result[col].astype(object)
        if is_num:
            result.loc[mask, col] = np.nan
        else:
            result.loc[mask, col] = "masked"
    return result

def apply_subgraph_sampling(edges: pd.DataFrame, p: float, seed: int) -> pd.DataFrame:
    if edges.empty or p <= 0:
        return edges.copy()
    rng = _rng(seed)
    nodes = np.array(sorted(set(edges["src_skill_id"]) | set(edges["dst_skill_id"])))
    target_size = min(len(nodes), max(1, int(np.ceil((1.0 - p) * len(nodes)))))

    adjacency = {node: set() for node in nodes}
    for src, dst in edges[["src_skill_id", "dst_skill_id"]].itertuples(index=False, name=None):
        adjacency[src].add(dst)
        adjacency[dst].add(src)

    start = rng.choice(nodes)
    keep_nodes = {start}
    frontier = [start]
    while len(keep_nodes) < target_size:
        if not frontier:
            remaining = np.array([node for node in nodes if node not in keep_nodes])
            if len(remaining) == 0:
                break
            start = rng.choice(remaining)
            keep_nodes.add(start)
            frontier.append(start)
            continue
        current = rng.choice(np.array(frontier))
        candidates = np.array(sorted(adjacency[current] - keep_nodes))
        if len(candidates) == 0:
            frontier.remove(current)
            continue
        nxt = rng.choice(candidates)
        keep_nodes.add(nxt)
        frontier.append(nxt)

    result = edges[edges["src_skill_id"].isin(keep_nodes) & edges["dst_skill_id"].isin(keep_nodes)].reset_index(drop=True)
    return result

def compute_dag_disruption_rate(original: pd.DataFrame, augmented: pd.DataFrame) -> float:
    original_edges = set(original[["src_skill_id", "dst_skill_id"]].itertuples(index=False, name=None))
    if not original_edges:
        return 0.0
    augmented_edges = set(augmented[["src_skill_id", "dst_skill_id"]].itertuples(index=False, name=None))
    lost = original_edges - augmented_edges
    reversed_edges = {(src, dst) for src, dst in original_edges if (dst, src) in augmented_edges and (src, dst) not in augmented_edges}
    ddr = len(lost | reversed_edges) / len(original_edges)
    return ddr

def sweep_ddr(
    edges: pd.DataFrame,
    augmentations=("node_drop", "edge_drop", "attr_mask", "subgraph"),
    ps=(0.05, 0.10, 0.20, 0.30),
    seeds=(2026,)
) -> pd.DataFrame:
    fns = {
        "node_drop": apply_node_drop,
        "edge_drop": apply_edge_drop,
        "attr_mask": apply_attribute_mask,
        "subgraph": apply_subgraph_sampling,
    }
    rows = []
    for aug in augmentations:
        if aug not in fns:
            continue
        for p in ps:
            for seed in seeds:
                augmented = fns[aug](edges, float(p), int(seed))
                rows.append({"augmentation": aug, "p": float(p), "seed": int(seed), "ddr": compute_dag_disruption_rate(edges, augmented)})
    return pd.DataFrame(rows)

def _bootstrap_mean_ci(values, seed: int, n_bootstrap: int = 1000):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_bootstrap, len(arr)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    dataset = cfg['dataset']['name']
    
    tab_dir = f"results/tables/{dataset}/fold_{args.fold}"
    fig_dir = f"results/figures/{dataset}/fold_{args.fold}"
    ensure_dir(tab_dir)
    ensure_dir(fig_dir)
    
    edges_path = f"{tab_dir}/E_pre_train_pruned.csv"
    try:
        edges = pd.read_csv(edges_path)
    except FileNotFoundError:
        edges = pd.DataFrame(columns=["src_skill_id", "dst_skill_id", "weight"])
        print(f"Warning: E_pre graph not found at {edges_path}. DDR will be 0.")

    # Sweep settings
    augmentations = ("node_drop", "edge_drop", "attr_mask", "subgraph")
    ps = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
    seeds = (args.seed,)
    n_bootstrap = 1000

    result = sweep_ddr(edges, augmentations=augmentations, ps=ps, seeds=seeds)
    result.insert(0, "fold", args.fold)
    result.insert(0, "dataset", dataset)
    
    summary_rows = []
    for (augmentation, p), part in result.groupby(["augmentation", "p"]):
        ci_low, ci_high = _bootstrap_mean_ci(part["ddr"], seed=args.seed, n_bootstrap=n_bootstrap)
        summary_rows.append({
            "dataset": dataset,
            "fold": args.fold,
            "augmentation": augmentation,
            "p": p,
            "ddr_mean": float(part["ddr"].mean()),
            "ddr_std": float(part["ddr"].std(ddof=1)) if len(part) > 1 else 0.0,
            "ddr_ci_low": ci_low,
            "ddr_ci_high": ci_high,
            "n": len(part),
        })
        
    summary = pd.DataFrame(summary_rows)
    
    result.to_csv(f"{tab_dir}/dag_disruption.csv", index=False)
    summary.to_csv(f"{tab_dir}/dag_disruption_summary.csv", index=False)
    
    # Plotting
    fig_path = f"{fig_dir}/fig7_ddr_sweep.pdf"
    if not summary.empty:
        plt.figure(figsize=(8, 6))
        for aug, part in summary.groupby("augmentation"):
            part = part.sort_values("p")
            plt.plot(part["p"], part["ddr_mean"], marker="o", label=aug)
            # plt.fill_between(part["p"], part["ddr_ci_low"], part["ddr_ci_high"], alpha=0.2)
            
        plt.xlabel("Graph Disruption Probability (p)")
        plt.ylabel("Mean DAG Disruption Rate (DDR)")
        plt.title(f"DDR Sweep: {dataset} (Fold {args.fold})")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
        
    print(f"DAG Disruption Rate (DDR) experiment for {dataset} completed.")
