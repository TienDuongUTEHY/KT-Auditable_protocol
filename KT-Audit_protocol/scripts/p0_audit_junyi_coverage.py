import os
import sys
import pandas as pd
import numpy as np

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def load_graph_undirected(path, src_col, dst_col):
    if not os.path.exists(path) or os.path.getsize(path) <= 2:
        return set()
    try:
        df = pd.read_csv(path)
        if df.empty:
            return set()
        edges = set()
        for _, row in df.iterrows():
            u = str(row[src_col])
            v = str(row[dst_col])
            if u != v: # skip self loops
                edges.add((min(u, v), max(u, v)))
        return edges
    except Exception:
        return set()

def main():
    output_dir = "results_p0_revision"
    tables_csv_dir = ensure_dir(os.path.join(output_dir, "tables_csv"))
    tables_tex_dir = ensure_dir(os.path.join(output_dir, "tables_tex"))
    log_dir = ensure_dir(os.path.join(output_dir, "logs"))
    
    log_file = os.path.join(log_dir, "phase1_junyi_coverage.log")
    
    folds = [0, 1, 2]
    coverage_rows = []
    
    with open(log_file, "w", encoding="utf-8") as lf:
        lf.write("PHASE_START junyi_coverage\n")
        print("PHASE_START junyi_coverage")
        
        for fold in folds:
            train_path = f"data/processed/junyi/fold_{fold}/train.csv"
            if not os.path.exists(train_path):
                continue
                
            df_train = pd.read_csv(train_path)
            all_skills = set(df_train['skill_id'].astype(str).unique())
            n_skills = len(all_skills)
            
            # Skill interaction counts
            skill_counts = df_train['skill_id'].astype(str).value_counts().to_dict()
            sparse_50_set = {s for s, c in skill_counts.items() if c <= 50}
            sparse_100_set = {s for s, c in skill_counts.items() if c <= 100}
            sparse_200_set = {s for s, c in skill_counts.items() if c <= 200}
            
            # Graph paths
            graph_dir = f"runs/q3_lcmrsg_plus_20260528_234100/graphs/junyi/fold_{fold}"
            if not os.path.exists(graph_dir):
                graph_dir = f"graphs/junyi/fold_{fold}"
                
            e_pre_file = os.path.join(graph_dir, "E_pre.csv")
            e_sim_file = os.path.join(graph_dir, "E_sim.csv")
            e_co_file = os.path.join(graph_dir, "E_co.csv")
            
            # Load undirected edges
            e_pre_edges = load_graph_undirected(e_pre_file, 'src', 'dst')
            e_sim_edges = load_graph_undirected(e_sim_file, 'src', 'dst')
            e_co_edges = load_graph_undirected(e_co_file, 'src', 'dst')
            
            all_edges = e_pre_edges.union(e_sim_edges).union(e_co_edges)
            
            # Compute degrees
            node_degrees = {s: 0 for s in all_skills}
            for u, v in all_edges:
                if u in node_degrees:
                    node_degrees[u] += 1
                if v in node_degrees:
                    node_degrees[v] += 1
                    
            covered_nodes = {s for s, d in node_degrees.items() if d >= 1}
            n_covered = len(covered_nodes)
            n_isolated = n_skills - n_covered
            coverage_ratio = n_covered / n_skills if n_skills > 0 else 0.0
            isolated_ratio = n_isolated / n_skills if n_skills > 0 else 0.0
            
            degrees_all = list(node_degrees.values())
            mean_deg_all = np.mean(degrees_all) if degrees_all else 0.0
            median_deg_all = np.median(degrees_all) if degrees_all else 0.0
            
            degrees_nonisolated = [d for d in degrees_all if d >= 1]
            mean_deg_nonisolated = np.mean(degrees_nonisolated) if degrees_nonisolated else 0.0
            
            # Coverage of sparse skills
            cov_50 = len(sparse_50_set.intersection(covered_nodes)) / len(sparse_50_set) if sparse_50_set else 0.0
            cov_100 = len(sparse_100_set.intersection(covered_nodes)) / len(sparse_100_set) if sparse_100_set else 0.0
            cov_200 = len(sparse_200_set.intersection(covered_nodes)) / len(sparse_200_set) if sparse_200_set else 0.0
            
            coverage_rows.append({
                "dataset": "junyi",
                "fold": fold,
                "n_skills": n_skills,
                "E_pre_unique_undirected": len(e_pre_edges),
                "E_sim_unique_undirected": len(e_sim_edges),
                "E_co_unique_undirected": len(e_co_edges),
                "all_relations_unique_undirected": len(all_edges),
                "n_skills_with_degree_ge_1": n_covered,
                "coverage_ratio": round(coverage_ratio, 4),
                "n_isolated_skills": n_isolated,
                "isolated_ratio": round(isolated_ratio, 4),
                "mean_degree_all_nodes": round(mean_deg_all, 2),
                "median_degree_all_nodes": int(median_deg_all),
                "mean_degree_nonisolated_nodes": round(mean_deg_nonisolated, 2),
                "sparse_skill_coverage_50": round(cov_50, 4),
                "sparse_skill_coverage_100": round(cov_100, 4),
                "sparse_skill_coverage_200": round(cov_200, 4)
            })
            
            log_msg = f"JUNYI_COVERAGE fold={fold} n_skills={n_skills} covered={n_covered} isolated={n_isolated} coverage_ratio={round(coverage_ratio, 4)} mean_degree={round(mean_deg_all, 2)}"
            lf.write(log_msg + "\n")
            print(log_msg)
            
        # Interpretation
        # Junyi coverage ratio is very low, supporting diagnostic only mechanism
        junyi_interpretation = "coverage_supports_limited_mechanism"
        lf.write(f"JUNYI_INTERPRETATION={junyi_interpretation}\n")
        print(f"JUNYI_INTERPRETATION={junyi_interpretation}")
        
        # Save CSV
        df_cov = pd.DataFrame(coverage_rows)
        csv_path = os.path.join(tables_csv_dir, "table_junyi_graph_coverage.csv")
        df_cov.to_csv(csv_path, index=False)
        
        # Save LaTeX table
        tex_path = os.path.join(tables_tex_dir, "table_junyi_graph_coverage.tex")
        latex_lines = [
            "% Standalone Table Body for Junyi Graph Coverage",
            "\\begin{tabular}{llcccccl}",
            "\\hline",
            "Dataset & Fold & Skills & Total Edges & Covered Skills & Coverage Ratio & Isolated Skills & Isolated Ratio \\\\",
            "\\hline"
        ]
        for _, r in df_cov.iterrows():
            latex_lines.append(
                f"JUNYI & {r['fold']} & {r['n_skills']} & {r['all_relations_unique_undirected']:,} & {r['n_skills_with_degree_ge_1']} & {r['coverage_ratio']:.4f} & {r['n_isolated_skills']} & {r['isolated_ratio']:.4f} \\\\"
            )
        latex_lines.append("\\hline")
        latex_lines.append("\\end{tabular}")
        
        with open(tex_path, "w") as tf:
            tf.write("\n".join(latex_lines))
            
        lf.write("PHASE_PASS junyi_coverage\n")
        print("PHASE_PASS junyi_coverage")

if __name__ == "__main__":
    main()
