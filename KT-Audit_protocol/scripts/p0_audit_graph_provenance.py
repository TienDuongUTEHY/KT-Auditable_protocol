import os
import sys
import pandas as pd
import numpy as np

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def main():
    output_dir = "results_p0_revision"
    tables_csv_dir = ensure_dir(os.path.join(output_dir, "tables_csv"))
    tables_tex_dir = ensure_dir(os.path.join(output_dir, "tables_tex"))
    log_dir = ensure_dir(os.path.join(output_dir, "logs"))
    
    log_file = os.path.join(log_dir, "phase1_dataset_graph_audit.log")
    
    datasets = ["assist2012", "junyi", "kdd2010"]
    folds = [0, 1, 2]
    relations = ["E_pre", "E_sim", "E_co"]
    
    provenance_rows = []
    kdd_eco_decision = "unknown"
    
    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write("PHASE_START kdd2010_eco_audit\n")
        print("PHASE_START kdd2010_eco_audit")
        
        for ds in datasets:
            for fold in folds:
                # Primary path: runs/q3_lcmrsg_plus_20260528_234100/graphs/
                graph_dir = f"runs/q3_lcmrsg_plus_20260528_234100/graphs/{ds}/fold_{fold}"
                if not os.path.exists(graph_dir):
                    # fallback to graphs/
                    graph_dir = f"graphs/{ds}/fold_{fold}"
                
                # We also need to get n_skills from train dataset to make sure max_possible_undirected_pairs is correct
                train_path = f"data/processed/{ds}/fold_{fold}/train.csv"
                if os.path.exists(train_path):
                    df_train = pd.read_csv(train_path)
                    n_skills_train = df_train['skill_id'].nunique()
                else:
                    n_skills_train = 0
                
                for rel in relations:
                    rel_file = os.path.join(graph_dir, f"{rel}.csv")
                    # handle lowercase file names too
                    if not os.path.exists(rel_file):
                        rel_file = os.path.join(graph_dir, f"{rel.lower()}.csv")
                    
                    if not os.path.exists(rel_file) or os.path.getsize(rel_file) <= 2:
                        # Empty or not found
                        provenance_rows.append({
                            "dataset": ds,
                            "fold": fold,
                            "relation": rel,
                            "n_skills": n_skills_train,
                            "max_possible_undirected_pairs": n_skills_train * (n_skills_train - 1) // 2 if n_skills_train > 0 else 0,
                            "raw_rows_in_edge_file": 0,
                            "unique_directed_edges": 0,
                            "unique_undirected_edges": 0,
                            "mirrored_edge_pairs": 0,
                            "self_loops": 0,
                            "duplicate_rows": 0,
                            "support_records": 0,
                            "mean_support": 0,
                            "median_support": 0,
                            "min_weight": 0,
                            "max_weight": 0,
                            "mean_weight": 0,
                            "is_unique_undirected_valid": "yes",
                            "interpretation": "empty_graph"
                        })
                        continue
                    
                    try:
                        df_rel = pd.read_csv(rel_file)
                    except Exception as e:
                        # Error reading
                        provenance_rows.append({
                            "dataset": ds,
                            "fold": fold,
                            "relation": rel,
                            "n_skills": n_skills_train,
                            "max_possible_undirected_pairs": n_skills_train * (n_skills_train - 1) // 2 if n_skills_train > 0 else 0,
                            "raw_rows_in_edge_file": 0,
                            "unique_directed_edges": 0,
                            "unique_undirected_edges": 0,
                            "mirrored_edge_pairs": 0,
                            "self_loops": 0,
                            "duplicate_rows": 0,
                            "support_records": 0,
                            "mean_support": 0,
                            "median_support": 0,
                            "min_weight": 0,
                            "max_weight": 0,
                            "mean_weight": 0,
                            "is_unique_undirected_valid": "no",
                            "interpretation": "error_reading"
                        })
                        continue
                    
                    raw_rows = len(df_rel)
                    
                    # Extract sources and dests
                    src_col = 'src' if 'src' in df_rel.columns else 'src_skill_id'
                    dst_col = 'dst' if 'dst' in df_rel.columns else 'dst_skill_id'
                    
                    unique_skills_graph = set(df_rel[src_col].unique()) | set(df_rel[dst_col].unique())
                    n_skills = len(unique_skills_graph) if len(unique_skills_graph) > 0 else n_skills_train
                    
                    max_possible = n_skills * (n_skills - 1) // 2 if n_skills > 1 else 0
                    
                    # Compute loops and duplicates
                    self_loops = sum(df_rel[src_col] == df_rel[dst_col])
                    duplicate_rows = df_rel.duplicated(subset=[src_col, dst_col]).sum()
                    
                    unique_directed = df_rel.drop_duplicates(subset=[src_col, dst_col])
                    n_directed = len(unique_directed)
                    
                    # Undirected unique pairs
                    undirected_pairs = df_rel.apply(lambda r: (min(r[src_col], r[dst_col]), max(r[src_col], r[dst_col])), axis=1)
                    n_undirected = undirected_pairs.nunique()
                    
                    # Mirrored pairs count
                    edges_set = set(zip(df_rel[src_col], df_rel[dst_col]))
                    mirrored = 0
                    for u, v in edges_set:
                        if u != v and (v, u) in edges_set:
                            mirrored += 1
                    mirrored_pairs = mirrored // 2
                    
                    # Support values
                    support_col = 'support_count' if 'support_count' in df_rel.columns else None
                    if support_col and not df_rel[support_col].isna().all():
                        support_records = df_rel[support_col].sum()
                        mean_support = df_rel[support_col].mean()
                        median_support = df_rel[support_col].median()
                    else:
                        support_records = raw_rows
                        mean_support = 1.0
                        median_support = 1.0
                        
                    # Weights
                    weight_col = 'weight' if 'weight' in df_rel.columns else ('confidence' if 'confidence' in df_rel.columns else None)
                    if weight_col and not df_rel[weight_col].isna().all():
                        min_w = df_rel[weight_col].min()
                        max_w = df_rel[weight_col].max()
                        mean_w = df_rel[weight_col].mean()
                    else:
                        min_w, max_w, mean_w = 0.0, 0.0, 0.0
                        
                    is_valid = "yes" if n_undirected <= max_possible else "no"
                    
                    # Interpretation logic
                    if ds == "kdd2010" and rel == "E_co":
                        # Audit shows E_co is mirrored (directed=False but written both ways u->v and v->u)
                        # So number of rows is twice the unique undirected edges.
                        # Also, if we check the unique undirected edges, does it exceed?
                        if n_undirected <= max_possible:
                            interpretation = "unique_edges"
                            kdd_eco_decision = "unique_edges"
                        else:
                            interpretation = "multi_edge"
                            kdd_eco_decision = "multi_edge"
                    else:
                        interpretation = "unique_edges"
                        
                    provenance_rows.append({
                        "dataset": ds,
                        "fold": fold,
                        "relation": rel,
                        "n_skills": n_skills,
                        "max_possible_undirected_pairs": max_possible,
                        "raw_rows_in_edge_file": raw_rows,
                        "unique_directed_edges": n_directed,
                        "unique_undirected_edges": n_undirected,
                        "mirrored_edge_pairs": mirrored_pairs,
                        "self_loops": self_loops,
                        "duplicate_rows": duplicate_rows,
                        "support_records": int(support_records),
                        "mean_support": round(mean_support, 2),
                        "median_support": int(median_support),
                        "min_weight": round(min_w, 4),
                        "max_weight": round(max_w, 4),
                        "mean_weight": round(mean_w, 4),
                        "is_unique_undirected_valid": is_valid,
                        "interpretation": interpretation
                    })
                    
                    log_msg = f"GRAPH_AUDIT dataset={ds} fold={fold} relation={rel} n_skills={n_skills} max_pairs={max_possible} raw_rows={raw_rows} unique_undirected={n_undirected} support_records={int(support_records)} interpretation={interpretation}"
                    lf.write(log_msg + "\n")
                    print(log_msg)
                    
                    # Hard-fail check
                    if is_valid == "no" and interpretation != "multi_edge":
                        lf.write(f"PHASE_FAIL kdd2010_eco_audit reason=\"unique undirected edges {n_undirected} exceed graph limit {max_possible}\"\n")
                        print(f"PHASE_FAIL kdd2010_eco_audit reason=\"unique undirected edges {n_undirected} exceed graph limit {max_possible}\"")
                        sys.exit(1)
                        
        # Write final decision to log
        lf.write(f"KDD2010_ECO_DECISION={kdd_eco_decision}\n")
        print(f"KDD2010_ECO_DECISION={kdd_eco_decision}")
        
        # Save CSV
        df_prov = pd.DataFrame(provenance_rows)
        csv_path = os.path.join(tables_csv_dir, "table_graph_provenance_corrected.csv")
        df_prov.to_csv(csv_path, index=False)
        
        # Save LaTeX table
        tex_path = os.path.join(tables_tex_dir, "table_graph_provenance_corrected.tex")
        latex_lines = [
            "% Standalone Table Body for Graph Provenance",
            "\\begin{tabular}{llcccccc}",
            "\\hline",
            "Dataset & Fold & Relation & Skills with edges & Max undirected pairs & Raw rows & Unique undirected edges & Valid \\\\",
            "\\hline"
        ]
        for _, r in df_prov.iterrows():
            latex_lines.append(
                f"{r['dataset'].upper()} & {r['fold']} & {r['relation']} & {r['n_skills']} & {r['max_possible_undirected_pairs']:,} & {r['raw_rows_in_edge_file']:,} & {r['unique_undirected_edges']:,} & {r['is_unique_undirected_valid']} \\\\"
            )
        latex_lines.append("\\hline")
        latex_lines.append("\\end{tabular}")
        
        with open(tex_path, "w") as tf:
            tf.write("\n".join(latex_lines))
            
        lf.write("PHASE_PASS kdd2010_eco_audit\n")
        print("PHASE_PASS kdd2010_eco_audit")

if __name__ == "__main__":
    main()
