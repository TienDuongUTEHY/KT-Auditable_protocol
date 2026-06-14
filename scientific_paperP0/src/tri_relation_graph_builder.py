"""
Ý NGHĨA TIẾN TRÌNH:
Xây dựng 3 đồ thị quan hệ: Tiên quyết (E_pre), Tương đồng (E_sim) và Đồng xuất hiện (E_co) từ tập train.
"""

import argparse
import pandas as pd
import numpy as np
import datetime
from src.common import load_config, ensure_dir

def build_e_pre(df_train, out_dir, dataset, fold=0):
    skill_times = df_train.groupby('skill_id')['timestamp'].median().sort_values()
    edges = []
    skills = skill_times.index.tolist()
    for i in range(len(skills)):
        for j in range(i + 1, len(skills)):
            edges.append({
                'dataset': dataset, 'fold_id': fold,
                'src_skill_id': skills[i], 'dst_skill_id': skills[j],
                'relation_type': 'E_pre', 'directed': True,
                'weight': 1.0, 'support_count': 10,
                'source_split': 'train', 'source_fold': fold, 'support_source': 'train',
                'support_interaction_ids_hash': 'hash', 'support_question_ids_hash': 'hash',
                'construction_method': 'temporal_heuristic', 'threshold': 0, 'confidence': 0.5,
                'created_at': datetime.datetime.now().isoformat()
            })
    columns = ['dataset', 'fold_id', 'src_skill_id', 'dst_skill_id', 'relation_type', 'directed', 'weight', 'support_count', 'source_split', 'source_fold', 'support_source', 'support_interaction_ids_hash', 'support_question_ids_hash', 'construction_method', 'threshold', 'confidence', 'created_at']
    df_edges = pd.DataFrame(edges, columns=columns)
    df_edges.to_csv(f"{out_dir}/E_pre_train.csv", index=False)
    print(f"  - E_pre built: {len(df_edges)} edges")
    return df_edges

def build_e_sim(df_train, out_dir, dataset, fold=0, threshold=0.1):
    skill_q_map = df_train.groupby('skill_id')['question_id'].apply(set).to_dict()
    skills = list(skill_q_map.keys())
    edges = []
    for i in range(len(skills)):
        for j in range(i + 1, len(skills)):
            s1, s2 = skills[i], skills[j]
            q1, q2 = skill_q_map[s1], skill_q_map[s2]
            intersection = len(q1.intersection(q2))
            union = len(q1.union(q2))
            jaccard = intersection / union if union > 0 else 0
            if jaccard >= threshold:
                edges.append({
                    'dataset': dataset, 'fold_id': fold,
                    'src_skill_id': s1, 'dst_skill_id': s2,
                    'relation_type': 'E_sim', 'directed': False,
                    'weight': jaccard, 'support_count': intersection,
                    'source_split': 'train', 'source_fold': fold, 'support_source': 'train',
                    'support_interaction_ids_hash': 'hash', 'support_question_ids_hash': 'hash',
                    'construction_method': 'jaccard', 'threshold': threshold, 'confidence': 1.0,
                    'created_at': datetime.datetime.now().isoformat()
                })
    columns = ['dataset', 'fold_id', 'src_skill_id', 'dst_skill_id', 'relation_type', 'directed', 'weight', 'support_count', 'source_split', 'source_fold', 'support_source', 'support_interaction_ids_hash', 'support_question_ids_hash', 'construction_method', 'threshold', 'confidence', 'created_at']
    df_edges = pd.DataFrame(edges, columns=columns)
    df_edges.to_csv(f"{out_dir}/E_sim_train.csv", index=False)
    print(f"  - E_sim built: {len(df_edges)} edges (threshold={threshold})")
    return df_edges

def build_e_co(df_train, out_dir, dataset, fold=0, min_count=2):
    learner_skills = df_train.groupby('learner_id')['skill_id'].apply(list).to_dict()
    co_counts = {}
    skill_counts = {}
    
    for skills in learner_skills.values():
        unique_skills = set(skills)
        for s in unique_skills:
            skill_counts[s] = skill_counts.get(s, 0) + 1
        for s1 in unique_skills:
            for s2 in unique_skills:
                if s1 < s2:
                    pair = (s1, s2)
                    co_counts[pair] = co_counts.get(pair, 0) + 1
                    
    total_learners = len(learner_skills)
    edges = []
    for (s1, s2), count in co_counts.items():
        if count >= min_count:
            p_s1 = skill_counts[s1] / total_learners
            p_s2 = skill_counts[s2] / total_learners
            p_co = count / total_learners
            pmi = np.log(p_co / (p_s1 * p_s2))
            if pmi > 0:
                edges.append({
                    'dataset': dataset, 'fold_id': fold,
                    'src_skill_id': s1, 'dst_skill_id': s2,
                    'relation_type': 'E_co', 'directed': False,
                    'weight': pmi, 'support_count': count,
                    'source_split': 'train', 'source_fold': fold, 'support_source': 'train',
                    'support_interaction_ids_hash': 'hash', 'support_question_ids_hash': 'hash',
                    'construction_method': 'pmi', 'threshold': 0, 'confidence': 1.0,
                    'created_at': datetime.datetime.now().isoformat()
                })
                
    columns = ['dataset', 'fold_id', 'src_skill_id', 'dst_skill_id', 'relation_type', 'directed', 'weight', 'support_count', 'source_split', 'source_fold', 'support_source', 'support_interaction_ids_hash', 'support_question_ids_hash', 'construction_method', 'threshold', 'confidence', 'created_at']
    df_edges = pd.DataFrame(edges, columns=columns)
    df_edges.to_csv(f"{out_dir}/E_co_train.csv", index=False)
    print(f"  - E_co built: {len(df_edges)} edges (min_count={min_count})")
    return df_edges

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    dataset = cfg['dataset']['name']
    
    in_dir = f"data/processed/{dataset}/fold_{args.fold}"
    out_dir = f"results/tables/{dataset}/fold_{args.fold}"
    ensure_dir(out_dir)
    
    df_train = pd.read_csv(f"{in_dir}/train.csv")
    
    e_pre = build_e_pre(df_train, out_dir, dataset, fold=args.fold)
    e_sim = build_e_sim(df_train, out_dir, dataset, fold=args.fold)
    e_co = build_e_co(df_train, out_dir, dataset, fold=args.fold)
    
    columns = ['dataset', 'fold_id', 'src_skill_id', 'dst_skill_id', 'relation_type', 'directed', 'weight', 'support_count', 'source_split', 'source_fold', 'support_source', 'support_interaction_ids_hash', 'support_question_ids_hash', 'construction_method', 'threshold', 'confidence', 'created_at']
    prov = pd.concat([e_pre, e_sim, e_co]) if not e_pre.empty else pd.DataFrame(columns=columns)
    prov.to_csv(f"{out_dir}/edge_provenance.csv", index=False)
    print(f"Graph construction for {dataset} completed.")
