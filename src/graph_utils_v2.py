"""
graph_utils_v2.py
Build real E_sim via top-K cosine similarity; load adjacency; GCN aggregation.
Memory-safe for KDD2010 (1.28M rows).
"""
import os, gc
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity as cos_sim


def build_esim_topk(train_df, unique_skills, skill_to_id, k=5, max_learners=3000):
    """Compute top-K cosine-similarity edges from skill response profiles."""
    n = len(unique_skills)
    skill_list = list(unique_skills)

    # Sample learners to cap RAM usage
    all_learners = train_df['learner_id'].unique()
    if len(all_learners) > max_learners:
        rng = np.random.default_rng(42)
        sampled = rng.choice(all_learners, max_learners, replace=False)
    else:
        sampled = all_learners
    sampled_set = set(sampled.tolist())
    l2i = {l: i for i, l in enumerate(sampled)}

    sub = train_df[train_df['learner_id'].isin(sampled_set)]
    R = np.zeros((n, len(sampled)), dtype=np.float32)
    C = np.zeros((n, len(sampled)), dtype=np.float32)
    for row in sub.itertuples(index=False):
        si = skill_to_id.get(row.skill_id)
        li = l2i.get(row.learner_id)
        if si is not None and li is not None:
            R[si, li] += row.correct
            C[si, li] += 1
    C[C == 0] = 1
    R = R / C
    del C, sub; gc.collect()

    edges = []
    chunk = 100
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        S = cos_sim(R[i0:i1], R)          # (chunk, n)
        for ci in range(i1 - i0):
            gi = i0 + ci
            S[ci, gi] = -1.0              # zero self
            topk = np.argsort(S[ci])[-k:]
            for j in topk:
                w = float(S[ci, j])
                if w > 0.05:
                    edges.append((skill_list[gi], skill_list[j], w))
        del S; gc.collect()

    del R; gc.collect()
    return edges   # list of (src_skill_id, dst_skill_id, weight)


def save_esim(edges, tab_dir, dataset, fold):
    """Persist E_sim to E_sim_train.csv (overwrites placeholder)."""
    fpath = os.path.join(tab_dir, 'E_sim_train.csv')
    rows = []
    for src, dst, w in edges:
        rows.append({'src_skill_id': src, 'dst_skill_id': dst,
                     'weight': round(w, 6), 'relation_type': 'E_sim',
                     'dataset': dataset, 'fold_id': fold})
    pd.DataFrame(rows).to_csv(fpath, index=False)
    return len(rows)


def load_adjacency(tab_dir, variant, unique_skills, skill_to_id):
    """
    Load edge files for a graph variant.
    Returns: adj (scipy sparse, n x n, symmetric), degree_map (str→float, normalized)
    """
    n = len(unique_skills)
    skill_list = list(unique_skills)

    files = []
    if variant in ('E_pre', 'E_pre_E_sim', 'E_pre_E_sim_E_co'):
        files.append('E_pre_train.csv')
    if variant in ('E_pre_E_sim', 'E_pre_E_sim_E_co'):
        files.append('E_sim_train.csv')
    if variant == 'E_pre_E_sim_E_co':
        files.append('E_co_train.csv')

    rows_i, cols_i, data_v = [], [], []
    deg = {str(sk): 0 for sk in skill_list}

    for fname in files:
        fpath = os.path.join(tab_dir, fname)
        if not os.path.exists(fpath):
            continue
        try:
            df = pd.read_csv(fpath)
            if df.empty:
                continue
            for row in df.itertuples(index=False):
                src = getattr(row, 'src_skill_id', None)
                dst = getattr(row, 'dst_skill_id', None)
                if src not in skill_to_id or dst not in skill_to_id:
                    continue
                i, j = skill_to_id[src], skill_to_id[dst]
                w = float(getattr(row, 'weight', 1.0) or 1.0)
                rows_i += [i, j]; cols_i += [j, i]; data_v += [w, w]
                deg[str(src)] = deg.get(str(src), 0) + 1
                deg[str(dst)] = deg.get(str(dst), 0) + 1
        except Exception:
            continue

    if rows_i:
        adj = sp.csr_matrix((data_v, (rows_i, cols_i)), shape=(n, n), dtype=np.float32)
    else:
        adj = sp.csr_matrix((n, n), dtype=np.float32)

    max_d = max(deg.values()) if deg else 1
    if max_d == 0: max_d = 1
    degree_map = {k: v / max_d for k, v in deg.items()}
    return adj, degree_map


def gcn_aggregate(embs, adj, n_rounds=1):
    """D^{-1}A aggregation in-place. embs: ndarray (n_skills, dim)."""
    row_sums = np.array(adj.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    D_inv = sp.diags(1.0 / row_sums)
    A_norm = D_inv.dot(adj)
    for _ in range(n_rounds):
        embs = 0.5 * embs + 0.5 * A_norm.dot(embs)
    return embs


def two_hop_degree(adj, degree_map, unique_skills):
    """Average degree of 1-hop neighbors for each skill."""
    deg_arr = np.array([degree_map.get(str(sk), 0.0) for sk in unique_skills], dtype=np.float32)
    nbr_deg = np.array(adj.dot(deg_arr)).flatten()
    cnt = np.array(adj.sum(axis=1)).flatten()
    cnt[cnt == 0] = 1.0
    return {str(sk): float(nbr_deg[i] / cnt[i]) for i, sk in enumerate(unique_skills)}
