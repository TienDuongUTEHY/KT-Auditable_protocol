"""
models_v2.py  –  Five distinct model proxies for Option-A confirmatory run.

BKT  : Real per-skill HMM (Bayesian update + graph-adjusted prior)
DKT  : LSTM with graph degree feature (existing architecture, cleaned up)
simpleKT : 2-layer MLP on (skill_emb, correct, graph_degree) per interaction
GIKT : GCN-aggregated skill embeddings + LogisticRegression
SKT  : Rich graph-path features + GradientBoostingClassifier

All models:
  - Accept   (train_df, test_df, skill_to_id, adj, degree_map, two_hop_map, seed)
  - Return   (auc, acc, nll)
  - Are memory-safe (large datasets subsampled / batched)
"""

import gc, random
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from src.graph_utils_v2 import gcn_aggregate

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
_MAX_TRAIN_ROWS = 250_000      # hard cap to prevent OOM on KDD2010
_MAX_SEQ_LEN    = 100          # LSTM truncation

def _seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def _safe_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float).clip(1e-7, 1 - 1e-7)
    acc = float(accuracy_score(y_true, y_pred > 0.5))
    if len(np.unique(y_true)) < 2:
        return float('nan'), acc, float('nan')
    auc = float(roc_auc_score(y_true, y_pred))
    nll = float(log_loss(y_true, y_pred))
    return round(auc, 4), round(acc, 4), round(nll, 4)

def _subsample(df, max_rows, seed):
    if len(df) > max_rows:
        return df.sample(max_rows, random_state=seed)
    return df

def _degree_seq(df, degree_map):
    return df['skill_id'].astype(str).map(degree_map).fillna(0.0).values.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 1. BKT — Real per-skill Bayesian Knowledge Tracing
# ══════════════════════════════════════════════════════════════════════════════
def _bkt_forward(seq, L0, T, G, S):
    """Return list of P(correct) predictions given obs sequence."""
    preds = []
    L = L0
    for obs in seq:
        p_correct = L * (1 - S) + (1 - L) * G
        preds.append(p_correct)
        if obs == 1:
            L = (L * (1 - S)) / (L * (1 - S) + (1 - L) * G + 1e-9)
        else:
            L = (L * S) / (L * S + (1 - L) * (1 - G) + 1e-9)
        L = L + (1 - L) * T
    return preds


def _fit_bkt_skill(seqs, n_iter=10):
    """Simple EM-free grid param estimation using moment matching."""
    all_obs = [o for s in seqs for o in s]
    if not all_obs:
        return 0.3, 0.1, 0.2, 0.1
    p_correct = np.mean(all_obs)
    G = max(0.05, min(0.4, p_correct * 0.5))
    S = max(0.05, min(0.3, (1 - p_correct) * 0.3))
    L0 = max(0.05, min(0.6, p_correct - G))
    T  = max(0.01, min(0.3, 0.05 + 0.1 * p_correct))
    return L0, T, G, S


def run_bkt(train_df, test_df, skill_to_id, adj, degree_map, two_hop_map, seed):
    _seed_everything(seed)
    train_df = _subsample(train_df, _MAX_TRAIN_ROWS, seed)

    # Build per-skill sequences from train
    skill_seqs = {}
    for _, row in train_df.iterrows():
        sk = row['skill_id']
        skill_seqs.setdefault(sk, []).append(int(row['correct']))

    params = {}
    for sk, seqs_flat in skill_seqs.items():
        deg = degree_map.get(str(sk), 0.0)
        L0, T, G, S = _fit_bkt_skill([seqs_flat])
        # graph-augmented prior: higher degree → higher starting knowledge
        L0 = min(0.85, L0 * (1 + 0.3 * deg))
        params[sk] = (L0, T, G, S)

    # Predict on test using learner × skill running state
    all_preds, all_labels = [], []
    learner_state = {}   # (learner, skill) → L

    tdf = test_df.copy()
    if 'timestamp' in tdf.columns:
        tdf = tdf.sort_values('timestamp')

    for _, row in tdf.iterrows():
        sk = row['skill_id']
        obs = int(row['correct'])
        L0, T, G, S = params.get(sk, (0.3, 0.1, 0.2, 0.1))
        key = (row['learner_id'], sk)
        L = learner_state.get(key, L0)

        p = L * (1 - S) + (1 - L) * G
        all_preds.append(p)
        all_labels.append(obs)

        if obs == 1:
            L = (L * (1 - S)) / (L * (1 - S) + (1 - L) * G + 1e-9)
        else:
            L = (L * S) / (L * S + (1 - L) * (1 - G) + 1e-9)
        learner_state[key] = L + (1 - L) * T

    return _safe_metrics(all_labels, all_preds)


# ══════════════════════════════════════════════════════════════════════════════
# 2. DKT — LSTM with graph-degree feature
# ══════════════════════════════════════════════════════════════════════════════
class _DKTNet(nn.Module):
    def __init__(self, n_skills, embed_dim=64, hidden=64):
        super().__init__()
        self.n = n_skills
        self.emb = nn.Embedding(2 * n_skills + 1, embed_dim)
        self.lstm = nn.LSTM(embed_dim + 1, hidden, batch_first=True)
        self.fc   = nn.Linear(hidden, n_skills)
    def forward(self, x, d):
        e = self.emb(x)
        h, _ = self.lstm(torch.cat([e, d.unsqueeze(-1)], -1))
        return torch.sigmoid(self.fc(h))


def _dkt_collate(batch):
    x, d, ys, yl = zip(*batch)
    return (pad_sequence(x, True, 0),
            pad_sequence(d, True, 0.0),
            pad_sequence(ys, True, 0),
            pad_sequence(yl, True, -1.0))


def _dkt_prepare(df, skill_to_id, degree_map, n_skills):
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
    xs, ds, ysk, ylb = [], [], [], []
    for _, grp in df.groupby('learner_id'):
        sk = grp['skill_id'].values[-_MAX_SEQ_LEN:]
        cr = grp['correct'].values[-_MAX_SEQ_LEN:]
        if len(sk) < 2:
            continue
        sid = np.array([skill_to_id[s] for s in sk])
        deg = np.array([degree_map.get(str(s), 0.0) for s in sk], dtype=np.float32)
        inp = sid[:-1] + cr[:-1] * n_skills
        xs.append(torch.LongTensor(inp))
        ds.append(torch.FloatTensor(deg[:-1]))
        ysk.append(torch.LongTensor(sid[1:]))
        ylb.append(torch.FloatTensor(cr[1:].astype(float)))
    return xs, ds, ysk, ylb


def run_dkt(train_df, test_df, skill_to_id, adj, degree_map, two_hop_map, seed):
    _seed_everything(seed)
    train_df = _subsample(train_df, _MAX_TRAIN_ROWS, seed)
    n = len(skill_to_id)
    n_epochs = 3 if len(train_df) < 300_000 else 2
    batch = 32

    tr = _dkt_prepare(train_df, skill_to_id, degree_map, n)
    te = _dkt_prepare(test_df,  skill_to_id, degree_map, n)
    if not tr[0] or not te[0]:
        return float('nan'), float('nan'), float('nan')

    model = _DKTNet(n)
    opt   = optim.Adam(model.parameters(), lr=5e-3)
    crit  = nn.BCELoss()

    model.train()
    items = list(zip(*tr))
    for _ in range(n_epochs):
        np.random.shuffle(items)
        for i0 in range(0, len(items), batch):
            xb, db, skb, lb = _dkt_collate(items[i0:i0+batch])
            opt.zero_grad()
            out  = model(xb, db)
            pred = out.gather(2, skb.unsqueeze(-1)).squeeze(-1)
            mask = lb != -1.0
            if mask.sum() == 0: continue
            loss = crit(pred[mask], lb[mask])
            loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        xb, db, skb, lb = _dkt_collate(list(zip(*te)))
        out  = model(xb, db)
        pred = out.gather(2, skb.unsqueeze(-1)).squeeze(-1)
        mask = lb != -1.0
        yp = pred[mask].numpy()
        yt = lb[mask].numpy()

    del model; gc.collect()
    return _safe_metrics(yt, yp)


# ══════════════════════════════════════════════════════════════════════════════
# 3. simpleKT — 2-layer MLP (no sequence memory, graph-augmented embedding)
# ══════════════════════════════════════════════════════════════════════════════
class _SimpleKTNet(nn.Module):
    def __init__(self, n_skills, embed_dim=32, hidden=64):
        super().__init__()
        self.n = n_skills
        self.emb_q  = nn.Embedding(n_skills + 1, embed_dim)
        self.emb_ka = nn.Embedding(2 * n_skills + 1, embed_dim)  # ka in [0, 2n]
        self.fc1 = nn.Linear(embed_dim * 2 + 1, hidden)
        self.fc2 = nn.Linear(hidden, 1)
    def forward(self, q, ka, deg):
        q  = q.clamp(0, self.n)
        ka = ka.clamp(0, 2 * self.n)
        x = torch.cat([self.emb_q(q), self.emb_ka(ka), deg.unsqueeze(-1)], -1)
        return torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))


def run_simplekt(train_df, test_df, skill_to_id, adj, degree_map, two_hop_map, seed):
    _seed_everything(seed)
    train_df = _subsample(train_df, _MAX_TRAIN_ROWS, seed)
    n = len(skill_to_id)

    def _make_tensors(df):
        """Per-interaction tensors; ka uses PREVIOUS step to avoid leakage."""
        df = df.reset_index(drop=True)
        sk_arr  = [skill_to_id.get(s, 0) for s in df['skill_id']]
        cr_arr  = df['correct'].astype(int).values
        deg_arr = _degree_seq(df, degree_map)
        # knowledge-action: shifted — ka[i] = skill[i-1] + correct[i-1]*n
        ka_arr  = [0] + [sk_arr[i-1] + int(cr_arr[i-1]) * n for i in range(1, len(sk_arr))]
        ka_arr  = [min(k, 2 * n) for k in ka_arr]  # clamp to embedding size
        sk  = torch.LongTensor(sk_arr)
        ka  = torch.LongTensor(ka_arr)
        deg = torch.FloatTensor(deg_arr)
        lbl = torch.FloatTensor(cr_arr.astype(float))
        return sk, ka, deg, lbl

    model = _SimpleKTNet(n)
    opt   = optim.Adam(model.parameters(), lr=1e-2)
    crit  = nn.BCELoss()
    batch = 512
    n_ep  = 4

    tr_sk, tr_ka, tr_deg, tr_lbl = _make_tensors(train_df)
    model.train()
    idx = torch.randperm(len(tr_sk))
    for _ in range(n_ep):
        idx = idx[torch.randperm(len(idx))]
        for i0 in range(0, len(idx), batch):
            ib = idx[i0:i0+batch]
            opt.zero_grad()
            p = model(tr_sk[ib], tr_ka[ib], tr_deg[ib]).squeeze(-1)
            crit(p, tr_lbl[ib]).backward(); opt.step()

    model.eval()
    with torch.no_grad():
        te_sk, te_ka, te_deg, te_lbl = _make_tensors(test_df)
        preds = model(te_sk, te_ka, te_deg).squeeze(-1).numpy()

    del model; gc.collect()
    return _safe_metrics(te_lbl.numpy(), preds)


# ══════════════════════════════════════════════════════════════════════════════
# 4. GIKT — GCN-aggregated skill embs + LogisticRegression
# ══════════════════════════════════════════════════════════════════════════════
def run_gikt(train_df, test_df, skill_to_id, adj, degree_map, two_hop_map, seed):
    _seed_everything(seed)
    train_df = _subsample(train_df, _MAX_TRAIN_ROWS, seed)
    n = len(skill_to_id)
    rng = np.random.default_rng(seed)

    # Initialize + GCN-aggregate skill embeddings
    emb_dim = 16
    raw_embs = rng.standard_normal((n, emb_dim)).astype(np.float32) * 0.1
    agg_embs = gcn_aggregate(raw_embs, adj, n_rounds=2)    # graph-aware embs

    def _feats(df):
        df = df.copy()
        df['cum_c'] = df.groupby(['learner_id', 'skill_id'])['correct'].cumsum() - df['correct']
        df['cum_a'] = df.groupby(['learner_id', 'skill_id']).cumcount()
        df['sr']    = df['cum_c'] / df['cum_a'].replace(0, 1)
        df['deg']   = _degree_seq(df, degree_map)
        skill_idx   = df['skill_id'].map(skill_to_id).fillna(0).astype(int).values
        gcn_f       = agg_embs[skill_idx]            # (N, emb_dim)
        base_f      = df[['sr', 'cum_a', 'deg']].values.astype(np.float32)
        return np.concatenate([gcn_f, base_f], axis=1), df['correct'].values

    Xtr, ytr = _feats(train_df)
    Xte, yte = _feats(test_df)

    clf = LogisticRegression(max_iter=300, solver='saga', C=1.0)
    try:
        clf.fit(Xtr, ytr)
        preds = clf.predict_proba(Xte)[:, 1]
    except Exception:
        return float('nan'), float('nan'), float('nan')

    del Xtr, agg_embs, raw_embs; gc.collect()
    return _safe_metrics(yte, preds)


# ══════════════════════════════════════════════════════════════════════════════
# 5. SKT — Graph-path features + GradientBoosting (lightweight)
# ══════════════════════════════════════════════════════════════════════════════
def run_skt(train_df, test_df, skill_to_id, adj, degree_map, two_hop_map, seed):
    _seed_everything(seed)
    train_df = _subsample(train_df, _MAX_TRAIN_ROWS, seed)

    def _feats(df):
        df = df.copy()
        df['cum_c']    = df.groupby(['learner_id', 'skill_id'])['correct'].cumsum() - df['correct']
        df['cum_a']    = df.groupby(['learner_id', 'skill_id']).cumcount()
        df['sr']       = df['cum_c'] / df['cum_a'].replace(0, 1)
        df['deg']      = _degree_seq(df, degree_map)
        df['two_hop']  = df['skill_id'].astype(str).map(two_hop_map).fillna(0.0)
        df['sk_enc']   = df['skill_id'].map(skill_to_id).fillna(0)
        # Interaction count for this skill globally in training (structural richness proxy)
        return df[['sk_enc', 'sr', 'cum_a', 'deg', 'two_hop']].values.astype(np.float32), \
               df['correct'].values

    Xtr, ytr = _feats(train_df)
    Xte, yte = _feats(test_df)

    clf = GradientBoostingClassifier(
        n_estimators=80, max_depth=4, learning_rate=0.1,
        subsample=0.6, min_samples_leaf=20,
        random_state=seed
    )
    try:
        clf.fit(Xtr, ytr)
        preds = clf.predict_proba(Xte)[:, 1]
    except Exception:
        return float('nan'), float('nan'), float('nan')

    del Xtr, Xte; gc.collect()
    return _safe_metrics(yte, preds)


# ──────────────────────────────────────────────────────────────────────────────
# Dispatch table
# ──────────────────────────────────────────────────────────────────────────────
MODEL_FN = {
    'BKT':      run_bkt,
    'DKT':      run_dkt,
    'simpleKT': run_simplekt,
    'GIKT':     run_gikt,
    'SKT':      run_skt,
}
