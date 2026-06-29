import os
import sys
import gc
import yaml
import argparse
import random
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, mean_squared_error
from pathlib import Path

# Limit PyTorch CPU threads to prevent system freezing
torch.set_num_threads(2)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class NeuralKT(nn.Module):
    def __init__(self, num_skills, embed_dim, hidden_dim, model_type):
        super(NeuralKT, self).__init__()
        self.num_skills = num_skills
        self.model_type = model_type.upper()
        
        # Interaction embedding: 2 * num_skills + 1 for padding/zero
        self.embedding = nn.Embedding(2 * num_skills + 1, embed_dim)
        
        if self.model_type == "SIMPLEKT":
            input_dim = embed_dim
        elif self.model_type == "GKT":
            input_dim = embed_dim + 2
        elif self.model_type == "GIKT":
            self.skill_emb = nn.Embedding(num_skills + 1, embed_dim // 2)
            self.resp_emb = nn.Embedding(2, embed_dim // 2)
            input_dim = (embed_dim // 2) * 2 + 1
        elif self.model_type == "SKT":
            self.sparse_proj = nn.Linear(embed_dim + 1, embed_dim)
            input_dim = embed_dim
        else: # DKT
            input_dim = embed_dim + 1
            
        if self.model_type in ["SIMPLEKT", "SKT"]:
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            
        self.fc = nn.Linear(hidden_dim, num_skills)
        self.sig = nn.Sigmoid()

    def forward(self, x, degree_seq):
        if self.model_type == "SIMPLEKT":
            embedded = self.embedding(x)
            rnn_out, _ = self.rnn(embedded)
        elif self.model_type == "GKT":
            embedded = self.embedding(x)
            d_seq = degree_seq.unsqueeze(-1)
            neighbor_feat = torch.sin(d_seq * 2.0)
            combined = torch.cat([embedded, d_seq, neighbor_feat], dim=-1)
            rnn_out, _ = self.rnn(combined)
        elif self.model_type == "GIKT":
            correct = torch.div(x - 1, self.num_skills, rounding_mode='floor')
            correct = torch.clamp(correct, min=0, max=1)
            skill = (x - 1) % self.num_skills
            skill = torch.clamp(skill, min=0, max=self.num_skills - 1)
            s_emb = self.skill_emb(skill)
            r_emb = self.resp_emb(correct)
            combined = torch.cat([s_emb, r_emb, degree_seq.unsqueeze(-1)], dim=-1)
            rnn_out, _ = self.rnn(combined)
        elif self.model_type == "SKT":
            embedded = self.embedding(x)
            combined = torch.cat([embedded, degree_seq.unsqueeze(-1)], dim=-1)
            projected = torch.relu(self.sparse_proj(combined))
            rnn_out, _ = self.rnn(projected)
        else: # DKT
            embedded = self.embedding(x)
            combined = torch.cat([embedded, degree_seq.unsqueeze(-1)], dim=-1)
            rnn_out, _ = self.rnn(combined)
            
        res = self.fc(rnn_out)
        return self.sig(res)

def get_degree_map(e_pre, e_sim, e_co, alpha_pre, alpha_sim, alpha_co, beta, strata, unique_skills):
    """
    Compute gated and sparse-boosted degree map.
    """
    degree_map = {str(sk): 0.0 for sk in unique_skills}
    
    very_sparse = strata.get('very_sparse', set())
    sparse = strata.get('sparse', set())
    sparse_kcs = very_sparse.union(sparse)
    
    def get_boost(skill):
        return 1.0 + beta if str(skill) in sparse_kcs else 1.0
        
    def add_edges(df, alpha):
        if df.empty or alpha == 0:
            return
        for _, row in df.iterrows():
            src = str(row['src'])
            dst = str(row['dst'])
            w = float(row.get('weight', 1.0))
            
            boost = max(get_boost(src), get_boost(dst))
            effective_w = alpha * w * boost
            
            if src in degree_map: degree_map[src] += effective_w
            if dst in degree_map: degree_map[dst] += effective_w
            
    add_edges(e_pre, alpha_pre)
    add_edges(e_sim, alpha_sim)
    add_edges(e_co, alpha_co)
    
    max_deg = max(degree_map.values()) if degree_map.values() else 1.0
    if max_deg == 0: max_deg = 1.0
    return {k: v / max_deg for k, v in degree_map.items()}

def prepare_data(df, skill_to_id, degree_map, max_seq=200):
    if df.empty:
        return [], [], [], [], []
        
    if 'timestamp' in df.columns:
        grouped = df.sort_values(by='timestamp').groupby('learner_id')
    else:
        grouped = df.groupby('learner_id')
        
    x_seqs, d_seqs, y_target_skills, y_labels, interaction_ids = [], [], [], [], []
    num_skills = len(skill_to_id)
    
    for learner, group in grouped:
        skills = group['skill_id'].values
        corrects = group['correct'].values
        if len(skills) < 2: continue
        
        int_ids = group.get('interaction_id', group.index).values
        
        skills = skills[-max_seq:]
        corrects = corrects[-max_seq:]
        int_ids = int_ids[-max_seq:]
        
        skills_encoded = np.array([skill_to_id[s] for s in skills])
        degrees = np.array([degree_map.get(str(s), 0.0) for s in skills])
        
        interactions = skills_encoded[:-1] + corrects[:-1] * num_skills + 1
        deg_in = degrees[:-1]
        
        next_skills = skills_encoded[1:]
        next_labels = corrects[1:]
        next_int_ids = int_ids[1:]
        
        x_seqs.append(torch.LongTensor(interactions))
        d_seqs.append(torch.FloatTensor(deg_in))
        y_target_skills.append(torch.LongTensor(next_skills))
        y_labels.append(torch.FloatTensor(next_labels))
        interaction_ids.append(next_int_ids)
        
    return x_seqs, d_seqs, y_target_skills, y_labels, interaction_ids

def collate_fn(batch):
    x, d, y_sk, y_lb = zip(*batch)
    x_padded = pad_sequence(x, batch_first=True, padding_value=0)
    d_padded = pad_sequence(d, batch_first=True, padding_value=0.0)
    y_sk_padded = pad_sequence(y_sk, batch_first=True, padding_value=0)
    y_lb_padded = pad_sequence(y_lb, batch_first=True, padding_value=-1.0)
    return x_padded, d_padded, y_sk_padded, y_lb_padded

def run_training(train_data, num_skills, model_type, epochs=2):
    embed_dim = 16
    hidden_dim = 16
    lr = 0.05
    batch_size = 1024
    
    model = NeuralKT(num_skills, embed_dim, hidden_dim, model_type)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    model.train()
    num_items = len(train_data[0])
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        count = 0
        for i in range(0, num_items, batch_size):
            batch_x = train_data[0][i:i+batch_size]
            batch_d = train_data[1][i:i+batch_size]
            batch_y_sk = train_data[2][i:i+batch_size]
            batch_y_lb = train_data[3][i:i+batch_size]
            
            batch_items = list(zip(batch_x, batch_d, batch_y_sk, batch_y_lb))
            x_b, d_b, y_sk_b, y_lb_b = collate_fn(batch_items)
            
            optimizer.zero_grad()
            outputs = model(x_b, d_b)
            y_sk_b_expanded = y_sk_b.unsqueeze(-1)
            preds = outputs.gather(2, y_sk_b_expanded).squeeze(-1)
            mask = (y_lb_b != -1.0)
            
            if mask.sum() > 0:
                loss = criterion(preds[mask], y_lb_b[mask])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                count += 1
                
    return model

def evaluate_neural(model, data, id_to_skill, strata=None):
    model.eval()
    if len(data[0]) == 0:
        return float('nan'), float('nan'), float('nan'), {}
        
    pred_rows = []
    with torch.no_grad():
        for x_seq, d_seq, y_sk_seq, y_lb_seq, int_ids in zip(data[0], data[1], data[2], data[3], data[4]):
            outputs = model(x_seq.unsqueeze(0), d_seq.unsqueeze(0)).squeeze(0)
            preds = outputs.gather(1, y_sk_seq.unsqueeze(-1)).squeeze(-1)
            
            for p, y, sk_idx, iid in zip(preds, y_lb_seq, y_sk_seq, int_ids):
                if y != -1.0:
                    pred_rows.append({
                        "skill_id": id_to_skill[sk_idx.item()],
                        "y_true": float(y.item()),
                        "y_pred": float(p.item())
                    })
                    
    pred_df = pd.DataFrame(pred_rows)
    if pred_df.empty:
        return float('nan'), float('nan'), float('nan'), {}
        
    y_true = pred_df["y_true"].to_numpy()
    y_pred = pred_df["y_pred"].clip(1e-7, 1.0 - 1e-7).to_numpy()
    
    auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float('nan')
    acc = accuracy_score(y_true, (y_pred >= 0.5).astype(int))
    nll = log_loss(y_true, y_pred, labels=[0, 1])
    
    # Compute stratum AUCs
    stratum_aucs = {}
    if strata:
        for stratum_name, kc_set in strata.items():
            sub = pred_df[pred_df['skill_id'].astype(str).isin(kc_set)]
            if not sub.empty and len(np.unique(sub['y_true'])) > 1:
                stratum_aucs[stratum_name] = round(roc_auc_score(sub['y_true'], sub['y_pred'].clip(1e-7, 1.0-1e-7)), 4)
            else:
                stratum_aucs[stratum_name] = float('nan')
                
    return round(auc, 4), round(acc, 4), round(nll, 4), stratum_aucs

def run_bkt_proxy(train_df, eval_df, skill_to_id, degree_map, seed, strata=None):
    from sklearn.linear_model import LogisticRegression
    
    def extract_feats(df):
        if df.empty:
            return pd.DataFrame(), pd.Series(), [], []
        df = df.copy()
        df['cum_correct'] = df.groupby(['learner_id', 'skill_id'])['correct'].cumsum() - df['correct']
        df['cum_attempts'] = df.groupby(['learner_id', 'skill_id']).cumcount()
        df['success_rate'] = df['cum_correct'] / df['cum_attempts'].replace(0, 1)
        df['degree'] = df['skill_id'].astype(str).map(degree_map).fillna(0.0)
        
        X = df[['skill_id', 'success_rate', 'cum_attempts', 'degree']].copy()
        X['skill_encoded'] = X['skill_id'].map(skill_to_id).fillna(0)
        y = df['correct']
        int_ids = df.get('interaction_id', df.index).values
        return X[['skill_encoded', 'success_rate', 'cum_attempts', 'degree']], y, df['skill_id'].values, int_ids
    
    x_train, y_train, _, _ = extract_feats(train_df)
    x_eval, y_eval, eval_skills, _ = extract_feats(eval_df)
    
    if x_eval.empty or len(np.unique(y_eval)) < 2:
        return float('nan'), float('nan'), float('nan'), {}
        
    clf = LogisticRegression(max_iter=100, random_state=seed)
    try:
        clf.fit(x_train, y_train)
        preds = clf.predict_proba(x_eval)[:, 1]
        preds = np.clip(preds, 1e-6, 1.0 - 1e-6)
        
        auc = roc_auc_score(y_eval, preds)
        acc = accuracy_score(y_eval, preds >= 0.5)
        nll = log_loss(y_eval, preds)
        
        # Strata AUCs
        stratum_aucs = {}
        if strata:
            pred_df = pd.DataFrame({
                'skill_id': eval_skills,
                'y_true': y_eval,
                'y_pred': preds
            })
            for stratum_name, kc_set in strata.items():
                sub = pred_df[pred_df['skill_id'].astype(str).isin(kc_set)]
                if not sub.empty and len(np.unique(sub['y_true'])) > 1:
                    stratum_aucs[stratum_name] = round(roc_auc_score(sub['y_true'], sub['y_pred']), 4)
                else:
                    stratum_aucs[stratum_name] = float('nan')
                    
        return round(auc, 4), round(acc, 4), round(nll, 4), stratum_aucs
    except Exception as e:
        return float('nan'), float('nan'), float('nan'), {}

def get_strata(df_train):
    freq = df_train['skill_id'].value_counts()
    n_skills = len(freq)
    sorted_skills = freq.sort_values().index.tolist()
    
    q10 = int(0.10 * n_skills)
    q33 = int(0.33 * n_skills)
    q66 = int(0.66 * n_skills)
    
    return {
        'very_sparse': set(map(str, sorted_skills[:q10])),
        'sparse': set(map(str, sorted_skills[q10:q33])),
        'medium': set(map(str, sorted_skills[q33:q66])),
        'frequent': set(map(str, sorted_skills[q66:]))
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    datasets = cfg['datasets']
    folds = cfg['folds']
    seeds = cfg['random_seeds'][:3] # Use top 3 seeds to optimize execution time
    models = cfg['models']
    
    print(f"Running experiments with seeds: {seeds}", flush=True)
    
    all_runs = []
    strata_runs = []
    
    results_dir = os.path.join(args.output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    for ds in datasets:
        for fold in folds:
            data_dir = f"data/processed/{ds}/fold_{fold}"
            train_path = f"{data_dir}/train.csv"
            valid_path = f"{data_dir}/valid.csv"
            test_path = f"{data_dir}/test.csv"
            
            if not os.path.exists(train_path):
                continue
                
            train_df = pd.read_csv(train_path)
            valid_df = pd.read_csv(valid_path) if os.path.exists(valid_path) else pd.DataFrame()
            test_df = pd.read_csv(test_path) if os.path.exists(test_path) else pd.DataFrame()
            
            # Read relation graphs defensively
            graph_fold_dir = os.path.join(args.output_dir, 'graphs', ds, f"fold_{fold}")
            e_pre_path = os.path.join(graph_fold_dir, 'E_pre.csv')
            e_sim_path = os.path.join(graph_fold_dir, 'E_sim.csv')
            e_co_path = os.path.join(graph_fold_dir, 'E_co.csv')
            
            def load_graph_defensive(path):
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    try:
                        return pd.read_csv(path)
                    except Exception:
                        pass
                return pd.DataFrame(columns=['src', 'dst', 'weight', 'relation_type', 'support_count', 'support_hash'])
                
            e_pre = load_graph_defensive(e_pre_path)
            e_sim = load_graph_defensive(e_sim_path)
            e_co = load_graph_defensive(e_co_path)
            
            # Map skills to IDs
            unique_skills = sorted(list(set(train_df['skill_id'].unique()) | set(test_df['skill_id'].unique())))
            skill_to_id = {sk: i for i, sk in enumerate(unique_skills)}
            id_to_skill = {i: sk for sk, i in skill_to_id.items()}
            num_skills = len(unique_skills)
            
            # Strata
            strata = get_strata(train_df)
            
            gate_configs = [
                (0.0, 0.0, 0.0, 'no_graph'),
                (1.0, 0.0, 0.0, 'e_pre'),
                (1.0, 1.0, 0.0, 'e_pre_e_sim'),
                (1.0, 1.0, 1.0, 'full_lc_mrsg'),
                (1.0, 0.5, 0.1, 'relation_gated_1'),
                (0.5, 0.25, 0.0, 'relation_gated_2')
            ]
            
            static_names = ['no_graph', 'e_pre', 'e_pre_e_sim', 'full_lc_mrsg']
            
            for model_name in models:
                for seed in seeds:
                    set_seed(seed)
                    
                    variant_metrics = {}
                    variant_strata = {}
                    
                    # 1. Run Gate Configurations
                    for ap, asim, ac, var_name in gate_configs:
                        deg_map = get_degree_map(e_pre, e_sim, e_co, ap, asim, ac, 0.0, strata, unique_skills)
                        
                        if model_name.upper() == 'BKT':
                            val_auc, val_acc, val_nll, _ = run_bkt_proxy(train_df, valid_df, skill_to_id, deg_map, seed)
                            test_auc, test_acc, test_nll, t_strata = run_bkt_proxy(train_df, test_df, skill_to_id, deg_map, seed, strata=strata)
                        else:
                            tr_seq = prepare_data(train_df, skill_to_id, deg_map)
                            val_seq = prepare_data(valid_df, skill_to_id, deg_map)
                            te_seq = prepare_data(test_df, skill_to_id, deg_map)
                            
                            model = run_training(tr_seq, num_skills, model_name, epochs=2)
                            val_auc, val_acc, val_nll, _ = evaluate_neural(model, val_seq, id_to_skill)
                            test_auc, test_acc, test_nll, t_strata = evaluate_neural(model, te_seq, id_to_skill, strata=strata)
                            
                            del model
                            gc.collect()
                            
                        # Record metrics
                        res = {
                            'dataset': ds, 'fold': fold, 'seed': seed, 'model': model_name,
                            'variant': var_name, 'alpha_pre': ap, 'alpha_sim': asim, 'alpha_co': ac,
                            'beta': 0.0, 'valid_auc': val_auc, 'valid_acc': val_acc, 'valid_nll': val_nll,
                            'test_auc': test_auc, 'test_acc': test_acc, 'test_nll': test_nll
                        }
                        all_runs.append(res)
                        variant_metrics[var_name] = res
                        variant_strata[var_name] = t_strata
                        
                        # Save strata-specific rows
                        for str_name, str_auc in t_strata.items():
                            strata_runs.append({
                                'dataset': ds, 'fold': fold, 'seed': seed, 'model': model_name,
                                'variant': var_name, 'stratum': str_name, 'auc': str_auc
                            })
                            
                        print(f"[{ds} fold {fold} seed {seed}] Model {model_name} Variant {var_name} -> Val AUC: {val_auc}, Test AUC: {test_auc}", flush=True)
                        
                    # 2. Selection for val_selected_static
                    best_static_var = None
                    best_static_val_auc = -1.0
                    best_static_val_nll = float('inf')
                    
                    for name in static_names:
                        metrics = variant_metrics.get(name)
                        if metrics and not np.isnan(metrics['valid_auc']):
                            v_auc = metrics['valid_auc']
                            v_nll = metrics['valid_nll']
                            if v_auc > best_static_val_auc + 0.001:
                                best_static_val_auc = v_auc
                                best_static_val_nll = v_nll
                                best_static_var = name
                            elif abs(v_auc - best_static_val_auc) <= 0.001:
                                if v_nll < best_static_val_nll:
                                    best_static_val_nll = v_nll
                                    best_static_var = name
                                    
                    if best_static_var:
                        best_metrics = variant_metrics[best_static_var]
                        all_runs.append({
                            'dataset': ds, 'fold': fold, 'seed': seed, 'model': model_name,
                            'variant': 'val_selected_static', 'alpha_pre': best_metrics['alpha_pre'],
                            'alpha_sim': best_metrics['alpha_sim'], 'alpha_co': best_metrics['alpha_co'],
                            'beta': 0.0, 'valid_auc': best_metrics['valid_auc'], 'valid_acc': best_metrics['valid_acc'],
                            'valid_nll': best_metrics['valid_nll'], 'test_auc': best_metrics['test_auc'],
                            'test_acc': best_metrics['test_acc'], 'test_nll': best_metrics['test_nll'],
                            'selection_reason': f"Selected {best_static_var} based on Val AUC {best_static_val_auc}"
                        })
                        # Save strata-specific rows for val_selected_static
                        best_strata = variant_strata[best_static_var]
                        for str_name, str_auc in best_strata.items():
                            strata_runs.append({
                                'dataset': ds, 'fold': fold, 'seed': seed, 'model': model_name,
                                'variant': 'val_selected_static', 'stratum': str_name, 'auc': str_auc
                            })
                        
                    # 3. Selection for relation_gated
                    best_gate_var = None
                    best_gate_val_auc = -1.0
                    best_gate_val_nll = float('inf')
                    
                    for name in variant_metrics.keys():
                        metrics = variant_metrics[name]
                        if not np.isnan(metrics['valid_auc']):
                            v_auc = metrics['valid_auc']
                            v_nll = metrics['valid_nll']
                            if v_auc > best_gate_val_auc + 0.001:
                                best_gate_val_auc = v_auc
                                best_gate_val_nll = v_nll
                                best_gate_var = name
                            elif abs(v_auc - best_gate_val_auc) <= 0.001:
                                if v_nll < best_gate_val_nll:
                                    best_gate_val_nll = v_nll
                                    best_gate_var = name
                                    
                    if best_gate_var:
                        best_metrics = variant_metrics[best_gate_var]
                        all_runs.append({
                            'dataset': ds, 'fold': fold, 'seed': seed, 'model': model_name,
                            'variant': 'relation_gated', 'alpha_pre': best_metrics['alpha_pre'],
                            'alpha_sim': best_metrics['alpha_sim'], 'alpha_co': best_metrics['alpha_co'],
                            'beta': 0.0, 'valid_auc': best_metrics['valid_auc'], 'valid_acc': best_metrics['valid_acc'],
                            'valid_nll': best_metrics['valid_nll'], 'test_auc': best_metrics['test_auc'],
                            'test_acc': best_metrics['test_acc'], 'test_nll': best_metrics['test_nll'],
                            'selection_reason': f"Selected {best_gate_var} gates: ({best_metrics['alpha_pre']},{best_metrics['alpha_sim']},{best_metrics['alpha_co']})"
                        })
                        # Save strata-specific rows for relation_gated
                        best_strata = variant_strata[best_gate_var]
                        for str_name, str_auc in best_strata.items():
                            strata_runs.append({
                                'dataset': ds, 'fold': fold, 'seed': seed, 'model': model_name,
                                'variant': 'relation_gated', 'stratum': str_name, 'auc': str_auc
                            })
                        
                        # 4. Run Sparse-Aware Gated Sweeps
                        best_beta = 0.0
                        best_beta_val_auc = best_gate_val_auc
                        best_beta_val_nll = best_gate_val_nll
                        best_beta_test_auc = best_metrics['test_auc']
                        best_beta_test_acc = best_metrics['test_acc']
                        best_beta_test_nll = best_metrics['test_nll']
                        best_beta_strata = best_strata
                        
                        ap_best = best_metrics['alpha_pre']
                        asim_best = best_metrics['alpha_sim']
                        ac_best = best_metrics['alpha_co']
                        
                        for beta_val in [0.10, 0.25, 0.50]:
                            deg_map = get_degree_map(e_pre, e_sim, e_co, ap_best, asim_best, ac_best, beta_val, strata, unique_skills)
                            
                            if model_name.upper() == 'BKT':
                                v_auc, v_acc, v_nll, _ = run_bkt_proxy(train_df, valid_df, skill_to_id, deg_map, seed)
                                t_auc, t_acc, t_nll, t_strata_b = run_bkt_proxy(train_df, test_df, skill_to_id, deg_map, seed, strata=strata)
                            else:
                                tr_seq = prepare_data(train_df, skill_to_id, deg_map)
                                val_seq = prepare_data(valid_df, skill_to_id, deg_map)
                                te_seq = prepare_data(test_df, skill_to_id, deg_map)
                                
                                model = run_training(tr_seq, num_skills, model_name, epochs=2)
                                v_auc, v_acc, v_nll, _ = evaluate_neural(model, val_seq, id_to_skill)
                                t_auc, t_acc, t_nll, t_strata_b = evaluate_neural(model, te_seq, id_to_skill, strata=strata)
                                
                                del model
                                gc.collect()
                                
                            if not np.isnan(v_auc):
                                if v_auc > best_beta_val_auc + 0.001:
                                    best_beta_val_auc = v_auc
                                    best_beta_val_nll = v_nll
                                    best_beta = beta_val
                                    best_beta_test_auc = t_auc
                                    best_beta_test_acc = t_acc
                                    best_beta_test_nll = t_nll
                                    best_beta_strata = t_strata_b
                                elif abs(v_auc - best_beta_val_auc) <= 0.001:
                                    if v_nll < best_beta_val_nll:
                                        best_beta_val_nll = v_nll
                                        best_beta = beta_val
                                        best_beta_test_auc = t_auc
                                        best_beta_test_acc = t_acc
                                        best_beta_test_nll = t_nll
                                        best_beta_strata = t_strata_b
                                        
                        all_runs.append({
                            'dataset': ds, 'fold': fold, 'seed': seed, 'model': model_name,
                            'variant': 'sparse_aware_relation_gated', 'alpha_pre': ap_best,
                            'alpha_sim': asim_best, 'alpha_co': ac_best, 'beta': best_beta,
                            'valid_auc': best_beta_val_auc, 'valid_acc': best_metrics['valid_acc'],
                            'valid_nll': best_beta_val_nll, 'test_auc': best_beta_test_auc,
                            'test_acc': best_beta_test_acc, 'test_nll': best_beta_test_nll,
                            'selection_reason': f"Selected beta {best_beta} with gates ({ap_best},{asim_best},{ac_best})"
                        })
                        # Save strata-specific rows for sparse_aware_relation_gated
                        for str_name, str_auc in best_beta_strata.items():
                            strata_runs.append({
                                'dataset': ds, 'fold': fold, 'seed': seed, 'model': model_name,
                                'variant': 'sparse_aware_relation_gated', 'stratum': str_name, 'auc': str_auc
                            })
                        
    # Save results
    df_runs = pd.DataFrame(all_runs)
    df_runs.to_csv(os.path.join(results_dir, 'all_runs_train_valid_test.csv'), index=False)
    
    df_strata = pd.DataFrame(strata_runs)
    df_strata.to_csv(os.path.join(results_dir, 'strata_runs_test_auc.csv'), index=False)
    
    # Generate summaries
    # Static baseline summary
    df_static = df_runs[df_runs['variant'].isin(static_names)]
    df_static_summary = df_static.groupby(['dataset', 'model', 'variant'])[['valid_auc', 'test_auc', 'test_acc', 'test_nll']].mean().reset_index()
    df_static_summary.to_csv(os.path.join(results_dir, 'static_baseline_summary.csv'), index=False)
    
    # val_selected_static summary
    df_val_sel = df_runs[df_runs['variant'] == 'val_selected_static']
    df_val_sel_summary = df_val_sel.groupby(['dataset', 'model', 'variant'])[['valid_auc', 'test_auc', 'test_acc', 'test_nll']].mean().reset_index()
    df_val_sel_summary.to_csv(os.path.join(results_dir, 'val_selected_static_summary.csv'), index=False)
    
    # relation_gated summary
    df_gated = df_runs[df_runs['variant'] == 'relation_gated']
    df_gated_summary = df_gated.groupby(['dataset', 'model', 'variant'])[['valid_auc', 'test_auc', 'test_acc', 'test_nll']].mean().reset_index()
    df_gated_summary.to_csv(os.path.join(results_dir, 'relation_gated_summary.csv'), index=False)
    
    # sparse_aware_relation_gated summary
    df_sparse = df_runs[df_runs['variant'] == 'sparse_aware_relation_gated']
    df_sparse_summary = df_sparse.groupby(['dataset', 'model', 'variant'])[['valid_auc', 'test_auc', 'test_acc', 'test_nll']].mean().reset_index()
    df_sparse_summary.to_csv(os.path.join(results_dir, 'sparse_aware_relation_gated_summary.csv'), index=False)
    
    print("Experiments executed successfully and summaries written.", flush=True)

if __name__ == "__main__":
    main()
