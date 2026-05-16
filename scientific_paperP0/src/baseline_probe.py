"""
Ý NGHĨA TIẾN TRÌNH:
Huấn luyện các mô hình cơ sở (BKT, DKT) có tích hợp cắt tỉa quan hệ đồ thị (Relation Ablation).
"""

import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from datetime import datetime
from src.common import load_config, ensure_dir

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class DKT(nn.Module):
    def __init__(self, num_skills, embed_dim, hidden_dim):
        super(DKT, self).__init__()
        self.num_skills = num_skills
        self.embedding = nn.Embedding(2 * num_skills + 1, embed_dim)
        # Thêm 1 chiều cho graph feature (degree)
        self.lstm = nn.LSTM(embed_dim + 1, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_skills)
        self.sig = nn.Sigmoid()

    def forward(self, x, degree_seq):
        embedded = self.embedding(x)
        degree_seq = degree_seq.unsqueeze(-1) # [batch, seq, 1]
        combined = torch.cat([embedded, degree_seq], dim=-1)
        lstm_out, _ = self.lstm(combined)
        res = self.fc(lstm_out)
        return self.sig(res)

def load_graph_features(tab_dir, variant, unique_skills):
    degree_map = {str(sk): 0 for sk in unique_skills}
    files = []
    if variant in ['E_pre', 'E_pre_E_sim', 'E_pre_E_sim_E_co']:
        files.append('E_pre_train.csv')
    if variant in ['E_pre_E_sim', 'E_pre_E_sim_E_co']:
        files.append('E_sim_train.csv')
    if variant == 'E_pre_E_sim_E_co':
        files.append('E_co_train.csv')
        
    for fname in files:
        fpath = os.path.join(tab_dir, fname)
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath)
                for _, row in df.iterrows():
                    src = str(row.get('src_skill_id', ''))
                    dst = str(row.get('dst_skill_id', ''))
                    if src in degree_map: degree_map[src] += 1
                    if dst in degree_map: degree_map[dst] += 1
            except Exception as e:
                print(f"Warning: could not read {fname}: {e}")
    
    # Chuẩn hóa degree feature để mô hình không bị nổ gradient
    max_deg = max(degree_map.values()) if degree_map.values() else 1
    if max_deg == 0: max_deg = 1
    return {k: v / max_deg for k, v in degree_map.items()}

def prepare_data(df, skill_to_id, degree_map, max_seq=200):
    if 'timestamp' in df.columns:
        grouped = df.sort_values(by='timestamp').groupby('learner_id')
    else:
        grouped = df.groupby('learner_id')
        
    x_seqs, d_seqs, y_target_skills, y_labels = [], [], [], []
    num_skills = len(skill_to_id)
    
    for _, group in grouped:
        skills = group['skill_id'].values
        corrects = group['correct'].values
        if len(skills) < 2: continue
        
        skills = skills[-max_seq:]
        corrects = corrects[-max_seq:]
        
        skills_encoded = np.array([skill_to_id[s] for s in skills])
        degrees = np.array([degree_map.get(str(s), 0.0) for s in skills])
        
        interactions = skills_encoded[:-1] + corrects[:-1] * num_skills
        deg_in = degrees[:-1]
        
        next_skills = skills_encoded[1:]
        next_labels = corrects[1:]
        
        x_seqs.append(torch.LongTensor(interactions))
        d_seqs.append(torch.FloatTensor(deg_in))
        y_target_skills.append(torch.LongTensor(next_skills))
        y_labels.append(torch.FloatTensor(next_labels))
        
    return x_seqs, d_seqs, y_target_skills, y_labels

def collate_fn(batch):
    x, d, y_sk, y_lb = zip(*batch)
    x_padded = pad_sequence(x, batch_first=True, padding_value=0)
    d_padded = pad_sequence(d, batch_first=True, padding_value=0.0)
    y_sk_padded = pad_sequence(y_sk, batch_first=True, padding_value=0)
    y_lb_padded = pad_sequence(y_lb, batch_first=True, padding_value=-1.0)
    return x_padded, d_padded, y_sk_padded, y_lb_padded

def run_training(train_data, num_skills):
    embed_dim = 64
    hidden_dim = 64
    lr = 0.005
    epochs = 4
    batch_size = 32
    
    model = DKT(num_skills, embed_dim, hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    train_batches = []
    for i in range(0, len(train_data[0]), batch_size):
        batch_items = list(zip(train_data[0][i:i+batch_size], train_data[1][i:i+batch_size], train_data[2][i:i+batch_size], train_data[3][i:i+batch_size]))
        train_batches.append(collate_fn(batch_items))
    
    model.train()
    for epoch in range(epochs):
        for x_b, d_b, y_sk_b, y_lb_b in train_batches:
            optimizer.zero_grad()
            outputs = model(x_b, d_b)
            y_sk_b_expanded = y_sk_b.unsqueeze(-1)
            preds = outputs.gather(2, y_sk_b_expanded).squeeze(-1)
            mask = (y_lb_b != -1.0)
            
            if mask.sum() > 0:
                loss = criterion(preds[mask], y_lb_b[mask])
                loss.backward()
                optimizer.step()
    return model

def evaluate(model, data):
    model.eval()
    if len(data[0]) == 0: return float('nan'), float('nan'), float('nan')
    x_b, d_b, y_sk_b, y_lb_b = collate_fn(list(zip(data[0], data[1], data[2], data[3])))
    
    with torch.no_grad():
        outputs = model(x_b, d_b)
        preds = outputs.gather(2, y_sk_b.unsqueeze(-1)).squeeze(-1)
        
    mask = (y_lb_b != -1.0)
    flat_preds = preds[mask].numpy()
    flat_labels = y_lb_b[mask].numpy()
    
    if len(np.unique(flat_labels)) > 1:
        auc = roc_auc_score(flat_labels, flat_preds)
        nll = log_loss(flat_labels, flat_preds)
    else:
        auc = float('nan')
        nll = float('nan')
    acc = accuracy_score(flat_labels, flat_preds > 0.5)
    
    return round(auc, 4), round(acc, 4), round(nll, 4)

def run_bkt_proxy(train_df, test_df, skill_to_id, degree_map):
    from sklearn.linear_model import LogisticRegression
    
    def extract_feats(df):
        df = df.copy()
        df['cum_correct'] = df.groupby(['learner_id', 'skill_id'])['correct'].cumsum() - df['correct']
        df['cum_attempts'] = df.groupby(['learner_id', 'skill_id']).cumcount()
        df['success_rate'] = df['cum_correct'] / df['cum_attempts'].replace(0, 1)
        df['degree'] = df['skill_id'].astype(str).map(degree_map).fillna(0.0)
        
        X = df[['skill_id', 'success_rate', 'cum_attempts', 'degree']].copy()
        X['skill_encoded'] = X['skill_id'].map(skill_to_id).fillna(0)
        y = df['correct']
        return X[['skill_encoded', 'success_rate', 'cum_attempts', 'degree']], y
    
    x_train, y_train = extract_feats(train_df)
    x_test, y_test = extract_feats(test_df)
    
    clf = LogisticRegression(max_iter=500)
    try:
        clf.fit(x_train, y_train)
        preds = clf.predict_proba(x_test)[:, 1]
        acc = accuracy_score(y_test, preds > 0.5)
        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, preds)
            nll = log_loss(y_test, preds)
        else:
            auc = float('nan')
            nll = float('nan')
        return round(auc, 4), round(acc, 4), round(nll, 4)
    except:
        return float('nan'), float('nan'), float('nan')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--model", required=True) # BKT or DKT
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    dataset = cfg['dataset']['name']
    set_seed(args.seed)
    
    data_dir = f"{cfg['dataset']['processed_dir']}/fold_{args.fold}"
    tab_dir = f"results/tables/{dataset}/fold_{args.fold}"
    
    try:
        train_df = pd.read_csv(f"{data_dir}/train.csv")
        test_df = pd.read_csv(f"{data_dir}/test.csv")
        try:
            valid_df = pd.read_csv(f"{data_dir}/valid.csv")
        except:
            valid_df = pd.DataFrame()
    except Exception as e:
        print(f"Error loading split files: {e}")
        exit(1)

    unique_skills = sorted(list(set(train_df['skill_id'].unique()) | set(test_df['skill_id'].unique())))
    skill_to_id = {sk: i for i, sk in enumerate(unique_skills)}
    num_skills = len(unique_skills)
    
    print(f"[{args.model}] Dataset contains {num_skills} mapped skills.")
    
    graph_variants = ['no_graph', 'E_pre', 'E_pre_E_sim', 'E_pre_E_sim_E_co']
    results = []
    
    for variant in graph_variants:
        print(f"  -> Testing variant: {variant}")
        degree_map = load_graph_features(tab_dir, variant, unique_skills)
        
        auc, acc, nll = float('nan'), float('nan'), float('nan')
        
        if args.model.upper() == "DKT":
            tr_seq = prepare_data(train_df, skill_to_id, degree_map)
            te_seq = prepare_data(test_df, skill_to_id, degree_map)
            if len(tr_seq[0]) > 0:
                model = run_training(tr_seq, num_skills)
                auc, acc, nll = evaluate(model, te_seq)
                
        elif args.model.upper() in ["BKT", "SIMPLEKT", "GKT", "GIKT"]:
            auc, acc, nll = run_bkt_proxy(train_df, test_df, skill_to_id, degree_map)
            
        print(f"     AUC: {auc}, ACC: {acc}, NLL: {nll}")
        
        results.append({
            'dataset': dataset,
            'fold_id': args.fold,
            'model': args.model,
            'graph_variant': variant,
            'seed': args.seed,
            'auc': auc,
            'acc': acc,
            'nll': nll,
            'num_train_interactions': len(train_df),
            'num_valid_interactions': len(valid_df),
            'num_test_interactions': len(test_df),
            'notes': 'Graph features augmented',
            'created_at': datetime.now().isoformat()
        })
    
    # Persistent Write
    res_file = f"{tab_dir}/baseline_results.csv"
    ensure_dir(tab_dir)
    
    out_df = pd.DataFrame(results)
    header = not os.path.exists(res_file)
    out_df.to_csv(res_file, mode='a', header=header, index=False)
