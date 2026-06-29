# -*- coding: utf-8 -*-
import os
import sys
import gc
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

# Setup seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# Model Architecture
class NeuralKT(nn.Module):
    def __init__(self, num_skills, embed_dim, hidden_dim, model_type):
        super(NeuralKT, self).__init__()
        self.num_skills = num_skills
        self.model_type = model_type.upper()
        self.embedding = nn.Embedding(2 * num_skills + 1, embed_dim)
        
        if self.model_type == "SIMPLEKT":
            input_dim = embed_dim
        else: # DKT
            input_dim = embed_dim + 1
            
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True) if self.model_type == "SIMPLEKT" else nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_skills)
        self.sig = nn.Sigmoid()

    def forward(self, x, degree_seq):
        embedded = self.embedding(x)
        if self.model_type == "SIMPLEKT":
            rnn_out, _ = self.rnn(embedded)
        else: # DKT
            combined = torch.cat([embedded, degree_seq.unsqueeze(-1)], dim=-1)
            rnn_out, _ = self.rnn(combined)
            
        res = self.fc(rnn_out)
        return self.sig(res)

def get_degree_map(e_pre, e_sim, e_co, alpha_pre, alpha_sim, alpha_co, beta, strata, unique_skills):
    degree_map = {str(sk): 0.0 for sk in unique_skills}
    very_sparse = strata.get('very_sparse', set())
    sparse = strata.get('sparse', set())
    sparse_kcs = very_sparse.union(sparse)
    
    def get_boost(skill):
        return 1.0 + beta if str(skill) in sparse_kcs else 1.0
        
    def add_edges(df, alpha):
        if df is None or df.empty or alpha == 0:
            return
        src_col = 'src' if 'src' in df.columns else 'source'
        dst_col = 'dst' if 'dst' in df.columns else 'target'
        weight_col = 'weight' if 'weight' in df.columns else 'pmi'
        
        for _, row in df.iterrows():
            src = str(row[src_col])
            dst = str(row[dst_col])
            w = float(row.get(weight_col, 1.0))
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
        if len(skills) < 2:
            continue
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

def run_training_early_stopping(train_data, val_data, num_skills, id_to_skill, model_type, max_epochs, patience, min_delta, checkpoint_path, device):
    embed_dim = 16
    hidden_dim = 16
    lr = 0.05
    batch_size = 4096 if len(train_data[0]) > 2000 else 1024
    
    model = NeuralKT(num_skills, embed_dim, hidden_dim, model_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    best_val_auc = -1.0
    best_epoch = 0
    patience_counter = 0
    num_items = len(train_data[0])
    
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        count = 0
        
        indices = np.arange(num_items)
        np.random.shuffle(indices)
        
        for idx in range(0, num_items, batch_size):
            batch_idx = indices[idx:idx+batch_size]
            batch_x = [train_data[0][k] for k in batch_idx]
            batch_d = [train_data[1][k] for k in batch_idx]
            batch_y_sk = [train_data[2][k] for k in batch_idx]
            batch_y_lb = [train_data[3][k] for k in batch_idx]
            
            x_b, d_b, y_sk_b, y_lb_b = collate_fn(list(zip(batch_x, batch_d, batch_y_sk, batch_y_lb)))
            x_b, d_b, y_sk_b, y_lb_b = x_b.to(device), d_b.to(device), y_sk_b.to(device), y_lb_b.to(device)
            
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
                
        train_loss = epoch_loss / max(count, 1)
        val_auc, _, _ = evaluate_neural(model, val_data, id_to_skill, device=device)
        
        if not np.isnan(val_auc):
            if val_auc > best_val_auc + min_delta:
                best_val_auc = val_auc
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    if os.path.exists(checkpoint_path) and best_epoch > 0:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        torch.save(model.state_dict(), checkpoint_path)
        
    return model, best_epoch, best_val_auc

def evaluate_neural(model, data, id_to_skill, device='cpu'):
    model.eval()
    if len(data[0]) == 0:
        return float('nan'), float('nan'), float('nan')
        
    pred_rows = []
    num_items = len(data[0])
    batch_size = 1024
    
    with torch.no_grad():
        for idx_start in range(0, num_items, batch_size):
            batch_x = data[0][idx_start:idx_start+batch_size]
            batch_d = data[1][idx_start:idx_start+batch_size]
            batch_y_sk = data[2][idx_start:idx_start+batch_size]
            batch_y_lb = data[3][idx_start:idx_start+batch_size]
            
            x_padded, d_padded, y_sk_padded, _ = collate_fn(list(zip(batch_x, batch_d, batch_y_sk, batch_y_lb)))
            x_padded, d_padded, y_sk_padded = x_padded.to(device), d_padded.to(device), y_sk_padded.to(device)
            
            outputs = model(x_padded, d_padded)
            y_sk_expanded = y_sk_padded.unsqueeze(-1)
            preds_padded = outputs.gather(2, y_sk_expanded).squeeze(2)
            preds_np = preds_padded.cpu().numpy()
            
            for b_idx in range(len(batch_x)):
                orig_len = len(batch_x[b_idx])
                p = preds_np[b_idx, :orig_len]
                y = batch_y_lb[b_idx].numpy()
                sk = batch_y_sk[b_idx].numpy()
                
                mask = (y != -1.0)
                for val_p, val_y, val_sk in zip(p[mask], y[mask], sk[mask]):
                    pred_rows.append({
                        "y_true": float(val_y),
                        "y_pred": float(val_p)
                    })
                      
    pred_df = pd.DataFrame(pred_rows)
    if pred_df.empty:
        return float('nan'), float('nan'), float('nan')
        
    y_true = pred_df["y_true"].to_numpy()
    y_pred = pred_df["y_pred"].clip(1e-7, 1.0 - 1e-7).to_numpy()
    
    auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float('nan')
    acc = accuracy_score(y_true, (y_pred >= 0.5).astype(int))
    nll = log_loss(y_true, y_pred, labels=[0, 1])
    
    return auc, acc, nll

def build_controlled_eco_graph(train_df, fold, k_min=3, pmi_min=0.25, top_k=50):
    learner_skills = defaultdict(list)
    for lid, sid in zip(train_df['learner_id'], train_df['skill_id']):
        learner_skills[lid].append(sid)
        
    co_counts = {}
    s_counts = {}
    for sks in learner_skills.values():
        usks = set(sks)
        for s in usks:
            s_counts[s] = s_counts.get(s, 0) + 1
        usks_list = list(usks)
        for i in range(len(usks_list)):
            for j in range(i + 1, len(usks_list)):
                s1, s2 = min(usks_list[i], usks_list[j]), max(usks_list[i], usks_list[j])
                co_counts[(s1, s2)] = co_counts.get((s1, s2), 0) + 1
                
    total_learners = len(learner_skills)
    co_edges = []
    
    for (s1, s2), count in co_counts.items():
        if count >= k_min:
            p_s1 = s_counts[s1] / total_learners
            p_s2 = s_counts[s2] / total_learners
            p_co = count / total_learners
            pmi = np.log(p_co / (p_s1 * p_s2))
            if pmi >= pmi_min:
                co_edges.append((s1, s2, pmi))
                
    if top_k is not None:
        skill_edges = defaultdict(list)
        for s1, s2, pmi in co_edges:
            skill_edges[s1].append((s2, pmi))
            skill_edges[s2].append((s1, pmi))
        topk_edges = set()
        for s, edges in skill_edges.items():
            edges.sort(key=lambda x: x[1], reverse=True)
            for neighbor, pmi in edges[:top_k]:
                topk_edges.add((min(s, neighbor), max(s, neighbor)))
        final_edges = [e for e in co_edges if (e[0], e[1]) in topk_edges]
    else:
        final_edges = co_edges
        
    mirrored_edges = []
    for s1, s2, pmi in final_edges:
        mirrored_edges.append({'src': s1, 'dst': s2, 'weight': pmi, 'relation_type': 'E_co'})
        mirrored_edges.append({'src': s2, 'dst': s1, 'weight': pmi, 'relation_type': 'E_co'})
        
    return pd.DataFrame(mirrored_edges) if mirrored_edges else pd.DataFrame(columns=['src', 'dst', 'weight', 'relation_type'])

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running No-Epre sensitivity training for KDD2010 on device: {device}...", flush=True)
    
    # Load dataset fold 0
    ds = "kdd2010"
    fold = 0
    data_dir = f"data/processed/{ds}/fold_{fold}"
    train_path = f"{data_dir}/train.csv"
    valid_path = f"{data_dir}/valid.csv"
    test_path = f"{data_dir}/test.csv"
    
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    test_df = pd.read_csv(test_path)
    
    unique_skills = sorted(list(set(train_df['skill_id'].unique()) | set(test_df['skill_id'].unique())))
    skill_to_id = {sk: i for i, sk in enumerate(unique_skills)}
    id_to_skill = {i: sk for sk, i in skill_to_id.items()}
    num_skills = len(unique_skills)
    
    # Fetch strata
    freq = train_df['skill_id'].value_counts()
    strata = {
        'very_sparse': set(map(str, freq[freq <= 50].index)),
        'sparse': set(map(str, freq[(freq > 50) & (freq <= 100)].index)),
        'medium': set(map(str, freq[(freq > 100) & (freq <= 200)].index)),
        'frequent': set(map(str, freq[freq > 200].index))
    }
    
    # Load raw graphs
    graph_fold_dir = f"results_ejel_hau_revision_20260624_225226/graphs/{ds}/fold_{fold}"
    e_pre = pd.DataFrame(columns=['src', 'dst', 'weight', 'relation_type']) # Epre is set to empty for no-Epre
    e_sim = pd.read_csv(os.path.join(graph_fold_dir, 'Esim_edges.csv'))
    
    # Build controlled density Eco graph for KDD2010
    print("Building controlled Eco graph for KDD2010 Fold 0...", flush=True)
    e_co = build_controlled_eco_graph(train_df, fold=0, k_min=3, pmi_min=0.25, top_k=50)
    
    # Configurations without Epre
    no_epre_configs = [
        (0.0, 0.0, 1.0, 0.0, 'e_co_only'),
        (0.0, 1.0, 0.0, 0.0, 'e_sim_only'),
        (0.0, 1.0, 1.0, 0.0, 'e_sim_e_co')
    ]
    
    kdd_results = []
    
    backbones = ["dkt", "simplekt"]
    max_epochs = 100
    
    for bb in backbones:
        patience = 10 if bb == "dkt" else 3
        min_delta = 0.0001
        
        # 1. Start with no_graph
        best_no_epre_val_auc = 0.8068 if bb == "dkt" else 0.8036
        best_no_epre_test_auc = 0.7597 if bb == "dkt" else 0.7576
        best_no_epre_var = "no_graph"
        
        # 2. Run new candidates
        for ap, asim, ac, beta, var_name in no_epre_configs:
            print(f"Running candidate {var_name} for KDD2010 {bb} Fold 0 Seed 42...", flush=True)
            set_seed(42)
            
            degree_map = get_degree_map(e_pre, e_sim, e_co, ap, asim, ac, beta, strata, unique_skills)
            tr_seq = prepare_data(train_df, skill_to_id, degree_map)
            val_seq = prepare_data(valid_df, skill_to_id, degree_map)
            te_seq = prepare_data(test_df, skill_to_id, degree_map)
            
            chk_dir = "results/ejel_gA_experiments/checkpoints/no_epre"
            os.makedirs(chk_dir, exist_ok=True)
            chk_path = os.path.join(chk_dir, f"kdd2010_{bb}_{var_name}_best.pt")
            
            model, best_ep, val_auc = run_training_early_stopping(
                tr_seq, val_seq, num_skills, id_to_skill, bb, max_epochs, patience, min_delta, chk_path, device
            )
            test_auc, _, _ = evaluate_neural(model, te_seq, id_to_skill, device=device)
            print(f"Candidate {var_name} Validation AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}", flush=True)
            
            if val_auc > best_no_epre_val_auc:
                best_no_epre_val_auc = val_auc
                best_no_epre_test_auc = test_auc
                best_no_epre_var = var_name
                
            del model
            gc.collect()
            torch.cuda.empty_cache()
            
        original_test_auc = 0.7647 if bb == "dkt" else 0.7576
        original_selected = "e_pre_e_sim" if bb == "dkt" else "no_graph"
        no_graph_test_auc = 0.7597 if bb == "dkt" else 0.7576
        delta_vs_no_graph = best_no_epre_test_auc - no_graph_test_auc
        survives = "Yes" if delta_vs_no_graph >= 0.005 else "No"
        
        kdd_results.append({
            "Dataset": "KDD2010",
            "Backbone": bb.upper(),
            "Original Selected": original_selected,
            "Original Test AUC": f"{original_test_auc:.4f}",
            "Best No-Epre Selected": best_no_epre_var,
            "No-Epre Test AUC": f"{best_no_epre_test_auc:.4f}",
            "Delta vs No-Graph": f"{delta_vs_no_graph:.4f}",
            "Survives >= 0.005": survives
        })
        
    print("\nNo-Epre training for KDD2010 finished successfully.")
    
    # 3. Read old CSV and append KDD2010 results
    old_csv_path = "results_ejel_hau_revision_20260624_225226/tables/no_epre_L2_sensitivity.csv"
    if os.path.exists(old_csv_path):
        df_old = pd.read_csv(old_csv_path)
        
        # Convert df_old columns and dataset upper casing
        new_rows = []
        for _, r in df_old.iterrows():
            orig_test = float(r['original_test_auc'])
            no_epre_test = float(r['best_no_epre_test_auc'])
            delta = float(r['delta_vs_no_graph'])
            surv = r['survives_delta_gt_005']
            survives_str = "Yes" if str(surv).lower() in ['true', 'yes', '1'] else "No"
            
            new_rows.append({
                "Dataset": r['dataset'].upper(),
                "Backbone": r['backbone'].upper(),
                "Original Selected": r['original_selected_candidate'],
                "Original Test AUC": f"{orig_test:.4f}",
                "Best No-Epre Selected": r['best_no_epre_candidate'],
                "No-Epre Test AUC": f"{no_epre_test:.4f}",
                "Delta vs No-Graph": f"{delta:.4f}",
                "Survives >= 0.005": survives_str
            })
            
        # Append KDD2010 rows
        new_rows.extend(kdd_results)
        df_new = pd.DataFrame(new_rows)
        
        # Save updated CSV
        df_new.to_csv("results/ejel_gA_experiments/no_epre_L2_sensitivity.csv", index=False)
        
        # Write LaTeX table
        latex_lines = [
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{No-Epre Sensitivity Analysis and L2 Residual Risk Assessment.}",
            "\\label{tab:no_epre_L2_sensitivity}",
            "\\begin{tabular}{llccccccl}",
            "\\toprule",
            "Dataset & Backbone & Original Selected & Original Test AUC & Best No-Epre Selected & No-Epre Test AUC & Delta vs No-Graph & Survives $\\ge$ 0.005 \\\\",
            "\\midrule"
        ]
        for _, r in df_new.iterrows():
            latex_lines.append(
                f"{r['Dataset']} & {r['Backbone']} & {r['Original Selected']} & {r['Original Test AUC']} & {r['Best No-Epre Selected']} & {r['No-Epre Test AUC']} & {r['Delta vs No-Graph']} & {r['Survives >= 0.005']} \\\\"
            )
        latex_lines.append("\\bottomrule")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")
        
        os.makedirs("tables", exist_ok=True)
        with open("tables/table_no_epre_L2_sensitivity.tex", "w", encoding="utf-8") as f:
            f.write("\n".join(latex_lines) + "\n")
        with open("results/ejel_gA_experiments/tables/table_no_epre_L2_sensitivity.tex", "w", encoding="utf-8") as f:
            f.write("\n".join(latex_lines) + "\n")
            
        print("Updated no_epre_L2_sensitivity.csv and table_no_epre_L2_sensitivity.tex successfully.")

if __name__ == "__main__":
    main()
