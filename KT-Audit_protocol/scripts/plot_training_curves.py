import os
import sys
import gc
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

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
        self.embedding = nn.Embedding(2 * num_skills + 1, embed_dim)
        
        if self.model_type == "SIMPLEKT":
            input_dim = embed_dim
        else: # DKT
            input_dim = embed_dim + 1
            
        if self.model_type == "SIMPLEKT":
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            
        self.fc = nn.Linear(hidden_dim, num_skills)
        self.sig = nn.Sigmoid()

    def forward(self, x, degree_seq):
        if self.model_type == "SIMPLEKT":
            embedded = self.embedding(x)
            rnn_out, _ = self.rnn(embedded)
        else: # DKT
            embedded = self.embedding(x)
            combined = torch.cat([embedded, degree_seq.unsqueeze(-1)], dim=-1)
            rnn_out, _ = self.rnn(combined)
            
        res = self.fc(rnn_out)
        return self.sig(res)

def collate_fn(batch):
    x, d, y_sk, y_lb = zip(*batch)
    x_padded = pad_sequence(x, batch_first=True, padding_value=0)
    d_padded = pad_sequence(d, batch_first=True, padding_value=0.0)
    y_sk_padded = pad_sequence(y_sk, batch_first=True, padding_value=0)
    y_lb_padded = pad_sequence(y_lb, batch_first=True, padding_value=-1.0)
    return x_padded, d_padded, y_sk_padded, y_lb_padded

def prepare_data(df, skill_to_id, degree_map, max_seq=200):
    if df.empty:
        return [], [], [], [], []
    grouped = df.sort_values(by='timestamp').groupby('learner_id') if 'timestamp' in df.columns else df.groupby('learner_id')
    
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

def evaluate_neural(model, data, id_to_skill):
    model.eval()
    if len(data[0]) == 0:
        return float('nan')
        
    pred_rows = []
    with torch.no_grad():
        for x_seq, d_seq, y_sk_seq, y_lb_seq, int_ids in zip(data[0], data[1], data[2], data[3], data[4]):
            outputs = model(x_seq.unsqueeze(0), d_seq.unsqueeze(0)).squeeze(0)
            preds = outputs.gather(1, y_sk_seq.unsqueeze(-1)).squeeze(-1)
            
            for p, y, sk_idx in zip(preds, y_lb_seq, y_sk_seq):
                if y != -1.0:
                    pred_rows.append({
                        "y_true": float(y.item()),
                        "y_pred": float(p.item())
                    })
                    
    pred_df = pd.DataFrame(pred_rows)
    if pred_df.empty:
        return float('nan')
        
    y_true = pred_df["y_true"].to_numpy()
    y_pred = pred_df["y_pred"].clip(1e-7, 1.0 - 1e-7).to_numpy()
    try:
        return roc_auc_score(y_true, y_pred)
    except:
        return float('nan')

def get_degree_map_helper(graph_fold_dir, ds, ap, asim, ac, unique_skills):
    degree_map = {str(sk): 0.0 for sk in unique_skills}
    
    def load_graph(path):
        if os.path.exists(path) and os.path.getsize(path) > 2:
            return pd.read_csv(path)
        return pd.DataFrame()
        
    e_pre = load_graph(os.path.join(graph_fold_dir, 'E_pre.csv'))
    e_sim = load_graph(os.path.join(graph_fold_dir, 'E_sim.csv'))
    e_co = load_graph(os.path.join(graph_fold_dir, 'E_co.csv'))
    
    def add_edges(df, alpha):
        if df.empty or alpha == 0: return
        src_col = 'src' if 'src' in df.columns else 'src_skill_id'
        dst_col = 'dst' if 'dst' in df.columns else 'dst_skill_id'
        weight_col = 'weight' if 'weight' in df.columns else 'confidence'
        
        for _, row in df.iterrows():
            src = str(row[src_col])
            dst = str(row[dst_col])
            w = float(row.get(weight_col, 1.0))
            if src in degree_map: degree_map[src] += alpha * w
            if dst in degree_map: degree_map[dst] += alpha * w
            
    add_edges(e_pre, ap)
    add_edges(e_sim, asim)
    add_edges(e_co, ac)
    
    max_deg = max(degree_map.values()) if degree_map.values() else 1.0
    if max_deg == 0: max_deg = 1.0
    return {k: v / max_deg for k, v in degree_map.items()}

def main():
    print("=== PLOTTING REPRESENTATIVE MULTI-EPOCH TRAINING CURVES ===")
    
    datasets = ["assist2012", "junyi"]
    models = ["dkt", "simplekt"]
    fold = 1
    seed = 42
    epochs = 10
    
    selected_gates_map = {
        ("assist2012", "dkt"): (1.0, 1.0, 0.0), # e_pre_e_sim
        ("assist2012", "simplekt"): (1.0, 0.0, 0.0), # e_pre
        ("junyi", "dkt"): (1.0, 0.0, 0.0), # e_pre
        ("junyi", "simplekt"): (1.0, 1.0, 0.0), # e_pre_e_sim
    }
    
    curve_data = []
    
    # Grid setup for 2x2 summary plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    ax_idx = 0
    
    for ds in datasets:
        data_dir = f"data/processed/{ds}/fold_{fold}"
        train_df = pd.read_csv(f"{data_dir}/train.csv")
        valid_df = pd.read_csv(f"{data_dir}/valid.csv")
        test_df = pd.read_csv(f"{data_dir}/test.csv")
        
        unique_skills = sorted(list(set(train_df['skill_id'].unique()) | set(test_df['skill_id'].unique())))
        skill_to_id = {sk: i for i, sk in enumerate(unique_skills)}
        id_to_skill = {i: sk for sk, i in skill_to_id.items()}
        num_skills = len(unique_skills)
        
        graph_fold_dir = f"runs/q3_lcmrsg_plus_20260528_234100/graphs/{ds}/fold_{fold}"
        if not os.path.exists(graph_fold_dir):
            graph_fold_dir = f"graphs/{ds}/fold_{fold}"
            
        for backbone in models:
            ap, asim, ac = selected_gates_map[(ds, backbone)]
            deg_map = get_degree_map_helper(graph_fold_dir, ds, ap, asim, ac, unique_skills)
            
            # Prepare datasets
            tr_seq = prepare_data(train_df, skill_to_id, deg_map)
            val_seq = prepare_data(valid_df, skill_to_id, deg_map)
            
            # Setup model
            embed_dim = 16
            hidden_dim = 16
            lr = 0.05
            batch_size = 1024
            
            set_seed(seed)
            model = NeuralKT(num_skills, embed_dim, hidden_dim, backbone)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.BCELoss()
            
            epoch_logs = []
            num_items = len(tr_seq[0])
            
            # Train and evaluate epoch-by-epoch
            for epoch in range(1, epochs + 1):
                model.train()
                epoch_loss = 0.0
                count = 0
                for i in range(0, num_items, batch_size):
                    batch_x = tr_seq[0][i:i+batch_size]
                    batch_d = tr_seq[1][i:i+batch_size]
                    batch_y_sk = tr_seq[2][i:i+batch_size]
                    batch_y_lb = tr_seq[3][i:i+batch_size]
                    
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
                
                mean_train_loss = epoch_loss / count if count > 0 else 0.0
                val_auc = evaluate_neural(model, val_seq, id_to_skill)
                epoch_logs.append((epoch, mean_train_loss, val_auc))
                
                curve_data.append({
                    "dataset": ds,
                    "model": backbone,
                    "epoch": epoch,
                    "train_loss": mean_train_loss,
                    "valid_auc": val_auc
                })
                
            del model
            gc.collect()
            
            # Convert to DataFrame for plotting
            df_curve = pd.DataFrame(epoch_logs, columns=["epoch", "train_loss", "valid_auc"])
            
            # Save individual plot PDF
            fig_ind, ax_loss = plt.subplots(figsize=(6, 4))
            ax_auc = ax_loss.twinx()
            
            color_loss = '#C44E52'
            color_auc = '#4C72B0'
            
            ax_loss.plot(df_curve["epoch"], df_curve["train_loss"], marker='o', color=color_loss, label='Train Loss')
            ax_auc.plot(df_curve["epoch"], df_curve["valid_auc"], marker='s', color=color_auc, label='Valid AUC')
            
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('Train Loss', color=color_loss)
            ax_loss.tick_params(axis='y', labelcolor=color_loss)
            ax_auc.set_ylabel('Validation AUC', color=color_auc)
            ax_auc.tick_params(axis='y', labelcolor=color_auc)
            
            plt.title(f"{ds.upper()} - {backbone.upper()} Learning Curves (Fold 1)")
            plt.tight_layout()
            
            out_dir_ind = "outputs/figures/supplementary/training_curves"
            os.makedirs(out_dir_ind, exist_ok=True)
            plt.savefig(f"{out_dir_ind}/curve_{ds}_{backbone}_fold1_seed42.pdf")
            plt.close(fig_ind)
            print(f"Saved individual curve: {out_dir_ind}/curve_{ds}_{backbone}_fold1_seed42.pdf")
            
            # Plot on the 2x2 grid
            ax = axes[ax_idx]
            ax_twin = ax.twinx()
            ax.plot(df_curve["epoch"], df_curve["train_loss"], marker='o', color=color_loss, label='Train Loss', linewidth=1.5)
            ax_twin.plot(df_curve["epoch"], df_curve["valid_auc"], marker='s', color=color_auc, label='Valid AUC', linewidth=1.5)
            
            ax.set_xlabel('Epoch')
            if ax_idx % 2 == 0:
                ax.set_ylabel('Train Loss', color=color_loss)
            ax.tick_params(axis='y', labelcolor=color_loss)
            
            if ax_idx % 2 == 1:
                ax_twin.set_ylabel('Validation AUC', color=color_auc)
            ax_twin.tick_params(axis='y', labelcolor=color_auc)
            
            ax.set_title(f"{ds.upper()} - {backbone.upper()} (Fold 1)")
            ax.set_xticks(range(1, epochs + 1))
            ax_idx += 1
            
    # Save the 2x2 summary grid
    fig.suptitle("LC-MRSG++ Representative Learning Curves", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/training_curves_summary.pdf", bbox_inches='tight')
    plt.close(fig)
    print("Saved training curves summary grid: figures/training_curves_summary.pdf")
    
    # Save CSV table of the curves
    df_curves_all = pd.DataFrame(curve_data)
    os.makedirs("results_p0_revision/tables_csv", exist_ok=True)
    df_curves_all.to_csv("results_p0_revision/tables_csv/table_epoch_sanity_curves.csv", index=False)
    print("Saved curve data CSV: results_p0_revision/tables_csv/table_epoch_sanity_curves.csv")
    print("=== FINISHED PLOTTING TRAINING CURVES ===")

if __name__ == "__main__":
    main()
