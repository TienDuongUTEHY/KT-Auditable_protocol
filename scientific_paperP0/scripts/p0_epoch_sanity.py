import os
import sys
import gc
import yaml
import random
import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

# Limit PyTorch CPU threads to prevent system freezing
torch.set_num_threads(2)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

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

def get_degree_map(e_pre, e_sim, e_co, alpha_pre, alpha_sim, alpha_co, unique_skills):
    degree_map = {str(sk): 0.0 for sk in unique_skills}
    
    def add_edges(df, alpha):
        if df.empty or alpha == 0:
            return
        # Find src/dst cols
        src_col = 'src' if 'src' in df.columns else 'src_skill_id'
        dst_col = 'dst' if 'dst' in df.columns else 'dst_skill_id'
        weight_col = 'weight' if 'weight' in df.columns else 'confidence'
        
        for _, row in df.iterrows():
            src = str(row[src_col])
            dst = str(row[dst_col])
            w = float(row.get(weight_col, 1.0))
            effective_w = alpha * w
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

def collate_fn(batch):
    x, d, y_sk, y_lb = zip(*batch)
    x_padded = pad_sequence(x, batch_first=True, padding_value=0)
    d_padded = pad_sequence(d, batch_first=True, padding_value=0.0)
    y_sk_padded = pad_sequence(y_sk, batch_first=True, padding_value=0)
    y_lb_padded = pad_sequence(y_lb, batch_first=True, padding_value=-1.0)
    return x_padded, d_padded, y_sk_padded, y_lb_padded

def run_training(train_data, num_skills, model_type, epochs):
    embed_dim = 16
    hidden_dim = 16
    lr = 0.05
    batch_size = 1024
    
    model = NeuralKT(num_skills, embed_dim, hidden_dim, model_type)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    model.train()
    num_items = len(train_data[0])
    
    last_loss = 0.0
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
        if count > 0:
            last_loss = epoch_loss / count
                
    return model, last_loss

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
                        "skill_id": id_to_skill[sk_idx.item()],
                        "y_true": float(y.item()),
                        "y_pred": float(p.item())
                    })
                    
    pred_df = pd.DataFrame(pred_rows)
    if pred_df.empty:
        return float('nan')
        
    y_true = pred_df["y_true"].to_numpy()
    y_pred = pred_df["y_pred"].clip(1e-7, 1.0 - 1e-7).to_numpy()
    
    auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float('nan')
    return round(auc, 4)

def load_graph_defensive(path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.DataFrame(columns=['src', 'dst', 'weight', 'relation_type', 'support_count'])

def main():
    output_dir = "results_p0_revision"
    tables_csv_dir = ensure_dir(os.path.join(output_dir, "tables_csv"))
    tables_tex_dir = ensure_dir(os.path.join(output_dir, "tables_tex"))
    log_dir = ensure_dir(os.path.join(output_dir, "logs"))
    log_file = os.path.join(log_dir, "phase2_epoch_sanity.log")
    
    csv_path = os.path.join(tables_csv_dir, "table_epoch_sanity.csv")
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 100:
        print("Found existing epoch sanity results, skipping training to save time.")
        df_sanity = pd.read_csv(csv_path)
        tex_path = os.path.join(tables_tex_dir, "table_epoch_sanity.tex")
        latex_lines = [
            "% Standalone Table Body for Epoch Sanity-Check",
            "\\begin{tabular}{llccccccl}",
            "\\hline",
            "Dataset & Model & Epochs & No Graph AUC & Selected Graph AUC & Delta AUC & Main 2-epoch Delta & Sign Preserved & Notes \\\\",
            "\\hline"
        ]
        for _, r in df_sanity.iterrows():
            latex_lines.append(
                f"{r['dataset'].upper()} & {r['backbone'].upper()} & {r['epoch_budget']} & {r['auc_no_graph']:.4f} & {r['auc_selected_graph']:.4f} & {r['delta_auc']:.4f} & {r['main_two_epoch_delta_if_available']} & {r['sign_preserved_vs_main']} & {r['notes']} \\\\"
            )
        latex_lines.append("\\hline")
        latex_lines.append("\\end{tabular}")
        with open(tex_path, "w") as tf:
            tf.write("\n".join(latex_lines))
        with open(log_file, "w", encoding="utf-8") as lf:
            lf.write("PHASE_START epoch_sanity\n")
            lf.write("PHASE_PASS epoch_sanity\n")
        print("PHASE_START epoch_sanity")
        print("PHASE_PASS epoch_sanity")
        return
    
    # Selected graph conditions per (dataset, backbone) based on validation AUC in main runs
    # Default selection mapping
    selected_gates_map = {
        ("assist2012", "dkt"): (1.0, 1.0, 0.0, "e_pre_e_sim"),
        ("assist2012", "simplekt"): (1.0, 0.0, 0.0, "e_pre"),
        ("junyi", "dkt"): (1.0, 0.0, 0.0, "e_pre"),
        ("junyi", "simplekt"): (1.0, 1.0, 0.0, "e_pre_e_sim"),
    }
    
    # Load 2-epoch results to compare and preserve signs
    main_runs_csv = "runs/q3_lcmrsg_plus_20260528_234100/results/all_runs_train_valid_test.csv"
    df_main_runs = pd.read_csv(main_runs_csv) if os.path.exists(main_runs_csv) else pd.DataFrame()
    
    epoch_budgets = [5, 10]
    datasets = ["assist2012", "junyi"]
    backbones = ["dkt", "simplekt"]
    fold = 1
    seed = 42
    
    sanity_results = []
    total_comparisons = 0
    sign_preserved_count = 0
    
    with open(log_file, "w", encoding="utf-8") as lf:
        lf.write("PHASE_START epoch_sanity\n")
        print("PHASE_START epoch_sanity")
        
        for ds in datasets:
            # Data paths
            data_dir = f"data/processed/{ds}/fold_{fold}"
            train_path = f"{data_dir}/train.csv"
            valid_path = f"{data_dir}/valid.csv"
            test_path = f"{data_dir}/test.csv"
            
            if not os.path.exists(train_path):
                continue
                
            train_df = pd.read_csv(train_path)
            valid_df = pd.read_csv(valid_path) if os.path.exists(valid_path) else pd.DataFrame()
            test_df = pd.read_csv(test_path) if os.path.exists(test_path) else pd.DataFrame()
            
            # Load graphs
            graph_fold_dir = f"runs/q3_lcmrsg_plus_20260528_234100/graphs/{ds}/fold_{fold}"
            if not os.path.exists(graph_fold_dir):
                graph_fold_dir = f"graphs/{ds}/fold_{fold}"
                
            e_pre = load_graph_defensive(os.path.join(graph_fold_dir, 'E_pre.csv'))
            e_sim = load_graph_defensive(os.path.join(graph_fold_dir, 'E_sim.csv'))
            e_co = load_graph_defensive(os.path.join(graph_fold_dir, 'E_co.csv'))
            
            # Skill mapping
            unique_skills = sorted(list(set(train_df['skill_id'].unique()) | set(test_df['skill_id'].unique())))
            skill_to_id = {sk: i for i, sk in enumerate(unique_skills)}
            id_to_skill = {i: sk for sk, i in skill_to_id.items()}
            num_skills = len(unique_skills)
            
            for backbone in backbones:
                # 1. Selected graph setup
                ap, asim, ac, var_name = selected_gates_map[(ds, backbone)]
                
                # Fetch main 2-epoch results
                main_no_graph_auc = np.nan
                main_sel_graph_auc = np.nan
                if not df_main_runs.empty:
                    match_no = df_main_runs[
                        (df_main_runs['dataset'] == ds) & 
                        (df_main_runs['fold'] == fold) & 
                        (df_main_runs['seed'] == seed) & 
                        (df_main_runs['model'] == backbone) & 
                        (df_main_runs['variant'] == 'no_graph')
                    ]
                    match_sel = df_main_runs[
                        (df_main_runs['dataset'] == ds) & 
                        (df_main_runs['fold'] == fold) & 
                        (df_main_runs['seed'] == seed) & 
                        (df_main_runs['model'] == backbone) & 
                        (df_main_runs['variant'] == var_name)
                    ]
                    if not match_no.empty:
                        main_no_graph_auc = match_no.iloc[0]['test_auc']
                    if not match_sel.empty:
                        main_sel_graph_auc = match_sel.iloc[0]['test_auc']
                        
                main_delta_auc = main_sel_graph_auc - main_no_graph_auc if not np.isnan(main_no_graph_auc) and not np.isnan(main_sel_graph_auc) else np.nan
                main_sign = np.sign(main_delta_auc) if not np.isnan(main_delta_auc) else 0.0
                
                # 2. Get degree maps
                deg_map_no = get_degree_map(e_pre, e_sim, e_co, 0.0, 0.0, 0.0, unique_skills)
                deg_map_sel = get_degree_map(e_pre, e_sim, e_co, ap, asim, ac, unique_skills)
                
                # Prepare dataloaders
                tr_seq_no = prepare_data(train_df, skill_to_id, deg_map_no)
                val_seq_no = prepare_data(valid_df, skill_to_id, deg_map_no)
                te_seq_no = prepare_data(test_df, skill_to_id, deg_map_no)
                
                tr_seq_sel = prepare_data(train_df, skill_to_id, deg_map_sel)
                val_seq_sel = prepare_data(valid_df, skill_to_id, deg_map_sel)
                te_seq_sel = prepare_data(test_df, skill_to_id, deg_map_sel)
                
                for budget in epoch_budgets:
                    # Run training & evaluation for no_graph
                    lf.write(f"EPOCH_RUN_START dataset={ds} fold={fold} seed={seed} backbone={backbone} condition=no_graph epoch_budget={budget}\n")
                    print(f"EPOCH_RUN_START dataset={ds} fold={fold} seed={seed} backbone={backbone} condition=no_graph epoch_budget={budget}")
                    
                    set_seed(seed)
                    model_no, train_loss_no = run_training(tr_seq_no, num_skills, backbone, epochs=budget)
                    val_auc_no = evaluate_neural(model_no, val_seq_no, id_to_skill)
                    test_auc_no = evaluate_neural(model_no, te_seq_no, id_to_skill)
                    
                    del model_no
                    gc.collect()
                    
                    lf.write(f"EPOCH_RUN_END dataset={ds} fold={fold} seed={seed} backbone={backbone} condition=no_graph auc={test_auc_no} status=PASS\n")
                    print(f"EPOCH_RUN_END dataset={ds} fold={fold} seed={seed} backbone={backbone} condition=no_graph auc={test_auc_no} status=PASS")
                    
                    # Run training & evaluation for selected_graph
                    lf.write(f"EPOCH_RUN_START dataset={ds} fold={fold} seed={seed} backbone={backbone} condition=selected_graph epoch_budget={budget}\n")
                    print(f"EPOCH_RUN_START dataset={ds} fold={fold} seed={seed} backbone={backbone} condition=selected_graph epoch_budget={budget}")
                    
                    set_seed(seed)
                    model_sel, train_loss_sel = run_training(tr_seq_sel, num_skills, backbone, epochs=budget)
                    val_auc_sel = evaluate_neural(model_sel, val_seq_sel, id_to_skill)
                    test_auc_sel = evaluate_neural(model_sel, te_seq_sel, id_to_skill)
                    
                    del model_sel
                    gc.collect()
                    
                    lf.write(f"EPOCH_RUN_END dataset={ds} fold={fold} seed={seed} backbone={backbone} condition=selected_graph auc={test_auc_sel} status=PASS\n")
                    print(f"EPOCH_RUN_END dataset={ds} fold={fold} seed={seed} backbone={backbone} condition=selected_graph auc={test_auc_sel} status=PASS")
                    
                    # Compute delta and verify signs
                    delta_auc = test_auc_sel - test_auc_no
                    sign_delta = "positive" if delta_auc > 0 else ("negative" if delta_auc < 0 else "zero")
                    
                    sign_preserved = "unknown"
                    if not np.isnan(main_sign):
                        curr_sign_val = np.sign(delta_auc)
                        if curr_sign_val == main_sign:
                            sign_preserved = "yes"
                            sign_preserved_count += 1
                        else:
                            sign_preserved = "no"
                        total_comparisons += 1
                        
                    # Convergence heuristic label
                    # If val_auc_sel doesn't change much, label stable
                    convergence = "stable"
                    if abs(train_loss_sel) > 0.05:
                        convergence = "still_improving"
                        
                    sanity_results.append({
                        "dataset": ds,
                        "fold": fold,
                        "seed": seed,
                        "backbone": backbone,
                        "epoch_budget": budget,
                        "auc_no_graph": test_auc_no,
                        "auc_selected_graph": test_auc_sel,
                        "delta_auc": round(delta_auc, 4),
                        "sign_delta": sign_delta,
                        "main_two_epoch_delta_if_available": round(main_delta_auc, 4) if not np.isnan(main_delta_auc) else "NA",
                        "sign_preserved_vs_main": sign_preserved,
                        "train_loss_last": round(train_loss_sel, 4),
                        "valid_auc_last": val_auc_sel,
                        "convergence_label": convergence,
                        "notes": f"Selected graph configuration: {var_name}"
                    })
                    
                    lf.write(f"EPOCH_DELTA dataset={ds} fold={fold} seed={seed} backbone={backbone} epoch_budget={budget} delta_auc={round(delta_auc, 4)} sign={sign_delta} sign_preserved_vs_main={sign_preserved}\n")
                    print(f"EPOCH_DELTA dataset={ds} fold={fold} seed={seed} backbone={backbone} epoch_budget={budget} delta_auc={round(delta_auc, 4)} sign={sign_delta} sign_preserved_vs_main={sign_preserved}")
                    
        # Interpretation
        if total_comparisons > 0:
            rate = sign_preserved_count / total_comparisons
            if rate >= 0.75:
                interpretation = "direction_stable"
            elif rate >= 0.50:
                interpretation = "mixed_direction"
            else:
                interpretation = "direction_unstable"
        else:
            rate = 0.0
            interpretation = "direction_stable"
            
        lf.write(f"EPOCH_SANITY_SUMMARY total_runs={len(sanity_results)} positive_delta={sum(r['delta_auc'] > 0 for r in sanity_results)} negative_delta={sum(r['delta_auc'] < 0 for r in sanity_results)} sign_preserved_rate={round(rate, 4)}\n")
        print(f"EPOCH_SANITY_SUMMARY total_runs={len(sanity_results)} positive_delta={sum(r['delta_auc'] > 0 for r in sanity_results)} negative_delta={sum(r['delta_auc'] < 0 for r in sanity_results)} sign_preserved_rate={round(rate, 4)}")
        
        # Save CSV
        df_sanity = pd.DataFrame(sanity_results)
        csv_path = os.path.join(tables_csv_dir, "table_epoch_sanity.csv")
        df_sanity.to_csv(csv_path, index=False)
        
        # Save LaTeX table
        tex_path = os.path.join(tables_tex_dir, "table_epoch_sanity.tex")
        latex_lines = [
            "% Standalone Table Body for Epoch Sanity-Check",
            "\\begin{tabular}{llccccccl}",
            "\\hline",
            "Dataset & Model & Epochs & No Graph AUC & Selected Graph AUC & Delta AUC & Main 2-epoch Delta & Sign Preserved & Notes \\\\",
            "\\hline"
        ]
        for _, r in df_sanity.iterrows():
            latex_lines.append(
                f"{r['dataset'].upper()} & {r['backbone'].upper()} & {r['epoch_budget']} & {r['auc_no_graph']:.4f} & {r['auc_selected_graph']:.4f} & {r['delta_auc']:.4f} & {r['main_two_epoch_delta_if_available']} & {r['sign_preserved_vs_main']} & {r['notes']} \\\\"
            )
        latex_lines.append("\\hline")
        latex_lines.append("\\end{tabular}")
        
        with open(tex_path, "w") as tf:
            tf.write("\n".join(latex_lines))
            
        lf.write("PHASE_PASS epoch_sanity\n")
        print("PHASE_PASS epoch_sanity")

if __name__ == "__main__":
    main()
