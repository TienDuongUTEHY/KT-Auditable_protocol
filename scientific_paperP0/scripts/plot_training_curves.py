import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

LOG_PATH = Path("outputs/diagnostics/training_logs.csv")
OUT_DIR = Path("outputs/figures/supplementary/training_curves")
OUT_DIR.mkdir(parents=True, exist_ok=True)

logs = pd.read_csv(LOG_PATH)

# Representative curves: full graph, fold 0, seed đầu tiên theo từng dataset-model
rep = logs[logs["graph_variant"].isin(["E_pre_E_sim_E_co", "LC-MRSG", "full"])]
if rep.empty:
    rep = logs.copy()

for (dataset, model), g0 in rep.groupby(["dataset", "model"]):
    # Chọn fold 0 nếu có; nếu không lấy fold nhỏ nhất
    fold = 0 if 0 in set(g0["fold"]) else sorted(g0["fold"].unique())[0]
    g1 = g0[g0["fold"] == fold]
    seed = sorted(g1["seed"].unique())[0]
    g = g1[g1["seed"] == seed].sort_values("epoch")

    if g.empty:
        continue

    for metric in ["train_loss", "valid_loss", "valid_auc"]:
        plt.figure(figsize=(6, 4))
        plt.plot(g["epoch"], g[metric], marker="o", linewidth=1)
        if g[metric].notna().any():
            if metric == "valid_auc":
                best_idx = g[metric].idxmax()
            else:
                best_idx = g[metric].idxmin()
                
            if not pd.isna(best_idx):
                plt.axvline(g.loc[best_idx, "epoch"], linestyle="--", linewidth=1)
        
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(f"{dataset} - {model} - fold {fold} - seed {seed} - {metric}")
        plt.tight_layout()
        out = OUT_DIR / f"curve_{dataset}_{model}_fold{fold}_seed{seed}_{metric}.pdf"
        plt.savefig(out)
        plt.close()
        print(f"Saved {out}")
