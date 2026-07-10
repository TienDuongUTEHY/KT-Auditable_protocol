import pandas as pd
import numpy as np
from pathlib import Path

LOG_PATH = Path("outputs/diagnostics/training_logs.csv")
PRED_PATH = Path("outputs/diagnostics/prediction_stats.csv")
OUT_PATH = Path("outputs/diagnostics/training_integrity_summary.csv")

logs = pd.read_csv(LOG_PATH)
preds = pd.read_csv(PRED_PATH)

required = [
    "dataset", "fold", "seed", "model", "graph_variant", "epoch",
    "train_loss", "valid_loss", "valid_auc", "valid_acc", "run_id"
]
missing = [c for c in required if c not in logs.columns]
if missing:
    raise ValueError(f"Missing required columns in training_logs.csv: {missing}")

rows = []
for run_id, g in logs.groupby("run_id"):
    g = g.sort_values("epoch")
    first = g.iloc[0]
    last = g.iloc[-1]
    best_valid_auc = g["valid_auc"].max()

    no_nan = np.isfinite(g[["train_loss", "valid_loss", "valid_auc", "valid_acc"]].to_numpy().astype(float)).all()
    loss_decreased = last["train_loss"] <= first["train_loss"] * 0.99
    auc_improved = best_valid_auc >= first["valid_auc"] + 0.005

    pred_row = preds[preds["run_id"] == run_id]
    pred_std = float(pred_row["pred_std"].iloc[0]) if len(pred_row) else np.nan
    non_constant = pred_std > 0.01

    status = "PASS" if (no_nan and non_constant and (loss_decreased or auc_improved)) else "WARN"

    rows.append({
        "run_id": run_id,
        "dataset": first["dataset"],
        "fold": first["fold"],
        "seed": first["seed"],
        "model": first["model"],
        "graph_variant": first["graph_variant"],
        "first_train_loss": first["train_loss"],
        "final_train_loss": last["train_loss"],
        "first_valid_auc": first["valid_auc"],
        "best_valid_auc": best_valid_auc,
        "pred_std": pred_std,
        "no_nan": no_nan,
        "loss_decreased_1pct": loss_decreased,
        "valid_auc_improved_0p005": auc_improved,
        "non_constant_prediction": non_constant,
        "status": status,
    })

summary = pd.DataFrame(rows)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(OUT_PATH, index=False)
print(summary["status"].value_counts(dropna=False))
print(f"Saved: {OUT_PATH}")
