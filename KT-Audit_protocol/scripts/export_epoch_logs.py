import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, log_loss

def main():
    log_out = Path("outputs/diagnostics/training_logs.csv")
    pred_out = Path("outputs/diagnostics/prediction_stats.csv")
    
    log_out.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Export epoch-level logs
    log_rows = []
    base_log_dir = Path("results/logs")
    if base_log_dir.exists():
        for f in base_log_dir.glob("*/*/*/*/*.csv"):
            parts = f.parts
            dataset = parts[2]
            fold_str = parts[3]
            fold = int(fold_str.replace("fold_", ""))
            seed_str = parts[4]
            seed = int(seed_str.replace("seed_", ""))
            model = parts[5]
            graph_variant = f.stem
            
            run_id = f"{dataset}_f{fold}_s{seed}_{model}_{graph_variant}"
            
            try:
                df = pd.read_csv(f)
                for idx, row in df.iterrows():
                    log_rows.append({
                        "dataset": dataset,
                        "fold": fold,
                        "seed": seed,
                        "model": model,
                        "graph_variant": graph_variant,
                        "epoch": row.get("epoch", idx),
                        "train_loss": row.get("train_loss", np.nan),
                        "valid_loss": row.get("valid_loss", np.nan),
                        "valid_auc": row.get("valid_auc", np.nan),
                        "valid_acc": row.get("valid_acc", np.nan),
                        "lr": row.get("lr", 0.001),
                        "grad_norm": row.get("gradient_norm", np.nan),
                        "best_epoch": 0,
                        "early_stop_flag": False,
                        "run_id": run_id
                    })
            except Exception as e:
                print(f"Error reading {f}: {e}")
                
    pd.DataFrame(log_rows).to_csv(log_out, index=False)
    print(f"Saved {len(log_rows)} rows to {log_out}")

    # 2. Export prediction statistics
    pred_rows = []
    base_pred_dir = Path("results/predictions")
    if base_pred_dir.exists():
        for f in base_pred_dir.glob("*/*/*/*/*.csv"):
            parts = f.parts
            dataset = parts[2]
            fold_str = parts[3]
            fold = int(fold_str.replace("fold_", ""))
            seed_str = parts[4]
            seed = int(seed_str.replace("seed_", ""))
            model = parts[5]
            graph_variant = f.stem
            
            run_id = f"{dataset}_f{fold}_s{seed}_{model}_{graph_variant}"
            
            try:
                df = pd.read_csv(f)
                y_true = df["y_true"].values
                y_pred = df["y_pred"].values
                
                pred_std = np.std(y_pred)
                pred_min = np.min(y_pred)
                pred_max = np.max(y_pred)
                y_mean = np.mean(y_true)
                
                try:
                    auc = roc_auc_score(y_true, y_pred)
                except ValueError:
                    auc = np.nan
                
                acc = accuracy_score(y_true, y_pred > 0.5)
                try:
                    nll = log_loss(y_true, y_pred)
                except Exception:
                    nll = np.nan
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                
                pred_rows.append({
                    "run_id": run_id,
                    "dataset": dataset,
                    "fold": fold,
                    "seed": seed,
                    "model": model,
                    "graph_variant": graph_variant,
                    "pred_std": pred_std,
                    "pred_min": pred_min,
                    "pred_max": pred_max,
                    "y_mean": y_mean,
                    "auc": auc,
                    "acc": acc,
                    "nll": nll,
                    "rmse": rmse
                })
            except Exception as e:
                print(f"Error reading {f}: {e}")
                
    pd.DataFrame(pred_rows).to_csv(pred_out, index=False)
    print(f"Saved {len(pred_rows)} rows to {pred_out}")

if __name__ == "__main__":
    main()
