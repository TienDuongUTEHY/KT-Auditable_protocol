# Additional Experiment Plan for LC-MRSG using Google Gravity / Colab

## 0. Revision Goals

The goal of this work package is to address 6 potential criticisms from reviewers:

1. Training-integrity diagnostics are currently questionable.
2. Clearly define the scope difference between dataset-scale audit and fold-level plots.
3. Eco provenance has a `FLAG` which needs to be correctly interpreted as an audit provenance flag, not held-out data leakage.
4. Sparse-skill claims must be written conservatively.
5. KDD2010 BKT is statistically significant but has a practically negligible effect.
6. The appendix contains too many tables and figures; representative figures should remain in the main text while full folds should be moved to the supplementary material.

Expected outcome: The reviewer sees the paper as honest, leakage-controlled, with realistic performance claims, and backed by training diagnostics in the appendix to remove doubts about training integrity.

---

## 1. Directory Structure

Create the following structure in the project:

```text
scientific_paper1/
  data/
    processed/
    splits/
  outputs/
    results/
      all_runs.csv
      full_graph_results.csv
      paired_tests.csv
      sparse_strata_auc.csv
    diagnostics/
      training_logs.csv
      prediction_stats.csv
      consistency_checks.csv
      leakage_audit.csv
      eco_provenance_raw.csv
      eco_provenance_fixed.csv
      density_audit.csv
    figures/
      main/
      appendix/
      supplementary/
  scripts/
    export_epoch_logs.py
    plot_training_curves.py
    audit_training_integrity.py
    mirror_eco_edges.py
    audit_eco_provenance.py
    build_sparse_skill_tables.py
    organize_figures_for_paper.py
  paper/
    LC_MRSG_Q3_revised_main.tex
```

---

## 2. Task A - Export Epoch-Level Training Logs

### 2.1. Rationale

The current strict training-integrity export is flagged as `FAIL` because the `training_loss_decrease` rule is not met in the probe log. However, since the prediction standard deviation is non-zero, we cannot conclude that the model suffers from constant-output collapse. A Q3 reviewer will likely accept this if we provide epoch-level logs or learning curves to prove that valid training progress occurred.

### 2.2. Mandatory Schema for `training_logs.csv`

Each row represents one epoch of a training run configuration:

```text
dataset,fold,seed,model,graph_variant,epoch,train_loss,valid_loss,valid_auc,valid_acc,lr,grad_norm,best_epoch,early_stop_flag,run_id
```

Example:

```text
assist2012,0,42,gikt,E_pre_E_sim_E_co,1,0.6941,0.6935,0.5012,0.6901,0.001,1.82,0,False,assist2012_f0_s42_gikt_full
assist2012,0,42,gikt,E_pre_E_sim_E_co,2,0.6813,0.6802,0.5634,0.6912,0.001,1.54,0,False,assist2012_f0_s42_gikt_full
```

### 2.3. Realistic Success Criteria (Relaxed Monotonic Rule)

We do not require the training loss to decrease at every single epoch. Instead, use a reviewer-safe check:

- `pred_std > 0.01` to exclude constant predictions.
- `final_train_loss <= first_train_loss * 0.99` OR `best_valid_auc >= first_valid_auc + 0.005`.
- No NaN/Inf values in `train_loss`, `valid_loss`, `valid_auc`, or `valid_acc`.
- `best_epoch` falls within the valid range of trained epochs.

### 2.4. Training Integrity Audit Script

Create `scripts/audit_training_integrity.py`:

```python
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

    no_nan = np.isfinite(g[["train_loss", "valid_loss", "valid_auc", "valid_acc"]].to_numpy()).all()
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
```

---

## 3. Task B - Plot Learning Curves

### 3.1. Objective

Generate learning curves for Appendix C or Supplementary Material:

- Training loss by epoch.
- Validation loss by epoch.
- Validation AUC by epoch.
- Mark the best epoch.

### 3.2. Learning Curves Script

Create `scripts/plot_training_curves.py`:

```python
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

LOG_PATH = Path("outputs/diagnostics/training_logs.csv")
OUT_DIR = Path("outputs/figures/supplementary/training_curves")
OUT_DIR.mkdir(parents=True, exist_ok=True)

logs = pd.read_csv(LOG_PATH)

# Representative curves: full graph, fold 0, first seed for each dataset-model
rep = logs[logs["graph_variant"].isin(["E_pre_E_sim_E_co", "LC-MRSG", "full"])]
if rep.empty:
    rep = logs.copy()

for (dataset, model), g0 in rep.groupby(["dataset", "model"]):
    # Select fold 0 if available; otherwise use the minimum fold index
    fold = 0 if 0 in set(g0["fold"]) else sorted(g0["fold"].unique())[0]
    g1 = g0[g0["fold"] == fold]
    seed = sorted(g1["seed"].unique())[0]
    g = g1[g1["seed"] == seed].sort_values("epoch")

    if g.empty:
        continue

    for metric in ["train_loss", "valid_loss", "valid_auc"]:
        plt.figure(figsize=(6, 4))
        plt.plot(g["epoch"], g[metric], marker="o", linewidth=1)
        if metric == "valid_auc":
            best_idx = g[metric].idxmax()
        else:
            best_idx = g[metric].idxmin()
        plt.axvline(g.loc[best_idx, "epoch"], linestyle="--", linewidth=1)
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(f"{dataset} - {model} - fold {fold} - seed {seed} - {metric}")
        plt.tight_layout()
        out = OUT_DIR / f"curve_{dataset}_{model}_fold{fold}_seed{seed}_{metric}.pdf"
        plt.savefig(out)
        plt.close()
        print(f"Saved {out}")
```

### 3.3. Compiling PDF Curves for LaTeX

After generating the PDF files, combine the 6-9 representative curves:

```bash
python scripts/plot_training_curves.py
```

Then include it in LaTeX:

```latex
\includegraphics[width=0.95\textwidth]{figures/training_curves_summary.pdf}
```

A placeholder for `figures/training_curves_summary.pdf` is already defined in the LaTeX file. Once the actual figures are generated, place them at that path.

---

## 4. Task C - Resolving the Eco Provenance FLAG

### 4.1. Scientific Interpretation

Do not write `Eco FAIL = leakage` in the paper. Instead, write:

> FLAG indicates one-direction storage or missing train-only support metadata in the raw provenance table. It is an audit/provenance flag, not observed held-out leakage.

### 4.2. Standard Schema for `eco_provenance_fixed.csv`

```text
dataset,fold,src_skill,dst_skill,weight,pmi,support_count,train_only_support,support_hash,is_mirrored,edge_source
```

### 4.3. Mirrored Undirected Eco Edges Script

Create `scripts/mirror_eco_edges.py`:

```python
import pandas as pd
from pathlib import Path

IN_PATH = Path("outputs/diagnostics/eco_provenance_raw.csv")
OUT_PATH = Path("outputs/diagnostics/eco_provenance_fixed.csv")

df = pd.read_csv(IN_PATH)

# Normalize column names if needed
rename_map = {
    "src": "src_skill",
    "dst": "dst_skill",
    "source": "src_skill",
    "target": "dst_skill",
}
df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

required = ["dataset", "fold", "src_skill", "dst_skill", "weight"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

for col, default in [
    ("support_count", -1),
    ("train_only_support", "UNKNOWN"),
    ("support_hash", "UNKNOWN"),
    ("edge_source", "train_cooccurrence"),
]:
    if col not in df.columns:
        df[col] = default

forward = df.copy()
forward["is_mirrored"] = False

reverse = df.copy()
reverse[["src_skill", "dst_skill"]] = reverse[["dst_skill", "src_skill"]]
reverse["is_mirrored"] = True

fixed = pd.concat([forward, reverse], ignore_index=True)
fixed = fixed.drop_duplicates(subset=["dataset", "fold", "src_skill", "dst_skill"], keep="first")
fixed.to_csv(OUT_PATH, index=False)

print("Raw edges:", len(df))
print("Fixed directed-storage edges:", len(fixed))
print(f"Saved: {OUT_PATH}")
```

### 4.4. Eco Provenance Re-Audit Script

Create `scripts/audit_eco_provenance.py`:

```python
import pandas as pd
from pathlib import Path

PATH = Path("outputs/diagnostics/eco_provenance_fixed.csv")
OUT = Path("outputs/diagnostics/eco_provenance_fixed_audit.csv")
df = pd.read_csv(PATH)

rows = []
for (dataset, fold), g in df.groupby(["dataset", "fold"]):
    pairs = set(zip(g["src_skill"], g["dst_skill"]))
    missing_reverse = 0
    for s, t in pairs:
        if (t, s) not in pairs and s != t:
            missing_reverse += 1

    train_only_ok = (g["train_only_support"].astype(str).str.upper() == "TRUE").mean()
    unknown_support = (g["train_only_support"].astype(str).str.upper() == "UNKNOWN").sum()

    status = "PASS" if missing_reverse == 0 and unknown_support == 0 else "FLAG"

    rows.append({
        "dataset": dataset,
        "fold": fold,
        "edges": len(g),
        "missing_reverse_edges": missing_reverse,
        "unknown_train_only_support_rows": int(unknown_support),
        "train_only_support_true_rate": train_only_ok,
        "status": status,
    })

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print(out)
print(f"Saved: {OUT}")
```

---

## 5. Task D - Sparse-Skill Claims

### 5.1. Target Text to Use in Paper

Use conservative phrasing:

> The evidence supports sparse-skill diagnostics rather than a universal claim that LC-MRSG improves sparse-skill prediction.

Do not use:

> LC-MRSG improves sparse-skill prediction.

### 5.2. Rationale

- ASSIST2012 has clearly defined sparse/very sparse strata.
- Junyi has no very-sparse or sparse test strata in the final export.
- KDD2010 exhibits unstable sparse-stratum results for multiple neural/graph-aware KT models.

---

## 6. Task E - KDD2010 BKT: Statistically Significant but Practically Negligible

### 6.1. Target Text to Use

In the Results and Discussion sections, write:

> KDD2010 BKT is statistically significant but practically tiny, with ΔAUC around +0.0001; it should not be treated as a meaningful performance improvement.

### 6.2. Proposed Interpretation Thresholds

- `|ΔAUC| < 0.001`: practically tiny.
- `0.001 <= |ΔAUC| < 0.005`: small.
- `0.005 <= |ΔAUC| < 0.01`: moderate for KT benchmarks.
- `|ΔAUC| >= 0.01`: practically meaningful, requires further CI and stability audits.

---

## 7. Task F - Reorganizing Main Text, Appendix, and Supplementary Material

### 7.1. Main Text Content

- Table dataset-scale audit.
- Table multi-fold performance.
- Table paired test.
- Representative pipeline figure.
- Representative relation-ablation figure.
- Leakage summary table/figure.
- Eco provenance table.
- Prerequisite density table.
- Sparse-strata representative figure/table.
- Training-integrity summary table.

### 7.2. Appendix Content

- Fold-level Eco provenance audit.
- Training-integrity by dataset-model.
- Naming convention of full fold-level figures.
- Training diagnostics reproducibility checklist.

### 7.3. Supplementary Material (External files)

- All fold-level figures for 3 datasets x 3 folds x 6 figure types.
- Full epoch logs.
- Full learning curves.
- Full prediction files (if allowed by the journal).

---

## 8. Master Prompt for Google Gravity / Antigravity

Copy and paste the following prompt into Google Gravity / Antigravity:

```text
You are editing the scientific_paper1 project for a knowledge tracing paper titled "Leakage-Controlled Multi-Relational Skill Graph Construction for Sparse-Skill Knowledge Tracing".

Objective: implement reviewer-safe diagnostics and artifacts for the LC-MRSG revision.

Tasks:
1. Export epoch-level training logs for every run with columns: dataset, fold, seed, model, graph_variant, epoch, train_loss, valid_loss, valid_auc, valid_acc, lr, grad_norm, best_epoch, early_stop_flag, run_id. Save to outputs/diagnostics/training_logs.csv.
2. Export prediction statistics for every run with columns: run_id, dataset, fold, seed, model, graph_variant, pred_std, pred_min, pred_max, y_mean, auc, acc, nll, rmse. Save to outputs/diagnostics/prediction_stats.csv.
3. Implement scripts/audit_training_integrity.py using the rule: pred_std > 0.01 and no NaN/Inf and either final_train_loss <= first_train_loss * 0.99 or best_valid_auc >= first_valid_auc + 0.005. Save outputs/diagnostics/training_integrity_summary.csv.
4. Implement scripts/plot_training_curves.py to generate representative learning curves for train_loss, valid_loss, and valid_auc. Save to outputs/figures/supplementary/training_curves/ and create figures/training_curves_summary.pdf for the paper.
5. Implement scripts/mirror_eco_edges.py to create mirrored undirected storage for Eco edges. Save outputs/diagnostics/eco_provenance_fixed.csv.
6. Implement scripts/audit_eco_provenance.py to verify mirrored Eco storage and train-only support metadata. Save outputs/diagnostics/eco_provenance_fixed_audit.csv.
7. Ensure manuscript wording uses: "supports sparse-skill diagnostics" instead of "improves sparse-skill prediction".
8. Ensure Results states: "KDD2010 BKT is statistically significant but practically tiny".
9. Organize figures so main text contains representative figures only; full folds go to appendix/supplementary.
10. Do not fabricate learning curves. If raw epoch logs are unavailable, keep the training-curve figure as a placeholder and report the limitation honestly.

Acceptance criteria:
- No claim of universal graph improvement.
- No claim that Eco FLAG is held-out leakage.
- Training diagnostics distinguish logging warning from constant-output collapse.
- LaTeX compiles without missing figures when figures/training_curves_summary.pdf is absent.
- All generated CSV and figure paths are documented in README or supplementary index.
```

---

## 9. Post-Execution LaTeX Guidelines

1. Copy the generated `figures/training_curves_summary.pdf` into the `figures/` directory beside the `.tex` file.
2. If the Eco fixed audit PASSES, update the Table Eco provenance entry from `FLAG` to `PASS_FIXED`, but explain that the raw export originally flagged it.
3. If the training integrity summary has multiple PASS entries, split Table 8 into two columns: `Strict probe status` and `Epoch-log status`.
4. If it still WARNS, keep the conservative wording: `training-log warning, not constant-output collapse`.

---

## 10. Pre-Submission Checklist

- [ ] Abstract does not state universal improvement.
- [ ] Introduction explicitly states graph leakage risks.
- [ ] Protocol includes split-first condition.
- [ ] Dataset-scale audits and fold-level plots are clearly distinguished by: `Scale audit reports benchmark-level processed availability; fold plots report representative fold artifacts.`
- [ ] Eco FLAG is explained as one-direction storage or missing metadata, not held-out leakage.
- [ ] Sparse-skill claims use the phrasing `supports sparse-skill diagnostics`.
- [ ] KDD2010 BKT is stated as statistically significant but practically tiny.
- [ ] Main text does not overload fold-level supplementary figures.
- [ ] Appendix includes training diagnostics or learning curves.
- [ ] Supplementary material contains all fold figures.
- [ ] LaTeX compiles cleanly.
