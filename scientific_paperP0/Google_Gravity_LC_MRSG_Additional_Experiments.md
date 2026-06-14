# Kế hoạch xử lý bổ sung cho LC-MRSG bằng Google Gravity / Colab

## 0. Mục tiêu chỉnh sửa

Mục tiêu của gói việc này là xử lý 6 điểm còn dễ bị reviewer bắt bẻ:

1. Training-integrity diagnostic còn gây nghi ngờ.
2. Cần giải thích rõ phạm vi giữa dataset-scale audit và fold-level plots.
3. Eco provenance đang có `FLAG` nhưng phải diễn giải đúng là cờ provenance, không phải held-out leakage.
4. Sparse-skill claim phải viết thận trọng.
5. KDD2010 BKT có ý nghĩa thống kê nhưng hiệu ứng thực tế quá nhỏ.
6. Bảng/hình phụ lục quá nhiều, cần tách representative figures ở main text và full folds ở supplementary.

Kết quả kỳ vọng sau khi chạy bổ sung: reviewer thấy bài báo trung thực, có kiểm soát rò rỉ, không thổi phồng kết quả, và có đủ phụ lục training diagnostics để giảm nghi ngờ về tính toàn vẹn huấn luyện.

---

## 1. Cấu trúc thư mục chuẩn

Tạo cấu trúc sau trong project:

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

## 2. Task A - Xuất epoch-level training logs

### 2.1. Lý do cần làm

Bản hiện tại có strict training-integrity export bị `FAIL` vì rule `training_loss_decrease` không được thỏa trong probe log. Tuy nhiên prediction standard deviation khác 0, nên chưa thể kết luận model bị constant-output collapse. Reviewer Q3 có thể chấp nhận nếu ta bổ sung epoch-level logs hoặc learning curves để chứng minh mô hình có quá trình học hợp lệ.

### 2.2. Schema bắt buộc của file `training_logs.csv`

Mỗi dòng là một epoch của một cấu hình chạy:

```text
dataset,fold,seed,model,graph_variant,epoch,train_loss,valid_loss,valid_auc,valid_acc,lr,grad_norm,best_epoch,early_stop_flag,run_id
```

Ví dụ:

```text
assist2012,0,42,gikt,E_pre_E_sim_E_co,1,0.6941,0.6935,0.5012,0.6901,0.001,1.82,0,False,assist2012_f0_s42_gikt_full
assist2012,0,42,gikt,E_pre_E_sim_E_co,2,0.6813,0.6802,0.5634,0.6912,0.001,1.54,0,False,assist2012_f0_s42_gikt_full
```

### 2.3. Điều kiện pass thực tế hơn strict monotonic rule

Không yêu cầu loss giảm ở mọi epoch. Dùng quy tắc reviewer-safe:

- `pred_std > 0.01` để loại constant prediction.
- `final_train_loss <= first_train_loss * 0.99` hoặc `best_valid_auc >= first_valid_auc + 0.005`.
- Không có NaN/Inf trong `train_loss`, `valid_loss`, `valid_auc`, `valid_acc`.
- `best_epoch` nằm trong khoảng epoch hợp lệ.

### 2.4. Code kiểm tra training integrity

Tạo file `scripts/audit_training_integrity.py`:

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

## 3. Task B - Vẽ learning curves

### 3.1. Mục tiêu

Tạo learning curves để đưa vào Appendix C hoặc Supplementary:

- Train loss theo epoch.
- Valid loss theo epoch.
- Valid AUC theo epoch.
- Đánh dấu best epoch.

### 3.2. Code vẽ learning curves

Tạo file `scripts/plot_training_curves.py`:

```python
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

### 3.3. File tổng hợp để chèn vào LaTeX

Sau khi có nhiều PDF curve, có thể ghép 6-9 curves đại diện thành một file:

```bash
python scripts/plot_training_curves.py
```

Sau đó chèn vào LaTeX:

```latex
\includegraphics[width=0.95\textwidth]{figures/training_curves_summary.pdf}
```

Trong file LaTeX tôi đã để sẵn placeholder `figures/training_curves_summary.pdf`. Khi có hình thật, chỉ cần đặt file đúng đường dẫn đó.

---

## 4. Task C - Sửa Eco provenance FLAG

### 4.1. Diễn giải khoa học

Trong bài báo, không viết `Eco FAIL = leakage`. Viết đúng:

> FLAG indicates one-direction storage or missing train-only support metadata in the raw provenance table. It is an audit/provenance flag, not observed held-out leakage.

### 4.2. Schema chuẩn cho `eco_provenance_fixed.csv`

```text
dataset,fold,src_skill,dst_skill,weight,pmi,support_count,train_only_support,support_hash,is_mirrored,edge_source
```

### 4.3. Code mirror undirected Eco edges

Tạo file `scripts/mirror_eco_edges.py`:

```python
import pandas as pd
from pathlib import Path

IN_PATH = Path("outputs/diagnostics/eco_provenance_raw.csv")
OUT_PATH = Path("outputs/diagnostics/eco_provenance_fixed.csv")

df = pd.read_csv(IN_PATH)

# Chuẩn hóa tên cột nếu cần
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

### 4.4. Code audit lại Eco provenance

Tạo file `scripts/audit_eco_provenance.py`:

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

## 5. Task D - Sparse-skill claim

### 5.1. Câu chữ cần dùng trong bài

Dùng câu an toàn:

> The evidence supports sparse-skill diagnostics rather than a universal claim that LC-MRSG improves sparse-skill prediction.

Tránh dùng:

> LC-MRSG improves sparse-skill prediction.

### 5.2. Lý do

- ASSIST2012 có sparse/very sparse strata rõ.
- Junyi không có very-sparse hoặc sparse test strata trong final export.
- KDD2010 có kết quả sparse-stratum không ổn định đối với nhiều neural/graph-aware models.

---

## 6. Task E - KDD2010 BKT: thống kê có ý nghĩa nhưng hiệu ứng nhỏ

### 6.1. Câu chữ cần dùng

Trong Results và Discussion viết:

> KDD2010 BKT is statistically significant but practically tiny, with ΔAUC around +0.0001; it should not be treated as a meaningful performance improvement.

### 6.2. Ngưỡng diễn giải đề xuất

- `|ΔAUC| < 0.001`: practically tiny.
- `0.001 <= |ΔAUC| < 0.005`: small.
- `0.005 <= |ΔAUC| < 0.01`: moderate for KT benchmarks.
- `|ΔAUC| >= 0.01`: practically meaningful, cần kiểm tra thêm CI và stability.

---

## 7. Task F - Tổ chức lại main text, appendix, supplementary

### 7.1. Main text chỉ giữ

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

### 7.2. Appendix giữ

- Fold-level Eco provenance audit.
- Training-integrity by dataset-model.
- Naming convention của full fold-level figures.
- Training diagnostics reproducibility checklist.

### 7.3. Supplementary để ngoài bài

- Tất cả fold-level figures cho 3 datasets x 3 folds x 6 figure types.
- Full epoch logs.
- Full learning curves.
- Full prediction files nếu journal cho phép.

---

## 8. Prompt tổng thể cho Google Gravity / Antigravity

Sao chép prompt dưới đây vào Google Gravity / Antigravity:

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

## 9. Sau khi chạy xong cần cập nhật LaTeX

1. Copy file `figures/training_curves_summary.pdf` vào thư mục `figures/` cùng cấp file `.tex`.
2. Nếu Eco fixed audit đã PASS, cập nhật Table Eco provenance từ `FLAG` sang `PASS_FIXED` nhưng vẫn giải thích raw export ban đầu có FLAG.
3. Nếu training integrity summary có nhiều PASS, cập nhật Table 8 thành hai cột: `Strict probe status` và `Epoch-log status`.
4. Nếu vẫn WARN, giữ wording thận trọng: `training-log warning, not constant-output collapse`.

---

## 10. Checklist trước khi nộp Q3

- [ ] Abstract không nói universal improvement.
- [ ] Introduction nêu rõ graph leakage risk.
- [ ] Protocol có split-first condition.
- [ ] Dataset-scale audit và fold-level plots được phân biệt bằng đúng câu: `Scale audit reports benchmark-level processed availability; fold plots report representative fold artifacts.`
- [ ] Eco FLAG được giải thích là one-direction storage / missing metadata, không phải held-out leakage.
- [ ] Sparse-skill claim dùng `supports sparse-skill diagnostics`.
- [ ] KDD2010 BKT ghi rõ statistically significant but practically tiny.
- [ ] Main text không quá tải hình phụ lục.
- [ ] Appendix có training diagnostics hoặc learning curves.
- [ ] Supplementary có full fold figures.
- [ ] LaTeX compile sạch.
```
