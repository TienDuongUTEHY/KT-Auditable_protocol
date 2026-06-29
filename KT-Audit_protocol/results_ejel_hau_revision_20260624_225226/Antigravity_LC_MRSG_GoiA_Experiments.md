# ANTIGRAVITY PROMPT — GÓI A: HOÀN THIỆN THÍ NGHIỆM BẮT BUỘC CHO LC-MRSG/EJEL

## 0. Vai trò và mục tiêu

Bạn là một tác nhân lập trình/nghiên cứu thực nghiệm làm việc trong repository của bài báo:

**LC-MRSG: An Auditable Protocol for Relation-Aware Skill-Graph Construction in e-Learning Knowledge Tracing**

Mục tiêu của tác vụ này là **hoàn thiện các thí nghiệm bắt buộc trước khi sửa manuscript EJEL**, gồm:

1. Chạy bổ sung **KDD2010 neural early-stopping** với `Eco` đã kiểm soát mật độ.
2. Hoàn tất **Junyi đủ 3 folds × 3 seeds** cho DKT và simpleKT dưới cùng quy trình early-stopping.
3. Xuất lại các bảng/tệp kết quả phục vụ sửa manuscript:
   - `selected-config`
   - `best-graph-vs-no-graph`
   - `training-budget / two-epoch vs early-stopping` nếu có dữ liệu two-epoch cũ
   - `practical threshold |ΔAUC| >= 0.005`
   - `Holm-adjusted p-values`
4. Không sửa nội dung manuscript ở giai đoạn này, trừ khi cần tạo bảng `.tex`/`.csv` đầu ra để người viết đưa vào bản thảo sau.

Yêu cầu quan trọng: **không được dùng test set để chọn cấu hình đồ thị**. Mọi lựa chọn candidate phải dựa trên validation AUC; test set chỉ được dùng một lần để báo cáo kết quả cuối cùng của cấu hình đã chọn.

---

## 1. Bối cảnh vấn đề cần khắc phục

Bản manuscript hiện tại đã chuyển sang định vị LC-MRSG như một **audit/governance protocol** thay vì một kiến trúc SOTA. Tuy nhiên còn tồn tại vấn đề lớn:

- KDD2010 là dataset duy nhất có đủ ba quan hệ hiệu lực: `Epre + Esim + Eco`.
- Bảng relation-availability/density đã báo cáo KDD2010 có đủ ba quan hệ, nhưng phần neural early-stopping chính hiện mới có ASSIST2012 và Junyi.
- Vì vậy claim `relation-aware` hoặc `tri-relational` chưa được kiểm chứng bằng neural early-stopping trên chính dataset có đủ ba quan hệ.

Tác vụ này cần tạo bằng chứng thực nghiệm sạch cho điểm đó.

---

## 2. Nguyên tắc bất biến của thí nghiệm

### 2.1. Split-first và chống leakage

Mọi pipeline phải tuân thủ:

```text
D_train, D_valid, D_test được tách trước.
Graph candidate chỉ được build từ D_train.
Candidate graph được freeze trước khi đánh giá validation.
Validation chỉ dùng để chọn candidate.
Test chỉ dùng để báo cáo cuối cùng.
```

Không được:

- build `Eco`, `Esim`, edge weight, support count từ validation/test;
- tune threshold theo test AUC;
- chọn candidate theo test AUC;
- trộn prediction của nhiều candidate rồi chọn theo test;
- dùng file processed mà trong đó graph đã được tạo từ full data nếu không có audit chứng minh train-only.

### 2.2. Early stopping

Dùng cùng quy trình early-stopping đã áp dụng cho ASSIST2012/Junyi trong bản sửa trước. Nếu repo đã có config sẵn, ưu tiên tái sử dụng để đảm bảo nhất quán.

Nếu chưa rõ config, đặt mặc định tối thiểu như sau và ghi lại trong `run_manifest.csv`:

```yaml
early_stopping:
  monitor: valid_auc
  mode: max
  patience: 10
  min_delta: 0.0001
  restore_best_checkpoint: true
  max_epochs: 200
```

Mỗi run phải lưu:

- best epoch;
- best validation AUC;
- test AUC tại checkpoint tốt nhất theo validation;
- training log theo epoch;
- đường dẫn checkpoint hoặc hash checkpoint;
- seed, fold, dataset, backbone, candidate.

### 2.3. Seeds và folds

Dùng thống nhất:

```text
folds = [0, 1, 2]
seeds = [42, 2024, 2025]
backbones = [DKT, simpleKT]
```

Mỗi dataset/backbone/candidate cần có đủ 3 × 3 = 9 runs, trừ khi đã có kết quả hợp lệ từ trước. Nếu có kết quả cũ, phải kiểm tra schema và log trước khi tái sử dụng.

---

## 3. Ma trận thí nghiệm bắt buộc

## 3.1. Junyi — hoàn tất đủ folds/seeds

Hiện cần kiểm tra lại Junyi vì log trước đó có dấu hiệu chưa đủ 3 folds × 3 seeds.

### Việc cần làm

1. Quét toàn bộ thư mục kết quả hiện có để xác định các run Junyi đã hoàn thành.
2. Tạo `missing_runs_report.csv` liệt kê mọi run thiếu theo khóa:

```text
dataset, backbone, fold, seed, candidate, status, reason
```

3. Chạy bù mọi run thiếu cho:

```text
dataset = Junyi
backbones = DKT, simpleKT
folds = 0, 1, 2
seeds = 42, 2024, 2025
candidate set = cùng candidate set đã dùng ở ASSIST2012/Junyi hiện tại
```

### Candidate set cho Junyi

Dùng candidate set hiện có trong repo. Nếu cần chuẩn hóa, dùng tối thiểu:

```text
no_graph
Epre
Eco_only
Epre_plus_Eco
full_LC_MRSG nếu Esim tồn tại, nếu Esim = 0 thì full phải được ghi rõ là effective bi-relational
relation_gated_1
relation_gated_2
sparse_aware_relation_gated nếu repo đã có
validation_selected_static nếu repo đã có
```

Vì Junyi hiện có `Esim = 0` trong processed single-skill setting, mọi bảng phải ghi rõ relation availability để tránh ngụ ý Junyi là tri-relational.

---

## 3.2. KDD2010 — chạy neural early-stopping với Eco-controlled

KDD2010 là dataset trọng tâm của gói này vì có đủ `Epre + Esim + Eco`.

### 3.2.1. Backbones

Chạy:

```text
DKT
simpleKT
```

### 3.2.2. Folds/seeds

Chạy:

```text
folds = 0, 1, 2
seeds = 42, 2024, 2025
```

### 3.2.3. Candidate set cho KDD2010

Tối thiểu cần có:

```text
no_graph
Epre
Epre_plus_Esim
Eco_controlled_primary
full_LC_MRSG_controlled = Epre + Esim + Eco_controlled_primary
relation_gated_controlled nếu repo hỗ trợ
```

Không nên dùng Eco default quá dày làm candidate chính duy nhất. Có thể giữ `Eco_default` như diagnostic phụ, nhưng main comparison phải dùng Eco đã kiểm soát density.

### 3.2.4. Eco-controlled configurations

Tạo hoặc kiểm tra các cấu hình `Eco` sau. Chọn ít nhất một cấu hình primary trước khi chạy main experiment. Không được chọn primary theo test AUC.

Khuyến nghị chạy 3 cấu hình để có sensitivity:

```yaml
eco_controlled_configs:
  eco_c1_high_coverage_low_density:
    k_min: 2
    pmi_min: 0.25
    top_k: 10
    expected_density_region: approximately_0.01_to_0.03
    expected_skill_coverage_region: high

  eco_c2_balanced:
    k_min: 3
    pmi_min: 0.25
    top_k: 50
    expected_density_region: approximately_0.05_to_0.10
    expected_skill_coverage_region: high

  eco_c3_sparse_control:
    k_min: 10
    pmi_min: 0.00
    top_k: 20
    expected_density_region: approximately_0.02_to_0.04
    expected_skill_coverage_region: medium
```

Nếu compute hạn chế, chọn `eco_c2_balanced` làm primary vì cân bằng giữa density và skill coverage. Ghi rõ lựa chọn này trong `run_manifest.csv` và `kdd2010_eco_density_audit.csv`.

### 3.2.5. Không chọn threshold bằng test

Quy tắc chọn Eco primary:

- Được chọn trước dựa trên density/coverage audit từ train graph.
- Hoặc được chọn bằng validation AUC.
- Tuyệt đối không chọn bằng test AUC.

---

## 4. Định nghĩa density bắt buộc phải đồng bộ

Tạo file `kdd2010_eco_density_audit.csv` và ghi rõ công thức.

### 4.1. Với Eco và Esim không hướng

```text
density_undirected = unique_undirected_edges / (n_skill_train * (n_skill_train - 1) / 2)
```

Trong đó:

```text
n_skill_train = số skill có mặt trong train fold sau preprocessing và mapping.
unique_undirected_edges = số cạnh không hướng duy nhất sau khi loại trùng.
```

### 4.2. Với Epre có hướng

```text
density_directed = unique_directed_edges / (n_skill_train * (n_skill_train - 1))
```

Trong đó:

```text
unique_directed_edges = số cạnh có hướng duy nhất sau khi loại trùng self-loop.
```

### 4.3. Cột bắt buộc trong density audit

```text
dataset
fold
relation
config_name
k_min
pmi_min
top_k
n_skill_full
n_skill_train
raw_edge_rows
unique_edges
edge_directionality
max_possible_edges
density
skill_coverage
built_from_train_only
notes
```

### 4.4. Mục tiêu khắc phục lỗi 0.807 vs 0.723

Sau khi chạy audit, tạo file `density_consistency_report.md` giải thích rõ:

- số nào là mean across folds;
- số nào là fold-specific/config-specific;
- mẫu số density dùng `n_skill_train` hay `n_skill_full`;
- vì sao các số trong main/supplementary cũ có thể khác nhau;
- số cuối cùng đề xuất dùng cho manuscript.

---

## 5. Chuẩn hóa schema kết quả

Mọi prediction file cần có schema tối thiểu:

```text
dataset
fold
seed
backbone
candidate
user_id
item_id
skill_id
y_true
y_score
split
```

Nếu có timestamp hoặc sequence index, giữ thêm:

```text
timestamp
sequence_index
```

Mọi run-level result cần có schema:

```text
dataset
fold
seed
backbone
candidate
selected_by_validation
valid_auc
test_auc
test_brier nếu có
test_ece nếu có
best_epoch
num_epochs_run
early_stop_patience
checkpoint_path
prediction_path
graph_config_path
graph_hash
status
notes
```

---

## 6. Validation-selected config

Tạo bảng `selected_config_early_stopping.csv`.

### 6.1. Quy tắc chọn

Với mỗi:

```text
dataset, backbone, fold, seed
```

chọn candidate có `valid_auc` cao nhất.

Nếu hòa nhau trong sai số rất nhỏ, áp dụng tie-breaker đã định trước:

```text
1. no_graph nếu chênh lệch valid_auc <= 0.0001 so với best candidate
2. candidate đơn giản hơn trước candidate phức tạp hơn
3. Epre trước full nếu cùng AUC
4. ghi rõ tie_break_applied = true
```

Lý do: tránh ép dùng graph khi validation evidence không đủ.

### 6.2. Cột bắt buộc

```text
dataset
backbone
fold
seed
selected_candidate
selected_valid_auc
selected_test_auc
no_graph_valid_auc
no_graph_test_auc
delta_selected_vs_no_graph
selected_is_no_graph
is_tautological_delta
tie_break_applied
notes
```

### 6.3. Diễn giải bắt buộc

Nếu `selected_candidate == no_graph`, thì:

```text
delta_selected_vs_no_graph = 0 theo định nghĩa
is_tautological_delta = true
```

Không được diễn giải dòng này như bằng chứng trực tiếp rằng graph không giúp. Diễn giải đúng là:

```text
validation-only selection tự loại graph; đây là một governance output của protocol.
```

---

## 7. Best-available-graph vs no-graph

Tạo bảng `best_graph_vs_no_graph_early_stopping.csv` để có so sánh không tầm thường.

### 7.1. Quy tắc chọn best graph

Với mỗi:

```text
dataset, backbone, fold, seed
```

chọn candidate graph có `valid_auc` cao nhất trong tập candidate không phải no_graph.

```text
graph_candidates = all candidates where candidate != no_graph
best_graph = argmax(valid_auc among graph_candidates)
```

Sau đó báo cáo test AUC của `best_graph` và so sánh với no_graph.

### 7.2. Cột bắt buộc

```text
dataset
backbone
fold
seed
best_graph_candidate
best_graph_valid_auc
best_graph_test_auc
no_graph_valid_auc
no_graph_test_auc
delta_best_graph_vs_no_graph
relation_types_effective
contains_Epre
contains_Esim
contains_Eco
eco_config_name
notes
```

### 7.3. Ý nghĩa

Bảng này trả lời câu hỏi:

```text
Nếu buộc phải chọn một graph candidate bằng validation, graph tốt nhất có tạo thêm giá trị so với no_graph không?
```

Bảng này tách biệt khỏi selected-config, nơi validation có thể chọn no_graph.

---

## 8. Practical threshold và phân loại kết quả

Ngưỡng practical được cố định:

```text
practical_threshold_auc = 0.005
```

Tạo cột `classification` theo quy tắc:

```text
if is_tautological_delta == true:
    classification = "selection-no-graph tautology; governance selection outcome"
elif delta_auc >= 0.005 and adjusted_p < 0.05:
    classification = "statistically and practically positive"
elif delta_auc <= -0.005 and adjusted_p < 0.05:
    classification = "statistically and practically negative"
elif abs(delta_auc) < 0.005:
    classification = "diagnostic negligible"
else:
    classification = "practically notable but statistically non-confirmatory"
```

Không được gọi các dòng `selected == no_graph` là `near-zero graph effect`; phải gọi là `selection-no-graph outcome`.

---

## 9. Holm-adjusted p-values

Tạo bảng `neural_summary_practical_holm.csv`.

### 9.1. So sánh chính

So sánh chính nên gồm:

```text
selected_config_vs_no_graph
best_available_graph_vs_no_graph
```

Tối thiểu báo cáo cho:

```text
ASSIST2012 × DKT
ASSIST2012 × simpleKT
Junyi × DKT
Junyi × simpleKT
KDD2010 × DKT
KDD2010 × simpleKT
```

Nếu chưa có ASSIST2012 đủ dữ liệu trong repo, không chạy lại trừ khi thiếu. Nhưng phải chuẩn hóa lại bảng bằng cùng script.

### 9.2. P-value thô

Ưu tiên dùng paired bootstrap hoặc paired test trên run-level deltas.

Khuyến nghị:

1. Tính `delta_auc` theo từng run `(fold, seed)`.
2. Với mỗi `(dataset, backbone, comparison_type)`, có vector 9 deltas.
3. Tính mean delta, CI 95% bằng bootstrap trên 9 run-level deltas với fixed seed.
4. Tính p-value hai phía kiểm tra `delta = 0` bằng paired permutation hoặc Wilcoxon signed-rank nếu phù hợp.
5. Sau đó Holm correction trên toàn bộ family của neural comparisons.

Nếu đã có prediction-level paired bootstrap/DeLong trong repo, có thể dùng lại, nhưng phải ghi rõ phương pháp trong `stats_method_report.md`.

### 9.3. Holm correction thủ công nếu thiếu statsmodels

Nếu không có `statsmodels`, implement Holm như sau:

```python
def holm_adjust(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [None] * m
    running_max = 0.0
    for rank, i in enumerate(order):
        adjusted = (m - rank) * pvals[i]
        running_max = max(running_max, adjusted)
        adj[i] = min(running_max, 1.0)
    return adj
```

### 9.4. Cột bắt buộc trong summary

```text
dataset
backbone
comparison_type
n_runs
mean_delta_auc
ci95_low
ci95_high
raw_p
holm_p
practical_threshold
classification
selected_no_graph_count
non_graph_candidate_count
notes
```

---

## 10. Training-budget comparison nếu có dữ liệu two-epoch

Nếu repo còn kết quả two-epoch cũ, tạo bảng:

```text
neural_summary_two_epoch_vs_early_stopping.csv
```

Cột bắt buộc:

```text
dataset
backbone
comparison_type
two_epoch_mean_delta
early_stopping_mean_delta
early_stopping_ci_low
early_stopping_ci_high
sign_change
stability_label
notes
```

Quy tắc `stability_label`:

```text
if abs(early_stopping_mean_delta) < 0.005:
    stability_label = "near-zero under early stopping"
elif sign(two_epoch_mean_delta) != sign(early_stopping_mean_delta):
    stability_label = "sign changed"
else:
    stability_label = "directionally stable"
```

Nếu không có two-epoch data, tạo `two_epoch_missing_report.md` và không dựng bảng giả.

---

## 11. Đầu ra bắt buộc

Tạo thư mục:

```text
results/ejel_gA_experiments/
```

Trong đó phải có:

```text
results/ejel_gA_experiments/run_manifest.csv
results/ejel_gA_experiments/missing_runs_report.csv
results/ejel_gA_experiments/kdd2010_eco_density_audit.csv
results/ejel_gA_experiments/density_consistency_report.md
results/ejel_gA_experiments/selected_config_early_stopping.csv
results/ejel_gA_experiments/best_graph_vs_no_graph_early_stopping.csv
results/ejel_gA_experiments/neural_summary_practical_holm.csv
results/ejel_gA_experiments/neural_summary_two_epoch_vs_early_stopping.csv hoặc two_epoch_missing_report.md
results/ejel_gA_experiments/stats_method_report.md
results/ejel_gA_experiments/reproducibility_manifest.md
```

Tạo thêm bảng LaTeX nếu repo dùng LaTeX:

```text
tables/table_selected_config_early_stopping.tex
tables/table_best_graph_vs_no_graph_early_stopping.tex
tables/table_neural_summary_practical_holm.tex
tables/table_kdd2010_eco_density_audit.tex
```

Không tự động chèn vào manuscript ở giai đoạn này.

---

## 12. Kiểm tra chất lượng bắt buộc trước khi kết thúc

Tạo file:

```text
results/ejel_gA_experiments/quality_gate_report.md
```

Nội dung phải trả lời rõ từng câu:

### 12.1. Completeness

```text
[ ] Junyi đã đủ 3 folds × 3 seeds cho DKT chưa?
[ ] Junyi đã đủ 3 folds × 3 seeds cho simpleKT chưa?
[ ] KDD2010 đã đủ 3 folds × 3 seeds cho DKT chưa?
[ ] KDD2010 đã đủ 3 folds × 3 seeds cho simpleKT chưa?
[ ] Mỗi run có prediction file không?
[ ] Mỗi run có valid_auc/test_auc không?
```

### 12.2. Leakage control

```text
[ ] Graph được build từ train only chưa?
[ ] Validation/test có bị dùng để tạo edge/support/weight không?
[ ] Candidate graph có được freeze trước validation không?
[ ] Test có bị dùng để chọn candidate/threshold không?
[ ] Repo/log có hash hoặc metadata chứng minh graph config không?
```

### 12.3. Density consistency

```text
[ ] Density formula đã thống nhất chưa?
[ ] KDD2010 Eco default/fold-specific/mean-across-folds đã phân biệt rõ chưa?
[ ] Số 0.807 và 0.723 cũ đã được giải thích hoặc sửa chưa?
[ ] n_skill_train/n_skill_full đã ghi rõ chưa?
```

### 12.4. Interpretation readiness

```text
[ ] selected == no_graph đã được đánh dấu tautological chưa?
[ ] Có bảng best-available-graph vs no-graph chưa?
[ ] Có practical threshold |ΔAUC| >= 0.005 chưa?
[ ] Có Holm p chưa?
[ ] Có classification đúng chưa?
```

---

## 13. Quy tắc không được làm

Không được:

1. Sửa manuscript chính khi chưa được yêu cầu.
2. Chọn Eco threshold bằng test AUC.
3. Xóa kết quả cũ mà không backup.
4. Gộp các dòng `selected == no_graph` vào nhóm `near-zero graph effect`.
5. Gọi Junyi/ASSIST2012 là tri-relational nếu `Esim = 0`.
6. Gọi KDD2010 là bằng chứng tri-relational nếu chưa chạy neural early-stopping.
7. Tạo bảng giả hoặc điền số suy đoán.
8. Bỏ qua run lỗi; phải ghi vào `run_manifest.csv` và `missing_runs_report.csv`.

---

## 14. Quy trình thực hiện đề xuất

Thực hiện theo thứ tự:

### Step 1 — Inspect repository

- Xác định cấu trúc repo.
- Tìm script train/evaluate hiện có.
- Tìm config dataset/model/candidate hiện có.
- Tìm thư mục kết quả cũ.
- Không sửa code khi chưa hiểu pipeline.

### Step 2 — Backup

Tạo backup kết quả cũ:

```text
results_backup_before_ejel_gA_<timestamp>/
```

### Step 3 — Build missing-runs report

Tạo `missing_runs_report.csv` cho Junyi và KDD2010.

### Step 4 — Implement or verify Eco-controlled graph builder

- Kiểm tra graph builder hiện tại có hỗ trợ `k_min`, `pmi_min`, `top_k` không.
- Nếu chưa có, thêm tham số theo cách không phá vỡ config cũ.
- Tạo density audit cho từng fold trước khi train.

### Step 5 — Run missing Junyi experiments

- Chạy bù các fold/seed/candidate còn thiếu.
- Lưu prediction và logs.

### Step 6 — Run KDD2010 experiments

- Chạy DKT/simpleKT với candidate set có Eco-controlled.
- Ưu tiên chạy `no_graph`, `Epre`, `Epre_plus_Esim`, `Eco_controlled_primary`, `full_LC_MRSG_controlled` trước.
- Sau đó mới chạy gated nếu compute cho phép.

### Step 7 — Aggregate

- Tạo run manifest.
- Tạo selected-config table.
- Tạo best-graph-vs-no-graph table.
- Tính CI, p-value, Holm p.
- Tạo classification theo practical threshold.

### Step 8 — Quality gate

- Sinh `quality_gate_report.md`.
- Nếu còn thiếu run, ghi rõ thiếu gì, vì sao, và bảng nào không được dùng làm main evidence.

---

## 15. Gợi ý cấu trúc script nếu cần tạo mới

Nếu repo chưa có script tổng hợp, tạo các script sau:

```text
scripts/ejel_gA/01_scan_runs.py
scripts/ejel_gA/02_build_eco_density_audit.py
scripts/ejel_gA/03_run_missing_experiments.py
scripts/ejel_gA/04_collect_run_manifest.py
scripts/ejel_gA/05_selected_config.py
scripts/ejel_gA/06_best_graph_vs_no_graph.py
scripts/ejel_gA/07_stats_holm.py
scripts/ejel_gA/08_export_tables.py
scripts/ejel_gA/09_quality_gate_report.py
```

Các script phải có CLI rõ ràng, ví dụ:

```bash
python scripts/ejel_gA/01_scan_runs.py --results_dir results --out results/ejel_gA_experiments/missing_runs_report.csv
python scripts/ejel_gA/02_build_eco_density_audit.py --dataset kdd2010 --folds 0 1 2 --out results/ejel_gA_experiments/kdd2010_eco_density_audit.csv
python scripts/ejel_gA/05_selected_config.py --manifest results/ejel_gA_experiments/run_manifest.csv --out results/ejel_gA_experiments/selected_config_early_stopping.csv
python scripts/ejel_gA/06_best_graph_vs_no_graph.py --manifest results/ejel_gA_experiments/run_manifest.csv --out results/ejel_gA_experiments/best_graph_vs_no_graph_early_stopping.csv
python scripts/ejel_gA/07_stats_holm.py --selected results/ejel_gA_experiments/selected_config_early_stopping.csv --best_graph results/ejel_gA_experiments/best_graph_vs_no_graph_early_stopping.csv --out results/ejel_gA_experiments/neural_summary_practical_holm.csv
python scripts/ejel_gA/09_quality_gate_report.py --base results/ejel_gA_experiments --out results/ejel_gA_experiments/quality_gate_report.md
```

---

## 16. Báo cáo cuối cùng cần in ra màn hình

Khi hoàn thành, in ra summary dạng sau:

```text
EJEL Gói A completed.

Datasets completed:
- Junyi DKT: <n_completed>/<n_required>
- Junyi simpleKT: <n_completed>/<n_required>
- KDD2010 DKT: <n_completed>/<n_required>
- KDD2010 simpleKT: <n_completed>/<n_required>

Key outputs:
- results/ejel_gA_experiments/run_manifest.csv
- results/ejel_gA_experiments/selected_config_early_stopping.csv
- results/ejel_gA_experiments/best_graph_vs_no_graph_early_stopping.csv
- results/ejel_gA_experiments/neural_summary_practical_holm.csv
- results/ejel_gA_experiments/quality_gate_report.md

Warnings:
- <list any incomplete runs or leakage/density concerns>
```

---

## 17. Tiêu chí hoàn thành cuối cùng

Chỉ coi tác vụ hoàn thành khi:

1. KDD2010 có neural early-stopping cho DKT và simpleKT, đủ folds/seeds hoặc có báo cáo thiếu rõ ràng.
2. Junyi được hoàn tất đủ folds/seeds hoặc có báo cáo thiếu rõ ràng.
3. Có bảng selected-config phân biệt rõ `selected == no_graph`.
4. Có bảng best-available-graph vs no-graph.
5. Có practical threshold `|ΔAUC| >= 0.005`.
6. Có Holm-adjusted p-values.
7. Có density audit giải thích được khác biệt mean/fold/config và công thức density.
8. Có quality gate report xác nhận chống leakage.
9. Không có số liệu giả, không có selection bằng test set.

