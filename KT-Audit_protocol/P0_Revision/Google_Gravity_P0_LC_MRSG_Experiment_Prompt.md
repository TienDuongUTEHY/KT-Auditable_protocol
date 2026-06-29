# PROMPT CHO GOOGLE GRAVITY / ANTIGRAVITY

## Nhiệm vụ tổng quát

Bạn là một tác nhân lập trình nghiên cứu khoa học. Hãy tự động rà soát, bổ sung, chạy kiểm tra và xuất toàn bộ kết quả thực nghiệm tối thiểu để hoàn thiện bài báo P0:

**LC-MRSG++: Leakage-Controlled and Validation-Guided Multi-Relational Skill Graph Construction for Sparse-Skill Knowledge Tracing**

Mục tiêu không phải tạo thêm một bài performance/SOTA, mà là hoàn thiện một bài **protocol/audit paper** đạt chất lượng Q3 uy tín. Tất cả kết quả performance chỉ được xem là **diagnostic evidence**. Không mở rộng sang P1/P2: không triển khai calibration/ECE/Brier, không adaptive stratification, không SSA-CL/InfoNCE, không learning-path recommendation.

---

## 0. Nguyên tắc bắt buộc

1. **Không sửa raw dataset.** Chỉ đọc dữ liệu gốc và ghi kết quả vào thư mục `results_p0_revision/`.
2. **Không overwrite kết quả cũ.** Nếu file đã tồn tại, ghi bản mới có timestamp.
3. **Không bịa số liệu.** Nếu không đọc được file hoặc không tính được chỉ số, ghi `NA` kèm lý do trong log.
4. **Fail fast.** Nếu audit phát hiện lỗi nghiêm trọng ở graph provenance hoặc split leakage, dừng trước khi chạy epoch sanity.
5. **Tất cả log phải dễ đọc.** Mỗi giai đoạn phải có dòng:
   - `PHASE_START`
   - `PHASE_PASS` hoặc `PHASE_FAIL`
   - `PHASE_SUMMARY`
6. **Mọi bảng phải xuất cả CSV và LaTeX.** CSV dùng kiểm tra; `.tex` dùng ráp vào manuscript.
7. **Mọi artifact chính phải có SHA256 hash.** Tạo manifest cuối cùng.
8. **Không thay đổi claim khoa học.** Chỉ báo cáo kết quả theo hướng thận trọng: graph usefulness is dataset- and backbone-dependent.

---

## 1. Cấu trúc thư mục cần tạo

Hãy tạo cấu trúc sau trong root của repository:

```text
results_p0_revision/
  logs/
    master_run_<TIMESTAMP>.log
    phase0_scope_decision.log
    phase1_dataset_graph_audit.log
    phase1_esim_trace.log
    phase1_junyi_coverage.log
    phase2_epoch_sanity.log
    phase3_table_generation.log
    phase4_manifest_validation.log
  configs/
    p0_revision_config.yaml
    p0_validation_candidates.yaml
    p0_epoch_sanity_config.yaml
  tables_csv/
    table_dataset_statistics.csv
    table_graph_provenance_corrected.csv
    table_esim_trace.csv
    table_junyi_graph_coverage.csv
    table_leakage_audit_L1_L6.csv
    table_validation_candidates_prespecified.csv
    table_selected_relation_variants.csv
    table_main_auc_delta_holm.csv
    table_sparse_bins_descriptive.csv
    table_epoch_sanity.csv
    table_reproducibility_checklist.csv
  tables_tex/
    table_dataset_statistics.tex
    table_graph_provenance_corrected.tex
    table_esim_trace.tex
    table_junyi_graph_coverage.tex
    table_leakage_audit_L1_L6.tex
    table_validation_candidates_prespecified.tex
    table_selected_relation_variants.tex
    table_main_auc_delta_holm.tex
    table_sparse_bins_descriptive.tex
    table_epoch_sanity.tex
    table_reproducibility_checklist.tex
  manifests/
    sha256_manifest.csv
    run_environment.txt
    git_state.txt
    final_definition_of_done.md
  supplementary/
    README_SUPPLEMENTARY_P0.md
    reviewer_response_notes.md
```

Nếu repository đã có cấu trúc riêng, vẫn tạo thư mục trên để gom kết quả cuối.

---

## 2. Tự động phát hiện repository

Trước khi chạy, hãy kiểm tra:

```text
- Có file cấu hình dataset không?
- Có thư mục data/ hoặc datasets/ không?
- Có thư mục outputs/results/tables hiện hữu không?
- Có script train/evaluate hiện hữu không?
- Có file graph provenance hiện hữu không?
- Có split specification hiện hữu không?
- Có bảng kết quả AUC hiện hữu không?
```

Ghi vào `logs/master_run_<TIMESTAMP>.log`:

```text
[REPO_SCAN]
root=<absolute path>
python_version=<version>
git_commit=<hash or NA>
found_data_dirs=<list>
found_result_dirs=<list>
found_training_scripts=<list>
found_graph_files=<list>
found_split_files=<list>
found_auc_files=<list>
```

Nếu không phát hiện được script train/evaluate, không tự ý viết lại toàn bộ model; hãy tạo wrapper và ghi rõ cần người dùng chỉ đường dẫn.

---

## 3. Phase 0 — Chốt phạm vi P0

### 3.1 Việc cần làm

Tạo `configs/p0_revision_config.yaml` với các biến:

```yaml
paper_scope:
  main_contribution: "split-first graph construction + leakage/E_co audit + validation-guided relation selection"
  performance_role: "diagnostic evidence only"
  not_claiming:
    - "state-of-the-art KT architecture"
    - "universal graph improvement"
    - "true prerequisite discovery"
    - "calibration or pedagogical usefulness"
    - "learning-path recommendation"
    - "SSA-CL / InfoNCE representation learning"
  defer_to_future_or_other_papers:
    P1:
      - "adaptive stratification"
      - "calibration: ECE/Brier"
      - "full sparse-skill evaluation methodology"
    P2:
      - "SSA-CL"
      - "InfoNCE"
      - "graph augmentation strategies"
    P3:
      - "distillation"
      - "cross-dataset transfer"
```

### 3.2 Log bắt buộc

Ghi vào `phase0_scope_decision.log`:

```text
PHASE_START phase0_scope_decision
P0_MAIN_CONTRIBUTION=construction/audit/selection protocol
PERFORMANCE_ROLE=diagnostic evidence only
DEFER_CALIBRATION_TO_P1=true
DEFER_ADAPTIVE_STRATIFICATION_TO_P1=true
DEFER_SSA_CL_TO_P2=true
PHASE_PASS phase0_scope_decision
```

---

## 4. Phase 1 — Kiểm tra đúng đắn dữ liệu và đồ thị

Phase này là bắt buộc trước mọi claim kết quả. Nếu fail, không chạy Phase 2.

---

### 4.1 Audit dataset statistics và disclosure subsampling

#### Mục tiêu

Khôi phục bảng thống kê dataset và phát hiện có dùng subset/subsampling hay không.

#### Datasets cần xử lý

```text
- ASSIST2012
- Junyi
- KDD2010
```

Tên folder có thể khác nhau. Hãy tự dò các alias:

```text
assist2012, assistments2012, assist_2012, ASSIST2012
junyi, junyi_academy, Junyi
kdd2010, algebra2008, bridge_to_algebra, KDDCup2010
```

#### Chỉ số cần tính

Với mỗi dataset và mỗi split/fold nếu có:

```text
- dataset
- fold
- split_type
- n_users_total
- n_users_train
- n_users_valid
- n_users_test
- n_questions_total
- n_questions_train
- n_questions_valid
- n_questions_test
- n_skills_total
- n_skills_train
- n_skills_valid
- n_skills_test
- n_interactions_total
- n_interactions_train
- n_interactions_valid
- n_interactions_test
- user_skill_density_train
- mean_interactions_per_user_train
- median_interactions_per_user_train
- mean_interactions_per_skill_train
- median_interactions_per_skill_train
- sparse_50_count
- sparse_100_count
- sparse_200_count
- frequent_500_count
- suspected_subsampling: yes/no/unknown
- subsampling_reason: compute_budget/unknown/not_applicable
```

#### Đối chiếu quy mô tham chiếu

Ghi cảnh báo nếu quy mô nhỏ hơn đáng kể so với các mốc đang được dùng trong bản thảo:

```text
ASSIST2012 expected interactions ≈ 2.7M
Junyi expected interactions ≈ 16.2M
KDD2010 expected interactions: compute from local processed dataset; if there is an original raw file, compare processed/raw ratio.
```

Không hard-fail chỉ vì quy mô nhỏ; chỉ hard-fail nếu không có disclosure subsampling.

#### Output

```text
tables_csv/table_dataset_statistics.csv
tables_tex/table_dataset_statistics.tex
logs/phase1_dataset_graph_audit.log
```

#### Log bắt buộc

```text
PHASE_START dataset_statistics
DATASET_STATS dataset=ASSIST2012 users_total=... interactions_total=... suspected_subsampling=...
DATASET_STATS dataset=Junyi users_total=... interactions_total=... suspected_subsampling=...
DATASET_STATS dataset=KDD2010 users_total=... interactions_total=... suspected_subsampling=...
DATASET_STATS_OUTPUT=results_p0_revision/tables_csv/table_dataset_statistics.csv
PHASE_PASS dataset_statistics
```

---

### 4.2 Audit KDD2010 E_co bất thường

#### Vấn đề cần giải quyết

Bản nhận xét nêu rủi ro: `KDD2010 E_co = 658,943` cạnh. Nếu `|C|≈493`, số cặp KC--KC vô hướng tối đa là `C(|C|,2)≈121,278`, nên cần xác định con số kia là:

```text
(a) support records,
(b) directed/mirrored edges,
(c) multi-edge graph,
(d) |C| thực tế khác,
(e) lỗi aggregation/mirroring.
```

#### Chỉ số phải tính cho từng dataset/fold/relation

```text
- dataset
- fold
- relation: E_pre/E_sim/E_co
- n_skills
- max_possible_undirected_pairs = n_skills * (n_skills - 1) / 2
- raw_rows_in_edge_file
- unique_directed_edges
- unique_undirected_edges
- mirrored_edge_pairs
- self_loops
- duplicate_rows
- support_records
- mean_support
- median_support
- min_weight
- max_weight
- mean_weight
- is_unique_undirected_valid = unique_undirected_edges <= max_possible_undirected_pairs
- interpretation: unique_edges/support_records/multi_edge/error
```

#### Hard-fail rule

Fail nếu:

```text
unique_undirected_edges > max_possible_undirected_pairs
```

trừ khi graph được định nghĩa rõ là multi-edge và có cột `support_records` riêng.

#### Output

```text
tables_csv/table_graph_provenance_corrected.csv
tables_tex/table_graph_provenance_corrected.tex
logs/phase1_dataset_graph_audit.log
```

#### Log bắt buộc

```text
PHASE_START kdd2010_eco_audit
GRAPH_AUDIT dataset=KDD2010 fold=0 relation=E_co n_skills=... max_pairs=... raw_rows=... unique_undirected=... support_records=... interpretation=...
GRAPH_AUDIT dataset=KDD2010 fold=1 relation=E_co ...
GRAPH_AUDIT dataset=KDD2010 fold=2 relation=E_co ...
KDD2010_ECO_DECISION=<unique_edges|support_records|multi_edge|error_fixed|error_unfixed>
PHASE_PASS kdd2010_eco_audit
```

Nếu fail:

```text
PHASE_FAIL kdd2010_eco_audit reason="unique undirected edges exceed graph limit and no multi-edge definition found"
STOP_BEFORE_PHASE2=true
```

---

### 4.3 Trace E_sim pipeline

#### Vấn đề cần giải quyết

Bản nhận xét nêu mâu thuẫn: một bảng nói `top-k=20`, nhưng bảng khác báo `E_sim=0` trên ASSIST2012/Junyi. Cần log số cạnh sau từng bước.

#### Các bước cần trace

Với mỗi dataset/fold:

```text
- n_skills
- embedding_file_found: yes/no
- similarity_matrix_shape
- candidate_pairs_before_threshold
- threshold_theta_sim
- pairs_after_threshold
- top_k
- pairs_after_top_k
- final_E_sim_edges
- reason_if_zero
```

#### Quyết định sau trace

Tự động gán một trong hai quyết định:

```text
A. E_sim_active: E_sim có cạnh trên ít nhất 2/3 datasets.
B. E_sim_empty_effective: E_sim rỗng ở >=2 datasets; manuscript phải dùng nhãn E_sim^eff hoặc "empty E_sim branch" và không claim similarity benefit.
```

Không tự ý đổi thuật toán nếu không có flag cho phép. Mặc định chỉ audit. Nếu muốn sửa pipeline top-k, tạo đề xuất riêng trong log, không chạy thay đổi chính.

#### Output

```text
tables_csv/table_esim_trace.csv
tables_tex/table_esim_trace.tex
logs/phase1_esim_trace.log
```

#### Log bắt buộc

```text
PHASE_START esim_trace
ESIM_TRACE dataset=ASSIST2012 fold=0 before_threshold=... after_threshold=... after_topk=... final_edges=... reason_if_zero=...
ESIM_TRACE dataset=Junyi fold=0 ...
ESIM_TRACE dataset=KDD2010 fold=0 ...
ESIM_DECISION=<E_sim_active|E_sim_empty_effective>
PHASE_PASS esim_trace
```

---

### 4.4 Junyi graph coverage và node cô lập

#### Mục tiêu

Giải thích vì sao Junyi graph rất thưa nhưng vẫn có gain ở một số backbone. Không được diễn giải quá mức.

#### Chỉ số cần tính

Với Junyi, mỗi fold và mỗi relation/candidate graph:

```text
- n_skills
- E_pre_unique_undirected
- E_sim_unique_undirected
- E_co_unique_undirected
- all_relations_unique_undirected
- n_skills_with_degree_ge_1
- coverage_ratio
- n_isolated_skills
- isolated_ratio
- mean_degree_all_nodes
- median_degree_all_nodes
- mean_degree_nonisolated_nodes
- sparse_skill_coverage_50
- sparse_skill_coverage_100
- sparse_skill_coverage_200
```

#### Output

```text
tables_csv/table_junyi_graph_coverage.csv
tables_tex/table_junyi_graph_coverage.tex
logs/phase1_junyi_coverage.log
```

#### Log bắt buộc

```text
PHASE_START junyi_coverage
JUNYI_COVERAGE fold=0 n_skills=... covered=... isolated=... coverage_ratio=... mean_degree=...
JUNYI_COVERAGE fold=1 ...
JUNYI_COVERAGE fold=2 ...
JUNYI_INTERPRETATION=<diagnostic_only|coverage_supports_limited_mechanism|insufficient_explanation>
PHASE_PASS junyi_coverage
```

---

### 4.5 Leakage audit L1--L6

#### L1--L6 cần kiểm tra

```text
L1 Edge-construction leakage: mọi edge support phải thuộc train split.
L2 Q-matrix/provenance leakage: Q-matrix dùng để tạo edge không được lấy từ held-out logs ngoài quy tắc công bố dataset.
L3 Temporal leakage: nếu có timestamp, không có future evidence trong graph construction.
L4 Cold-start neighborhood leakage: test-only skills/items không được nhận neighborhood từ held-out evidence.
L5 Co-occurrence leakage: E_co support/count/PMI/NPMI chỉ từ train fold.
L6 Selection leakage: validation chọn candidate; test không được dùng để chọn graph/hyperparameter.
```

#### Output

```text
tables_csv/table_leakage_audit_L1_L6.csv
tables_tex/table_leakage_audit_L1_L6.tex
```

#### Log bắt buộc

```text
PHASE_START leakage_audit_L1_L6
LEAKAGE_AUDIT dataset=ASSIST2012 fold=0 L1=PASS L2=PASS L3=PASS L4=PASS L5=PASS L6=PASS notes=...
...
PHASE_PASS leakage_audit_L1_L6
```

Hard-fail nếu bất kỳ L1, L5, L6 fail.

---

## 5. Phase 2 — Epoch sanity-check tối thiểu

### 5.1 Mục tiêu

Xử lý rủi ro fixed two-epoch undertraining. Không dùng phase này để tìm SOTA, không thay kết quả confirmatory chính, chỉ kiểm tra **dấu của ΔAUC** có ổn định ở ngân sách 5 và 10 epoch không.

### 5.2 Thiết kế bắt buộc

Chạy đúng subset đang dùng trong paper, không đổi preprocessing.

```text
Datasets: chọn 2 dataset chính có đủ dữ liệu và đại diện. Mặc định: ASSIST2012 và Junyi. Nếu Junyi quá nặng, dùng KDD2010 thay thế nhưng phải ghi lý do.
Backbones: DKT, simpleKT
Graph conditions: no_graph, selected_graph
Epoch budgets: 5, 10
Folds: dùng 3 folds nếu compute cho phép; nếu không, tối thiểu fold 0 nhưng phải ghi limitation.
Seeds: dùng 3 seeds nếu compute cho phép; nếu không, tối thiểu seed chính trong paper nhưng phải ghi limitation.
Early stopping: nếu pipeline có sẵn, bật early stopping nhưng vẫn ghi max epoch budget.
```

### 5.3 Output cần tính

```text
- dataset
- fold
- seed
- backbone
- epoch_budget
- auc_no_graph
- auc_selected_graph
- delta_auc = selected_graph - no_graph
- sign_delta: positive/zero/negative
- main_two_epoch_delta_if_available
- sign_preserved_vs_main: yes/no/unknown
- train_loss_last
- valid_auc_last
- convergence_label: stable/still_improving/unstable/unknown
- notes
```

### 5.4 Output files

```text
tables_csv/table_epoch_sanity.csv
tables_tex/table_epoch_sanity.tex
logs/phase2_epoch_sanity.log
```

### 5.5 Log bắt buộc

```text
PHASE_START epoch_sanity
EPOCH_RUN_START dataset=ASSIST2012 fold=0 seed=42 backbone=DKT condition=no_graph epoch_budget=5
EPOCH_RUN_END dataset=ASSIST2012 fold=0 seed=42 backbone=DKT condition=no_graph auc=... status=PASS
EPOCH_RUN_START dataset=ASSIST2012 fold=0 seed=42 backbone=DKT condition=selected_graph epoch_budget=5
EPOCH_RUN_END dataset=ASSIST2012 fold=0 seed=42 backbone=DKT condition=selected_graph auc=... status=PASS
EPOCH_DELTA dataset=ASSIST2012 fold=0 seed=42 backbone=DKT epoch_budget=5 delta_auc=... sign=... sign_preserved_vs_main=...
...
EPOCH_SANITY_SUMMARY total_runs=... positive_delta=... negative_delta=... sign_preserved_rate=...
PHASE_PASS epoch_sanity
```

### 5.6 Quy tắc diễn giải tự động

Tạo biến `epoch_sanity_interpretation`:

```text
- "direction_stable" nếu >=75% comparisons giữ cùng dấu với kết quả chính.
- "mixed_direction" nếu 50--74% giữ dấu.
- "direction_unstable" nếu <50% giữ dấu.
```

Nếu `direction_unstable`, manuscript phải hạ claim mạnh:

```text
The fixed-budget graph effect should be interpreted as a two-epoch diagnostic result only; longer-budget sanity checks do not support a stable selected-graph direction.
```

Nếu `direction_stable`, manuscript có thể viết:

```text
The longer-budget sanity check preserves the direction of the selected-graph effect in most inspected settings, supporting the use of the fixed-budget results as protocol diagnostics rather than SOTA-tuned model comparisons.
```

---

## 6. Phase 3 — Tạo bảng tổng hợp cho manuscript

### 6.1 Bảng validation candidates pre-specified

Tạo `p0_validation_candidates.yaml` và bảng `.csv/.tex` với 9 candidate configurations.

Nếu không tìm thấy config thật trong repo, tạo bảng template nhưng đánh dấu `status=TO_VERIFY`.

Cấu trúc bảng:

```text
candidate_id
candidate_name
relations_enabled
selection_gate
beta_or_weighting
pre_specified_yes_no
test_used_for_selection_yes_no
notes
```

Yêu cầu: `pre_specified_yes_no=yes` và `test_used_for_selection_yes_no=no` cho mọi candidate, nếu có bằng chứng timestamp/hash.

### 6.2 Bảng main AUC delta + CI + Holm

Tạo bảng ngắn cho main text:

```text
- dataset
- backbone
- mean_auc_no_graph
- mean_auc_selected_graph
- mean_delta_auc
- ci95_low
- ci95_high
- raw_p
- holm_p
- holm_significant_yes_no
- practical_label: negligible/small/moderate
- interpretation: confirmatory/diagnostic_only/not_supported
```

Nếu file AUC cũ đã tồn tại, đọc và chuẩn hóa. Không tính lại nếu thiếu prediction files; ghi `NA` và log.

### 6.3 Bảng selected relation variants

```text
- dataset
- backbone
- selected_candidate_most_frequent
- n_fold_seed_observations
- selection_frequency
- selected_relations
- includes_E_co_yes_no
- includes_E_sim_effective_yes_no
- notes
```

Nếu `E_sim` rỗng, dùng nhãn `E_sim^eff=empty`.

### 6.4 Bảng sparse bins descriptive

```text
- dataset
- bin_name: <=50, <=100, <=200, >500
- n_skills
- n_test_interactions
- effective_sample_size
- reliability_flag: reliable/limited/insufficient
- mean_degree_E_pre
- mean_degree_E_co
- coverage_ratio
```

Không chạy adaptive stratification.

### 6.5 Bảng reproducibility checklist

```text
artifact
path
status: available/missing/partial
sha256
purpose
used_in_main_text_yes_no
```

### 6.6 Log bắt buộc

```text
PHASE_START table_generation
TABLE_CREATED table_dataset_statistics rows=...
TABLE_CREATED table_graph_provenance_corrected rows=...
TABLE_CREATED table_main_auc_delta_holm rows=...
TABLE_CREATED table_epoch_sanity rows=...
TABLE_CREATED table_reproducibility_checklist rows=...
PHASE_PASS table_generation
```

---

## 7. Phase 4 — Manifest, validation, Definition of Done

### 7.1 SHA256 manifest

Tạo `manifests/sha256_manifest.csv` cho mọi file trong `results_p0_revision/`:

```text
relative_path
sha256
size_bytes
modified_time
```

### 7.2 Run environment

Tạo `manifests/run_environment.txt`:

```text
python_version
platform
pip_freeze_or_conda_env
cuda_available
gpu_name_or_NA
git_commit
git_branch
uncommitted_changes_yes_no
```

### 7.3 Definition of Done

Tạo `manifests/final_definition_of_done.md` với checklist:

```markdown
# Definition of Done — P0 LC-MRSG++ Revision

- [ ] Dataset statistics table created and internally consistent.
- [ ] Subsampling disclosure written if processed data are smaller than expected.
- [ ] KDD2010 E_co resolved as unique edges / support records / multi-edge / fixed error.
- [ ] E_sim trace completed; manuscript label adjusted if E_sim is empty.
- [ ] Junyi coverage and isolated-node analysis completed.
- [ ] L1--L6 leakage audit PASS, especially L1, L5, L6.
- [ ] Epoch sanity-check completed or limitation explicitly logged.
- [ ] Main ΔAUC table uses CI + Holm correction only; heavy statistics moved to supplementary.
- [ ] P0/P1/P2/P3 boundary table included in manuscript.
- [ ] No claim of SOTA, universal graph improvement, calibration, SSA-CL, or learning-path recommendation.
- [ ] Reproducibility checklist and SHA256 manifest completed.
- [ ] References 2024+ manually verified before submission.
```

### 7.4 Validation script

Tạo hoặc chạy `scripts/p0_validate_outputs.py` để kiểm tra:

```text
- mọi file bắt buộc có tồn tại không;
- không có hard-fail trong log;
- table_graph_provenance_corrected không có unique_undirected_edges > max_possible_undirected_pairs;
- leakage L1/L5/L6 không fail;
- table_epoch_sanity có ít nhất một kết quả hoặc có limitation rõ;
- sha256 manifest có đủ file chính;
- bảng LaTeX không rỗng.
```

Log:

```text
PHASE_START manifest_validation
VALIDATION_CHECK name=required_files status=PASS
VALIDATION_CHECK name=graph_limits status=PASS
VALIDATION_CHECK name=leakage_L1_L5_L6 status=PASS
VALIDATION_CHECK name=epoch_sanity status=PASS_OR_LIMITED
VALIDATION_CHECK name=latex_tables_nonempty status=PASS
PHASE_PASS manifest_validation
```

---

## 8. Scripts cần tạo nếu chưa có

Nếu repository chưa có các script tương ứng, hãy tạo trong `scripts/`:

```text
scripts/p0_repo_scan.py
scripts/p0_audit_dataset_stats.py
scripts/p0_audit_graph_provenance.py
scripts/p0_trace_esim_pipeline.py
scripts/p0_audit_junyi_coverage.py
scripts/p0_audit_leakage_L1_L6.py
scripts/p0_epoch_sanity.py
scripts/p0_generate_tables.py
scripts/p0_build_manifest.py
scripts/p0_validate_outputs.py
scripts/p0_run_all_revision.py
```

`p0_run_all_revision.py` phải là entrypoint chính:

```bash
python scripts/p0_run_all_revision.py --config results_p0_revision/configs/p0_revision_config.yaml
```

Trong mỗi script, dùng logging chuẩn:

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
```

Mỗi phase cần ghi log ra console và file.

---

## 9. Quy tắc viết bảng LaTeX

1. Bảng `.tex` không được chứa `\begin{table}` nếu sẽ được `\input{}` vào manuscript có wrapper riêng. Nếu cần standalone, tạo hai bản: `_body.tex` và `_table.tex`.
2. Số quá dài dùng `\scriptsize` hoặc `\resizebox{\textwidth}{!}{...}`.
3. Mọi bảng phải có cột `Notes` nếu có `NA` hoặc `TO_VERIFY`.
4. Không dùng dấu phẩy kiểu Việt Nam trong số thập phân; dùng dấu chấm.
5. Với KDD2010 E_co, bảng phải có cả `support_records` và `unique_undirected_edges` để tránh hiểu nhầm.

---

## 10. Nội dung reviewer-note tự động sinh

Tạo `supplementary/reviewer_response_notes.md` với các đoạn sẵn để đưa vào response letter:

```markdown
## Dataset and graph-provenance correction
We added a dataset-statistics and graph-provenance audit table reporting users, questions, skills, interactions, split sizes, user-skill density, unique graph edges, support records, isolated nodes, and SHA256 provenance.

## KDD2010 E_co clarification
We clarified whether the previously large KDD2010 E_co count referred to unique KC--KC edges, directed/mirrored rows, support records, or multi-edge records. The revised manuscript now reports unique KC--KC edges separately from support records.

## E_sim effective relation clarification
We traced the E_sim pipeline and now label empty similarity branches as E_sim^eff=empty where applicable. The manuscript no longer interprets empty E_sim branches as evidence that similarity edges improved prediction.

## Epoch sanity-check
We added a longer-budget sanity check for DKT and simpleKT under no-graph and validation-selected graph conditions at 5 and 10 epochs. This check is reported as a stability audit, not as SOTA tuning.

## Scope control
We moved calibration, adaptive stratification, SSA-CL, and learning-path recommendation outside P0 and stated them as future or separate-paper work.
```

---

## 11. Quy tắc cập nhật manuscript sau khi chạy

Sau khi chạy xong, cập nhật file LaTeX bằng cách copy các bảng `.tex` từ:

```text
results_p0_revision/tables_tex/
```

vào thư mục `tables/` của Overleaf hoặc giữ nguyên path và gọi:

```latex
\input{results_p0_revision/tables_tex/table_dataset_statistics.tex}
\input{results_p0_revision/tables_tex/table_graph_provenance_corrected.tex}
\input{results_p0_revision/tables_tex/table_leakage_audit_L1_L6.tex}
\input{results_p0_revision/tables_tex/table_main_auc_delta_holm.tex}
\input{results_p0_revision/tables_tex/table_selected_relation_variants.tex}
\input{results_p0_revision/tables_tex/table_sparse_bins_descriptive.tex}
\input{results_p0_revision/tables_tex/table_epoch_sanity.tex}
\input{results_p0_revision/tables_tex/table_reproducibility_checklist.tex}
```

Nếu bảng nào thiếu, manuscript phải hiển thị placeholder `TO BE FILLED AFTER P0 REVISION RUN` thay vì bịa kết quả.

---

## 12. Lệnh chạy cuối cùng

Hãy triển khai toàn bộ pipeline, sau đó chạy:

```bash
python scripts/p0_run_all_revision.py --config results_p0_revision/configs/p0_revision_config.yaml 2>&1 | tee results_p0_revision/logs/master_run_$(date +%Y%m%d_%H%M%S).log
```

Kết thúc phải in ra console:

```text
P0_REVISION_PIPELINE_FINISHED
STATUS=<PASS|PASS_WITH_LIMITATIONS|FAIL>
KEY_OUTPUT_DIR=results_p0_revision/
NEXT_ACTION=Copy tables_tex into LaTeX manuscript and update the interpretation paragraphs according to epoch_sanity_interpretation.
```

---

## 13. Không được làm

Không được tự động làm các việc sau:

```text
- Không thêm calibration/ECE/Brier vào P0.
- Không thêm adaptive sparse stratification.
- Không thêm SSA-CL/InfoNCE.
- Không thêm learning-path recommendation experiments.
- Không claim SOTA.
- Không dùng test set để chọn graph.
- Không thay đổi raw dataset.
- Không xóa kết quả cũ.
- Không sửa kết quả để đẹp hơn.
```

---

## 14. Tiêu chí thành công

Pipeline đạt `PASS` nếu:

```text
- Dataset stats đủ 3 datasets.
- KDD2010 E_co được phân định rõ: unique edges / support records / multi-edge / lỗi đã sửa.
- E_sim trace có quyết định rõ.
- Junyi graph coverage có số skill covered và isolated nodes.
- Leakage audit L1--L6 không fail ở L1/L5/L6.
- Epoch sanity có kết quả hoặc limitation rõ.
- Tất cả bảng CSV và LaTeX được xuất.
- SHA256 manifest hoàn chỉnh.
- Definition of Done được tạo.
```

Pipeline đạt `PASS_WITH_LIMITATIONS` nếu epoch sanity chưa chạy đủ folds/seeds vì compute budget nhưng các audit dữ liệu/graph đều pass và limitation được ghi rõ.

Pipeline `FAIL` nếu có lỗi graph leakage, KDD2010 E_co vượt giới hạn mà không giải thích được, hoặc selection dùng test evidence.
