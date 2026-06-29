# LC-MRSG Q3 Upgrade — Experimental Automation Plan for Google Antigravity

**Project:** `lc_mrsg_q3_upgrade`  
**Paper:** *Leakage-Controlled Multi-Relational Skill Graph Construction for Sparse-Skill Knowledge Tracing*  
**Goal:** build the missing experimental evidence required to make the manuscript defensible for a Scopus Q3 journal submission.

---

## 0. Core instruction for Antigravity

Build an automated, reproducible experiment pipeline that:

1. Reuses the current LC-MRSG graph-construction/audit protocol.
2. Fixes the technical weaknesses identified by the supervisor checklist.
3. Re-runs or re-evaluates all required experiments.
4. Exports publication-ready CSV, LaTeX tables, figures, and a final diagnostic report.
5. Produces evidence for:
   - full-scale dataset preprocessing;
   - non-degenerate model training;
   - constrained prerequisite graph density;
   - top-k similarity graph construction;
   - sparse-skill predictive performance;
   - multi-fold and multi-seed reliability;
   - statistical significance;
   - leakage audit L1–L6;
   - reproducibility.

The pipeline must prioritize scientific validity over producing optimistic results. If a model or graph variant does not improve performance, report it honestly.

---

## 1. Background: why this additional experiment package is needed

The current manuscript is already strong as a **protocol/audit paper**. It includes LC-MRSG, five-type leakage auditing, \(E_{co}\) provenance checks, top-k similarity diagnostics, noise injection, temporal split evaluation, and DDR sweeps.

However, the supervisor's checklist identifies several remaining risks before targeting a Scopus Q3 journal:

1. Some graph-based models, especially GIKT/SKT/simpleKT, show near-zero standard deviation across seeds.
2. Some processed dataset sizes may still be smaller than accepted benchmark scales.
3. \(E_{pre}\) density is suspiciously close to 0.5 across datasets.
4. Sparse-skill evidence is currently stronger as **structural coverage** than as **predictive improvement**.
5. Top-k similarity evidence must include the number of retained similarity edges, not only AUC.
6. Q3-level evidence should include multi-fold results, confidence intervals, and statistical tests.

---

## 2. Required input files

Antigravity should expect the following input folders and files.

```text
data/
  raw/
    assist2012/
    junyi/
    kdd2010/
  processed/
    assist2012/
    junyi/
    kdd2010/

graphs/
  existing_fold0_exports/
  qmatrix/
  metadata/

results_existing/
  144_run_export.csv
  table1_dataset_stats.csv
  table2_graph_comparison.csv
  table3_baseline_metrics.csv
  TopK_Similarity_Evaluation.csv
  Noise_Injection_Robustness.csv
  Temporal_Split_Evaluation.csv
  figures/
```

If any file is missing, create a warning in:

```text
results/reports/missing_inputs.md
```

---

## 3. Global configuration

Create a config file:

```text
configs/q3_upgrade.yaml
```

with the following minimum fields:

```yaml
project_name: lc_mrsg_q3_upgrade
random_seeds: [42, 43, 44, 2024, 2025]
folds: [0, 1, 2]
datasets:
  assist2012:
    target_min_learners: 40000
    target_min_interactions: 1000000
    target_min_kcs: 250
  junyi:
    target_min_learners: 100000
    target_min_interactions: 5000000
    target_min_kcs: 40
  kdd2010:
    target_min_learners: 500
    target_min_interactions: 500000
    target_min_kcs: 400

models:
  - bkt
  - dkt
  - gikt
  - skt
  - simplekt

graph_variants:
  - no_graph
  - e_pre_raw
  - e_pre_pruned
  - e_pre_transitive_reduced
  - e_pre_e_sim_topk
  - e_pre_e_sim_topk_e_co

topk_similarity:
  k_values: [3, 5, 10, 20]

prerequisite_pruning:
  topk_out: [3, 5, 10]
  density_targets:
    assist2012: 0.05
    kdd2010: 0.05
    junyi: 0.20

sparse_thresholds:
  very_sparse_max: 50
  sparse_max: 100
  medium_max: 500

noise_injection:
  rates: [0.05, 0.10, 0.20, 0.50, 1.00]

splits:
  learner_level: true
  temporal_80_20: true

metrics:
  - auc
  - accuracy
  - nll
  - rmse
  - brier
  - ece
```

---

# PART A — Data and preprocessing validation

## Task A1. Verify processed dataset scale

### Objective

Confirm whether the processed datasets are large enough for a Q3-level manuscript.

### Required output

```text
results/tables/q3_table_dataset_scale.csv
results/tables/q3_table_dataset_scale.tex
results/reports/dataset_scale_audit.md
```

### Required columns

```text
dataset
raw_num_learners
processed_num_learners
raw_num_interactions
processed_num_interactions
raw_num_kcs
processed_num_kcs
retention_rate_learners
retention_rate_interactions
retention_rate_kcs
status
notes
```

### Acceptance criteria

Mark `status = PASS` only if:

```text
KDD2010: processed learners >= 500 and interactions >= 500000 and KCs >= 400
ASSIST2012: processed learners >= 40000 or a justified documented subset is used
JUNYI: processed KCs >= 40 or the compact-skill limitation is explicitly documented
```

If the dataset does not pass, generate:

```text
results/reports/preprocessing_failure_reason.md
```

and list exact filters causing the reduction.

---

## Task A2. Re-run preprocessing using standard settings if scale fails

### Objective

If any dataset is too small, re-run preprocessing with less restrictive filters.

### Required implementation

Create:

```text
src/preprocess/preprocess_assist2012.py
src/preprocess/preprocess_junyi.py
src/preprocess/preprocess_kdd2010.py
```

Each script must:

1. Load raw dataset.
2. Map learner IDs, question IDs, and KC IDs.
3. Preserve Q-matrix mapping.
4. Preserve timestamps or sequence order.
5. Filter only invalid rows.
6. Export learner-level and temporal-ready processed data.

### Required output

```text
data/processed/{dataset}/interactions.csv
data/processed/{dataset}/qmatrix.csv
data/processed/{dataset}/metadata.json
results/tables/q3_table_dataset_preprocessing.tex
```

### Required metadata

Each `metadata.json` must contain:

```json
{
  "dataset": "...",
  "num_learners": 0,
  "num_questions": 0,
  "num_kcs": 0,
  "num_interactions": 0,
  "min_sequence_length": 0,
  "filters_applied": [],
  "timestamp_available": true,
  "qmatrix_source": "static/train-only/expert",
  "notes": ""
}
```

---

# PART B — Model training integrity and zero-variance debugging

## Task B1. Diagnose zero-variance rows

### Objective

Detect whether GIKT, SKT, and simpleKT are genuinely training or producing degenerate constant predictions.

### Required output

```text
results/tables/q3_zero_variance_diagnosis.csv
results/tables/q3_zero_variance_diagnosis.tex
results/reports/zero_variance_debug_report.md
```

### Required columns

```text
dataset
model
graph_variant
seed
auc
accuracy
nll
prediction_mean
prediction_std
prediction_min
prediction_max
positive_rate
train_loss_start
train_loss_end
valid_loss_best
num_trainable_params
gradient_norm_mean
status
diagnosis
```

### Diagnostics

Set `status = FAIL` if any of the following holds:

```text
prediction_std < 1e-4
abs(train_loss_start - train_loss_end) < 1e-4
gradient_norm_mean == 0
auc_std_across_seeds == 0 for a neural model
```

### Required plots

```text
results/figures/zero_variance/{dataset}_{model}_{graph_variant}_loss_curve.pdf
results/figures/zero_variance/{dataset}_{model}_{graph_variant}_prediction_hist.pdf
```

---

## Task B2. Fix model training or graph integration if zero variance is found

### Objective

Ensure that each model actually learns from data and, where relevant, receives graph input.

### Required checks

For each model:

#### BKT

- Check slip, guess, transition, prior parameters.
- Verify that predictions are not constant.

#### DKT

- Verify sequence input.
- Verify correctness embedding.
- Verify train loss decreases.

#### GIKT / SKT

- Verify graph adjacency is loaded.
- Verify graph embeddings are used in prediction.
- Verify graph variant changes at least one tensor used by the model.

#### simpleKT

- Verify implementation is not a placeholder.
- Verify model uses item/skill embeddings.
- Verify training matches pyKT-style input conventions.

### Required output

```text
results/reports/model_training_integrity_report.md
results/tables/q3_model_training_integrity.csv
```

### Acceptance criteria

For each dataset and model:

```text
prediction_std > 1e-4
train_loss_end < train_loss_start
auc_std_across_seeds > 0 for neural models unless deterministic mode is explicitly justified
```

---

# PART C — Leakage audit L1–L6

## Task C1. Re-run leakage audit on every fold and dataset

### Objective

Produce a full leakage audit table across datasets, folds, graph variants, and relation types.

### Leakage definitions

Use the following checks:

```text
L1 Edge leakage: edge support contains valid/test evidence.
L2 Q-matrix leakage: Q-matrix is derived from full response logs.
L3 Temporal leakage: edge support uses interactions after train cutoff.
L4 Cold-start / learner-overlap leakage: strict no-learner-overlap or sparse-KC neighborhood violation.
L5 Co-occurrence leakage: E_co support counts/PMI include held-out support.
L6 Hyperparameter leakage: threshold selected using test performance.
```

### Required output

```text
results/tables/q3_leakage_audit_full.csv
results/tables/q3_leakage_audit_full.tex
results/figures/leakage/q3_leakage_audit_heatmap.pdf
results/reports/leakage_audit_report.md
```

### Required columns

```text
dataset
fold
graph_variant
relation
L1_edge
L2_qmatrix
L3_temporal
L4_coldstart
L5_cooccurrence
L6_hyperparameter
num_failed_edges
num_failed_sparse_kcs
notes
```

### Acceptance criteria

For a Q3-ready manuscript:

```text
L1 PASS
L2 PASS
L3 PASS
L5 PASS
L6 PASS
L4 may be WARN only if the manuscript explicitly frames it as a cold-start limitation.
```

If L4 is FAIL, export exact causes:

```text
results/tables/q3_L4_failure_details.csv
```

---

## Task C2. Compute co-occurrence leakage ratio

### Objective

Add a quantitative measure for co-occurrence leakage.

### Formula

\[
\rho_{co} =
\frac{
\left|\{(i,j)\in E_{co}: S_{co}(i,j)\cap(D_{valid}\cup D_{test})\neq\emptyset\}\right|
}{
|E_{co}|
}
\]

### Required output

```text
results/tables/q3_cooccurrence_leakage_ratio.csv
results/tables/q3_cooccurrence_leakage_ratio.tex
```

### Required columns

```text
dataset
fold
num_eco_edges
num_edges_with_heldout_support
rho_co
status
```

### Acceptance criteria

```text
rho_co = 0
```

---

# PART D — E_co provenance audit and sparse structural support

## Task D1. Produce a clearly labeled E_co provenance audit module

### Objective

Create a table with the four required E_co checks.

### Required output

```text
results/tables/q3_eco_provenance_audit.csv
results/tables/q3_eco_provenance_audit.tex
```

### Required columns

```text
dataset
fold
symmetry_pass
train_only_support_pass
weight_mean
weight_std
weight_median
weight_max
sparse_kc_coverage
num_sparse_kcs
num_sparse_kcs_with_eco_neighbor
status
```

### Required checks

1. **Symmetry**
\[
(c_i,c_j,w)\in E_{co} \Rightarrow (c_j,c_i,w)\in E_{co}
\]

2. **Train-only support**
\[
S_{co}(i,j)\subseteq D_{train}
\]

3. **Weight distribution**
Report mean, std, median, max and export histogram.

4. **Sparse-KC coverage**
\[
\mathrm{Cov}*{co}(C_s)=
\frac{
|\{c\in C_s:\deg*{E_{co}}(c)>0\}|
}{
|C_s|
}
\]

### Required figures

```text
results/figures/eco/q3_eco_weight_distribution_{dataset}.pdf
results/figures/eco/q3_eco_weight_cdf_{dataset}.pdf
```

---

## Task D2. Compute degree support by sparse-skill stratum

### Objective

Connect sparse-skill structural coverage with a mechanism.

### Required output

```text
results/tables/q3_stratum_degree_support.csv
results/tables/q3_stratum_degree_support.tex
results/figures/sparse/q3_stratum_degree_support.pdf
```

### Required columns

```text
dataset
fold
stratum
num_kcs
avg_e_pre_degree
avg_e_sim_degree
avg_e_co_degree
median_e_co_degree
sparse_isolation_rate_pre
sparse_isolation_rate_sim
sparse_isolation_rate_co
```

### Strata

Use:

```text
very_sparse: n_train(c) <= 50
sparse: 50 < n_train(c) <= 100
medium: 100 < n_train(c) <= 500
frequent: n_train(c) > 500
```

### Interpretation requirement

Export a short written interpretation:

```text
results/reports/stratum_degree_support_interpretation.md
```

The report must answer:

1. Do sparse KCs receive structural support from \(E_{co}\)?
2. Is \(E_{co}\) support stronger for sparse or frequent KCs?
3. Does this differ by dataset?
4. Does graph coverage justify keeping “Sparse-Skill” in the title?

---

# PART E — Prerequisite graph density correction

## Task E1. Diagnose why E_pre density is exactly or nearly 0.5

### Objective

Determine whether the prerequisite graph construction has an artificial top-50% rule or another artifact.

### Required output

```text
results/reports/e_pre_density_diagnosis.md
results/tables/q3_e_pre_density_diagnosis.csv
results/figures/e_pre/q3_pre_score_distribution_{dataset}.pdf
```

### Required columns

```text
dataset
fold
num_kcs
num_e_pre_edges
density
score_threshold
score_min
score_median
score_max
construction_rule
suspected_artifact
```

### Artifact warning

Set `suspected_artifact = TRUE` if:

```text
density is exactly 0.5000 across multiple datasets
or construction rule keeps a fixed top percentage of possible edges
```

---

## Task E2. Implement constrained prerequisite graph variants

### Objective

Create more pedagogically plausible prerequisite graphs.

### Variants

```text
e_pre_raw
e_pre_top3
e_pre_top5
e_pre_top10
e_pre_confidence_0_70
e_pre_confidence_0_80
e_pre_transitive_reduced
```

### Required output

```text
graphs/{dataset}/fold_{fold}/e_pre_variants/
results/tables/q3_e_pre_pruning_summary.csv
results/tables/q3_e_pre_pruning_summary.tex
results/figures/e_pre/q3_e_pre_density_vs_auc.pdf
```

### Required columns

```text
dataset
fold
variant
nodes
edges
density
avg_degree
dag_pass
num_cycles_before_pruning
num_cycles_after_pruning
ddr_edge_drop_0_1
auc_dkt
auc_simplekt
notes
```

### Acceptance criteria

Target density:

```text
ASSIST2012: density <= 0.05
KDD2010: density <= 0.05
JUNYI: density <= 0.20
```

If predictive performance decreases, report the trade-off honestly.

---

# PART F — Top-k similarity graph repair

## Task F1. Export number of top-k similarity edges retained

### Objective

Resolve the inconsistency where threshold-based \(E_{sim}=0\), but top-k similarity is said to repair the relation.

### Required output

```text
results/tables/q3_topk_similarity_edges.csv
results/tables/q3_topk_similarity_edges.tex
results/figures/sim/q3_topk_similarity_edges_by_dataset.pdf
```

### Required columns

```text
dataset
fold
k
num_nodes_with_similarity
num_similarity_edges
density
avg_similarity_score
median_similarity_score
auc_dkt
auc_simplekt
```

### Required k values

```text
k = 3, 5, 10, 20
```

### Interpretation report

```text
results/reports/topk_similarity_interpretation.md
```

Answer:

1. Does top-k similarity actually produce non-empty \(E_{sim}\)?
2. Does increasing k increase density too aggressively?
3. Which k is best by validation AUC?
4. Does top-k similarity help more on datasets where threshold similarity was empty?

---

# PART G — Sparse-skill predictive ablation

## Task G1. Compute stratified AUC, ACC, NLL, and RMSE

### Objective

Justify the “Sparse-Skill” claim using predictive evidence, not only structural coverage.

### Required output

```text
results/tables/q3_sparse_skill_predictive_ablation.csv
results/tables/q3_sparse_skill_predictive_ablation.tex
results/figures/sparse/q3_sparse_skill_auc_by_graph.pdf
results/reports/sparse_skill_predictive_report.md
```

### Required columns

```text
dataset
fold
seed
model
graph_variant
stratum
num_test_interactions
auc
accuracy
nll
rmse
```

### Models

```text
bkt
dkt
gikt
skt
simplekt
```

### Graph variants

```text
no_graph
e_pre_pruned
e_pre_e_sim_topk
e_pre_e_sim_topk_e_co
```

### Strata

```text
very_sparse
sparse
medium
frequent
```

### Required summary table

Create:

```text
results/tables/q3_sparse_skill_summary_mean_std.tex
```

with:

```text
dataset
model
graph_variant
AUC_very_sparse_mean±std
AUC_sparse_mean±std
AUC_medium_mean±std
AUC_frequent_mean±std
```

### Interpretation requirements

The report must state:

1. Whether graph variants improve sparse-skill prediction.
2. Whether \(E_{co}\) helps sparse KCs more than frequent KCs.
3. Whether the gains are consistent across datasets.
4. Whether the paper should keep “Sparse-Skill” in the title.

---

# PART H — Multi-fold and multi-seed confirmatory evaluation

## Task H1. Run confirmatory evaluation across folds and seeds

### Objective

Move from fold-0 diagnostic evidence to Q3-level confirmatory evidence.

### Required design

Minimum Q3 design:

```text
datasets = assist2012, junyi, kdd2010
folds = 0, 1, 2
seeds = 42, 43, 44, 2024, 2025
models = dkt, gikt, skt, simplekt
graph_variants = no_graph, e_pre_pruned, e_pre_e_sim_topk, e_pre_e_sim_topk_e_co
```

If compute is limited, use:

```text
datasets = assist2012, junyi, kdd2010
folds = 0, 1, 2
seeds = 42, 43, 44
models = dkt, simplekt
graph_variants = no_graph, e_pre_pruned, full
```

### Required output

```text
results/tables/q3_multifold_confirmatory_results.csv
results/tables/q3_multifold_confirmatory_results.tex
results/reports/multifold_confirmatory_report.md
```

### Required columns

```text
dataset
fold
seed
model
graph_variant
auc
accuracy
nll
rmse
brier
ece
train_time_sec
```

### Summary formula

For each setting:

\[
\bar{x} =
\frac{1}{KF}
\sum_{k=1}^{K}\sum_{f=1}^{F} x_{k,f}
\]

\[
s =
\sqrt{
\frac{1}{KF-1}
\sum_{k=1}^{K}\sum_{f=1}^{F}
(x_{k,f}-\bar{x})^2
}
\]

Export mean ± std across fold-seed pairs.

---

# PART I — Statistical significance tests

## Task I1. Paired tests for key graph comparisons

### Objective

Provide statistical evidence for Q3-level claims.

### Comparisons

For each dataset and model:

```text
no_graph vs e_pre_pruned
no_graph vs e_pre_e_sim_topk
no_graph vs e_pre_e_sim_topk_e_co
e_pre_pruned vs e_pre_e_sim_topk_e_co
threshold_similarity vs topk_similarity
temporal_no_graph vs temporal_full
```

### Required tests

Implement:

1. Paired t-test over fold-seed pairs.
2. Wilcoxon signed-rank test.
3. Bootstrap 95% confidence interval for \(\Delta\)AUC.
4. Effect size using Cohen's \(d\).

### Required output

```text
results/tables/q3_statistical_tests.csv
results/tables/q3_statistical_tests.tex
results/reports/statistical_tests_report.md
```

### Required columns

```text
dataset
model
comparison
metric
mean_delta
ci95_low
ci95_high
paired_t_p
wilcoxon_p
cohens_d
interpretation
```

### Interpretation rule

Do not claim improvement unless:

```text
mean_delta > 0
ci95_low > 0
p < 0.05
```

If gains are not significant, write:

```text
"Graph relations show diagnostic value but do not provide statistically reliable predictive gains under this setting."
```

---

# PART J — Calibration and probability quality

## Task J1. Compute Brier score and ECE

### Objective

Add probability-quality evidence beyond AUC and ACC.

### Required output

```text
results/tables/q3_calibration_metrics.csv
results/tables/q3_calibration_metrics.tex
results/figures/calibration/q3_reliability_diagram_{dataset}_{model}_{graph}.pdf
results/reports/calibration_report.md
```

### Metrics

Brier score:

\[
\mathrm{Brier} =
\frac{1}{N}
\sum_{i=1}^{N}
(\hat{p}_i-y_i)^2
\]

Expected calibration error:

\[
\mathrm{ECE} =
\sum_{b=1}^{B}
\frac{|B_b|}{N}
\left|
\mathrm{acc}(B_b)-\mathrm{conf}(B_b)
\right|
\]

Use:

```text
B = 10 bins
```

### Required columns

```text
dataset
fold
seed
model
graph_variant
brier
ece
nll
auc
```

### Interpretation

Report whether graph relations improve or worsen calibration.

---

# PART K — Temporal split and learner-level split comparison

## Task K1. Compare learner-level and temporal evaluation

### Objective

Clarify whether graph conclusions hold under deployment-style temporal evaluation.

### Required output

```text
results/tables/q3_split_strategy_comparison.csv
results/tables/q3_split_strategy_comparison.tex
results/reports/split_strategy_comparison_report.md
```

### Required columns

```text
dataset
model
graph_variant
split_strategy
auc_mean
auc_std
acc_mean
acc_std
nll_mean
nll_std
```

### Split strategies

```text
learner_level
temporal_80_20
```

### Required interpretation

Answer:

1. Does graph gain shrink under temporal split?
2. Does temporal split expose leakage or overfitting?
3. Which split should be used for final claims?

---

# PART L — Robustness and graph noise

## Task L1. Re-run noise injection after model debugging

### Objective

Ensure noise robustness results are meaningful after fixing any degenerate model training.

### Required output

```text
results/tables/q3_noise_injection_robustness.csv
results/tables/q3_noise_injection_robustness.tex
results/figures/noise/q3_noise_robustness_curves.pdf
```

### Required columns

```text
dataset
fold
seed
model
base_graph_variant
noise_rate
auc
delta_auc
accuracy
nll
```

### Noise rates

```text
0.00
0.05
0.10
0.20
0.50
1.00
```

### Robustness slope

\[
\mathrm{Slope}_{noise}
=

\frac{
\mathrm{AUC}*{100\%}-\mathrm{AUC}*{0\%}
}{
1.0
}
\]

Export:

```text
results/tables/q3_noise_robustness_slope.csv
```

---

# PART M — DDR and prerequisite structural validity

## Task M1. Re-run DDR on constrained E_pre variants

### Objective

Show whether pruning makes prerequisite graphs structurally more valid.

### Required output

```text
results/tables/q3_ddr_prerequisite_variants.csv
results/tables/q3_ddr_prerequisite_variants.tex
results/figures/ddr/q3_ddr_sweep_by_variant.pdf
```

### Required columns

```text
dataset
fold
e_pre_variant
perturbation_type
perturbation_probability
mean_ddr
std_ddr
dag_pass_rate
```

### Perturbation types

```text
edge_drop
node_drop
subgraph
attr_mask
```

### Probability values

```text
0.1, 0.2, 0.3, 0.4, 0.5
```

### Interpretation

Answer:

1. Is raw \(E_{pre}\) too dense to be pedagogically meaningful?
2. Does constrained \(E_{pre}\) reduce DDR?
3. Which prerequisite variant should be used in final experiments?

---

# PART N — Reproducibility package

## Task N1. Create a reproducibility manifest

### Objective

Make the Q3 revision reproducible.

### Required output

```text
REPRODUCIBILITY.md
results/tables/q3_reproducibility_manifest.tex
```

### Required content

```text
1. Software environment
2. Python version
3. Package versions
4. Dataset download/source notes
5. Preprocessing commands
6. Graph construction commands
7. Training commands
8. Evaluation commands
9. Random seeds
10. Expected output files
11. Hardware used
12. Known limitations
```

### Required command examples

```bash
python src/preprocess/preprocess_assist2012.py --config configs/q3_upgrade.yaml
python src/graphs/build_lc_mrsg.py --dataset assist2012 --fold 0 --config configs/q3_upgrade.yaml
python src/audit/run_leakage_audit.py --dataset assist2012 --fold 0
python src/train/run_q3_experiment.py --dataset assist2012 --model dkt --graph_variant e_pre_e_sim_topk_e_co --fold 0 --seed 42
python src/evaluate/compute_sparse_skill_auc.py --dataset assist2012
python src/reports/build_q3_report.py
```

---

# PART O — Final report generation

## Task O1. Generate final Q3 diagnostic report

### Objective

Create a single report that can be used to update the manuscript.

### Required output

```text
results/reports/q3_upgrade_final_report.md
```

### Required sections

```text
1. Dataset scale audit
2. Model training integrity and zero-variance diagnosis
3. Leakage audit L1-L6
4. E_co provenance audit
5. Sparse stratum degree support
6. E_pre density diagnosis and pruning
7. Top-k similarity edge retention
8. Sparse-skill predictive ablation
9. Multi-fold confirmatory performance
10. Statistical significance tests
11. Calibration metrics
12. Temporal vs learner-level split comparison
13. Noise robustness
14. DDR structural validity
15. Reproducibility manifest
16. Recommended manuscript changes
17. Tables and figures ready for LaTeX
18. Remaining limitations
```

---

# PART P — LaTeX export

## Task P1. Export all publication-ready tables

### Required output folder

```text
paper/tables_q3/
```

### Required tables

```text
q3_table_dataset_scale.tex
q3_table_dataset_preprocessing.tex
q3_leakage_audit_full.tex
q3_cooccurrence_leakage_ratio.tex
q3_eco_provenance_audit.tex
q3_stratum_degree_support.tex
q3_e_pre_density_diagnosis.tex
q3_e_pre_pruning_summary.tex
q3_topk_similarity_edges.tex
q3_sparse_skill_summary_mean_std.tex
q3_multifold_confirmatory_results.tex
q3_statistical_tests.tex
q3_calibration_metrics.tex
q3_split_strategy_comparison.tex
q3_noise_injection_robustness.tex
q3_ddr_prerequisite_variants.tex
q3_reproducibility_manifest.tex
```

---

## Task P2. Export all publication-ready figures

### Required output folder

```text
paper/figures_q3/
```

### Required figures

```text
q3_dataset_scale_bar.pdf
q3_leakage_audit_heatmap.pdf
q3_eco_weight_distribution_assist2012.pdf
q3_eco_weight_distribution_junyi.pdf
q3_eco_weight_distribution_kdd2010.pdf
q3_stratum_degree_support.pdf
q3_e_pre_density_vs_auc.pdf
q3_topk_similarity_edges_by_dataset.pdf
q3_sparse_skill_auc_by_graph.pdf
q3_reliability_diagram_assist2012_dkt_full.pdf
q3_noise_robustness_curves.pdf
q3_ddr_sweep_by_variant.pdf
```

---

# PART Q — Manuscript update guidance

## Task Q1. Produce manuscript-ready paragraphs

### Required output

```text
paper/text_q3/
  abstract_revision.md
  introduction_revision.md
  method_revision.md
  results_revision.md
  discussion_revision.md
  limitations_revision.md
  conclusion_revision.md
```

### Required writing principles

1. Do not overclaim.
2. Separate graph validity from model superiority.
3. Report L4 warning honestly if it remains unresolved.
4. Do not claim sparse-skill improvement unless stratified AUC supports it.
5. If graph variants do not improve AUC significantly, frame the contribution as auditability and reproducibility.
6. State exactly which results are fold-level diagnostics and which are multi-fold confirmatory results.

---

# 4. Priority order

## P0 — Show-stopper

Must finish before any submission.

```text
1. Zero-variance debugging and model training integrity
2. Dataset scale audit and re-preprocessing if needed
3. E_pre density diagnosis and constrained E_pre variants
```

## P1 — Q3-critical

Strongly required for Q3.

```text
4. Sparse-skill predictive ablation
5. Multi-fold and multi-seed confirmatory evaluation
6. Statistical significance tests
7. Top-k similarity edge-retention table
```

## P2 — Q3-polish

Important for strengthening the manuscript.

```text
8. E_co provenance audit module labeling
9. Stratum-level degree support table
10. Calibration metrics
11. Temporal vs learner-level split comparison
12. Noise robustness rerun after model debugging
13. DDR on constrained E_pre variants
```

## P3 — Reproducibility

Required for reviewer trust.

```text
14. Reproducibility manifest
15. Final diagnostic report
16. LaTeX table and figure export
17. Manuscript-ready revised paragraphs
```

---

# 5. Final acceptance checklist

Antigravity should mark the project as Q3-ready only if all of the following are true:

```text
[ ] No neural model has degenerate constant predictions.
[ ] Dataset scale is either benchmark-level or explicitly justified.
[ ] E_pre density artifact is explained and constrained variants are tested.
[ ] Top-k similarity produces non-empty E_sim and reports edge counts.
[ ] Leakage audit L1, L2, L3, L5, L6 pass.
[ ] L4 is PASS or clearly reported as a cold-start limitation.
[ ] E_co provenance audit includes four labeled checks.
[ ] Sparse-skill predictive AUC is reported by stratum.
[ ] Multi-fold and multi-seed results are reported.
[ ] Statistical tests include CI and p-values.
[ ] Calibration metrics are included.
[ ] Temporal split is compared with learner-level split.
[ ] Noise robustness is re-run after model debugging.
[ ] DDR is computed for constrained E_pre variants.
[ ] All tables are exported as CSV and LaTeX.
[ ] All figures are exported as PDF.
[ ] A final Q3 diagnostic report is generated.
[ ] Manuscript-ready revised text is generated.
```

---

# 6. Expected final folder structure

```text
lc_mrsg_q3_upgrade/
  configs/
    q3_upgrade.yaml

  src/
    preprocess/
    graphs/
    audit/
    train/
    evaluate/
    stats/
    reports/

  data/
    raw/
    processed/

  graphs/
    assist2012/
    junyi/
    kdd2010/

  results/
    tables/
    figures/
    reports/

  paper/
    tables_q3/
    figures_q3/
    text_q3/

  supplementary/
    all_runs_q3.csv
    per_seed_predictions_q3.csv
    leakage_audit_q3.csv
    graph_exports_manifest_q3.csv

  REPRODUCIBILITY.md
```

---

# 7. Final note for implementation

This upgrade package should not be optimized to make LC-MRSG look better than all baselines. It should be optimized to make the study scientifically defensible.

The final paper should be able to say:

> “LC-MRSG provides a reproducible and leakage-controlled way to construct and audit multi-relational skill graphs. Its benefit is not that every graph variant always improves AUC, but that graph-enhanced KT experiments become more transparent, testable, and less vulnerable to hidden structural leakage. Where graph variants improve sparse-skill prediction, the claim is supported by stratified AUC, multi-fold evaluation, and statistical tests; where they do not, the paper reports the limitation explicitly.”
