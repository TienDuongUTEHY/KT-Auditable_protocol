# ANTIGRAVITY PROMPT — PACKAGE A: COMPLETING MANDATORY EXPERIMENTS FOR LC-MRSG/EJEL

## 0. Role and Objectives

You are a programming agent / experimental researcher working in the repository of the paper:

**LC-MRSG: An Auditable Protocol for Relation-Aware Skill-Graph Construction in e-Learning Knowledge Tracing**

The goal of this task is to **complete the mandatory experiments before revising the EJEL manuscript**, including:

1. Running additional **KDD2010 neural early-stopping** with density-controlled `Eco`.
2. Completing **Junyi with 3 folds × 3 seeds** for DKT and simpleKT under the same early-stopping protocol.
3. Re-exporting results tables/files for the manuscript revision:
   - `selected-config`
   - `best-graph-vs-no-graph`
   - `training-budget / two-epoch vs early-stopping` if historical two-epoch data exists
   - `practical threshold |ΔAUC| >= 0.005`
   - `Holm-adjusted p-values`
4. Do not modify the manuscript text at this stage, except for generating the output `.tex`/`.csv` tables for the authors to insert later.

Important requirement: **do not use the test set to select graph configurations**. All candidate selections must be based on validation AUC; the test set must only be evaluated once to report the final performance of the selected configuration.

---

## 1. Background of Issues to Address

The current manuscript positions LC-MRSG as an **audit/governance protocol** rather than a SOTA architecture. However, a major issue remains:

- KDD2010 is the only dataset that has all three active relations: `Epre + Esim + Eco`.
- The relation-availability/density table reported that KDD2010 has all three relations, but the main neural early-stopping results currently only cover ASSIST2012 and Junyi.
- Therefore, the claim of being `relation-aware` or `tri-relational` has not yet been verified by neural early-stopping on the dataset that actually contains all three relations.

This task must generate clean experimental evidence to address this point.

---

## 2. Invariant Principles of the Experiments

### 2.1. Split-first and Leakage Prevention

Every pipeline must comply with:

```text
D_train, D_valid, D_test are split beforehand.
Graph candidates must only be built from D_train.
Candidate graphs must be frozen before validation evaluation.
Validation is only used to select candidates.
Test is only used for the final report.
```

Do not:

- build `Eco`, `Esim`, edge weights, or support counts from validation/test data;
- tune thresholds using test AUC;
- select candidates based on test AUC;
- combine predictions of multiple candidates and select based on test set;
- use processed files where the graph was generated from full data without an audit proving train-only construction.

### 2.2. Early Stopping

Use the same early-stopping protocol applied to ASSIST2012/Junyi in the previous revision. If the repo already has configurations, prioritize reusing them to ensure consistency.

If the configuration is unclear, use the following minimum defaults and record them in `run_manifest.csv`:

```yaml
early_stopping:
  monitor: valid_auc
  mode: max
  patience: 10
  min_delta: 0.0001
  restore_best_checkpoint: true
  max_epochs: 200
```

Each run must save:

- best epoch;
- best validation AUC;
- test AUC at the best validation checkpoint;
- epoch-level training log;
- checkpoint path or checkpoint hash;
- seed, fold, dataset, backbone, candidate.

### 2.3. Seeds and Folds

Use consistently:

```text
folds = [0, 1, 2]
seeds = [42, 2024, 2025]
backbones = [DKT, simpleKT]
```

Each dataset/backbone/candidate combination requires 3 × 3 = 9 runs, unless valid results already exist. Existing results must be verified for schema and log consistency before reuse.

---

## 3. Mandatory Experiment Matrix

### 3.1. Junyi — Completing Folds and Seeds

Currently, we need to verify Junyi because the previous log suggests fewer than 3 folds × 3 seeds were completed.

### Tasks to Perform

1. Scan all existing result directories to identify completed Junyi runs.
2. Generate `missing_runs_report.csv` listing all missing runs using the keys:

```text
dataset, backbone, fold, seed, candidate, status, reason
```

3. Run all missing runs for:

```text
dataset = Junyi
backbones = DKT, simpleKT
folds = 0, 1, 2
seeds = 42, 2024, 2025
candidate set = same candidate set used in current ASSIST2012/Junyi experiments
```

### Candidate Set for Junyi

Use the candidate set currently available in the repo. If standardization is required, use at least:

```text
no_graph
Epre
Eco_only
Epre_plus_Eco
full_LC_MRSG (if Esim exists; if Esim = 0, full must be clearly labeled as effective bi-relational)
relation_gated_1
relation_gated_2
sparse_aware_relation_gated (if available in repo)
validation_selected_static (if available in repo)
```

Since Junyi has `Esim = 0` in the processed single-skill setting, all tables must clearly report the relation availability to avoid implying Junyi is tri-relational.

---

### 3.2. KDD2010 — Neural Early-Stopping with Density-Controlled Eco

KDD2010 is the core dataset of this package because it has all three relationships: `Epre + Esim + Eco`.

### 3.2.1. Backbones

Run:

```text
DKT
simpleKT
```

### 3.2.2. Folds/seeds

Run:

```text
folds = [0, 1, 2]
seeds = [42, 2024, 2025]
```

### 3.2.3. Candidate Set for KDD2010

At a minimum, include:

```text
no_graph
Epre
Epre_plus_Esim
Eco_controlled_primary
full_LC_MRSG_controlled = Epre + Esim + Eco_controlled_primary
relation_gated_controlled (if supported by repo)
```

Do not use the default dense Eco as the sole primary candidate. You can keep `Eco_default` as a diagnostic reference, but the main comparisons must use the density-controlled Eco.

### 3.2.4. Eco-Controlled Configurations

Create or verify the following `Eco` configurations. Select at least one primary configuration before running the main experiments. Do not select the primary configuration using test AUC.

It is recommended to evaluate 3 configurations for sensitivity checks:

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

If compute is limited, select `eco_c2_balanced` as the primary configuration because it balances density and skill coverage. Explicitly document this choice in `run_manifest.csv` and `kdd2010_eco_density_audit.csv`.

### 3.2.5. Do Not Select Thresholds Using Test Data

Rules for selecting the Eco primary configuration:

- It must be selected beforehand based on the density/coverage audit of the training graph.
- Or selected based on validation AUC.
- Under no circumstances select it using test AUC.

---

## 4. Standardizing Density Definitions

Create `kdd2010_eco_density_audit.csv` and document the formulas clearly.

### 4.1. For Undirected Eco and Esim

```text
density_undirected = unique_undirected_edges / (n_skill_train * (n_skill_train - 1) / 2)
```

Where:

```text
n_skill_train = number of skills present in the training fold after preprocessing and mapping.
unique_undirected_edges = number of unique undirected edges after removing duplicates.
```

### 4.2. For Directed Epre

```text
density_directed = unique_directed_edges / (n_skill_train * (n_skill_train - 1))
```

Where:

```text
unique_directed_edges = number of unique directed edges after removing self-loops.
```

### 4.3. Mandatory Columns in the Density Audit

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

### 4.4. Resolving the 0.807 vs 0.723 Discrepancy

After running the audit, create `density_consistency_report.md` explaining:

- which values represent means across folds;
- which values are fold-specific or config-specific;
- whether the density denominator uses `n_skill_train` or `n_skill_full`;
- why the numbers in previous versions of the main text or supplementary material differed;
- the final recommended numbers to be used in the manuscript.

---

## 5. Standardizing Results Schema

All prediction files must contain at least the following schema:

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

If timestamps or sequence indices exist, retain them:

```text
timestamp
sequence_index
```

All run-level results must use the schema:

```text
dataset
fold
seed
backbone
candidate
selected_by_validation
valid_auc
test_auc
test_brier (if available)
test_ece (if available)
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

## 6. Validation-Selected Configuration

Create `selected_config_early_stopping.csv`.

### 6.1. Selection Rule

For each:

```text
dataset, backbone, fold, seed
```

select the candidate with the highest `valid_auc`.

If there is a tie within a very small threshold, apply the predefined tie-breaker:

```text
1. no_graph if the difference in valid_auc is <= 0.0001 compared to the best candidate
2. choose the simpler candidate over the more complex one
3. choose Epre over full graph if AUC is identical
4. explicitly record tie_break_applied = true
```

Rationale: Avoid forcing the use of graphs when validation evidence is insufficient.

### 6.2. Mandatory Columns

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

### 6.3. Mandatory Interpretation

If `selected_candidate == no_graph`, then:

```text
delta_selected_vs_no_graph = 0 by definition
is_tautological_delta = true
```

Do not interpret this row as direct evidence that graphs do not help. The correct interpretation is:

```text
validation-only selection automatically rejected the graph; this is a governance output of the protocol.
```

---

## 7. Best-Available-Graph vs No-Graph

Create `best_graph_vs_no_graph_early_stopping.csv` to provide a non-trivial comparison.

### 7.1. Best Graph Selection Rule

For each:

```text
dataset, backbone, fold, seed
```

select the candidate graph with the highest `valid_auc` among all candidates except `no_graph`.

```text
graph_candidates = all candidates where candidate != no_graph
best_graph = argmax(valid_auc among graph_candidates)
```

Then report the test AUC of `best_graph` and compare it against `no_graph`.

### 7.2. Mandatory Columns

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

### 7.3. Rationale

This table answers the question:

```text
If forced to select a graph candidate using validation data, does the best graph add value compared to no_graph?
```

This table is separate from the selected-config table, where validation might select `no_graph`.

---

## 8. Practical Threshold and Classification

The practical threshold is fixed at:

```text
practical_threshold_auc = 0.005
```

Create a `classification` column based on the following rules:

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

Do not refer to `selected == no_graph` rows as `near-zero graph effect`; they must be termed `selection-no-graph outcome`.

---

## 9. Holm-Adjusted P-Values

Create `neural_summary_practical_holm.csv`.

### 9.1. Main Comparisons

The primary comparisons should include:

```text
selected_config_vs_no_graph
best_available_graph_vs_no_graph
```

At a minimum, report for:

```text
ASSIST2012 × DKT
ASSIST2012 × simpleKT
Junyi × DKT
Junyi × simpleKT
KDD2010 × DKT
KDD2010 × simpleKT
```

If the repo lacks sufficient ASSIST2012 data, do not rerun unless missing, but standardize the table using the same script.

### 9.2. Raw P-Values

Prioritize using paired bootstrap or paired tests on run-level deltas.

Recommended steps:

1. Calculate `delta_auc` for each run `(fold, seed)`.
2. For each `(dataset, backbone, comparison_type)`, obtain a vector of 9 deltas.
3. Calculate the mean delta and 95% CI using bootstrapping over the 9 run-level deltas with a fixed seed.
4. Compute two-sided p-values for testing `delta = 0` using paired permutation or Wilcoxon signed-rank tests.
5. Apply Holm correction across the family of neural comparisons.

If prediction-level paired bootstrap or DeLong tests are already in the repo, they can be reused but the method must be documented in `stats_method_report.md`.

### 9.3. Manual Holm Correction

If `statsmodels` is not available, implement Holm correction manually as follows:

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

### 9.4. Mandatory Columns in the Summary

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

## 10. Training-Budget Comparison (Two-Epoch vs Early-Stopping)

If historical two-epoch results exist in the repo, create the table:

```text
neural_summary_two_epoch_vs_early_stopping.csv
```

Mandatory columns:

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

`stability_label` rules:

```text
if abs(early_stopping_mean_delta) < 0.005:
    stability_label = "near-zero under early stopping"
elif sign(two_epoch_mean_delta) != sign(early_stopping_mean_delta):
    stability_label = "sign changed"
else:
    stability_label = "directionally stable"
```

If no two-epoch data exists, create `two_epoch_missing_report.md` instead of fabricating data.

---

## 11. Mandatory Outputs

Create directory:

```text
results/ejel_gA_experiments/
```

It must contain:

```text
results/ejel_gA_experiments/run_manifest.csv
results/ejel_gA_experiments/missing_runs_report.csv
results/ejel_gA_experiments/kdd2010_eco_density_audit.csv
results/ejel_gA_experiments/density_consistency_report.md
results/ejel_gA_experiments/selected_config_early_stopping.csv
results/ejel_gA_experiments/best_graph_vs_no_graph_early_stopping.csv
results/ejel_gA_experiments/neural_summary_practical_holm.csv
results/ejel_gA_experiments/neural_summary_two_epoch_vs_early_stopping.csv OR two_epoch_missing_report.md
results/ejel_gA_experiments/stats_method_report.md
results/ejel_gA_experiments/reproducibility_manifest.md
```

Also generate LaTeX tables if the repo uses LaTeX:

```text
tables/table_selected_config_early_stopping.tex
tables/table_best_graph_vs_no_graph_early_stopping.tex
tables/table_neural_summary_practical_holm.tex
tables/table_kdd2010_eco_density_audit.tex
```

Do not automatically insert them into the manuscript at this stage.

---

## 12. Mandatory Quality Gate Checks

Create file:

```text
results/ejel_gA_experiments/quality_gate_report.md
```

It must address each of the following points:

### 12.1. Completeness

```text
[ ] Is Junyi complete with 3 folds × 3 seeds for DKT?
[ ] Is Junyi complete with 3 folds × 3 seeds for simpleKT?
[ ] Is KDD2010 complete with 3 folds × 3 seeds for DKT?
[ ] Is KDD2010 complete with 3 folds × 3 seeds for simpleKT?
[ ] Does each run have a prediction file?
[ ] Does each run have valid_auc/test_auc?
```

### 12.2. Leakage Control

```text
[ ] Was the graph built from train data only?
[ ] Were validation/test sets used to generate edges/support/weights?
[ ] Was the candidate graph frozen before validation?
[ ] Was the test set used to select candidates or thresholds?
[ ] Does the repo/log contain configuration hashes or metadata?
```

### 12.3. Density Consistency

```text
[ ] Is the density formula consistent?
[ ] Are KDD2010 default Eco, fold-specific, and mean-across-folds clearly distinguished?
[ ] Has the previous 0.807 vs 0.723 discrepancy been explained or corrected?
[ ] Are n_skill_train and n_skill_full documented?
```

### 12.4. Interpretation Readiness

```text
[ ] Are selected == no_graph rows marked as tautological?
[ ] Is there a best-available-graph vs no-graph table?
[ ] Is the practical threshold |ΔAUC| >= 0.005 applied?
[ ] Are Holm p-values computed?
[ ] Are classifications correct?
```

---

## 13. Strictly Forbidden Actions

Do not:

1. Edit the main manuscript text until requested.
2. Select Eco thresholds using test set AUC.
3. Delete older results without backing them up.
4. Group `selected == no_graph` rows into the `near-zero graph effect` category.
5. Refer to Junyi/ASSIST2012 as tri-relational if `Esim = 0`.
6. Refer to KDD2010 as tri-relational evidence if neural early-stopping has not been run.
7. Fabricate or guess table entries.
8. Skip failed runs; they must be logged in `run_manifest.csv` and `missing_runs_report.csv`.

---

## 14. Recommended Workflow

Execute in order:

### Step 1 — Inspect Repository

- Identify the directory structure.
- Locate existing training/evaluation scripts.
- Find existing configurations for datasets, models, and candidates.
- Locate older results folders.
- Do not modify code until the pipeline is understood.

### Step 2 — Backup

Create a backup of older results:

```text
results_backup_before_ejel_gA_<timestamp>/
```

### Step 3 — Build Missing-Runs Report

Create `missing_runs_report.csv` for Junyi and KDD2010.

### Step 4 — Implement or Verify Eco-Controlled Graph Builder

- Check if the current graph builder supports `k_min`, `pmi_min`, and `top_k`.
- If not, add these parameters without breaking backward compatibility.
- Perform a density audit for each fold before training.

### Step 5 — Run Missing Junyi Experiments

- Run missing folds, seeds, and candidates.
- Save predictions and logs.

### Step 6 — Run KDD2010 Experiments

- Run DKT/simpleKT with the density-controlled Eco candidates.
- Prioritize running `no_graph`, `Epre`, `Epre_plus_Esim`, `Eco_controlled_primary`, and `full_LC_MRSG_controlled` first.
- Run gated models afterward if compute budget allows.

### Step 7 — Aggregate Results

- Create the run manifest.
- Generate the selected-config table.
- Generate the best-graph-vs-no-graph table.
- Calculate CIs, p-values, and Holm p-values.
- Classify results based on the practical threshold.

### Step 8 — Quality Gate

- Generate `quality_gate_report.md`.
- Document any incomplete runs, explaining why, and mark corresponding tables as supplementary diagnostics only.

---

## 15. Recommended Script Structure

If the repo lacks compilation scripts, create them under:

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

Provide clear CLIs for these scripts, for example:

```bash
python scripts/ejel_gA/01_scan_runs.py --results_dir results --out results/ejel_gA_experiments/missing_runs_report.csv
python scripts/ejel_gA/02_build_eco_density_audit.py --dataset kdd2010 --folds 0 1 2 --out results/ejel_gA_experiments/kdd2010_eco_density_audit.csv
python scripts/ejel_gA/05_selected_config.py --manifest results/ejel_gA_experiments/run_manifest.csv --out results/ejel_gA_experiments/selected_config_early_stopping.csv
python scripts/ejel_gA/06_best_graph_vs_no_graph.py --manifest results/ejel_gA_experiments/run_manifest.csv --out results/ejel_gA_experiments/best_graph_vs_no_graph_early_stopping.csv
python scripts/ejel_gA/07_stats_holm.py --selected results/ejel_gA_experiments/selected_config_early_stopping.csv --best_graph results/ejel_gA_experiments/best_graph_vs_no_graph_early_stopping.csv --out results/ejel_gA_experiments/neural_summary_practical_holm.csv
python scripts/ejel_gA/09_quality_gate_report.py --base results/ejel_gA_experiments --out results/ejel_gA_experiments/quality_gate_report.md
```

---

## 16. Final Console Summary

When complete, output a summary in the following format:

```text
EJEL Package A completed.

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

## 17. Final Definition of Done

The task is only considered complete when:

1. KDD2010 has neural early-stopping evaluations for DKT and simpleKT, across all folds/seeds or with clear missing reports.
2. Junyi is complete across all folds/seeds or with clear missing reports.
3. The selected-config table clearly distinguishes the `selected == no_graph` cases.
4. The best-available-graph vs no-graph table is generated.
5. The practical threshold `|ΔAUC| >= 0.005` is applied.
6. Holm-adjusted p-values are computed.
7. A density audit is provided, explaining the formula and the mean/fold/config differences.
8. A quality gate report confirms leakage control.
9. No data is fabricated, and the test set was not used for configurations.
