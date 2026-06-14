# Antigravity Task Plan — LC-MRSG Q3 Experimental Fix

## Purpose

Use this task plan to let Google Antigravity repair and complete the remaining experiments for the LC-MRSG paper. The plan is based on the supervisor review of LC-MRSG v2/v3.

The review identifies six practical priorities:

1. **P0 — Zero-variance/model-integrity problem**: GIKT/GKT/simpleKT previously showed identical AUC and `std=0.0000`, suggesting possible constant predictions, metric bugs, or graph bypass.
2. **P1 — Dataset-scale problem**: KDD2010 and Junyi processed subsets were previously too small; full-scale or clearly documented benchmark-scale preprocessing is required.
3. **P2 — E_pre density artifact**: `E_pre` density was exactly `0.5000` across datasets, suggesting an artificial construction rule.
4. **P3 — Sparse-skill predictive evidence**: sparse-skill coverage is not enough; the paper needs stratified AUC/ACC/NLL/RMSE by training-fold skill frequency.
5. **P4 — E_co provenance module clarity**: the four checks must be reported explicitly: symmetry, train-only support, weight distribution, sparse-KC coverage.
6. **P5 — Consistency checks**: numerical inconsistencies across Table 3, zero-variance table, sparse AUC table, and noise curves must be detected before manuscript update.

---

## Required execution command

```bash
pip install -r requirements.txt
bash run_all_q3_fix.sh configs/q3_fix_config.yaml
```

---

## Required outputs to update the paper

### 1. Main confirmatory performance

Use:

```text
results/q3_fix/tables/multifold_confirmatory_results.tex
results/q3_fix/tables/paired_tests_no_vs_full.tex
```

This replaces older performance tables. The final experiment should be described as:

```text
3 datasets × 3 folds × 5 seeds × 5 models × 4 graph variants = 900 runs
```

### 2. Zero-variance diagnosis

Use:

```text
results/q3_fix/tables/zero_variance_diagnosis_summary.tex
results/q3_fix/tables/zero_variance_diagnosis_full.csv
```

Acceptance criterion:

```text
prediction_std > 1e-4
train_loss decreases where logs are available
no neural model has identical AUC with std=0.0000 across all seeds/folds unless deterministic behavior is explicitly justified
```

### 3. Dataset scale audit

Use:

```text
results/q3_fix/tables/dataset_scale_audit.tex
results/q3_fix/reports/dataset_scale_audit.md
```

Target scale:

```text
KDD2010: >= 500 learners, >= 500K interactions, >= 400 skills
ASSIST2012: >= 40K learners or justified processed subset
Junyi: >= 40 skills or explicitly framed as compact subset
```

If a dataset is below target, the manuscript must say “processed subset” and list preprocessing filters.

### 4. E_pre pruning and density correction

Use:

```text
results/q3_fix/tables/e_pre_pruning_summary.tex
```

Acceptance target:

```text
ASSIST2012 and KDD2010: E_pre density <= 0.05
Junyi: E_pre density <= 0.20 due to compact skill vocabulary
```

If pruning lowers AUC, report it honestly as a trade-off between structural plausibility and predictive behavior.

### 5. E_co provenance audit

Use:

```text
results/q3_fix/tables/eco_provenance_audit.tex
```

The manuscript must explicitly name the four checks:

```text
Check 1: Symmetry
Check 2: Train-only support
Check 3: Weight distribution
Check 4: Sparse-KC coverage
```

### 6. Sparse-skill stratified AUC

Use:

```text
results/q3_fix/tables/sparse_skill_summary_mean_std.tex
```

Strata must be computed from **training-fold interactions only**:

```text
Very Sparse: n_train(c) <= 50
Sparse: 50 < n_train(c) <= 100
Medium: 100 < n_train(c) <= 500
Frequent: n_train(c) > 500
```

The title “Sparse-Skill Knowledge Tracing” is well-justified only if the paper reports either predictive gains or a careful limitation statement for sparse strata.

### 7. Consistency checks before manuscript update

Use:

```text
results/q3_fix/tables/paper_consistency_checks.tex
results/q3_fix/tables/consistency_main_vs_zero_bad_rows.csv
results/q3_fix/tables/consistency_sparse_vs_overall_flags.csv
```

The manuscript should not be updated until all WARN rows have been inspected.

---

## Manuscript edits after running scripts

Add or replace the following sections:

### Experimental Setup

Add:

```latex
All main experiments are repeated over three learner-level folds and five random seeds, giving fifteen fold-seed runs per model--dataset--graph configuration. With three datasets, five backbones, and four graph variants, the final confirmatory evaluation contains 900 runs.
```

### E_co provenance subsection

Add:

```latex
The $E_{co}$ provenance module contains four checks. First, symmetry requires every retained co-occurrence edge to have its reverse edge with the same weight. Second, train-only support requires the support steps for each edge to be contained in the training fold. Third, the weight-distribution check reports the mean, standard deviation, median and maximum of the PMI-based edge weights. Fourth, sparse-KC coverage measures the proportion of training-fold sparse skills that receive at least one $E_{co}$ neighbor.
```

### E_pre pruning subsection

Add:

```latex
The purpose of prerequisite pruning is not to maximize AUC at any cost. It is a structural correction step: a very dense prerequisite graph may appear predictive but is difficult to interpret pedagogically. We therefore report both graph density and predictive behavior, treating pruning as a trade-off between graph sparsity and downstream KT performance.
```

### Sparse-skill result subsection

Add:

```latex
Sparse-skill groups are defined from training-fold interactions only. This prevents frequency information from validation or test interactions from influencing the sparse-skill evaluation. The stratified results therefore measure how models behave on skills with limited training evidence, rather than on skills identified using full-data statistics.
```

### Discussion

Add:

```latex
The corrected results should be interpreted as conditional graph utility. LC-MRSG helps in some model--dataset pairs, remains nearly neutral in others, and may slightly reduce AUC in a few settings. This pattern is useful for educational AI applications because it discourages a simplistic assumption that adding more graph edges always improves learning prediction.
```

---

## Final acceptance checklist

Antigravity should mark the experiment package complete only if:

```text
[ ] No model has unexplained constant predictions.
[ ] Dataset sizes are benchmark-level or explicitly justified as processed subsets.
[ ] E_pre density is no longer an unexplained 0.5000 artifact.
[ ] Top-k or constrained E_pre variants are exported.
[ ] E_co four-check provenance table is exported.
[ ] Sparse-skill AUC/ACC/NLL/RMSE by stratum is exported.
[ ] Main and supplementary tables pass consistency checks.
[ ] The paper text uses 900 runs and 15 fold-seed runs/config consistently.
```
