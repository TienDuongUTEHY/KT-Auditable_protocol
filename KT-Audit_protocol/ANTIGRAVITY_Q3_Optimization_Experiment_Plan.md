# ANTIGRAVITY MISSION FILE
# Q3-Safe LC-MRSG++ Optimization Experiment Plan

**Project:** Leakage-Controlled Multi-Relational Skill Graph Construction for Sparse-Skill Knowledge Tracing  
**Target manuscript quality:** Q3 Scopus safety level, approximately 8.5--8.8/10 after revision  
**Main instruction:** Improve the paper's methodological strength and empirical defensibility without test leakage, p-hacking, cherry-picking, or overstating results.

---

## 0. Executive Goal

The current paper has a strong methodological contribution but weak and selective predictive gains. The latest manuscript reports that full LC-MRSG improves AUC in only 6 of 15 model-dataset comparisons and only 2 positive effects are statistically significant. Therefore, the next experiment must **not** try to force the claim that the full graph is always better.

Instead, rebuild the contribution around a stronger and more defensible method:

> **LC-MRSG++: a leakage-controlled, validation-selected, relation-gated graph construction and audit protocol for Knowledge Tracing.**

The new experiment should answer this sharper question:

> Can a leakage-controlled graph protocol decide, using only training/validation evidence, when prerequisite, similarity, and co-occurrence relations should be used, down-weighted, pruned, or disabled?

The target is not merely to increase AUC. The target is to make the paper safer for Q3 by improving:

1. Algorithmic clarity.
2. Validation-only graph selection.
3. Sparse-skill diagnostics.
4. Provenance completeness.
5. Statistical reporting.
6. Reproducibility.
7. Conservative and honest claims.

---

## 1. Non-Negotiable Research Integrity Rules

The agent must obey all rules below.

### 1.1 No test leakage

Do not use validation or test records to build any graph relation.  
Do not use test AUC/ACC/NLL/RMSE to choose graph variants, thresholds, relation weights, seeds, epochs, or models.  
Do not tune hyperparameters on test results.  
Do not use test labels in sparse-stratum assignment unless the stratum definition is computed from train-only interaction frequencies.

### 1.2 Validation-only selection

All graph selection must be performed by validation metrics only.  
The test set must be used once per locked configuration.

### 1.3 No cherry-picking

Report all datasets: ASSIST2012, Junyi, KDD2010.  
Report all five backbones: BKT, DKT, simpleKT, GIKT, SKT.  
Report all folds and seeds.  
Report negative, neutral, and positive cases.

### 1.4 Do not overwrite previous evidence

Create a new output directory:

```bash
runs/q3_lcmrsg_plus_YYYYMMDD_HHMMSS/
```

Never overwrite the existing 900-run result package.

### 1.5 Safety in Antigravity

Before running commands, create a git branch and backup manifest.

```bash
git status
git checkout -b q3-lcmrsg-plus-final
mkdir -p backups/q3_lcmrsg_plus
python - <<'PY'
import os, hashlib, json, time
manifest=[]
for root, dirs, files in os.walk('.'):
    if root.startswith('./.git') or root.startswith('./runs'):
        continue
    for f in files:
        path=os.path.join(root,f)
        try:
            h=hashlib.sha256(open(path,'rb').read()).hexdigest()
            manifest.append({'path':path,'sha256':h})
        except Exception:
            pass
json.dump({'created':time.ctime(),'files':manifest}, open('backups/q3_lcmrsg_plus/pre_run_manifest.json','w'), indent=2)
PY
```

Do not run destructive commands such as `rm -rf`, disk cleanup scripts, or cache deletion without explicit human approval.

---

## 2. Current Weaknesses to Solve

The new mission must address these weaknesses.

| Weakness | Current problem | New solution |
|---|---|---|
| Weak predictive gain | Full graph improves only a minority of model-dataset pairs | Replace fixed full graph claim with validation-selected LC-MRSG++ |
| Full graph sometimes hurts | Eco may introduce noisy edges | Add relation gating, threshold search, and fallback to simpler graph |
| Method section too textual | Current protocol has equations but limited executable algorithm | Add Algorithm 1, Algorithm 2, Algorithm 3 |
| Statistical reporting incomplete | One-tailed p-values alone are not enough | Add 95% CI, two-tailed p, Holm correction, effect size |
| Provenance FLAG | Eco table stores one-direction edges and missing support metadata | Export mirrored edges and train-only support summaries |
| Sparse-skill claim is fragile | Junyi has no very-sparse/sparse test strata | Use sparse-aware diagnostics, not overstated sparse claims |
| Reviewer may ask why full graph | Full graph is not always best | Show LC-MRSG++ learns when to disable noisy relation channels |

---

## 3. New Method Name and Claim

Use the following method name in code and paper:

```text
LC-MRSG++
```

Full expansion:

```text
Leakage-Controlled Multi-Relational Skill Graph Construction with Validation-Guided Relation Selection
```

Recommended paper title after the additional experiment:

```text
A Leakage-Controlled and Validation-Guided Protocol for Multi-Relational Skill Graph Construction in Knowledge Tracing
```

Alternative title if the sparse analysis remains central:

```text
Leakage-Controlled Multi-Relational Skill Graph Construction with Sparse-Skill Diagnostics for Knowledge Tracing
```

Do not title the paper as if it proposes a universally better KT architecture.

---

## 4. Experimental Design Overview

### 4.1 Existing static graph variants

Keep these four static variants as baselines:

```text
no_graph
E_pre
E_pre_E_sim
full_lc_mrsg = E_pre + E_sim + E_co
```

### 4.2 New LC-MRSG++ variants

Add three new variants.

#### Variant A: `val_selected_static`

For each dataset, fold, seed, and model, select one of the four existing graph variants using validation AUC. Then evaluate the selected variant on the locked test set.

Purpose: demonstrate that a leakage-controlled graph protocol should not blindly use full graph.

#### Variant B: `relation_gated`

Use train-only graph construction, but apply validation-selected relation weights:

```text
G = alpha_pre * E_pre + alpha_sim * E_sim + alpha_co * E_co
```

Search alpha values on validation only:

```text
alpha_pre in {0, 0.25, 0.50, 0.75, 1.00}
alpha_sim in {0, 0.25, 0.50, 0.75, 1.00}
alpha_co  in {0, 0.10, 0.25, 0.50, 1.00}
```

Important: `alpha_co` has a smaller grid because co-occurrence edges are the most likely to be noisy.

#### Variant C: `sparse_aware_relation_gated`

Add sparse-aware edge scaling for skills with low train frequency:

```text
w_ij_final = w_ij * gamma_i * gamma_j
```

where:

```text
gamma_i = 1 + beta * I(skill_i is sparse_or_very_sparse)
```

Search beta on validation only:

```text
beta in {0.00, 0.10, 0.25, 0.50}
```

This is designed to strengthen the sparse-skill part of the paper without inventing new data.

---

## 5. Main Research Questions for the Final Revision

The new experiment should support these research questions.

### RQ1: Leakage control

Can LC-MRSG++ construct fold-specific multi-relational skill graphs using only train-available evidence while passing L1--L6 leakage and provenance checks?

### RQ2: Validation-guided relation selection

Does validation-guided graph selection reduce the negative effects observed when the full graph is always used?

### RQ3: Sparse-skill behavior

Does sparse-aware relation gating improve or stabilize AUC on sparse and very-sparse skill strata without harming frequent skills?

### RQ4: Graph relation usefulness

Which relations are actually useful by dataset and model: prerequisite, similarity, co-occurrence, or no graph?

### RQ5: Reproducibility

Can the full artifact be reproduced with graph checksums, split hashes, support summaries, and statistical reports?

---

## 6. Required Implementation Tasks

Create or update the following files.

```text
scripts/q3_lcmrsg_plus_build_graphs.py
scripts/q3_lcmrsg_plus_run_experiments.py
scripts/q3_lcmrsg_plus_analyze.py
scripts/q3_lcmrsg_plus_render_tables.py
scripts/q3_lcmrsg_plus_export_appendix.py
scripts/q3_lcmrsg_plus_sanity_checks.py
tests/test_q3_lcmrsg_plus_leakage.py
tests/test_q3_lcmrsg_plus_statistics.py
configs/q3_lcmrsg_plus.yaml
README_Q3_LCMRSG_PLUS.md
```

If the repository already has equivalent scripts, modify the existing scripts instead of duplicating code unnecessarily. Preserve backward compatibility.

---

## 7. Algorithm 1: Split-First LC-MRSG++ Graph Construction and Audit

Insert this algorithm into the manuscript after the existing mathematical protocol.

```text
Algorithm 1: Split-First LC-MRSG++ Graph Construction and Audit

Input:
  Interaction log D
  Q-matrix Q
  Fold index f
  Train/validation/test split: D_train^f, D_valid^f, D_test^f
  Relation thresholds theta_pre, theta_sim, theta_co
  Top-k pruning parameter k

Output:
  Fold-specific graph G_f
  Audit report A_f
  Provenance report P_f

1. Freeze the fold split before graph construction.
2. Compute split hashes H_train, H_valid, H_test.
3. Build prerequisite candidate edges E_pre_raw using D_train^f and Q only.
4. Apply top-k outgoing pruning to E_pre_raw.
5. Apply transitive reduction to produce E_pre while preserving DAG status.
6. Build similarity edges E_sim using train-available item/skill features only.
7. Build co-occurrence edges E_co using train-only learner sequences.
8. Store support summaries for every edge:
      edge_id, relation_type, source_skill, target_skill,
      train_support_count, train_support_hash,
      weight, fold, dataset
9. If E_co is intended to be undirected, mirror edges explicitly:
      (i, j) and (j, i) must both be stored or marked by an undirected flag.
10. Run L1--L6 leakage checks:
      L1 edge-support split check
      L2 Q-matrix availability check
      L3 temporal-order check
      L4 cold-start boundary check
      L5 co-occurrence support check
      L6 validation-selection/test-isolation check
11. Export G_f, A_f, P_f, and graph checksums.
12. Return G_f, A_f, P_f.
```

---

## 8. Algorithm 2: Validation-Guided Relation Selection

This algorithm is the key improvement over the current fixed full-graph claim.

```text
Algorithm 2: Validation-Guided Relation Selection

Input:
  Candidate graph variants V = {no_graph, E_pre, E_pre_E_sim, full_lc_mrsg}
  Dataset d
  Fold f
  Seed s
  Model m
  Training split D_train^f
  Validation split D_valid^f

Output:
  Selected graph variant v_star
  Locked test configuration C_star

1. For each v in V:
      a. Train model m using D_train^f and graph v.
      b. Evaluate on D_valid^f.
      c. Store validation AUC, ACC, NLL, RMSE.
2. Select v_star by this ordered criterion:
      a. Highest validation AUC.
      b. If tie within epsilon=0.001, choose lower validation NLL.
      c. If still tied, choose the simpler graph in this order:
         no_graph < E_pre < E_pre_E_sim < full_lc_mrsg.
3. Lock v_star before touching the test set.
4. Evaluate the locked configuration once on D_test^f.
5. Export selected variant, validation metrics, test metrics, and selection reason.
```

Expected reviewer benefit: this directly solves the criticism that full LC-MRSG is not always useful.

---

## 9. Algorithm 3: Sparse-Aware Relation-Gated LC-MRSG++

Use this only if it can be implemented without excessive runtime. It should be considered the main new experimental method if resources allow.

```text
Algorithm 3: Sparse-Aware Relation-Gated LC-MRSG++

Input:
  E_pre, E_sim, E_co from Algorithm 1
  Training skill frequencies F_train
  Validation split D_valid
  Candidate relation gates alpha_pre, alpha_sim, alpha_co
  Candidate sparse boost beta

Output:
  Sparse-aware weighted graph G_plus
  Selected gates alpha_star and beta_star

1. Define skill strata from train-only frequency F_train:
      very_sparse: bottom 10% of skills by train frequency
      sparse: 10%--33%
      medium: 33%--66%
      frequent: top 34%
2. For each candidate tuple (alpha_pre, alpha_sim, alpha_co, beta):
      a. Construct weighted graph:
         w_final(e) = alpha_relation(e) * w(e) * sparse_boost(e)
      b. sparse_boost(e) = 1 + beta if either endpoint is very_sparse or sparse.
      c. Train model on D_train.
      d. Evaluate on D_valid.
3. Select the candidate by:
      a. Highest validation AUC.
      b. If tie within epsilon=0.001, choose lower validation NLL.
      c. If still tied, choose smaller alpha_co and smaller beta.
4. Lock selected gates.
5. Evaluate once on D_test.
6. Export selected gates, sparse-stratum metrics, and graph audit report.
```

---

## 10. L1--L6 Audit Checks

Implement and export this audit table for every dataset/fold/variant.

| Check | Name | Required PASS condition |
|---|---|---|
| L1 | Edge-support split check | Every edge support set is train-only |
| L2 | Q-matrix availability check | Q entries used by graph are available before validation/test |
| L3 | Temporal check | No future interactions used for earlier evaluation |
| L4 | Cold-start boundary check | No held-out-only skill/item creates train graph edge |
| L5 | Co-occurrence support check | Eco support computed only from train sequences |
| L6 | Selection isolation check | No test metric used for graph selection, threshold selection, early stopping, or relation gates |

Export:

```text
outputs/audit/leakage_audit_l1_l6.csv
outputs/audit/graph_provenance_complete.csv
outputs/audit/graph_checksum_manifest.json
outputs/audit/selection_isolation_report.csv
```

---

## 11. Statistical Reporting Upgrade

Create `scripts/q3_lcmrsg_plus_analyze.py` to compute the following.

### 11.1 Paired tests

For each dataset-model pair:

```text
selected_or_gated_vs_no_graph
selected_or_gated_vs_full_lc_mrsg
selected_or_gated_vs_best_static_baseline
```

Report:

```text
n_pairs
mean_auc_baseline
mean_auc_method
delta_auc
95% bootstrap CI for delta_auc
paired t-test p-value, one-tailed
paired t-test p-value, two-tailed
Wilcoxon signed-rank p-value
Cohen's d for paired differences
Holm-corrected p-value across 15 comparisons
significant_005_uncorrected
significant_005_holm
```

### 11.2 Bootstrap confidence interval

Use 10,000 bootstrap resamples over fold-seed pairs.

```python
for b in range(10000):
    sample = random sample with replacement from paired differences
    boot_delta[b] = mean(sample)
ci_low, ci_high = percentile(boot_delta, [2.5, 97.5])
```

### 11.3 Multiple comparison correction

Use Holm correction over 15 dataset-model comparisons. Do not hide uncorrected results; report both.

### 11.4 Practical significance threshold

Use this interpretation:

```text
|delta_auc| < 0.001: negligible
0.001 <= |delta_auc| < 0.003: small
0.003 <= |delta_auc| < 0.007: moderate
|delta_auc| >= 0.007: practically meaningful
```

This will prevent reviewers from criticizing tiny significant effects such as KDD2010 BKT.

---

## 12. Minimal Additional Runtime Strategy

Because the project has already run too many experiments, use a staged strategy.

### Stage A: Re-analysis without retraining

First, inspect whether existing logs contain validation metrics.

Search for files containing:

```text
valid_auc
val_auc
validation_auc
valid_loss
val_loss
```

If validation metrics exist for the existing 900 runs, compute `val_selected_static` without retraining.

Expected output:

```text
outputs/q3_plus/stage_A_val_selected_static_results.csv
outputs/q3_plus/stage_A_val_selected_static_paired_tests.csv
```

### Stage B: Targeted rerun only if validation metrics are missing

If validation metrics are not available, rerun the existing 4 static graph variants but export train/valid/test metrics separately. This rerun is justified because the previous CSV contains only final aggregate metrics and cannot support validation-only graph selection.

Run:

```text
3 datasets × 3 folds × 5 seeds × 5 models × 4 static variants = 900 runs
```

This is a controlled rerun, not an exploratory rerun.

### Stage C: Gated variant only for promising models

Run `relation_gated` and `sparse_aware_relation_gated` only for these priority pairs:

```text
ASSIST2012: simpleKT, GIKT, SKT
Junyi: BKT, DKT, GIKT, SKT
KDD2010: DKT, simpleKT, GIKT, SKT
```

Reason: these pairs are more likely to benefit from graph or sparse-aware behavior than BKT on all datasets.

If compute budget allows, run all five models for consistency. If not, clearly report targeted exploratory analysis as secondary.

---

## 13. Configuration File: `configs/q3_lcmrsg_plus.yaml`

Create this configuration file.

```yaml
project_name: q3_lcmrsg_plus
output_root: runs/q3_lcmrsg_plus
random_seeds: [42, 2024, 3407, 7, 123]
datasets:
  - assist2012
  - junyi
  - kdd2010
folds: [0, 1, 2]
models:
  - bkt
  - dkt
  - simplekt
  - gikt
  - skt
static_graph_variants:
  - no_graph
  - e_pre
  - e_pre_e_sim
  - full_lc_mrsg
new_variants:
  - val_selected_static
  - relation_gated
  - sparse_aware_relation_gated
selection:
  primary_metric: valid_auc
  tie_epsilon_auc: 0.001
  tie_breaker_1: valid_nll
  tie_breaker_2: graph_simplicity
  graph_simplicity_order:
    - no_graph
    - e_pre
    - e_pre_e_sim
    - full_lc_mrsg
relation_gates:
  alpha_pre: [0.0, 0.25, 0.5, 0.75, 1.0]
  alpha_sim: [0.0, 0.25, 0.5, 0.75, 1.0]
  alpha_co:  [0.0, 0.10, 0.25, 0.5, 1.0]
sparse_boost:
  beta: [0.0, 0.10, 0.25, 0.5]
  strata_source: train_only
  very_sparse_quantile: 0.10
  sparse_quantile: 0.33
statistics:
  bootstrap_resamples: 10000
  ci_level: 0.95
  multiple_comparison: holm
integrity:
  require_l1_l6_pass: true
  forbid_test_selection: true
  export_graph_checksums: true
  mirror_undirected_eco_edges: true
```

---

## 14. Expected Output Files

The experiment must produce these files.

```text
runs/q3_lcmrsg_plus_*/results/all_runs_train_valid_test.csv
runs/q3_lcmrsg_plus_*/results/static_baseline_summary.csv
runs/q3_lcmrsg_plus_*/results/val_selected_static_summary.csv
runs/q3_lcmrsg_plus_*/results/relation_gated_summary.csv
runs/q3_lcmrsg_plus_*/results/sparse_aware_relation_gated_summary.csv
runs/q3_lcmrsg_plus_*/statistics/paired_tests_with_ci.csv
runs/q3_lcmrsg_plus_*/statistics/holm_corrected_tests.csv
runs/q3_lcmrsg_plus_*/statistics/practical_significance_table.csv
runs/q3_lcmrsg_plus_*/sparse/sparse_stratum_summary.csv
runs/q3_lcmrsg_plus_*/audit/leakage_audit_l1_l6.csv
runs/q3_lcmrsg_plus_*/audit/graph_provenance_complete.csv
runs/q3_lcmrsg_plus_*/audit/selection_isolation_report.csv
runs/q3_lcmrsg_plus_*/latex/table_main_confirmatory.tex
runs/q3_lcmrsg_plus_*/latex/table_val_selected.tex
runs/q3_lcmrsg_plus_*/latex/table_paired_tests_ci.tex
runs/q3_lcmrsg_plus_*/latex/table_sparse_summary.tex
runs/q3_lcmrsg_plus_*/latex/table_audit_l1_l6.tex
runs/q3_lcmrsg_plus_*/figures/fig_relation_selection_heatmap.pdf
runs/q3_lcmrsg_plus_*/figures/fig_delta_auc_forestplot.pdf
runs/q3_lcmrsg_plus_*/figures/fig_sparse_delta_auc.pdf
```

---

## 15. Analysis Tables Required for the Paper

### Table A: Main confirmatory result

Columns:

```text
Dataset | Model | No graph | Full LC-MRSG | LC-MRSG++ selected | Delta selected-no | 95% CI | Holm p | Practical effect
```

### Table B: Selected relation pattern

Columns:

```text
Dataset | Model | Most selected variant | Selection frequency | Mean valid AUC | Mean test AUC | Interpretation
```

### Table C: Relation gates

Columns:

```text
Dataset | Model | alpha_pre | alpha_sim | alpha_co | beta | Test AUC | Sparse AUC | Interpretation
```

### Table D: Sparse-stratum result

Columns:

```text
Dataset | Model | Variant | Very sparse | Sparse | Medium | Frequent | Sparse gain vs no_graph
```

### Table E: Audit and provenance

Columns:

```text
Dataset | Fold | L1 | L2 | L3 | L4 | L5 | L6 | Eco mirrored | Support metadata | Status
```

---

## 16. Decision Rules After the New Experiment

Use these rules to avoid overstating the paper.

### Case 1: Strong improvement

If LC-MRSG++ selected/gated improves AUC in at least 8/15 comparisons and at least 4 are significant before Holm correction:

Claim:

```text
LC-MRSG++ improves graph-enhanced KT selectively and reduces harmful graph use through validation-guided relation selection.
```

### Case 2: Moderate improvement

If LC-MRSG++ improves AUC in 6--7/15 comparisons and 2--3 are significant:

Claim:

```text
LC-MRSG++ is primarily a leakage-controlled graph construction and validation-selection protocol. It reduces negative graph effects and provides reproducible evidence about when graph relations are useful.
```

### Case 3: Weak improvement

If LC-MRSG++ improves fewer than 6/15 comparisons:

Claim:

```text
The main contribution is a diagnostic and audit framework showing that graph relations are not uniformly beneficial in KT. This negative result is useful because it prevents unsupported graph-use claims.
```

Do not hide Case 3. A careful negative result can still be Q3-suitable if the audit, methodology, and statistical reporting are strong.

---

## 17. Manuscript Upgrade Plan After Results

After the experiment, update the paper as follows.

### 17.1 Abstract

Replace performance-centered wording with:

```text
This paper proposes LC-MRSG++, a split-first and validation-guided protocol for constructing, selecting, and auditing multi-relational skill graphs in KT. Rather than assuming that the full graph is always useful, LC-MRSG++ uses validation-only evidence to decide whether prerequisite, similarity, and co-occurrence relations should be activated, weighted, or disabled.
```

### 17.2 Contributions

Add explicit bullet contributions:

```text
1. A split-first multi-relational skill graph construction protocol with L1--L6 leakage checks.
2. A validation-guided relation-selection mechanism that avoids assuming the full graph is always best.
3. A sparse-aware relation-gating extension for long-tail skills.
4. A reproducibility package with graph provenance, support summaries, checksums, and statistical confidence intervals.
5. A full evaluation across 3 datasets, 3 folds, 5 seeds, 5 KT backbones, and static/adaptive graph variants.
```

### 17.3 Method

Insert Algorithm 1, Algorithm 2, and Algorithm 3.

### 17.4 Results

Reorder results:

```text
5.1 Static baseline replication
5.2 Validation-guided LC-MRSG++ selection
5.3 Relation-gating and sparse-aware extension
5.4 Statistical significance and confidence intervals
5.5 Leakage, provenance, and L1--L6 audit
5.6 Sparse-skill diagnostics
5.7 Failure cases and when graph relations should be disabled
```

### 17.5 Discussion

Emphasize:

```text
The scientific value is not that graph edges always improve KT. The value is that LC-MRSG++ makes graph usefulness testable, auditable, and selectable without test leakage.
```

---

## 18. Reviewer-Focused Quality Checklist

Before producing the final LaTeX, confirm every item.

```text
[ ] Abstract does not overclaim.
[ ] Title matches the evidence.
[ ] Algorithms are included.
[ ] L1--L6 audit table is included.
[ ] Validation-only selection is documented.
[ ] Test set is never used for graph selection.
[ ] 95% CI and effect sizes are reported.
[ ] Holm correction is reported.
[ ] Sparse-skill results are not overstated.
[ ] Eco provenance FLAG is explained, not hidden.
[ ] Graph artifacts include checksums.
[ ] Negative cases are discussed.
[ ] Appendix figures are readable.
[ ] References start on a new page.
```

---

## 19. Antigravity Execution Prompt

Paste the following into Antigravity as the main mission prompt.

```text
You are working on the LC-MRSG Knowledge Tracing project. Your mission is to implement and run one final Q3-safe experiment called LC-MRSG++.

Do not optimize using test results. Do not overwrite previous results. First inspect the project structure and existing result files. Create a new git branch q3-lcmrsg-plus-final and a new output directory under runs/q3_lcmrsg_plus_TIMESTAMP.

The current paper shows that fixed full LC-MRSG is not uniformly beneficial. Therefore, implement LC-MRSG++ as a leakage-controlled, validation-guided relation-selection protocol. The method must select graph variants or relation weights using validation metrics only, then evaluate the locked configuration once on the test set.

Implement or update these files:
- scripts/q3_lcmrsg_plus_build_graphs.py
- scripts/q3_lcmrsg_plus_run_experiments.py
- scripts/q3_lcmrsg_plus_analyze.py
- scripts/q3_lcmrsg_plus_render_tables.py
- scripts/q3_lcmrsg_plus_export_appendix.py
- scripts/q3_lcmrsg_plus_sanity_checks.py
- tests/test_q3_lcmrsg_plus_leakage.py
- tests/test_q3_lcmrsg_plus_statistics.py
- configs/q3_lcmrsg_plus.yaml
- README_Q3_LCMRSG_PLUS.md

Add three methods:
1. val_selected_static: select among no_graph, E_pre, E_pre_E_sim, full_lc_mrsg using validation AUC only.
2. relation_gated: validation-selected relation weights alpha_pre, alpha_sim, alpha_co.
3. sparse_aware_relation_gated: relation_gated plus train-only sparse-skill boost beta.

Run Stage A first: check whether validation metrics already exist in previous logs. If yes, compute val_selected_static without retraining. If not, run Stage B: rerun the four static variants to export train, validation, and test metrics separately. Then run Stage C for relation_gated and sparse_aware_relation_gated on the priority model-dataset pairs described in this markdown.

Produce all outputs listed in Section 14. Compute paired tests, 95% bootstrap CI, two-tailed and one-tailed p-values, Wilcoxon p-values, Cohen's d, Holm correction, and practical significance labels.

Implement L1--L6 leakage checks. L6 must prove that no test metric was used for graph selection, threshold selection, relation-gate selection, early stopping, or model selection.

At the end, generate a concise report:
1. Which method is best overall?
2. How often does LC-MRSG++ improve over no_graph?
3. How often does LC-MRSG++ improve over fixed full LC-MRSG?
4. Which relations are selected most often?
5. Are sparse skills improved?
6. Did all L1--L6 checks pass?
7. What claims are safe for a Q3 paper?

Do not claim universal improvement unless the results support it. If the results are modest, frame the contribution as validation-guided graph selection and leakage-controlled reproducibility.
```

---

## 20. Final Success Criterion for Q3 8.5--8.8

The paper can reasonably move from approximately 7.2/10 to 8.5--8.8/10 if the final revision achieves the following even if predictive gains remain modest:

```text
1. LC-MRSG++ is clearly algorithmic, not just descriptive.
2. The full graph weakness is solved by validation-guided selection.
3. The paper reports confidence intervals, effect sizes, and corrected p-values.
4. The graph artifact is reproducible with support summaries and checksums.
5. Sparse-skill analysis is presented carefully and honestly.
6. The title and abstract match the evidence.
7. The paper explains when graph relations should be disabled.
8. All L1--L6 leakage checks pass.
```

The safest final claim is:

```text
LC-MRSG++ does not assume that multi-relational skill graphs always improve KT. Instead, it provides a leakage-controlled and validation-guided protocol for constructing, selecting, auditing, and reporting graph relations in KT. This turns weak or mixed graph gains into a stronger methodological contribution for reproducible graph-enhanced knowledge tracing.
```

---

## 21. What Not to Do

Do not do any of the following:

```text
[ ] Do not select the best seed using test AUC.
[ ] Do not select graph thresholds using test AUC.
[ ] Do not remove negative model-dataset pairs.
[ ] Do not report only AUC and hide NLL/RMSE.
[ ] Do not describe KDD2010 BKT delta +0.0001 as practically meaningful.
[ ] Do not claim sparse-skill improvement on Junyi where sparse/very-sparse strata are unavailable.
[ ] Do not treat provenance FLAG as hidden leakage; explain it as an artifact completeness issue unless audit proves otherwise.
[ ] Do not rerun endlessly without a pre-registered decision rule.
```

---

## 22. Immediate Next Command Sequence

Use this sequence after the repository is opened in Antigravity.

```bash
# 1. Inspect repository
pwd
ls -lah
find . -maxdepth 3 -type f | sed 's#^./##' | sort | head -200

# 2. Create safe branch
git status
git checkout -b q3-lcmrsg-plus-final

# 3. Create config and output folders
mkdir -p configs scripts tests runs/q3_lcmrsg_plus

# 4. Search for validation metrics
python - <<'PY'
import os
keys=['valid_auc','val_auc','validation_auc','valid_loss','val_loss']
for root, dirs, files in os.walk('.'):
    if '.git' in root:
        continue
    for f in files:
        if f.endswith(('.csv','.json','.txt','.log','.tsv')):
            p=os.path.join(root,f)
            try:
                text=open(p,'r',encoding='utf-8',errors='ignore').read(5000).lower()
                if any(k in text for k in keys):
                    print(p)
            except Exception:
                pass
PY

# 5. Implement scripts and tests according to this mission file
# 6. Run sanity tests before long experiment
python scripts/q3_lcmrsg_plus_sanity_checks.py
pytest tests/test_q3_lcmrsg_plus_leakage.py -q
pytest tests/test_q3_lcmrsg_plus_statistics.py -q

# 7. Run Stage A/B/C according to availability of validation logs
python scripts/q3_lcmrsg_plus_run_experiments.py --config configs/q3_lcmrsg_plus.yaml

# 8. Analyze and render tables
python scripts/q3_lcmrsg_plus_analyze.py --run_dir runs/q3_lcmrsg_plus_*/
python scripts/q3_lcmrsg_plus_render_tables.py --run_dir runs/q3_lcmrsg_plus_*/
python scripts/q3_lcmrsg_plus_export_appendix.py --run_dir runs/q3_lcmrsg_plus_*/
```

---

## 23. Final Note for the Human Researcher

It is possible that the additional experiment still does not produce large AUC gains. That does not mean the paper fails. The strongest Q3-safe path is to stop trying to prove that full LC-MRSG always improves prediction, and instead prove that LC-MRSG++ is a rigorous protocol that tells researchers when graph relations are useful, when they are neutral, and when they should be disabled.

This is a stronger, more honest, and more publishable contribution than repeatedly rerunning experiments until one table looks better.
