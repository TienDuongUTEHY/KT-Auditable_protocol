# PROMPT FOR GOOGLE GRAVITY / ANTIGRAVITY

## General Mission

You are a scientific research programming agent. Automatically inspect, supplement, run, and export all minimal experimental results to complete Paper P0:

**LC-MRSG++: Leakage-Controlled and Validation-Guided Multi-Relational Skill Graph Construction for Sparse-Skill Knowledge Tracing**

The goal is not to create another performance/SOTA paper, but to complete a reputable Q3-quality **protocol/audit paper**. All performance results should only be viewed as **diagnostic evidence**. Do not expand to P1/P2: do not implement calibration/ECE/Brier, do not implement adaptive stratification, do not implement SSA-CL/InfoNCE, and do not implement learning-path recommendations.

---

## 0. Mandatory Principles

1. **Do not modify the raw dataset.** Only read original data and write results to the `results_p0_revision/` directory.
2. **Do not overwrite older results.** If a file already exists, write the new version with a timestamp.
3. **Do not fabricate data.** If a file cannot be read or a metric cannot be calculated, record `NA` with the reason in the logs.
4. **Fail fast.** If the audit detects a critical issue with graph provenance or split leakage, stop before running epoch sanity checks.
5. **All logs must be human-readable.** Each stage must contain the following lines:
   - `PHASE_START`
   - `PHASE_PASS` or `PHASE_FAIL`
   - `PHASE_SUMMARY`
6. **All tables must be exported as both CSV and LaTeX.** CSV is used for verification; `.tex` is used for insertion into the manuscript.
7. **All main artifacts must have a SHA256 hash.** Generate a final manifest.
8. **Do not alter the scientific claims.** Only report results conservatively: graph usefulness is dataset- and backbone-dependent.

---

## 1. Directory Structure to Create

Create the following structure in the root of the repository:

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

If the repository already has its own structure, still create the above directories to aggregate the final results.

---

## 2. Automated Repository Inspection

Before running, verify:

```text
- Are dataset configuration files present?
- Is the data/ or datasets/ directory present?
- Do the outputs/results/tables directories exist?
- Are training/evaluation scripts present?
- Is the graph provenance file present?
- Is the split specification present?
- Are AUC results tables present?
```

Record in `logs/master_run_<TIMESTAMP>.log`:

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

If no training/evaluation scripts are detected, do not rewrite models; create wrappers and prompt the user for paths.

---

## 3. Phase 0 — Scope Boundary Control

### 3.1 Tasks to Perform

Create `configs/p0_revision_config.yaml` with variables:

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

### 3.2 Mandatory Logs

Record in `phase0_scope_decision.log`:

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

## 4. Phase 1 — Data and Graph Correctness Audit

This phase is mandatory before any results are claimed. If it fails, do not proceed to Phase 2.

---

### 4.1 Audit Dataset Statistics and Disclosure Subsampling

#### Objective

Recover the dataset statistics table and detect if subsetting or subsampling was used.

#### Datasets to Process

```text
- ASSIST2012
- Junyi
- KDD2010
```

Folder names may vary. Check aliases:

```text
assist2012, assistments2012, assist_2012, ASSIST2012
junyi, junyi_academy, Junyi
kdd2010, algebra2008, bridge_to_algebra, KDDCup2010
```

#### Metrics to Compute

For each dataset and each split/fold (if applicable):

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

#### Reference Scale Verification

Log a warning if the scale is significantly smaller than the reference points used in the manuscript:

```text
ASSIST2012 expected interactions ≈ 2.7M
Junyi expected interactions ≈ 16.2M
KDD2010 expected interactions: compute from local processed dataset; if there is an original raw file, compare processed/raw ratio.
```

Do not hard-fail solely due to scale; only hard-fail if there is no disclosure of subsampling.

#### Output

```text
tables_csv/table_dataset_statistics.csv
tables_tex/table_dataset_statistics.tex
logs/phase1_dataset_graph_audit.log
```

#### Mandatory Logs

```text
PHASE_START dataset_statistics
DATASET_STATS dataset=ASSIST2012 users_total=... interactions_total=... suspected_subsampling=...
DATASET_STATS dataset=Junyi users_total=... interactions_total=... suspected_subsampling=...
DATASET_STATS dataset=KDD2010 users_total=... interactions_total=... suspected_subsampling=...
DATASET_STATS_OUTPUT=results_p0_revision/tables_csv/table_dataset_statistics.csv
PHASE_PASS dataset_statistics
```

---

### 4.2 Audit KDD2010 E_co Anomaly

#### Problem Description

The review raised a risk: `KDD2010 E_co = 658,943` edges. If `|C|≈493`, the maximum number of undirected KC--KC pairs is `C(|C|,2)≈121,278`. We need to determine if this number represents:

```text
(a) support records,
(b) directed/mirrored edges,
(c) multi-edge graph,
(d) different actual |C|,
(e) aggregation/mirroring error.
```

#### Metrics to Compute for Each Dataset/Fold/Relation

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

#### Hard-Fail Rule

Fail if:

```text
unique_undirected_edges > max_possible_undirected_pairs
```

unless the graph is explicitly defined as a multi-edge graph and has a separate `support_records` column.

#### Output

```text
tables_csv/table_graph_provenance_corrected.csv
tables_tex/table_graph_provenance_corrected.tex
logs/phase1_dataset_graph_audit.log
```

#### Mandatory Logs

```text
PHASE_START kdd2010_eco_audit
GRAPH_AUDIT dataset=KDD2010 fold=0 relation=E_co n_skills=... max_pairs=... raw_rows=... unique_undirected=... support_records=... interpretation=...
GRAPH_AUDIT dataset=KDD2010 fold=1 relation=E_co ...
GRAPH_AUDIT dataset=KDD2010 fold=2 relation=E_co ...
KDD2010_ECO_DECISION=<unique_edges|support_records|multi_edge|error_fixed|error_unfixed>
PHASE_PASS kdd2010_eco_audit
```

If it fails:

```text
PHASE_FAIL kdd2010_eco_audit reason="unique undirected edges exceed graph limit and no multi-edge definition found"
STOP_BEFORE_PHASE2=true
```

---

### 4.3 Trace E_sim Pipeline

#### Problem Description

The review noted a contradiction: one table specified `top-k=20`, but another reported `E_sim=0` on ASSIST2012/Junyi. We need to log the edge count after each step.

#### Trace Steps

For each dataset/fold:

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

#### Post-Trace Decision

Automatically assign one of the two decisions:

```text
A. E_sim_active: E_sim has edges in at least 2/3 of datasets.
B. E_sim_empty_effective: E_sim is empty in >=2 datasets; the manuscript must use the label E_sim^eff or "empty E_sim branch" and not claim similarity benefits.
```

Do not modify the algorithm unless a flag is explicitly provided. Default behavior is audit only. If proposing changes to the top-k pipeline, log them as a separate proposal; do not execute them.

#### Output

```text
tables_csv/table_esim_trace.csv
tables_tex/table_esim_trace.tex
logs/phase1_esim_trace.log
```

#### Mandatory Logs

```text
PHASE_START esim_trace
ESIM_TRACE dataset=ASSIST2012 fold=0 before_threshold=... after_threshold=... after_topk=... final_edges=... reason_if_zero=...
ESIM_TRACE dataset=Junyi fold=0 ...
ESIM_TRACE dataset=KDD2010 fold=0 ...
ESIM_DECISION=<E_sim_active|E_sim_empty_effective>
PHASE_PASS esim_trace
```

---

### 4.4 Junyi Graph Coverage and Isolated Nodes

#### Objective

Explain why the Junyi graph is extremely sparse but still yields performance gains on some backbones. Do not over-interpret.

#### Metrics to Compute

For Junyi, each fold and each relation/candidate graph:

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

#### Mandatory Logs

```text
PHASE_START junyi_coverage
JUNYI_COVERAGE fold=0 n_skills=... covered=... isolated=... coverage_ratio=... mean_degree=...
JUNYI_COVERAGE fold=1 ...
JUNYI_COVERAGE fold=2 ...
JUNYI_INTERPRETATION=<diagnostic_only|coverage_supports_limited_mechanism|insufficient_explanation>
PHASE_PASS junyi_coverage
```

---

### 4.5 Leakage Audit L1--L6

#### Audits to Run

```text
L1 Edge-construction leakage: all edge supports must come from the training split.
L2 Q-matrix/provenance leakage: Q-matrix used to build edges must not be derived from held-out logs outside dataset specifications.
L3 Temporal leakage: if timestamps exist, no future evidence is used in graph construction.
L4 Cold-start neighborhood leakage: test-only skills/items must not receive neighborhoods from held-out evidence.
L5 Co-occurrence leakage: E_co support/count/PMI/NPMI must be computed only from the training fold.
L6 Selection leakage: validation selects the candidate; the test set must not be used to choose graphs or hyperparameters.
```

#### Output

```text
tables_csv/table_leakage_audit_L1_L6.csv
tables_tex/table_leakage_audit_L1_L6.tex
```

#### Mandatory Logs

```text
PHASE_START leakage_audit_L1_L6
LEAKAGE_AUDIT dataset=ASSIST2012 fold=0 L1=PASS L2=PASS L3=PASS L4=PASS L5=PASS L6=PASS notes=...
...
PHASE_PASS leakage_audit_L1_L6
```

Hard-fail if any of L1, L5, or L6 fail.

---

## 5. Phase 2 — Minimum Epoch Sanity Checks

### 5.1 Objective

Address the risk of fixed two-epoch undertraining. Do not use this phase to find SOTA, nor to replace the main confirmatory results; only verify if the **sign of ΔAUC** remains stable at 5 and 10 epoch budgets.

### 5.2 Mandatory Settings

Use the exact subset used in the paper; do not modify preprocessing.

```text
Datasets: select 2 main representative datasets. Default: ASSIST2012 and Junyi. If Junyi is too large, use KDD2010 but document the reason.
Backbones: DKT, simpleKT
Graph conditions: no_graph, selected_graph
Epoch budgets: 5, 10
Folds: use 3 folds if compute allows; otherwise, at a minimum fold 0 but document the limitation.
Seeds: use 3 seeds if compute allows; otherwise, at a minimum the main seed in the paper but document the limitation.
Early stopping: if available in pipeline, enable early stopping but log the maximum epoch budget.
```

### 5.3 Metrics to Compute

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

### 5.4 Output Files

```text
tables_csv/table_epoch_sanity.csv
tables_tex/table_epoch_sanity.tex
logs/phase2_epoch_sanity.log
```

### 5.5 Mandatory Logs

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

### 5.6 Automated Interpretation Rules

Create the variable `epoch_sanity_interpretation`:

```text
- "direction_stable" if >=75% of comparisons retain the same sign as the main results.
- "mixed_direction" if 50--74% retain the same sign.
- "direction_unstable" if <50% retain the same sign.
```

If `direction_unstable`, lower the claims in the manuscript:

```text
The fixed-budget graph effect should be interpreted as a two-epoch diagnostic result only; longer-budget sanity checks do not support a stable selected-graph direction.
```

If `direction_stable`, write:

```text
The longer-budget sanity check preserves the direction of the selected-graph effect in most inspected settings, supporting the use of the fixed-budget results as protocol diagnostics rather than SOTA-tuned model comparisons.
```

---

## 6. Phase 3 — Generating Summary Tables for the Manuscript

### 6.1 Pre-Specified Validation Candidates Table

Create `p0_validation_candidates.yaml` and a `.csv/.tex` table with 9 candidate configurations.

If no actual configuration is found in the repo, generate a template table but mark `status=TO_VERIFY`.

Table Structure:

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

Requirement: `pre_specified_yes_no=yes` and `test_used_for_selection_yes_no=no` for all candidates, provided there is timestamp/hash evidence.

### 6.2 Main AUC Delta + CI + Holm Table

Create a summary table for the main text:

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

If the older AUC file exists, read and standardize it. Do not recompute if prediction files are missing; record `NA` and log.

### 6.3 Selected Relation Variants Table

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

If `E_sim` is empty, use the label `E_sim^eff=empty`.

### 6.4 Sparse Bins Descriptive Table

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

Do not run adaptive stratification.

### 6.5 Reproducibility Checklist Table

```text
artifact
path
status: available/missing/partial
sha256
purpose
used_in_main_text_yes_no
```

### 6.6 Mandatory Logs

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

## 7. Phase 4 — Manifest, Validation, and Definition of Done

### 7.1 SHA256 Manifest

Create `manifests/sha256_manifest.csv` for all files in `results_p0_revision/`:

```text
relative_path
sha256
size_bytes
modified_time
```

### 7.2 Run Environment

Create `manifests/run_environment.txt`:

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

Create `manifests/final_definition_of_done.md` with the checklist:

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

### 7.4 Validation Script

Create or run `scripts/p0_validate_outputs.py` to verify:

```text
- presence of all required files;
- no hard-fails in logs;
- table_graph_provenance_corrected does not contain unique_undirected_edges > max_possible_undirected_pairs;
- leakage audits L1/L5/L6 pass;
- table_epoch_sanity has at least one entry or documented limitation;
- sha256 manifest contains all core files;
- LaTeX tables are not empty.
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

## 8. Scripts to Create if Missing

If not present in the repository, create the following under `scripts/`:

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

`p0_run_all_revision.py` must be the main entry point:

```bash
python scripts/p0_run_all_revision.py --config results_p0_revision/configs/p0_revision_config.yaml
```

Use standard logging in all scripts:

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
```

Log each phase to both the console and the file.

---

## 9. LaTeX Table Rules

1. `.tex` tables must not contain `\begin{table}` environment if they are meant to be `\input{}` into a manuscript wrapper. If a standalone version is required, generate two versions: `_body.tex` and `_table.tex`.
2. Format long numbers with `\scriptsize` or wrap with `\resizebox{\textwidth}{!}{...}`.
3. Every table must contain a `Notes` column if `NA` or `TO_VERIFY` values are present.
4. Use dots instead of commas as decimal separators.
5. For KDD2010 E_co, report both `support_records` and `unique_undirected_edges` to prevent confusion.

---

## 10. Automatically Generated Reviewer Response Snippets

Create `supplementary/reviewer_response_notes.md` containing response snippets:

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

## 11. Post-Run Manuscript Update Guidelines

After completion, update LaTeX files by copying the generated `.tex` files from:

```text
results_p0_revision/tables_tex/
```

into the `tables/` directory of Overleaf, or keep the paths and invoke:

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

If a table is missing, use a placeholder: `TO BE FILLED AFTER P0 REVISION RUN` (do not fabricate results).

---

## 12. Final Execution Command

Implement the complete pipeline, then run:

```bash
python scripts/p0_run_all_revision.py --config results_p0_revision/configs/p0_revision_config.yaml 2>&1 | tee results_p0_revision/logs/master_run_$(date +%Y%m%d_%H%M%S).log
```

Upon completion, print to console:

```text
P0_REVISION_PIPELINE_FINISHED
STATUS=<PASS|PASS_WITH_LIMITATIONS|FAIL>
KEY_OUTPUT_DIR=results_p0_revision/
NEXT_ACTION=Copy tables_tex into LaTeX manuscript and update the interpretation paragraphs according to epoch_sanity_interpretation.
```

---

## 13. Strictly Prohibited Actions

Do not:

```text
- Add calibration/ECE/Brier to P0.
- Add adaptive sparse stratification.
- Add SSA-CL/InfoNCE.
- Add learning-path recommendation experiments.
- Claim SOTA.
- Use the test set to select graphs.
- Modify the raw dataset.
- Delete older results.
- Alter results to look better.
```

---

## 14. Success Criteria

The pipeline achieves `PASS` if:

```text
- Dataset stats are complete for all 3 datasets.
- KDD2010 E_co is resolved (unique edges, support records, multi-edge, or fixed error).
- E_sim trace has a clear decision.
- Junyi graph coverage has covered skill and isolated node status.
- Leakage audits L1--L6 do not fail on L1/L5/L6.
- Epoch sanity check is complete or limitations are logged.
- All CSV and LaTeX tables are exported.
- SHA256 manifest is complete.
- Definition of Done is generated.
```

The pipeline achieves `PASS_WITH_LIMITATIONS` if the epoch sanity checks could not be fully run across folds/seeds due to compute budget, but dataset/graph audits all pass and limitations are documented.

The pipeline `FAILS` if there is a graph leakage error, KDD2010 E_co count exceeds limit without explanation, or if selection uses test set evidence.
