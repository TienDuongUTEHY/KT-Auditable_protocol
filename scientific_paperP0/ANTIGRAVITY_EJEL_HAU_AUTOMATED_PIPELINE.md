# Google Antigravity Autopilot Plan: EJEL Hau-Response Experiments for LC-MRSG

## Mission

Create a fresh, fully reproducible output folder for the EJEL revision of LC-MRSG and run all experiments needed to address GS Hậu's comments P1--P7. The pipeline must be automatic, resumable, and auditable. Do not wait for manual supervision between stages. If a stage fails, write an error report, skip dependent stages, continue independent stages, and produce a final `RUN_STATUS.md` explaining what succeeded and what remains unresolved.

## Non-negotiable scientific rules

1. Do not fabricate or impute missing experimental results.
2. Do not overwrite earlier results. Always create a new timestamped folder.
3. Build all graph artefacts after the train/valid/test split.
4. Use validation data only for relation selection and early stopping.
5. Evaluate the selected graph once on the held-out test split.
6. Separate LR-KT proxy from neural KT backbones in every table and conclusion.
7. Report both statistical significance and practical magnitude.
8. Mark L2 Q-matrix/KC provenance as `unverified` unless a dataset-specific provenance document can be independently verified.
9. Treat sparse-skill results as descriptive unless the effective sample-size reliability flag allows stronger interpretation.
10. Do not use language such as `leakage-free guarantee`, `universally improves`, or `state-of-the-art` in generated manuscript text.

## Expected repository assumption

Use the existing project repository if available. If a required script does not exist, create it under `scripts/ejel_hau_revision/` with clear docstrings and logging. Do not modify stable baseline scripts unless absolutely necessary; prefer wrapper scripts.

Recommended structure to create if missing:

```text
scripts/ejel_hau_revision/
  00_make_run_dir.py
  01_dataset_and_split_audit.py
  02_build_train_only_graphs.py
  03_graph_density_and_relation_availability.py
  04_eco_threshold_sensitivity.py
  05_train_two_epoch_reference.py
  06_train_early_stopping.py
  07_select_relations_validation_only.py
  08_evaluate_selected_once.py
  09_bootstrap_and_holm.py
  10_sparse_bin_reliability.py
  11_no_epre_sensitivity.py
  12_proxy_vs_neural_reanalysis.py
  13_generate_tables_and_figures.py
  14_quality_gates.py
  15_update_markdown_manuscript.py
configs/ejel_hau_revision_config.yaml
```

## Output folder

Create:

```text
results_ejel_hau_revision_<YYYYMMDD_HHMMSS>/
  manifest/
  logs/
  configs/
  splits/
  graphs/
  predictions/
  epoch_logs/
  statistics/
  tables/
  figures/
  manuscript_ready/
  quality_gates/
```

At the end, compress the folder into:

```text
results_ejel_hau_revision_<YYYYMMDD_HHMMSS>.zip
```

## Stage 0 — Environment, run directory, and frozen manifest

### Goal

Create a reproducible run folder and record environment details.

### Actions

1. Read `configs/ejel_hau_revision_config.yaml`.
2. Create timestamped output folder.
3. Copy the config into `configs/` inside the output folder.
4. Record:
   - git commit hash if available;
   - Python version;
   - package versions;
   - CUDA/GPU/CPU information;
   - random seeds;
   - hostname and OS;
   - start time.
5. Write `manifest/frozen_manifest_start.json`.

### Outputs

- `manifest/frozen_manifest_start.json`
- `logs/stage_00_environment.log`

## Stage 1 — Dataset and split audit

### Goal

Fix the misleading `Density` issue and verify split statistics.

### Actions

1. Load processed datasets for ASSIST2012, Junyi, and KDD2010.
2. For each dataset and fold, compute:
   - train users;
   - validation users;
   - test users;
   - skills in full processed data;
   - skills in train;
   - total interactions;
   - train interactions;
   - validation interactions;
   - test interactions;
   - train interaction intensity per user-skill cell: `train_interactions / (train_users * train_skills)`;
   - number of skills with train interactions <= 50, <=100, <=200, <=500.
3. Confirm that the column is named `train_interaction_intensity_per_user_skill_cell`, not `density`.
4. If any true density is computed, assert that it lies in `[0, 1]`.
5. Export split audit tables.

### Outputs

- `tables/dataset_split_stats_revised.csv`
- `tables/dataset_split_stats_revised.md`
- `logs/stage_01_split_audit.log`

### Quality checks

- No column named simply `Density` remains in generated tables.
- If interaction intensity is greater than 1, write an explanation that repeated user-skill interactions are possible.

## Stage 2 — Train-only graph construction

### Goal

Rebuild graph candidates from training evidence only and export support/provenance files.

### Actions

For each dataset and fold:

1. Build `Epre` from admissible train-side prerequisite-like or KC metadata priors.
2. Build `Esim` from train-side skill-question incidence using the existing Jaccard-style rule.
3. Build `Eco` from train-side KC co-occurrence support.
4. Export raw support rows separately from unique edges.
5. For each relation, compute:
   - skills covered by relation;
   - max pairs;
   - raw rows;
   - unique edges;
   - edge density = unique_edges / max_pairs;
   - effective relation flag:
     - `absent` if unique_edges = 0;
     - `sparse` if 0 < density < 0.05;
     - `moderate` if 0.05 <= density < 0.50;
     - `dense` if 0.50 <= density < 0.80;
     - `very_dense` if density >= 0.80.
6. Run L1, L3, L4, L5 checks locally.
7. Mark L2 as `unverified` unless verified metadata provenance exists.
8. Run L6 only after relation selection in Stage 7.

### Outputs

- `graphs/<dataset>/fold_<k>/Epre_edges.csv`
- `graphs/<dataset>/fold_<k>/Esim_edges.csv`
- `graphs/<dataset>/fold_<k>/Eco_edges.csv`
- `graphs/<dataset>/fold_<k>/Eco_support_rows.csv`
- `tables/graph_provenance_with_density.csv`
- `tables/effective_relation_availability.csv`
- `tables/leakage_audit_partial_L1_L5.csv`

### Quality checks

- `unique_edges <= max_pairs` for every undirected relation.
- Esim absence on ASSIST2012 and Junyi is explicitly reported if still true.
- KDD2010 Eco density is explicitly reported.

## Stage 3 — Eco threshold sensitivity on KDD2010

### Goal

Address the concern that KDD2010 Eco is nearly fully connected.

### Actions

For KDD2010 folds 0--2, evaluate Eco construction under:

- `k_min_grid = [2, 3, 5, 10, 20, 50, 100]`
- `pmi_min_grid = [0.0, 0.25, 0.5, 1.0]`
- `topk_per_skill_grid = [None, 100, 50, 20, 10]`

For each setting compute:

- raw support rows;
- unique edges;
- edge density;
- mean, median, p25, p75 of PMI weights;
- fraction of skills with at least one Eco neighbour;
- whether density falls below 0.80, 0.50, and 0.25.

Optionally, if compute permits, run validation-only relation selection with a small neural subset to determine whether lower-density Eco variants remain useful.

### Outputs

- `tables/kdd2010_eco_threshold_sensitivity.csv`
- `tables/kdd2010_eco_threshold_sensitivity.md`
- `figures/kdd2010_eco_density_vs_threshold.png`
- `figures/kdd2010_eco_coverage_vs_threshold.png`

### Quality checks

- At least one threshold setting should demonstrate how density changes as the threshold becomes stricter.
- If all settings remain dense, report this honestly.

## Stage 4 — Two-epoch reference runs

### Goal

Preserve the current fixed-budget diagnostic evidence for direct comparison with convergence runs.

### Actions

For datasets `[assist2012, junyi]` at minimum, and `[kdd2010]` if compute permits, run or collect existing two-epoch outputs for neural backbones:

- DKT
- simpleKT
- GIKT
- SKT if compute permits

Use seeds `[42, 2024, 2025]` and at least fold `0`; ideally all three folds.

Export validation and test predictions for:

- no_graph;
- selected graph according to validation-only relation selection;
- relation candidates used in the current paper.

### Outputs

- `predictions/two_epoch/<dataset>/<backbone>/fold_<k>/seed_<s>_*.csv`
- `epoch_logs/two_epoch/<dataset>/<backbone>/fold_<k>/seed_<s>.csv`
- `tables/two_epoch_reference_auc.csv`

### Quality checks

- Every prediction file contains: `user_id`, `item_id` or `question_id`, `skill_id`, `y_true`, `y_pred`, `fold`, `seed`, `dataset`, `backbone`, `candidate`.

## Stage 5 — Early-stopping convergence runs

### Goal

Resolve the P1 blocker by determining whether graph effects survive longer, more realistic training.

### Actions

For at least:

- datasets: `assist2012`, `junyi`;
- backbones: `dkt`, `simplekt`;
- folds: at least `0`, ideally `[0,1,2]`;
- seeds: `[42, 2024, 2025]`.

Run:

- max epochs: 100;
- validation-AUC early stopping;
- patience: 10;
- min_delta: 0.0001;
- restore best checkpoint;
- save train loss and validation AUC per epoch.

If resources permit, extend to KDD2010, GIKT, and SKT.

### Outputs

- `predictions/early_stopping/<dataset>/<backbone>/fold_<k>/seed_<s>_*.csv`
- `epoch_logs/early_stopping/<dataset>/<backbone>/fold_<k>/seed_<s>.csv`
- `tables/early_stopping_auc.csv`
- `figures/learning_curves_early_stopping_<dataset>_<backbone>.png`

### Quality checks

- Every run records the selected epoch.
- No test metric is used for early stopping.
- If early stopping fails, use the best validation checkpoint within max epochs and mark the run.

## Stage 6 — Two-epoch versus early-stopping stability

### Goal

Create the central table for P1.

### Actions

For each dataset-backbone pair with both two-epoch and early-stopping predictions:

1. Compute no-graph test AUC.
2. Compute selected-graph test AUC.
3. Compute delta AUC.
4. Compute sign of delta AUC.
5. Compare sign under two-epoch and early-stopping budgets.
6. Bootstrap paired 95% CI for delta AUC.
7. Label:
   - `stable_positive`;
   - `stable_negative`;
   - `sign_changed`;
   - `near_zero_unstable`;
   - `insufficient_runs`.

### Outputs

- `tables/two_epoch_vs_early_stopping.csv`
- `tables/two_epoch_vs_early_stopping.md`
- `figures/two_epoch_vs_early_stopping_delta_auc.png`

### Decision rules

- If neural effects remain stable and practically meaningful, the manuscript may report early-stopping results as main evidence.
- If signs change or effects vanish, the manuscript must state that the predictive evidence is budget-bounded and diagnostic only.

## Stage 7 — Validation-only relation selection and L6 audit

### Goal

Ensure relation selection never uses test evidence.

### Actions

For each fold, seed, dataset, and backbone:

1. Build candidate graphs from training evidence only.
2. Evaluate candidates on validation set only.
3. Select the best candidate using validation AUC.
4. Freeze selected candidate.
5. Evaluate once on test.
6. Save selection logs with validation metrics for all candidates and one test metric for the selected candidate.
7. Mark L6 as PASS only if the test set was not used for candidate choice.

### Outputs

- `tables/validation_selection_logs.csv`
- `tables/selected_relation_frequency.csv`
- `tables/leakage_audit_L1_L6.csv`

### Quality checks

- No script should sort, rank, or choose candidates by test AUC.
- The selected candidate must be recoverable from validation logs alone.

## Stage 8 — LR-KT proxy versus neural KT reanalysis

### Goal

Resolve P2 by preventing LR-KT proxy from carrying the main conclusion.

### Actions

1. Rename `BKT-proxy` to `LR-KT proxy` in all generated tables.
2. Generate two main result tables:
   - neural-only table: DKT, simpleKT, GIKT, SKT;
   - proxy sanity-check table: LR-KT proxy only.
3. Count confirmatory rows separately:
   - neural-only statistical count;
   - neural-only practical count;
   - proxy statistical count.
4. Generate a paragraph explaining why proxy headroom can yield larger deltas and smaller p-values.

### Outputs

- `tables/main_auc_neural_only.csv`
- `tables/main_auc_neural_only.md`
- `tables/lr_kt_proxy_sanity.csv`
- `tables/lr_kt_proxy_sanity.md`
- `manuscript_ready/proxy_neural_interpretation_paragraph.md`

### Quality checks

- No generated claim says `five comparisons remain confirmatory` without specifying whether proxy rows are included.

## Stage 9 — Statistical and practical significance

### Goal

Resolve P4 by separating p-values from educational magnitude.

### Actions

1. For each comparison, compute paired bootstrap CI for delta AUC.
2. Apply Holm correction within the family of planned dataset-backbone comparisons.
3. Add practical-magnitude status using threshold `delta_auc >= 0.005`.
4. Optionally compute paired Cohen-style effect size over fold-seed deltas if enough paired observations exist.
5. Classify each row:
   - `confirmatory_and_practically_meaningful`;
   - `confirmatory_but_negligible`;
   - `diagnostic_practically_meaningful`;
   - `diagnostic_negligible`.

### Outputs

- `tables/statistical_vs_practical_significance.csv`
- `tables/statistical_vs_practical_significance.md`

### Quality checks

- Delta AUC below 0.005 is never described as educationally meaningful.

## Stage 10 — Sparse-bin reliability diagnostics

### Goal

Resolve P6 by making sparse-skill analysis descriptive unless effective sample sizes are sufficient.

### Actions

For each dataset, fold, seed, backbone, and candidate:

1. Count train interactions per skill.
2. Assign skills to bins:
   - <=50;
   - <=100;
   - <=200;
   - <=500;
   - >500.
3. For each bin, compute:
   - number of skills;
   - number of validation interactions;
   - number of test interactions;
   - no-graph AUC;
   - selected-graph AUC;
   - delta AUC;
   - bootstrap CI if N is sufficient.
4. Add reliability flag:
   - `Reliable` if N_test >= 1000;
   - `Limited` if 100 <= N_test < 1000;
   - `Insufficient` if N_test < 100;
   - `Not available` if no skills.
5. For Junyi, explicitly report if sparse bins contain zero skills.

### Outputs

- `tables/sparse_bin_reliability.csv`
- `tables/sparse_bin_reliability.md`
- `figures/sparse_bin_delta_auc_with_reliability.png`

### Quality checks

- No sparse-skill claim is marked confirmatory when reliability is Limited, Insufficient, or Not available.

## Stage 11 — No-Epre sensitivity for L2 residual risk

### Goal

Resolve P5 by quantifying whether conclusions depend on relations affected by Q-matrix metadata provenance.

### Actions

Run relation-selection/evaluation under a candidate grid that excludes Epre:

- no_graph;
- Eco only;
- Esim only if effective edges exist;
- Esim + Eco if effective edges exist;
- gated variants without Epre.

For each dataset and neural backbone:

1. Select candidate by validation AUC.
2. Evaluate once on test.
3. Compare with the original selected graph.
4. Report whether any practical gain survives without Epre.

### Outputs

- `tables/no_epre_L2_sensitivity.csv`
- `tables/no_epre_L2_sensitivity.md`
- `manuscript_ready/L2_residual_risk_paragraph.md`

### Quality checks

- The interpretation must state that this sensitivity does not prove L2 provenance, but bounds the dependence on Epre.

## Stage 12 — Reference and venue verification

### Goal

Resolve P7 reference and venue concerns.

### Actions

Manually or semi-automatically verify references from 2024 onward against trusted sources available to the environment. For each reference, record:

- title;
- authors;
- year;
- venue;
- volume/issue/pages/article number;
- DOI or official URL if available;
- verification source;
- status: `verified`, `needs_author_check`, or `remove_or_replace`.

Also check current EJEL instructions from the official journal website before final submission, including word limit, reference style, ethics/data/code statements, and AI-use policy.

### Outputs

- `tables/reference_verification_log.csv`
- `tables/ejel_submission_requirements_check.csv`

### Quality checks

- No 2024--2025 reference remains marked `needs_author_check` in the final submission.

## Stage 13 — Generate final tables, figures, and manuscript-ready Markdown

### Goal

Export the exact artefacts needed to update the manuscript.

### Actions

1. Generate Markdown tables:
   - `dataset_split_stats_revised.md`
   - `graph_provenance_with_density.md`
   - `two_epoch_vs_early_stopping.md`
   - `main_auc_neural_only.md`
   - `lr_kt_proxy_sanity.md`
   - `statistical_vs_practical_significance.md`
   - `kdd2010_eco_threshold_sensitivity.md`
   - `sparse_bin_reliability.md`
   - `no_epre_L2_sensitivity.md`
2. Generate figures at >=300 dpi.
3. Copy the revised manuscript template into `manuscript_ready/`.
4. Replace all `TBD_AFTER_RUN` placeholders where outputs exist.
5. Leave unresolved placeholders only if a stage failed, and explain in `RUN_STATUS.md`.

### Outputs

- `manuscript_ready/LC_MRSG_EJEL_MAIN_FILLED.md`
- `manuscript_ready/LC_MRSG_EJEL_SUPPLEMENTARY_FILLED.md`
- `figures/*.png`
- `tables/*.md`

## Stage 14 — Quality gates

### Goal

Prevent overclaiming and technical inconsistencies before the file is sent to GS Hậu or EJEL.

### Actions

Run automated text checks on manuscript-ready Markdown:

1. Fail if title contains `++` unless an explicit public predecessor citation exists.
2. Fail if `leakage-free guarantee` appears.
3. Fail if `universally improves`, `state-of-the-art`, or `leaderboard` claim appears as a contribution.
4. Fail if `BKT-proxy` remains; it should be `LR-KT proxy` unless canonical BKT is actually implemented.
5. Warn if `multi-relational` appears without nearby explanation of effective relation availability.
6. Warn if `sparse-skill` appears in Abstract without `diagnostic` or sample-size qualification.
7. Fail if a column called `Density` contains values >1.
8. Fail if a row with Delta AUC < 0.005 is called educationally meaningful.
9. Warn if any `TBD_AFTER_RUN` remains in the final manuscript.
10. Warn if references marked `VERIFY BEFORE SUBMISSION` remain.

### Outputs

- `quality_gates/quality_gates_report.md`
- `quality_gates/forbidden_phrase_scan.csv`
- `RUN_STATUS.md`

## Stage 15 — Final packaging

### Goal

Export a clean folder for manuscript writing and review.

### Actions

1. Freeze final manifest:
   - all stages status;
   - config;
   - generated files;
   - checksums;
   - runtime.
2. Create `README_FOR_AUTHOR.md` explaining how to use outputs.
3. Zip the result folder.

### Outputs

- `manifest/frozen_manifest_final.json`
- `README_FOR_AUTHOR.md`
- `results_ejel_hau_revision_<timestamp>.zip`

## Final RUN_STATUS template

At the end, write:

```markdown
# RUN_STATUS

## Overall status

- Completed stages:
- Failed stages:
- Skipped stages:

## Main scientific conclusions allowed

- P1 early-stopping stability:
- P2 proxy separation:
- P3 relation availability and Eco density:
- P4 practical significance:
- P5 L2 residual risk:
- P6 sparse-bin reliability:
- P7 quality gates:

## Files for manuscript update

- Main manuscript:
- Supplementary:
- Tables:
- Figures:
- Quality-gate report:

## Author decisions still required

- Repository/data availability link:
- Funding statement:
- Conflict of interest statement:
- Author contribution statement:
- EJEL final formatting check:
```

## Suggested single master command if scripts are implemented

```bash
python -m scripts.ejel_hau_revision.run_all \
  --config configs/ejel_hau_revision_config.yaml \
  --output-root results_ejel_hau_revision \
  --auto \
  --resume \
  --continue-on-independent-stage-error
```

If `run_all` does not exist, create it as an orchestrator that calls stages 0--15 in order and writes `RUN_STATUS.md`.
