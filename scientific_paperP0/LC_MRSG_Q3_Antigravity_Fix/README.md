# LC-MRSG Q3 Fix Package for Google Antigravity

This package converts the supervisor review into an executable experiment plan.
It is designed to address the remaining empirical weaknesses before journal submission.

## Quick start

```bash
cd LC_MRSG_Q3_Antigravity_Fix
pip install -r requirements.txt
bash run_all_q3_fix.sh configs/q3_fix_config.yaml
```

## Expected input layout

```text
data/processed/{dataset}/interactions.csv
data/processed/{dataset}/fold_{fold}/train.csv
results/predictions/{dataset}/fold_{fold}/seed_{seed}/{model}/{graph_variant}.csv
graphs/{dataset}/fold_{fold}/e_pre_scores.csv
graphs/{dataset}/fold_{fold}/e_co.csv
```

Prediction CSV files should include at least:

```text
y_true,y_pred,skill_id
```

Alternative column names are supported for many cases, e.g. `label`, `correct`, `prob`, `prediction`, `kc_id`.

## Main outputs

```text
results/q3_fix/tables/dataset_scale_audit.csv/.tex
results/q3_fix/tables/multifold_confirmatory_results.csv/.tex
results/q3_fix/tables/paired_tests_no_vs_full.csv/.tex
results/q3_fix/tables/zero_variance_diagnosis_summary.csv/.tex
results/q3_fix/tables/e_pre_pruning_summary.csv/.tex
results/q3_fix/tables/eco_provenance_audit.csv/.tex
results/q3_fix/tables/sparse_skill_summary_mean_std.csv/.tex
results/q3_fix/tables/paper_consistency_checks.csv/.tex
```

## Manuscript-use priorities

1. Replace the old main performance table with `multifold_confirmatory_results.tex`.
2. Add `paired_tests_no_vs_full.tex` after the main results table.
3. Add `zero_variance_diagnosis_summary.tex` in reproducibility/model-integrity subsection.
4. Add `e_pre_pruning_summary.tex` in graph-ablation subsection.
5. Add `sparse_skill_summary_mean_std.tex` to justify the “Sparse-Skill” title.
6. Add `eco_provenance_audit.tex` to make the four-check E_co module explicit.
7. Add `paper_consistency_checks.tex` as internal QC; only include in appendix if needed.
