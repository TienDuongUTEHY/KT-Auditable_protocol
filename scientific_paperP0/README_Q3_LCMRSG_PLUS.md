# LC-MRSG++: Leakage-Controlled, Validation-Guided Multi-Relational Skill Graph Construction

This reproducibility package implements the **LC-MRSG++** protocol, which enhances graph-based Knowledge Tracing (KT) through validation-guided relation gating, sparse-aware edge boosting, and leakage-controlled audits.

## Repository Structure

- `configs/q3_lcmrsg_plus.yaml`: Experiment configuration file.
- `scripts/q3_lcmrsg_plus_build_graphs.py`: Temporal-first graph construction and L1--L6 leakage checks.
- `scripts/q3_lcmrsg_plus_run_experiments.py`: Model training and validation-guided search (static selection, relation gating, sparse boosting).
- `scripts/q3_lcmrsg_plus_analyze.py`: Bootstrap confidence intervals, paired t-tests, Wilcoxon signed-rank tests, Cohen's d, Holm correction, and sparse-skill stratified summaries.
- `scripts/q3_lcmrsg_plus_render_tables.py`: Exporters for LaTeX tables A--E and publication-ready CSVs.
- `scripts/q3_lcmrsg_plus_export_appendix.py`: Exporters for publication figures (forest plots, heatmap, bar charts).
- `tests/`: Unit tests for verifying data split isolation and statistical helpers.

## Execution Guide

### 1. Run Sanity Checks
Ensure the configuration file and data structures are in place:
```bash
D:\scientific_paper1\miniconda3\envs\scientific_paper1\python.exe scripts/q3_lcmrsg_plus_sanity_checks.py
```

### 2. Run Unit Tests
Run the automated test suite:
```bash
D:\scientific_paper1\miniconda3\envs\scientific_paper1\python.exe -m pytest tests/
```

### 3. Run Build Graphs
Construct relation graphs and execute L1--L6 audits:
```bash
D:\scientific_paper1\miniconda3\envs\scientific_paper1\python.exe scripts/q3_lcmrsg_plus_build_graphs.py --config configs/q3_lcmrsg_plus.yaml --output_dir runs/q3_lcmrsg_plus_TIMESTAMP
```

### 4. Run Experiments
Execute model training and evaluation:
```bash
D:\scientific_paper1\miniconda3\envs\scientific_paper1\python.exe scripts/q3_lcmrsg_plus_run_experiments.py --config configs/q3_lcmrsg_plus.yaml --output_dir runs/q3_lcmrsg_plus_TIMESTAMP
```

### 5. Perform Statistical Analysis and Render LaTeX Tables/Figures
```bash
D:\scientific_paper1\miniconda3\envs\scientific_paper1\python.exe scripts/q3_lcmrsg_plus_analyze.py --run_dir runs/q3_lcmrsg_plus_TIMESTAMP
D:\scientific_paper1\miniconda3\envs\scientific_paper1\python.exe scripts/q3_lcmrsg_plus_render_tables.py --run_dir runs/q3_lcmrsg_plus_TIMESTAMP
D:\scientific_paper1\miniconda3\envs\scientific_paper1\python.exe scripts/q3_lcmrsg_plus_export_appendix.py --run_dir runs/q3_lcmrsg_plus_TIMESTAMP
```
