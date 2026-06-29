# Science Paper P0 — Multi-Relational Graphs & Knowledge Tracing

Source code supporting arguments and experiments for the topic of **constructing multi-relational skill graphs with controlled data leakage**, serving **Knowledge Tracing** when key concepts (KC) are **sparse**.

Internal configuration name in `configs/global.yaml`: `KT-Auditable_protocol`; title in the same file: **"Leakage-Controlled Multi-Relational Skill Graph Construction for Sparse-Skill Knowledge Tracing"**.

---

## Project Goals

- **Preprocessing and data splitting** by learner and chronological order (`learner_based_temporal`), with train/valid/test split ratios configured in `configs/global.yaml`.
- **Three types of skill relationships** are estimated solely from the **training set** (reducing leakage):
  - **E_pre** — prerequisite / suggested teaching sequence,
  - **E_sim** — similarity (e.g., Jaccard index on shared `question_id`s),
  - **E_co** — co-occurrence (e.g., smoothed positive PMI).
- **Auditing and Reporting**: Q-matrix provenance, DAG validation, "eco" co-occurrence report, **leakage auditing (L1–L5)**, graph statistics, "sparse skill" profiling, and baseline evaluation with **relation ablation** variants (`no_graph`, `E_pre`, etc.).
- **Baselines**: BKT, DKT (PyTorch); with a smoke test mode supported in the global configuration.

---

## Directory Structure (Summary)

| Directory / File | Role / Purpose |
|------------------|----------------|
| `configs/global.yaml` | Global settings: seeds, data splitting, graph parameters, sparse bins, baseline, and audit. |
| `configs/*.yaml` | Dataset-specific settings (e.g., `kdd2010.yaml`, `assist2012.yaml`, `junyi.yaml`, `algebra2005.yaml`, `synthetic.yaml`). |
| `src/` | Python pipeline (preprocessing, split, Q-matrix, graph construction, auditing, baseline, reports, figures). |
| `scripts/` | Shell scripts for reproduction (`reproduce_one_dataset.sh`), python runner (`run_planA_master.py`), etc. |
| `data/raw/`, `data/processed/` | Raw and processed datasets (usually ignored in git — need to prepare according to README of each dataset). |
| `results/` | CSV files, Markdown reports, and publication-ready artefacts (`results/paper_ready/`). |
| `logs/` | Execution logs (`run_log_*`). |
| `P0_Master_Diagnostic_Report.md` | Aggregated report example (leakage, ablation, eco, etc.). |

---

## Environment Requirements

- **Python** 3.x (the interpreter path in `scripts/run_planA_master.py` is currently hardcoded for a specific Windows environment — please update it to your python path or call `python -m ...` as in other scripts).
- Core libraries (refer to `scripts/run_planA_master.py` and `src/` code):
  - `numpy`, `pandas`, `PyYAML`
  - `torch`, `scikit-learn`
  - `matplotlib`, `seaborn`, `tqdm`
  - Depending on stage: `openml`, `tabulate`

Installation example:

```bash
pip install numpy pandas pyyaml torch scikit-learn matplotlib seaborn tqdm openml tabulate
```

---

## Quick Start (Single Dataset, Single Seed)

From the project root directory, run on Bash (Git Bash/WSL/Linux):

```bash
bash scripts/reproduce_one_dataset.sh <dataset_name> <seed>
# Example:
bash scripts/reproduce_one_dataset.sh kdd2010 2026
```

On PowerShell, you can run the equivalent steps (same order as in `scripts/reproduce_one_dataset.sh`):
`python -m src.preprocess --config configs/<dataset>.yaml`, followed by `split_checker`, `qmatrix_provenance`, `tri_relation_graph_builder`, audits, etc.

---

## Automated Run (Multi-Seed / Multi-Dataset)

- `scripts/run_planA_master.py`: Loops over datasets and seeds (need to adjust `PYTHON_PATH`, `DATASETS`, `SEEDS` to match your system).
- Other helper shells in `scripts/`: `run_mini_test.sh`, `run_synthetic_minimal.sh`.

---

## Exporting Publication Tables

The script `scripts/generate_publication_tables.py` and the `results/paper_ready/` directory are used to aggregate results for the paper manuscript.

---

## Notes

- Configuration merging: `src.common.load_config` merges `configs/global.yaml` with the dataset-specific configuration.
- Detailed reports for each dataset/fold are located in `results/reports/<dataset>/fold_0/` (DAG, leakage, eco, diagnostics, etc.).

---

## Authors / Citation

Add author info, version, and paper citation once the final manuscript is accepted.
