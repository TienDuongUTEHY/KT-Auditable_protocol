# Review of Project P0 against Design Document `P0_DUong.pdf`

Reference document: Roadmap P0 KTr-GK-CL v3.0 (specifically the experiments E0–E8, artifacts, baselines, and checklist figures/tables in the PDF).

**Scope:** Evaluate the gaps between the repository and the design; does not include modifying code in the session that created this file.

---

## 1. Mandatory Experiments (E0–E5)

### E0 — Dataset Statistics

**Design:** Junyi + ASSIST2012 + KDD Cup 2010; `#learners`, `#questions`, `#KC`, `#interactions`, average sequence length, and KC frequency distribution.

**Repository:** `preprocess` outputs `dataset_stats.csv` (Learners, Questions, Skills, Interactions, AvgSeqLen, QMatrixSource) — meets most requirements.

**Gaps / Needs Addition:**

- KC frequency distribution (table or histogram formatted for the paper) is not systematically exported as described in E0; a module to export this from `interactions.csv` / train split might be needed.
- `scripts/generate_publication_tables.py` builds Table 1 using `sparse_skill_summary` and estimates interactions approximately — deviates from the standard statistics in `dataset_stats.csv` (defined for Table 1 in the PDF).

---

### E1 — Five-type Leakage Audit

**Design:** No learner overlap; train-only graph edges for all three types; graph per fold; train-only co-occurrence; table for the 5 leakage types.

**Repository:** Has `leakage_audit_log.csv` + `leakage_audit_report.md`; splits by learner (no learner overlap between train/valid/test). L1/L5 based on `source_split` column; L2 based on Q-matrix source.

**Gaps / Logic Fixes (for subsequent implementation):**

- L3 (temporal) and L4 (cold-start) currently always PASS (placeholder) — does not reflect the spirit of "meaningful auditing" in the design.
- No explicit check to ensure test edges do not use information outside of train, nor cold-start neighborhood check according to the protocol definition (needs to adhere to the L1–L5 definitions in the full document).
- "No learner overlap" is guaranteed by the split logic, but it is not recorded as a check row in the leakage log (which the design emphasizes).

---

### E2 — Tri-relation Graph Audit

**Design:** `#E_pre`, `#E_sim`, `#E_co`, degree, density, components, cycles before/after pruning, audit table.

**Repository:** `graph_stats.csv` + `dag_report.md` / `dag_audit.csv` (cycles before/after pruning).

**Gaps:**

- `graph_statistics.py` does not output connected components (or equivalent per relation) — the design explicitly mentions "components".
- Some error rows in `graph_stats` when files are missing use raw file names — easy to confuse with standard rows (schema table needs to be cleaned up for the paper).

---

### E3 — E_co Quality (KDD Cup 2010 + Junyi)

**Design:** PMI/count distribution, symmetry pass, sparse-skill coverage by stratum.

**Repository:** `eco_audit.md` / `eco_audit.csv` (symmetry, train-only); `make_figures` outputs histogram/CDF of E_co weights.

**Gaps:**

- `eco_audit` does not yet integrate sparse-skill coverage by stratum as in E3; a unified artifact module/table is needed to match the stratum definition in the protocol.
- Count distribution (parallel to PMI) is not highlighted in the `eco_audit` artifact, unlike in the design description.

---

### E4 — Sparse-skill Diagnostic

**Design:** Frequency strata, graph coverage by stratum, baseline AUC/ACC by stratum.

**Repository:** `sparse_skill_profile.csv` / `.md` with strata split by p33/p66 percentiles (sparse / medium / dense), including Figure 3.

**Gaps:**

- `configs/global.yaml` defines bins (very_sparse, sparse, medium, frequent by count threshold) that are not used in `sparse_skill_profile.py` — protocol discrepancy between configuration and code.
- Lacks baseline results broken down by stratum (mandatory for E4 in PDF).
- Graph coverage per stratum has not been calculated or integrated into the profile (only skill/interaction statistics are present).

---

### E5 — Relation Ablation (1–2 datasets; simpleKT or DKT)

**Design:** `no_graph` vs `E_pre` vs `E_pre+E_sim` vs full; one modern strong baseline (simpleKT or AKT).

**Repository:** Has 4 ablation variants with DKT (actual) and BKT (logistic proxy). Has branches named SIMPLEKT / GKT / GIKT but they all run `run_bkt_proxy` — not actual simpleKT/GKT/GIKT models. No AKT model.

**Gaps (critical compared to E5 + section 7.2 of PDF):**

- Lacks a true strong baseline (such as pyKT simpleKT or AKT as suggested by the design).
- Default pipeline (`reproduce_one_dataset.sh`, `global.yaml`) only includes BKT + DKT — does not meet the "classic + strong" combo described in the PDF.
- Configuration has `baseline.smoke_mode` set with `max_epochs: 5` — only suitable for smoke tests, not sufficient for the mandatory ablation table if the venue requires rigorous training.

---

## 2. Optional Experiments (E6–E8)

| Section | Design | Repository Status |
|---|---|---|
| **E6** | DDR (Junyi, cite Tuan); line chart / heatmap | No DDR module or figures found. |
| **E7** | E_co threshold sensitivity (KDD); heatmap `k_co`, PMI vs count | No parameter sweeps or heatmaps found. |
| **E8** | Sanity GKT/GIKT with tri-graph | Model names are in CLI but they are not the actual GKT/GIKT models. |

---

## 3. Dataset per PDF (Section 7.1)

- **Mandatory:** ASSIST2012, Junyi, KDD2010 — repository has corresponding configurations.
- **Optional:** XES3G5M — no pipeline yet.
- **EdNet KT1:** Not mandatory — consistent with "not done".
- Repository adds **algebra2005**, **synthetic** — does not conflict with the PDF (internal extensions); should avoid confusing the story if the paper only commits to 3 main datasets.

---

## 4. Artifact Checklist (Section 8 of PDF) vs Repository

Most paths and file names already exist: `configs/*.yaml`, `split_hash.txt`, `qmatrix_provenance.md`, `edge_provenance.csv`, `leakage_audit_log.csv`, `graph_stats.csv`, `eco_audit.md`, `sparse_skill_profile.md`, `baseline_results.csv`, `reproduce_one_dataset.sh`, README (repository uses `README.MD` — casing difference compared to `readme.md` in PDF; usually does not affect GitHub).

**Quality of Artifact Content vs PDF Description:**

- **`qmatrix_provenance.md`:** Very minimalist (source + PASS/FAIL); the design suggests a train-only rule when Q is derived — needs expansion if Q has derivation logic from train.
- **`common.py`:** Mock file/DataFrame hash functions — if actual provenance relies on these, it will not meet the reproducibility / edge provenance with actual hashing requirements.
- **`run_planA_master.py`:** `PYTHON_PATH` is hardcoded for a specific machine — violates the spirit of a repository designed for supervisors/reviewers to rerun (PDF emphasizes config + log + minimal reproduction).

---

## 5. Target Figures & Tables (PDF List)

- **Figure 1** pipeline: Code has `fig1_pipeline.pdf` (overview + edge counts) — close to "overview"; graphics can be adjusted to fit the "leakage-controlled construction" story.
- **Table 1** dataset + graph availability: Need to synchronize the data sources (the publication script currently uses a sub-optimal source).
- **Table 2** leakage checklist: CSV logs exist, but paper-ready tables and actual checks (L3/L4) are needed.
- **Table 3** tri-relation audit: Has `graph_stats` + DAG — lacks components as mentioned.
- **Figure 2** E_co quality: Has weight distribution — lacks sparse coverage synchronized with E3.
- **Figure 3** sparse strata: Exists — lacks stratum-specific baselines.
- **Table 4** ablation: Has DKT/BKT data — lacks simpleKT/AKT baselines.
- **Figures 4–5** (DDR, sensitivity): Not yet in code as described in optional features.

---

## 6. Implementation Priorities (Proposal)

1. **E5 + §7.2:** Integrate simpleKT or AKT (pyKT) and run the ablation protocol; separate "diagnostic smoke" from "paper statistics".
2. **E4 + config:** Standardize the stratum definition (`sparse_skill` in YAML vs code); add stratum-specific metrics and graph coverage by stratum.
3. **E1:** Replace L3/L4 placeholders with checks following the protocol definition; explicitly log learner disjointness.
4. **E2:** Add connected components statistics (and compare before/after pruning if the protocol requires).
5. **E3:** Expand `eco_audit` with stratum-specific coverage and count distributions.
6. **Reproducibility:** Remove hardcoded Python path; use real provenance hashes; build Table 1 from `dataset_stats` + KC distribution.
7. **Optional E6–E7:** Implement DDR module (cite Tuan) and `k_co` grid search — if aligning with the complete roadmap.

---

## Notes

Evaluations are based on text extracted from `P0_DUong.pdf` in the workspace (some lines in the PDF had font extraction errors). If a complete PDF version detailing L1–L5, DDR, and cold-start is available, each PASS/FAIL condition should be re-evaluated to match the protocol closely.
