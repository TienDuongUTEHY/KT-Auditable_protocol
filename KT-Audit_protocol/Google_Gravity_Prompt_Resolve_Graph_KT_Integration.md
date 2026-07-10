# Google Gravity Task Prompt  
## Resolve the graph→KT integration mechanism in LC-MRSG without inventing methods or altering validated results

### 1. Role

Act as a senior machine-learning engineer, knowledge-tracing researcher, reproducibility auditor, and academic editor.

Your task is to inspect the full LC-MRSG project, determine exactly how the graph-derived knowledge-component representation \(h_c\) is integrated into DKT and simpleKT, and then update the manuscript and Supplementary Materials so that the method is fully reproducible.

Do **not** invent an architecture, formula, hyperparameter, or training procedure that is not implemented in the code used to generate the reported results.

---

## 2. Primary objective

Resolve the unresolved manuscript text:

> `[TODO: specify the graph→KT integration mechanism — how h_c is fused into DKT/simpleKT.]`

The final manuscript must explain, with code-grounded evidence:

1. How node or knowledge-component representations are initialized.
2. How graph representations \(h_c\) are computed.
3. How multiple relation types are combined.
4. How \(h_c\) is projected, normalized, gated, concatenated, or added to the ordinary KC embedding.
5. Where the fused representation enters DKT.
6. Where the fused representation enters simpleKT.
7. Which parameters are trainable.
8. Which graph artefacts are frozen.
9. How the no-graph baseline is implemented.
10. Whether graph and no-graph candidates are parameter-count comparable.
11. Whether the reported results were produced by the mechanism found in the code.
12. Whether any experiment must be rerun.

---

## 3. Non-negotiable constraints

### 3.1. Do not fabricate

Do not assume that the model uses:

- additive fusion;
- concatenation;
- gated fusion;
- LayerNorm;
- R-GCN;
- GCN;
- GAT;
- GraphSAGE;
- relation attention;
- shared embeddings;
- frozen graph encoders;
- end-to-end graph training.

Use these terms only when confirmed by code, configuration files, run scripts, checkpoints, logs, or result manifests.

### 3.2. Preserve the manuscript

Do not rewrite unrelated sections.

Keep all original claims, tables, figures, citations, and numerical results unchanged unless code inspection proves that a reported statement is incorrect or the experiments must be rerun.

### 3.3. Do not silently rerun experiments

First complete the audit.

Only rerun experiments when the decision rules in Section 10 require it.

Before rerunning, create a written audit report explaining why the existing results are no longer valid.

### 3.4. Maintain double-blind review

Do not insert:

- author names;
- personal GitHub URLs;
- institutional identifiers not already present;
- local machine usernames;
- absolute private file paths.

### 3.5. Mathematical notation

Use valid LaTeX notation.

Define every symbol when first introduced.

Do not add equations that describe a mechanism different from the implementation.

---

## 4. Files and directories to inspect

Search the entire repository recursively.

Prioritize files matching patterns such as:

```text
**/*dkt*.py
**/*simplekt*.py
**/*simple_kt*.py
**/*graph*.py
**/*encoder*.py
**/*relation*.py
**/*fusion*.py
**/*model*.py
**/*module*.py
**/*train*.py
**/*runner*.py
**/*experiment*.py
**/*config*.yaml
**/*config*.yml
**/*config*.json
**/*.toml
**/*.ini
**/*.csv
**/*.log
**/*.md
**/*.tex
**/*.docx
**/*manifest*
**/*result*
**/*checkpoint*
```

Also inspect:

- model constructors;
- forward functions;
- embedding layers;
- graph-loading utilities;
- relation-selection logic;
- graph cache and frozen graph files;
- optimizer parameter registration;
- early-stopping code;
- no-graph candidate definitions;
- DKT input construction;
- simpleKT embedding construction;
- all YAML/JSON experiment configurations;
- saved run metadata;
- result-generation scripts.

If both LaTeX and DOCX versions exist, treat the LaTeX source as the primary editable scientific source unless the project explicitly identifies DOCX as authoritative.

---

## 5. Required audit procedure

### Phase A — Locate the implementation

Identify the exact source files and functions responsible for:

1. Graph construction.
2. Graph loading.
3. Graph encoding.
4. Relation-wise message passing.
5. Relation aggregation.
6. Fusion with the KC embedding.
7. DKT forward computation.
8. simpleKT forward computation.
9. No-graph execution.
10. Candidate selection.
11. Training and early stopping.
12. Test evaluation.

For each component, record:

| Component | File | Class/function | Line range | Evidence |
|---|---|---|---|---|
| Graph encoder |  |  |  |  |
| Relation aggregation |  |  |  |  |
| KC fusion |  |  |  |  |
| DKT integration |  |  |  |  |
| simpleKT integration |  |  |  |  |
| No-graph baseline |  |  |  |  |
| Frozen/trainable status |  |  |  |  |

Use exact file paths and line numbers in the audit report.

---

### Phase B — Reconstruct the implemented equations

Translate the code into mathematical notation.

Use only the branch that matches the implementation.

#### Case 1: Additive fusion

Use this form only if confirmed:

\[
g_c = W_g h_c + b_g,
\]

\[
\widetilde e_c = e_c + g_c,
\]

or, if normalization is implemented:

\[
\widetilde e_c = \operatorname{LayerNorm}(e_c + g_c).
\]

#### Case 2: Concatenation fusion

Use this form only if confirmed:

\[
\widetilde e_c =
\phi\!\left(W_f[e_c\Vert h_c]+b_f\right).
\]

State the actual activation \(\phi\), or omit it if no activation is used.

#### Case 3: Scalar or vector gate

Use this form only if confirmed:

\[
\alpha_c =
\sigma\!\left(W_\alpha[e_c\Vert g_c]+b_\alpha\right),
\]

\[
\widetilde e_c =
e_c+\alpha_c\odot g_c.
\]

#### Case 4: Relation-wise aggregation

If relation-specific representations are implemented, define:

\[
h_c^{(r)} = \operatorname{Enc}_r(c;G_r),
\]

and reproduce the actual aggregation rule, for example:

\[
h_c = \sum_{r\in\mathcal R}\alpha_{c,r}h_c^{(r)},
\]

only if the code implements it.

If the code uses a simple sum, mean, maximum, concatenation, fixed coefficient, or learned gate, write the exact implemented rule.

#### Case 5: No graph encoder exists

If \(h_c\) is not produced by a neural graph encoder but is instead:

- a precomputed vector;
- a neighbor average;
- a degree/statistical feature;
- a graph-derived lookup;
- a manually aggregated embedding;

state that clearly and do not call it a graph neural network.

---

### Phase C — Verify DKT integration

Determine the exact input to DKT.

Check whether the model uses:

\[
x_t=[\widetilde e_{c_t}\Vert e_{r_t}],
\]

a one-hot interaction representation, a question-response embedding, or another construction.

Document:

- input dimensions;
- whether the response is embedded or encoded jointly;
- whether \(h_c\) is injected before the LSTM, after the LSTM, or at the prediction head;
- whether the same fusion parameters are shared across time;
- whether graph features affect only the current KC or all output KCs.

Do not use the generic LSTM equation unless it accurately describes the implementation.

---

### Phase D — Verify simpleKT integration

Determine exactly where the graph representation enters simpleKT.

Inspect whether it modifies:

- question embedding;
- concept embedding;
- question–concept embedding;
- response embedding;
- difficulty embedding;
- positional embedding;
- query/key/value construction;
- attention output;
- prediction head.

Document all simpleKT components left unchanged.

Do not write that the graph “replaces the concept embedding” unless the code does that.

---

### Phase E — Verify trainable and frozen components

Create a complete parameter-status table:

| Component | Trainable? | Frozen? | Evidence |
|---|---:|---:|---|
| Graph topology |  |  |  |
| Edge weights |  |  |  |
| Node embedding |  |  |  |
| Graph encoder |  |  |  |
| Projection layer |  |  |  |
| Fusion gate |  |  |  |
| DKT backbone |  |  |  |
| simpleKT backbone |  |  |  |
| Prediction head |  |  |  |

Verify parameter registration by checking the optimizer input and `requires_grad`.

---

### Phase F — Verify the no-graph baseline

Determine whether no-graph is implemented as:

1. \(g_c=\mathbf 0\);
2. graph module bypass;
3. a separate model class;
4. empty edge set;
5. identity graph;
6. disabled relation mask;
7. another mechanism.

Check whether the no-graph condition has fewer parameters.

If graph candidates add trainable parameters but no-graph removes them, determine whether the comparison is still methodologically acceptable.

Report:

- parameter count for each candidate;
- trainable parameter count;
- whether the prediction head and backbone dimensions remain identical;
- whether any dummy projection is retained in the no-graph condition.

Do not modify the architecture solely to equalize parameters unless a rerun is approved by the decision rules.

---

## 6. Required implementation-details table

Create a new Supplementary table titled:

> **Implementation details for graph-to-backbone integration**

Use fields supported by the code:

| Item | DKT | simpleKT |
|---|---|---|
| Backbone input representation |  |  |
| KC embedding dimension |  |  |
| Graph representation dimension |  |  |
| Graph encoder |  |  |
| Number of graph layers |  |  |
| Relation aggregation |  |  |
| Fusion mechanism |  |  |
| Fusion location |  |  |
| Normalization |  |  |
| Dropout |  |  |
| Trainable graph parameters |  |  |
| Frozen graph artefacts |  |  |
| No-graph implementation |  |  |
| Optimizer |  |  |
| Learning rate |  |  |
| Batch size |  |  |
| Maximum sequence length |  |  |
| Early-stopping metric |  |  |
| Early-stopping patience |  |  |

Do not leave guessed values.

Use `Not applicable`, `Not implemented`, or `Not recoverable from the current artefacts` when appropriate.

---

## 7. Manuscript modification

### 7.1. Replace the TODO paragraph

Find the paragraph beginning:

> **Integration into the KT backbone.**

Replace only the unresolved TODO and the minimum surrounding sentences needed for grammatical consistency.

The revised paragraph must include:

1. Definition of \(h_c\).
2. Relation aggregation.
3. Fusion equation.
4. DKT integration point.
5. simpleKT integration point.
6. Trainable/frozen distinction.
7. No-graph definition.
8. Scope limitation.

Use the following structure, but replace all bracketed fields with code-grounded content:

```text
Integration into the KT backbones. For each candidate relation set, [GRAPH ENCODER OR
GRAPH-DERIVED PROCEDURE] produces a representation \(h_c\in\mathbb{R}^{d_g}\) for each
train-mapped knowledge component \(c\). [EXPLAIN RELATION-SPECIFIC REPRESENTATIONS AND
THE ACTUAL AGGREGATION RULE]. The graph topology [AND EDGE WEIGHTS, IF TRUE] is frozen
after train-only construction, whereas [LIST ACTUAL TRAINABLE COMPONENTS] are optimized
on the training partition.

The graph representation is fused with the ordinary knowledge-component embedding using
[ACTUAL FUSION MECHANISM]:
[INSERT EXACT LATEX EQUATION(S)].

For DKT, [EXACT DKT INTEGRATION DESCRIPTION AND EQUATION]. For simpleKT, [EXACT
SIMPLEKT INTEGRATION DESCRIPTION]. [LIST THE BACKBONE COMPONENTS THAT REMAIN
UNCHANGED].

For the no-graph candidate, [EXACT NO-GRAPH IMPLEMENTATION]. All candidates use the same
data splits, sequence construction, training schedule, early-stopping criterion, and prediction
target. Accordingly, the reported performance conclusions are conditional on this integration
mechanism and are not claimed to generalize to alternative graph-to-backbone fusion designs.
```

Do not mention “R-GCN”, “LayerNorm”, “gate”, “projection”, or “joint optimization” unless verified.

---

### 7.2. Add a Supplementary subsection

Add a subsection after the fold-level split statistics or at another logically appropriate location:

> **Graph-to-backbone integration and implementation details**

This subsection should contain:

- one concise explanatory paragraph;
- the implementation-details table;
- parameter-count information;
- a precise no-graph definition;
- links to anonymous code paths or repository-relative paths, not author-identifying URLs.

Renumber later Supplementary sections and tables consistently.

---

### 7.3. Update cross-references

Update every affected reference:

- Supplementary section numbers;
- Supplementary table numbers;
- in-text references to per-run tables;
- captions;
- contents list, if present;
- labels and `\ref{}` commands in LaTeX.

Do not leave:

```text
TODO
??
Table ??
Figure ??
Appendix ??
Section ??
```

Run a repository-wide search before completion.

---

## 8. Consistency audit against reported results

For every reported run, verify that the stored configuration corresponds to the integration mechanism described in the revised manuscript.

Check at least:

- dataset;
- backbone;
- fold;
- seed;
- candidate;
- relation set;
- fusion setting;
- graph encoder;
- graph dimension;
- early stopping;
- checkpoint selection;
- test evaluation.

Create a machine-readable file:

```text
artifacts/integration_audit/run_configuration_consistency.csv
```

Required columns:

```text
dataset
backbone
fold
seed
candidate
result_file
config_file
checkpoint_file
graph_encoder
relation_aggregation
fusion_type
fusion_location
graph_dim
backbone_dim
early_stopping_metric
consistent_with_manuscript
issue
```

If an item cannot be recovered, mark it as `unknown`; do not infer it.

---

## 9. Parameter-count audit

Create:

```text
artifacts/integration_audit/parameter_counts.csv
```

Columns:

```text
backbone
candidate
total_parameters
trainable_parameters
graph_parameters
fusion_parameters
backbone_parameters
prediction_head_parameters
notes
```

Generate counts by instantiating models using the original configurations without training.

Also produce:

```text
artifacts/integration_audit/parameter_count_summary.md
```

Explain whether graph/no-graph differences affect the interpretation.

Do not claim strict architectural fairness if parameter counts differ materially.

---

## 10. Decision rules for additional experiments

### Outcome A — No rerun required

Do not rerun experiments when all conditions hold:

- the integration mechanism is clearly implemented;
- it is identical to the mechanism used for the reported runs;
- result configurations and checkpoints are consistent;
- DKT and simpleKT truly receive graph-derived representations;
- no-graph behavior is correctly documented;
- manuscript changes are descriptive only;
- no code changes affect model output.

Required action:

- update manuscript;
- update Supplementary;
- add implementation table;
- add audit reports;
- preserve all numerical results.

### Outcome B — Limited verification run required

Run a small verification only when:

- the implementation is clear but old run metadata is incomplete;
- checkpoints exist but configuration provenance must be confirmed;
- deterministic forward equivalence needs testing;
- no architecture change is needed.

Required verification:

- one dataset;
- one fold;
- one seed;
- DKT and simpleKT;
- no graph and one graph candidate.

Purpose:

- verify that the reconstructed configuration reproduces the stored output within numerical tolerance;
- do not replace main results unless mismatch is found.

Suggested tolerance:

```text
AUC absolute difference <= 1e-4
```

Record environment and random-seed details.

### Outcome C — Full rerun required

A full rerun is mandatory when any condition holds:

- the manuscript description would require changing code;
- the existing code does not use \(h_c\) for one or both backbones;
- the fusion mechanism differs across runs without documentation;
- graph candidates were generated by multiple incompatible implementations;
- result files cannot be linked to configurations;
- no-graph and graph branches contain a bug affecting predictions;
- a projection, gate, normalization, dimension, or insertion point must be changed;
- candidate selection used a different implementation from final test evaluation;
- test information influenced integration hyperparameter selection;
- reported tables cannot be reproduced from stored artefacts.

Full rerun scope:

```text
3 datasets
2 backbones
3 folds
3 seeds
all pre-specified candidates
validation-only selection
one final test evaluation after selection
```

Update at minimum:

- main validation-selection frequency table;
- main early-stopping effect table;
- sparse-bin diagnostic figure;
- no-\(E_{\mathrm{pre}}\) sensitivity table;
- Supplementary validation-selection logs;
- Supplementary best-available-graph table;
- Supplementary early-stopping comparison;
- Supplementary no-\(E_{\mathrm{pre}}\) checks;
- all derived confidence intervals and corrected p-values.

Do not run an add-vs-concat-vs-gate architecture study unless explicitly requested.

---

## 11. Required code tests

Add non-destructive tests under an appropriate test directory.

### Test 1 — No-graph equivalence

Verify that the no-graph branch produces the same fused KC representation as the ordinary KC embedding, according to the actual implementation.

### Test 2 — Graph contribution

Verify that a nonempty graph candidate can alter the KC representation.

### Test 3 — Shape consistency

Verify that fused representations have the dimensions expected by DKT and simpleKT.

### Test 4 — Frozen topology

Verify that graph topology and precomputed edge weights are not modified during training.

### Test 5 — Trainable parameter registration

Verify that all intended graph/fusion parameters appear in the optimizer and unintended parameters do not.

### Test 6 — Candidate isolation

Verify that switching a relation candidate changes only the intended relation inputs and does not change data splits, response labels, or test access.

### Test 7 — Test-set isolation

Verify that graph integration hyperparameters and candidate selection do not read test metrics.

Do not alter the scientific model merely to make tests pass.

---

## 12. Required outputs

Create the following files:

```text
artifacts/integration_audit/graph_kt_integration_audit.md
artifacts/integration_audit/run_configuration_consistency.csv
artifacts/integration_audit/parameter_counts.csv
artifacts/integration_audit/parameter_count_summary.md
artifacts/integration_audit/rerun_decision.md
artifacts/integration_audit/manuscript_replacement_text.md
artifacts/integration_audit/supplementary_integration_section.md
artifacts/integration_audit/search_for_unresolved_placeholders.txt
```

If manuscript files are editable in the repository, also create revised versions without overwriting originals:

```text
revised/main_blinded_integration_resolved.*
revised/supplementary_integration_resolved.*
```

Use the original file extension.

If the authoritative manuscript is DOCX, preserve:

- A4 paper size;
- 2.54 cm margins;
- EJEL formatting;
- editable text;
- editable Word equations where supported;
- existing tables and figures;
- all original content outside the approved changes.

If automatic DOCX editing is unsafe, update the LaTeX/source version and generate a precise change file rather than corrupting the DOCX.

---

## 13. Structure of the audit report

The file `graph_kt_integration_audit.md` must contain:

### 13.1. Executive conclusion

State one of:

- `No rerun required`
- `Limited verification run required`
- `Full rerun required`

Provide a concise reason.

### 13.2. Located implementation

List exact files, classes, functions, and line numbers.

### 13.3. Implemented graph representation

Explain how \(h_c\) is produced.

### 13.4. Relation aggregation

Explain how \(E_{\mathrm{pre}}\), \(E_{\mathrm{sim}}\), and \(E_{\mathrm{co}}\) are handled.

### 13.5. Fusion mechanism

Provide exact code-grounded equations.

### 13.6. DKT integration

Explain the input and insertion point.

### 13.7. simpleKT integration

Explain the input and insertion point.

### 13.8. Trainable versus frozen components

Provide the parameter-status table.

### 13.9. No-graph implementation

Explain exact behavior and parameter-count implications.

### 13.10. Consistency with reported runs

Summarize the CSV audit.

### 13.11. Manuscript changes

Show the old TODO paragraph and final replacement.

### 13.12. Supplementary changes

Show the new subsection and table.

### 13.13. Experiment decision

Apply the rules in Section 10.

### 13.14. Remaining risks

List unresolved evidence gaps.

---

## 14. Final validation checklist

Before declaring completion, verify:

- [ ] The TODO has been removed.
- [ ] The manuscript explains how \(h_c\) is produced.
- [ ] The fusion equation matches the implementation.
- [ ] DKT integration is explicit.
- [ ] simpleKT integration is explicit.
- [ ] Relation aggregation is explicit.
- [ ] Trainable and frozen components are distinguished.
- [ ] No-graph behavior is explicit.
- [ ] Parameter counts are reported.
- [ ] The Supplementary implementation table is complete.
- [ ] No unsupported architecture term was introduced.
- [ ] Every numerical result was preserved unless a rerun was required.
- [ ] All section and table references are valid.
- [ ] No unresolved `TODO`, `??`, or broken cross-reference remains.
- [ ] Double-blind information is preserved.
- [ ] The rerun decision is justified with evidence.
- [ ] Original manuscript files remain available unchanged.
- [ ] Revised files are saved separately.
- [ ] All tests pass.
- [ ] The audit report contains source-file and line-level evidence.

---

## 15. Final response format

At the end of the task, return:

```text
STATUS:
[No rerun required / Limited verification run required / Full rerun required]

IMPLEMENTED MECHANISM:
[One-paragraph technical summary]

FILES MODIFIED:
[List]

FILES CREATED:
[List]

MAIN MANUSCRIPT CHANGE:
[Exact replacement paragraph]

SUPPLEMENTARY CHANGE:
[Section and table added]

EXPERIMENT DECISION:
[Decision and evidence]

UNRESOLVED RISKS:
[List or None]
```

Do not report success unless the implementation has been traced to code and all unresolved placeholders have been checked.
