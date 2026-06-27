# Reviewer Response Notes — P0 Revision

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
