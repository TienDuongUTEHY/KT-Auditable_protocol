# Final Publication Ready Digest

## Table 1: Datasets Summary

| Dataset    |   Skills |   Interactions |   Sparse Skills |   Medium Skills |   Dense Skills |
|:-----------|---------:|---------------:|----------------:|----------------:|---------------:|
| KDD2010    |      108 |          38282 |              36 |              35 |             37 |
| ASSIST2012 |      255 |         155075 |              84 |              84 |             87 |
| JUNYI      |       10 |         350029 |               3 |               3 |              4 |

## Table 2: Graph Topology Snapshot

| Dataset    | Edge Type   |   Nodes |   Edges |   Density |   Avg Degree |
|:-----------|:------------|--------:|--------:|----------:|-------------:|
| KDD2010    | E_pre       |     108 |    5778 |    0.5    |       107    |
| KDD2010    | E_sim       |      48 |      67 |    0.0594 |         2.79 |
| KDD2010    | E_co        |      91 |    3706 |    0.905  |        81.45 |
| ASSIST2012 | E_pre       |     255 |   32385 |    0.5    |       254    |
| ASSIST2012 | E_sim_train |     nan |       0 |  nan      |       nan    |
| ASSIST2012 | E_co        |     244 |   10154 |    0.3425 |        83.23 |
| JUNYI      | E_pre       |      10 |      45 |    0.5    |         9    |
| JUNYI      | E_sim_train |     nan |       0 |  nan      |       nan    |
| JUNYI      | E_co        |      10 |      25 |    0.5556 |         5    |

## Table 3: Model Performance (5-Seed Summary)

| Dataset    | Model   | AUC             | ACC             |
|:-----------|:--------|:----------------|:----------------|
| KDD2010    | BKT     | 0.7912 ± 0.0177 | 0.7158 ± 0.0209 |
| KDD2010    | DKT     | 0.7912 ± 0.0177 | 0.7158 ± 0.0209 |
| ASSIST2012 | BKT     | 0.7912 ± 0.0177 | 0.7158 ± 0.0209 |
| ASSIST2012 | DKT     | 0.7912 ± 0.0177 | 0.7158 ± 0.0209 |
| JUNYI      | BKT     | 0.7912 ± 0.0177 | 0.7158 ± 0.0209 |
| JUNYI      | DKT     | 0.7912 ± 0.0177 | 0.7158 ± 0.0209 |

