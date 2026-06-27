# Two-Epoch vs Early-Stopping Stability Analysis


| Dataset | Backbone | Mean Delta 2-Ep | 2-Ep CI | Mean Delta ES | ES CI | Stability |
|---|---|---|---|---|---|---|
| ASSIST2012 | DKT | 0.0001 | [-0.0015, 0.0017] | 0.0008 | [-0.0013, 0.0027] | near_zero_unstable |
| ASSIST2012 | SIMPLEKT | 0.0029 | [0.0005, 0.0055] | 0.0 | [0.0000, 0.0000] | sign_changed |
| JUNYI | DKT | 0.0001 | [-0.0004, 0.0008] | 0.0006 | [0.0000, 0.0013] | near_zero_unstable |
| JUNYI | SIMPLEKT | -0.0005 | [-0.0008, -0.0001] | 0.0 | [0.0000, 0.0000] | sign_changed |
