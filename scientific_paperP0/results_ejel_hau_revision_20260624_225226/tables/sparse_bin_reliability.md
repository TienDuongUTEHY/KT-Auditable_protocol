# Sparse Bin Reliability Audit


| Dataset | Backbone | Bin | Num Skills | Test Interactions | AUC No | AUC Gated | Delta | Reliability |
|---|---|---|---|---|---|---|---|---|
| ASSIST2012 | DKT | <=50 | 112 | 371 | 0.6378 | 0.6504 | 0.0125 | Limited |
| ASSIST2012 | DKT | <=100 | 14 | 185 | 0.6366 | 0.6481 | 0.0115 | Limited |
| ASSIST2012 | DKT | <=200 | 26 | 710 | 0.6278 | 0.6271 | -0.0007 | Limited |
| ASSIST2012 | DKT | <=500 | 27 | 1844 | 0.6133 | 0.6154 | 0.0022 | Reliable |
| ASSIST2012 | DKT | >500 | 76 | 26220 | 0.6127 | 0.615 | 0.0023 | Reliable |
| ASSIST2012 | SIMPLEKT | <=50 | 112 | 371 | 0.692 | 0.692 | 0.0 | Limited |
| ASSIST2012 | SIMPLEKT | <=100 | 14 | 185 | 0.6622 | 0.6622 | 0.0 | Limited |
| ASSIST2012 | SIMPLEKT | <=200 | 26 | 710 | 0.6177 | 0.6177 | 0.0 | Limited |
| ASSIST2012 | SIMPLEKT | <=500 | 27 | 1844 | 0.6345 | 0.6345 | 0.0 | Reliable |
| ASSIST2012 | SIMPLEKT | >500 | 76 | 26220 | 0.6184 | 0.6184 | 0.0 | Reliable |
| JUNYI | DKT | <=50 | 0 | 0 | NA | NA | NA | Insufficient |
| JUNYI | DKT | <=100 | 0 | 0 | NA | NA | NA | Insufficient |
| JUNYI | DKT | <=200 | 1 | 13 | NA | NA | NA | Insufficient |
| JUNYI | DKT | <=500 | 2 | 138 | 0.7582 | 0.7835 | 0.0253 | Limited |
| JUNYI | DKT | >500 | 7 | 67097 | 0.6705 | 0.6707 | 0.0002 | Reliable |
| JUNYI | SIMPLEKT | <=50 | 0 | 0 | NA | NA | NA | Insufficient |
| JUNYI | SIMPLEKT | <=100 | 0 | 0 | NA | NA | NA | Insufficient |
| JUNYI | SIMPLEKT | <=200 | 1 | 13 | NA | NA | NA | Insufficient |
| JUNYI | SIMPLEKT | <=500 | 2 | 138 | 0.7829 | 0.7829 | 0.0 | Limited |
| JUNYI | SIMPLEKT | >500 | 7 | 67097 | 0.6719 | 0.6719 | 0.0 | Reliable |
