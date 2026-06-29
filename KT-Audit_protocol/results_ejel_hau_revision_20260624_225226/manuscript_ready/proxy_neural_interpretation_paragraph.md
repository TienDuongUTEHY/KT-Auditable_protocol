### Proxy and Neural KT Headroom Reanalysis

Predictive headrooms differ substantially between the traditional linear KT proxies (LR-KT proxy) and deep neural KT architectures (DKT, simpleKT). 
Traditional models often exhibit larger relative gains (delta AUC) when augmented with graph degree mappings, but this is a side-effect of their lower baseline performance. 
When evaluating high-capacity neural models, the delta AUC narrows, demonstrating that deep KT architectures already extract sequence representations that overlap with topological graph features. 
Therefore, all proxy metrics are reported strictly for diagnostic sanity checking and kept separate from neural KT results.
