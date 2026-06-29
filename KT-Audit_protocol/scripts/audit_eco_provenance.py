import pandas as pd
from pathlib import Path

PATH = Path("outputs/diagnostics/eco_provenance_fixed.csv")
OUT = Path("outputs/diagnostics/eco_provenance_fixed_audit.csv")
df = pd.read_csv(PATH)

rows = []
for (dataset, fold), g in df.groupby(["dataset", "fold"]):
    pairs = set(zip(g["src_skill"], g["dst_skill"]))
    missing_reverse = 0
    for s, t in pairs:
        if (t, s) not in pairs and s != t:
            missing_reverse += 1

    train_only_ok = (g["train_only_support"].astype(str).str.upper() == "TRUE").mean()
    unknown_support = (g["train_only_support"].astype(str).str.upper() == "UNKNOWN").sum()

    status = "PASS" if missing_reverse == 0 and unknown_support == 0 else "FLAG"

    rows.append({
        "dataset": dataset,
        "fold": fold,
        "edges": len(g),
        "missing_reverse_edges": missing_reverse,
        "unknown_train_only_support_rows": int(unknown_support),
        "train_only_support_true_rate": train_only_ok,
        "status": status,
    })

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print(out)
print(f"Saved: {OUT}")
