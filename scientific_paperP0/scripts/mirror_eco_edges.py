import pandas as pd
from pathlib import Path

IN_PATH = Path("outputs/diagnostics/eco_provenance_raw.csv")
OUT_PATH = Path("outputs/diagnostics/eco_provenance_fixed.csv")

# Ensure raw provenance exists by collecting from graph directories
if not IN_PATH.exists():
    IN_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw_edges = []
    base_dir = Path("graphs")
    for f in base_dir.glob("*/*/e_co.csv"):
        try:
            df_part = pd.read_csv(f)
            # rename fold_id to fold if necessary
            if "fold_id" in df_part.columns:
                df_part = df_part.rename(columns={"fold_id": "fold"})
            raw_edges.append(df_part)
        except Exception as e:
            pass
    if raw_edges:
        raw_df = pd.concat(raw_edges, ignore_index=True)
        raw_df.to_csv(IN_PATH, index=False)
        print(f"Generated raw file at {IN_PATH} with {len(raw_df)} rows")

df = pd.read_csv(IN_PATH)

# Chuẩn hóa tên cột nếu cần
rename_map = {
    "src": "src_skill",
    "dst": "dst_skill",
    "source": "src_skill",
    "target": "dst_skill",
}
df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

required = ["dataset", "fold", "src_skill", "dst_skill", "weight"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

if "support_source" in df.columns:
    df["train_only_support"] = (df["support_source"] == "train").astype(str)

for col, default in [
    ("support_count", -1),
    ("train_only_support", "UNKNOWN"),
    ("support_hash", "UNKNOWN"),
    ("edge_source", "train_cooccurrence"),
]:
    if col not in df.columns:
        df[col] = default

forward = df.copy()
forward["is_mirrored"] = False

reverse = df.copy()
reverse[["src_skill", "dst_skill"]] = reverse[["dst_skill", "src_skill"]]
reverse["is_mirrored"] = True

fixed = pd.concat([forward, reverse], ignore_index=True)
fixed = fixed.drop_duplicates(subset=["dataset", "fold", "src_skill", "dst_skill"], keep="first")
fixed.to_csv(OUT_PATH, index=False)

print("Raw edges:", len(df))
print("Fixed directed-storage edges:", len(fixed))
print(f"Saved: {OUT_PATH}")
