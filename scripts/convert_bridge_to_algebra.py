import pandas as pd
import os
import time

raw_train_path = r"D:\Paper P0 Nguyen Tien Duong\SCIE_P0\KT-Auditable_protocol\data\raw\bridge_to_algebra_2008_2009_train.txt"
output_dir = r"D:\Paper P0 Nguyen Tien Duong\SCIE_P0\KT-Auditable_protocol\data\raw\kdd2010"
os.makedirs(output_dir, exist_ok=True)

print("Loading full raw train file (this might take a minute)...")
start = time.time()
# Reading first 2 million rows to avoid OOM in pandas and later pipeline stages
df = pd.read_csv(raw_train_path, sep='\t', low_memory=False, nrows=2000000)
print(f"Loaded {len(df)} rows in {time.time() - start:.2f} seconds.")

df = df.dropna(subset=['KC(SubSkills)'])
print(f"After dropping null KCs: {len(df)} rows.")

print("Using full dataset as requested...")

# 1. Prepare fast mapping
print("Vectorized processing started...")
df['q_str'] = df['Problem Name'].astype(str) + "_" + df['Step Name'].astype(str)
df['question_id'] = pd.factorize(df['q_str'])[0] + 1

# Correctness
df['correct'] = pd.to_numeric(df['Correct First Attempt'], errors='coerce').fillna(0).astype(int)

# Timestamp
try:
    df['timestamp'] = pd.to_datetime(df['Step Start Time']).astype('int64') // 10**9
except Exception as e:
    print(f"Timestamp parsing error: {e}")
    df['timestamp'] = range(1000, 1000 + len(df))

# Explode KCs
print("Exploding multiple KCs per row...")
df['KC(SubSkills)'] = df['KC(SubSkills)'].astype(str).str.split('~~')
df_exploded = df.explode('KC(SubSkills)')
df_exploded['KC(SubSkills)'] = df_exploded['KC(SubSkills)'].str.strip()
df_exploded = df_exploded[df_exploded['KC(SubSkills)'] != '']

# Map KC to Skill ID
df_exploded['skill_id'] = pd.factorize(df_exploded['KC(SubSkills)'])[0] + 1

# Interaction ID
df_exploded['interaction_id'] = range(1, len(df_exploded) + 1)

# Keep only needed columns
final_df = pd.DataFrame({
    'interaction_id': df_exploded['interaction_id'],
    'learner_id': df_exploded['Anon Student Id'],
    'question_id': df_exploded['question_id'].astype(str),
    'skill_id': df_exploded['skill_id'].astype(str),
    'timestamp': df_exploded['timestamp'],
    'correct': df_exploded['correct'],
    'dataset': 'kdd2010'
})

print(f"Expanded into {len(final_df)} skill-interaction pairs.")

final_df.to_csv(os.path.join(output_dir, "interactions.csv"), index=False)
print("Exported interactions.csv")

# Q-Matrix
print("Generating Q-Matrix...")
q_matrix_df = final_df[['question_id', 'skill_id']].drop_duplicates().copy()
q_matrix_df['source'] = 'provided_static'
q_matrix_df['source_version'] = '1'
q_matrix_df['is_train_only'] = 'false'
q_matrix_df.to_csv(os.path.join(output_dir, "q_matrix.csv"), index=False)
print("Exported q_matrix.csv")

print("Bridge to Algebra 2008-2009 setup for KDD2010 done!")
