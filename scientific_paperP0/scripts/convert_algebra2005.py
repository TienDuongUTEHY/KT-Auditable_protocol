import pandas as pd
import os

raw_train_path = r"d:\Paper P0 Nguyen Tien Duong\Paper_P0\temp_data\train.csv"
output_dir = r"d:\Paper P0 Nguyen Tien Duong\Paper_P0\scientific_paperP0\data\raw\algebra2005"
os.makedirs(output_dir, exist_ok=True)

print("Loading raw train file...")
# Note: File seems tab-separated OR has a tab separator. Let's try reading with tab separator first.
df = pd.read_csv(raw_train_path, sep='\t')
print(f"Loaded {len(df)} rows.")

# In KDD2010 format, multiple KCs are separated by "~~"
# Drop rows without KCs? Often necessary for Knowledge Tracing.
df = df.dropna(subset=['KC(Default)'])
print(f"After dropping null KCs: {len(df)} rows.")

# Take a subset of say 50k rows to ensure very fast execution while retaining high statistical validity
# User wants medium test pace.
if len(df) > 50000:
    print("Subsetting to 50,000 rows for consistent medium-test performance.")
    df = df.head(50000).copy()

# 1. Interaction mapping
df['interaction_id'] = range(1, len(df) + 1)

# 2. Correctness mapping
df['correct'] = df['Correct First Attempt'].astype(int)

# 3. Timestamp conversion
try:
    df['timestamp_dt'] = pd.to_datetime(df['Step Start Time'])
    df['timestamp'] = (df['timestamp_dt'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
except Exception as e:
    print("Timestamp parse failed, defaulting to incremental order:", e)
    df['timestamp'] = range(1000, 1000 + len(df))

# 4. Prepare Q-Matrix extraction
skill_map = {}
skill_counter = 1
question_map = {}
question_counter = 1

processed_rows = []
qmatrix_entries = set()

for i, row in df.iterrows():
    q_str = f"{row['Problem Name']}_{row['Step Name']}"
    if q_str not in question_map:
        question_map[q_str] = str(question_counter)
        question_counter += 1
    q_id = question_map[q_str]
    
    kc_str = str(row['KC(Default)'])
    kcs = kc_str.split('~~')
    
    for kc in kcs:
        kc = kc.strip()
        if not kc: continue
        if kc not in skill_map:
            skill_map[kc] = str(skill_counter)
            skill_counter += 1
        s_id = skill_map[kc]
        
        # Add to interactions (one row per interaction_id, skill_id as per framework contract)
        processed_rows.append({
            'interaction_id': row['interaction_id'],
            'learner_id': row['Anon Student Id'],
            'question_id': q_id,
            'skill_id': s_id,
            'timestamp': row['timestamp'],
            'correct': row['correct'],
            'dataset': 'algebra2005'
        })
        
        # Add to qmatrix
        qmatrix_entries.add((q_id, s_id))

final_df = pd.DataFrame(processed_rows)
print(f"Expanded into {len(final_df)} skill-interaction pairs.")

# Export interactions
final_df.to_csv(os.path.join(output_dir, "interactions.csv"), index=False)
print("Exported interactions.csv")

# Export Q-Matrix
q_matrix_data = []
for q_id, s_id in qmatrix_entries:
    q_matrix_data.append({
        'question_id': q_id,
        'skill_id': s_id,
        'source': 'provided_static',
        'source_version': '1',
        'is_train_only': 'false'
    })
q_matrix_df = pd.DataFrame(q_matrix_data)
q_matrix_df.to_csv(os.path.join(output_dir, "q_matrix.csv"), index=False)
print("Exported q_matrix.csv")
print("Algebra2005 setup done!")
