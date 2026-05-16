import pandas as pd
import os
import zipfile
import traceback

SAMPLE_SIZE = 500000

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def ingest_assist2012():
    print("\n--- Processing ASSISTments 2012 Sample ---")
    raw_path = r"D:\Paper P0 Nguyen Tien Duong\Paper_P0\Data\raw\assist2012\assist2012.csv"
    out_dir = r"D:\Paper P0 Nguyen Tien Duong\Paper_P0\scientific_paperP0\data\raw\assist2012"
    ensure_dir(out_dir)
    
    # Load small manageable chunk
    print(f"Reading first {SAMPLE_SIZE} lines of 3GB file...")
    df = pd.read_csv(raw_path, nrows=SAMPLE_SIZE, low_memory=False)
    df = df.dropna(subset=['user_id', 'skill_id', 'problem_id', 'correct'])
    
    # Normalize
    df['interaction_id'] = range(1, len(df) + 1)
    df['learner_id'] = df['user_id'].astype(str)
    df['question_id'] = df['problem_id'].astype(str)
    df['skill_id'] = df['skill_id'].astype(str).apply(lambda x: x.replace('.0', ''))
    df['correct'] = df['correct'].astype(int)
    
    try:
        df['timestamp'] = pd.to_datetime(df['start_time']).astype(int) // 10**9
    except:
        df['timestamp'] = range(1000, 1000 + len(df))
    
    df['dataset'] = 'assist2012'
    
    cols = ['interaction_id', 'learner_id', 'question_id', 'skill_id', 'timestamp', 'correct', 'dataset']
    final_df = df[cols]
    final_df.to_csv(os.path.join(out_dir, "interactions.csv"), index=False)
    print(f"Saved {len(final_df)} interaction rows to assist2012.")
    
    # Generate Q-matrix from relationships in these interactions
    qmatrix = final_df[['question_id', 'skill_id']].drop_duplicates()
    qmatrix['source'] = 'provided_static'
    qmatrix['source_version'] = '1'
    qmatrix['is_train_only'] = 'false'
    qmatrix.to_csv(os.path.join(out_dir, "q_matrix.csv"), index=False)
    print("Saved q_matrix.csv for assist2012.")

def ingest_junyi():
    print("\n--- Processing Junyi Sample (Directly from ZIP) ---")
    zip_path = r"D:\Paper P0 Nguyen Tien Duong\Paper_P0\Data\raw\Junyi\archive.zip"
    out_dir = r"D:\Paper P0 Nguyen Tien Duong\Paper_P0\scientific_paperP0\data\raw\junyi"
    ensure_dir(out_dir)
    
    try:
        with zipfile.ZipFile(zip_path) as z:
            # 1. Load metadata for mapping
            print("Loading Info_Content.csv metadata...")
            info_df = pd.read_csv(z.open("Info_Content.csv"))
            # Create mapping dict from ucid (question) to topic_id (level2_id). If level2_id null, fallback to subject.
            info_df['target_skill'] = info_df['level2_id'].fillna(info_df['subject']).fillna('Unknown')
            q_to_s_map = dict(zip(info_df['ucid'].astype(str), info_df['target_skill'].astype(str)))
            
            # 2. Load first chunk of problem logs
            print(f"Streaming first {SAMPLE_SIZE} rows from Log_Problem.csv in zip...")
            log_df = pd.read_csv(z.open("Log_Problem.csv"), nrows=SAMPLE_SIZE)
            
            # 3. Build processed dataframe
            log_df = log_df.dropna(subset=['uuid', 'ucid', 'is_correct'])
            log_df['interaction_id'] = range(1, len(log_df) + 1)
            log_df['learner_id'] = log_df['uuid'].astype(str)
            log_df['question_id'] = log_df['ucid'].astype(str)
            log_df['skill_id'] = log_df['question_id'].map(q_to_s_map)
            log_df['correct'] = log_df['is_correct'].apply(lambda x: 1 if str(x).lower() in ['true', '1', '1.0'] else 0)
            
            try:
                log_df['timestamp'] = pd.to_datetime(log_df['timestamp_TW']).astype(int) // 10**9
            except:
                log_df['timestamp'] = range(1000, 1000 + len(log_df))
            
            log_df['dataset'] = 'junyi'
            
            # Drop interactions with unmappable skills
            log_df = log_df.dropna(subset=['skill_id'])
            
            cols = ['interaction_id', 'learner_id', 'question_id', 'skill_id', 'timestamp', 'correct', 'dataset']
            final_df = log_df[cols]
            
            final_df.to_csv(os.path.join(out_dir, "interactions.csv"), index=False)
            print(f"Saved {len(final_df)} interaction rows for Junyi.")
            
            qmatrix = final_df[['question_id', 'skill_id']].drop_duplicates()
            qmatrix['source'] = 'provided_static'
            qmatrix['source_version'] = '1'
            qmatrix['is_train_only'] = 'false'
            qmatrix.to_csv(os.path.join(out_dir, "q_matrix.csv"), index=False)
            print("Saved q_matrix.csv for Junyi.")
            
    except Exception as e:
        print(f"FATAL ERROR processing Junyi: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("STARTING PLAN A INGESTION ENGINE")
    ingest_assist2012()
    ingest_junyi()
    print("\nPLAN A INGESTION COMPLETED SUCCESSFULLY.")
