import pandas as pd
import os
import zipfile
import traceback

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def ingest_full_junyi():
    print("\n--- Processing FULL Junyi Dataset (Memory Safe) ---")
    zip_path = r"D:\Paper P0 Nguyen Tien Duong\Paper_P0\Data\raw\Junyi\archive.zip"
    out_dir = r"D:\Paper P0 Nguyen Tien Duong\SCIE_P0\scientific_paperP0_final\data\raw\junyi"
    ensure_dir(out_dir)
    
    try:
        with zipfile.ZipFile(zip_path) as z:
            # 1. Load metadata for mapping
            print("Loading Info_Content.csv metadata...")
            info_df = pd.read_csv(z.open("Info_Content.csv"))
            info_df['target_skill'] = info_df['level2_id'].fillna(info_df['subject']).fillna('Unknown')
            q_to_s_map = dict(zip(info_df['ucid'].astype(str), info_df['target_skill'].astype(str)))
            
            # 2. Process in chunks
            print("Processing Log_Problem.csv in chunks...")
            chunksize = 1000000
            chunks = pd.read_csv(z.open("Log_Problem.csv"), chunksize=chunksize, 
                                 usecols=['uuid', 'ucid', 'is_correct', 'timestamp_TW'])
            
            first_chunk = True
            total_rows = 0
            
            for chunk in chunks:
                chunk = chunk.dropna(subset=['uuid', 'ucid', 'is_correct'])
                chunk['learner_id'] = chunk['uuid'].astype(str)
                chunk['question_id'] = chunk['ucid'].astype(str)
                chunk['skill_id'] = chunk['question_id'].map(q_to_s_map)
                chunk['correct'] = chunk['is_correct'].apply(lambda x: 1 if str(x).lower() in ['true', '1', '1.0'] else 0)
                
                # Try parsing timestamp
                try:
                    chunk['timestamp'] = pd.to_datetime(chunk['timestamp_TW']).astype(int) // 10**9
                except:
                    chunk['timestamp'] = range(1000, 1000 + len(chunk))
                
                chunk['dataset'] = 'junyi'
                chunk = chunk.dropna(subset=['skill_id'])
                
                cols = ['learner_id', 'question_id', 'skill_id', 'timestamp', 'correct', 'dataset']
                final_chunk = chunk[cols]
                
                # Write to CSV
                mode = 'w' if first_chunk else 'a'
                header = first_chunk
                final_chunk.to_csv(os.path.join(out_dir, "interactions.csv"), mode=mode, header=header, index=False)
                
                total_rows += len(final_chunk)
                first_chunk = False
                print(f"Processed chunk... Total saved so far: {total_rows} rows")
            
            print(f"Finished extracting ALL interactions. Total rows: {total_rows}")
            
            # 3. Create Q-matrix from the final extracted dataset
            print("Generating q_matrix.csv...")
            # We don't need to load the whole interactions.csv again, we can just use info_df!
            qmatrix = info_df[['ucid', 'target_skill']].rename(columns={'ucid': 'question_id', 'target_skill': 'skill_id'})
            qmatrix = qmatrix.dropna()
            qmatrix['source'] = 'provided_static'
            qmatrix['source_version'] = 'full'
            qmatrix['is_train_only'] = 'false'
            qmatrix.to_csv(os.path.join(out_dir, "q_matrix.csv"), index=False)
            print("Saved q_matrix.csv for full Junyi.")
            
    except Exception as e:
        print(f"FATAL ERROR processing Junyi: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    ingest_full_junyi()
