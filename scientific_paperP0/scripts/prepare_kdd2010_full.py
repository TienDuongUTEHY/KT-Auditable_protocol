import os
import pandas as pd
from pathlib import Path

def main():
    src_path = r"D:\Paper P0 Nguyen Tien Duong\SCIE_P0\scientific_paperP0_final\data\raw\kdd2010\interactions.csv"
    dst_dir = Path("data/raw/kdd2010")
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading source full KDD2010 interactions from: {src_path}")
    df = pd.read_csv(src_path)
    
    print("Initial Statistics:")
    print(f" - Rows: {len(df)}")
    print(f" - Unique Learners: {df['learner_id'].nunique()}")
    print(f" - Unique Skills (KCs): {df['skill_id'].nunique()}")
    
    # Standard pyKT sequence length filtering: min_seq_len = 3
    print("Applying sequence length >= 3 filtering...")
    counts = df['learner_id'].value_counts()
    valid_learners = counts[counts >= 3].index
    df_filtered = df[df['learner_id'].isin(valid_learners)].copy()
    
    # Re-index interaction_id to be sequential
    df_filtered['interaction_id'] = range(1, len(df_filtered) + 1)
    
    print("Filtered Statistics:")
    print(f" - Rows: {len(df_filtered)}")
    print(f" - Unique Learners: {df_filtered['learner_id'].nunique()}")
    print(f" - Unique Skills (KCs): {df_filtered['skill_id'].nunique()}")
    
    # Save interactions
    df_filtered.to_csv(dst_dir / "interactions.csv", index=False)
    print("Saved filtered interactions to data/raw/kdd2010/interactions.csv")
    
    # Generate and save q_matrix.csv
    print("Generating Q-matrix...")
    qm = df_filtered[['question_id', 'skill_id']].drop_duplicates().copy()
    qm['source'] = 'provided_static'
    qm['source_version'] = '1'
    qm['is_train_only'] = 'false'
    
    qm.to_csv(dst_dir / "q_matrix.csv", index=False)
    print("Saved Q-matrix to data/raw/kdd2010/q_matrix.csv")
    print("Full scale KDD2010 setup complete!")

if __name__ == "__main__":
    main()
