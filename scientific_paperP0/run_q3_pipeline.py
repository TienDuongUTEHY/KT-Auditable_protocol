import os
import sys
import subprocess
import shutil
from pathlib import Path
import pandas as pd
import builtins

python_path = "D:\\scientific_paper1\\miniconda3\\envs\\scientific_paper1\\python.exe"

# Make prints unbuffered
def print(*args, **kwargs):
    kwargs['flush'] = True
    builtins.print(*args, **kwargs)

def run_cmd(args):
    print(f"Executing: {' '.join(args)}")
    res = subprocess.run(args, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error executing command: {res.stderr}")
    else:
        print(f"Success: {res.stdout.strip()}")
    return res

def main():
    datasets = ["kdd2010", "junyi", "assist2012"]
    folds = [0, 1, 2]
    seeds = [42, 43, 44, 2025, 2026]
    models = ["bkt", "dkt", "simplekt", "gikt", "skt"]
    
    # Ensure graphs directory is clean
    if os.path.exists("graphs"):
        shutil.rmtree("graphs")
    os.makedirs("graphs", exist_ok=True)
    
    # Ensure predictions, logs, and q3_fix directories exist (do NOT delete to support resuming!)
    for d in ["results/predictions", "results/logs", "results/q3_fix"]:
        os.makedirs(d, exist_ok=True)
                
    # 1. Run Preprocessing and Graph construction
    for ds in datasets:
        config = f"configs/{ds}.yaml"
        # Run preprocessing
        run_cmd([python_path, "-m", "src.preprocess", "--config", config])
        
        for fold in folds:
            print(f"\n==================== PROCESSING {ds} FOLD {fold} ====================")
            # Run split checker
            run_cmd([python_path, "-m", "src.split_checker", "--config", config, "--seed", "2026", "--fold", str(fold)])
            # Run qmatrix provenance
            run_cmd([python_path, "-m", "src.qmatrix_provenance", "--config", config, "--fold", str(fold)])
            # Run tri-relation graph builder
            run_cmd([python_path, "-m", "src.tri_relation_graph_builder", "--config", config, "--fold", str(fold)])
            # Run DAG audit (which does our top-5 pruning now!)
            run_cmd([python_path, "-m", "src.dag_audit", "--config", config, "--fold", str(fold)])
            # Run ECO audit
            run_cmd([python_path, "-m", "src.eco_audit", "--config", config, "--fold", str(fold)])
            # Run Leakage audit
            run_cmd([python_path, "-m", "src.leakage_audit", "--config", config, "--fold", str(fold)])
            # Run Graph statistics
            run_cmd([python_path, "-m", "src.graph_statistics", "--config", config, "--fold", str(fold)])
            # Run Sparse skill profile
            run_cmd([python_path, "-m", "src.sparse_skill_profile", "--config", config, "--fold", str(fold)])
            
            # Copy/rename graph files for the Q3 pipeline
            # Source: results/tables/{dataset}/fold_{fold}/E_pre_train.csv
            # Destination: graphs/{dataset}/fold_{fold}/e_pre_scores.csv (renaming columns)
            src_pre_path = Path(f"results/tables/{ds}/fold_{fold}/E_pre_train.csv")
            dst_pre_dir = Path(f"graphs/{ds}/fold_{fold}")
            dst_pre_dir.mkdir(parents=True, exist_ok=True)
            if src_pre_path.exists():
                df_pre = pd.read_csv(src_pre_path)
                df_pre = df_pre.rename(columns={"src_skill_id": "src", "dst_skill_id": "dst"})
                df_pre.to_csv(dst_pre_dir / "e_pre_scores.csv", index=False)
                print(f"Copied and renamed E_pre for {ds} fold {fold}")
                
            src_co_path = Path(f"results/tables/{ds}/fold_{fold}/E_co_train.csv")
            if src_co_path.exists():
                df_co = pd.read_csv(src_co_path)
                df_co = df_co.rename(columns={"src_skill_id": "src", "dst_skill_id": "dst"})
                df_co.to_csv(dst_pre_dir / "e_co.csv", index=False)
                print(f"Copied and renamed E_co for {ds} fold {fold}")
                
            # 2. Run Model Training and Predictions
            for model in models:
                # Load existing baseline results to skip completed runs
                completed_seeds = set()
                res_file = Path(f"results/tables/{ds}/fold_{fold}/baseline_results.csv")
                if res_file.exists():
                    try:
                        df_res = pd.read_csv(res_file)
                        if 'model' in df_res.columns and 'seed' in df_res.columns:
                            # A model-seed combination is complete if it has 4 variants
                            counts = df_res[df_res['model'].str.lower() == model.lower()].groupby('seed').size()
                            completed_seeds = set(counts[counts >= 4].index)
                            if completed_seeds:
                                print(f"Found completed seeds for model {model} in fold {fold}: {completed_seeds}")
                    except Exception as e:
                        print(f"Warning: could not read {res_file} to check completed seeds: {e}")
                
                for seed in seeds:
                    if seed in completed_seeds:
                        print(f"Skipping model {model} fold {fold} seed {seed} (already completed).")
                        continue
                    print(f"Training model {model} fold {fold} seed {seed}...")
                    run_cmd([python_path, "-m", "src.baseline_probe", "--config", config, "--fold", str(fold), "--model", model, "--seed", str(seed), "--epochs", "50"])
                    
            # 3. Generate figures and report
            run_cmd([python_path, "-m", "src.make_figures", "--config", config, "--fold", str(fold)])
            run_cmd([python_path, "-m", "src.report_generator", "--config", config, "--fold", str(fold)])

    # 4. Run the Q3 fix pipeline
    print("\n==================== RUNNING Q3 FIX DIAGNOSTIC PIPELINE ====================")
    run_cmd([python_path, "LC_MRSG_Q3_Antigravity_Fix/lc_mrsg_q3_fix_pipeline.py", "--config", "LC_MRSG_Q3_Antigravity_Fix/configs/q3_fix_config.yaml"])
    
    print("\n==================== ALL EXPERIMENTS COMPLETED SUCCESSFULLY ====================")
    print("Tables generated in: results/q3_fix/tables/")
    print("Reports generated in: results/q3_fix/reports/")

if __name__ == "__main__":
    main()
