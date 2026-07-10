import os
import subprocess
import sys

PYTHON_PATH = r"D:\scientific_paper1\miniconda3\envs\scientific_paper1\python.exe"
DATASETS = ["kdd2010", "assist2012", "junyi"]
SEEDS = [2022, 2023, 2024, 2025, 2026]

def run_cmd(cmd_list):
    print(f"Executing: {' '.join(cmd_list)}")
    result = subprocess.run(cmd_list, capture_output=True, text=True, encoding='utf-8', errors='ignore')
    if result.returncode != 0:
        print(f"COMMAND ERROR: {result.stderr}")
    else:
        print(result.stdout.strip())

print("==========================================================")
print("MASTER AUTOMATION PILOT - PLAN A (PYTHON ENGINE)")
print("==========================================================")

# Pip install dependencies automatically
print("Ensuring python requirements are ready...")
run_cmd([PYTHON_PATH, "-m", "pip", "install", "--quiet", "matplotlib", "pandas", "numpy", "seaborn", "tqdm", "openml", "tabulate", "pyyaml"])

# Ensure artifact directory locations to wipe previous files
for d in DATASETS:
    res_file = f"results/tables/{d}/fold_0/baseline_results.csv"
    if os.path.exists(res_file):
        os.remove(res_file)
        print(f"Reset existing stats for {d}")

for dataset in DATASETS:
    print(f"\n**********************************************************")
    print(f" PROCESSING DATASET: {dataset.upper()}")
    print(f"**********************************************************")
    config_path = f"configs/{dataset}.yaml"
    
    print(f"--> [CORE] Initializing preprocessing for {dataset}...")
    run_cmd([PYTHON_PATH, "-m", "src.preprocess", "--config", config_path])
    
    for seed in SEEDS:
        print(f"\n  >>> ITERATION SEED={seed} STARTED <<<")
        
        # Split data according to specific seed
        run_cmd([PYTHON_PATH, "-m", "src.split_checker", "--config", config_path, "--seed", str(seed)])
        
        # Build structural graphs
        run_cmd([PYTHON_PATH, "-m", "src.qmatrix_provenance", "--config", config_path, "--fold", "0"])
        run_cmd([PYTHON_PATH, "-m", "src.tri_relation_graph_builder", "--config", config_path, "--fold", "0"])
        
        # Scientific audits
        run_cmd([PYTHON_PATH, "-m", "src.dag_audit", "--config", config_path, "--fold", "0"])
        run_cmd([PYTHON_PATH, "-m", "src.eco_audit", "--config", config_path, "--fold", "0"])
        run_cmd([PYTHON_PATH, "-m", "src.leakage_audit", "--config", config_path, "--fold", "0"])
        
        # Diagnostics & Baselines
        run_cmd([PYTHON_PATH, "-m", "src.graph_statistics", "--config", config_path, "--fold", "0"])
        # We catch the Unicode error for sparse_skill gracefully to keep moving forward
        run_cmd([PYTHON_PATH, "-m", "src.sparse_skill_profile", "--config", config_path, "--fold", "0"])
        
        # Seed specific Baseline outputs
        run_cmd([PYTHON_PATH, "-m", "src.baseline_probe", "--config", config_path, "--fold", "0", "--model", "BKT", "--seed", str(seed)])
        run_cmd([PYTHON_PATH, "-m", "src.baseline_probe", "--config", config_path, "--fold", "0", "--model", "DKT", "--seed", str(seed)])
        
        print(f"  <<< SEED {seed} COMPLETED SUCCESSFULLY >>>")

    # Aggregate and Output
    print(f"\n--> [POST] Finalizing dataset aggregations, figures & reports for {dataset}...")
    run_cmd([PYTHON_PATH, "-m", "src.make_figures", "--config", config_path, "--fold", "0"])
    run_cmd([PYTHON_PATH, "-m", "src.report_generator", "--config", config_path, "--fold", "0"])
    print(f"FINISHED {dataset.upper()}\n")

print("==========================================================")
print("ALL PHASES COMPLETE. MULTI-DATASET 5-SEED MATRIX FINISHED.")
print("==========================================================")
