import os
import sys
import yaml
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/q3_lcmrsg_plus.yaml")
    args = parser.parse_args()
    
    print("Running sanity checks...")
    
    # 1. Check configuration file
    if not os.path.exists(args.config):
        print(f"FAIL: Configuration file {args.config} not found.")
        sys.exit(1)
        
    try:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        print("PASS: Config loaded successfully.")
    except Exception as e:
        print(f"FAIL: Config parsing failed: {e}")
        sys.exit(1)
        
    # 2. Check structure
    required_keys = ['project_name', 'output_root', 'random_seeds', 'datasets', 'folds', 'models', 'static_graph_variants']
    missing_keys = [k for k in required_keys if k not in cfg]
    if missing_keys:
        print(f"FAIL: Missing configuration keys: {missing_keys}")
        sys.exit(1)
    print("PASS: All required config keys are present.")
    
    # 3. Check data splits existence
    for ds in cfg['datasets']:
        for fold in cfg['folds']:
            path = f"data/processed/{ds}/fold_{fold}/train.csv"
            if not os.path.exists(path):
                print(f"FAIL: Train split missing for {ds} fold {fold} at {path}")
                sys.exit(1)
    print("PASS: All required data split CSV files are present.")
    
    print("Sanity checks PASSED. Ready to run experiments.")

if __name__ == "__main__":
    main()
