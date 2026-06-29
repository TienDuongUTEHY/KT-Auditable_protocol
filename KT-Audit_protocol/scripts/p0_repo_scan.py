import os
import sys
import subprocess
import glob

def get_git_commit():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        return commit
    except Exception:
        return "NA"

def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    python_version = sys.version.replace("\n", " ")
    git_commit = get_git_commit()

    # Search for items
    found_data_dirs = [d for d in ["data", "datasets"] if os.path.exists(os.path.join(root, d))]
    
    found_result_dirs = []
    for r_dir in ["outputs", "results", os.path.join("results", "tables"), "runs"]:
        if os.path.exists(os.path.join(root, r_dir)):
            found_result_dirs.append(r_dir)

    found_training_scripts = []
    for s_file in glob.glob(os.path.join(root, "scripts", "*train*.py")) + glob.glob(os.path.join(root, "scripts", "*run*.py")) + glob.glob(os.path.join(root, "src", "*.py")):
        found_training_scripts.append(os.path.relpath(s_file, root))

    found_graph_files = []
    for g_file in glob.glob(os.path.join(root, "graphs", "**", "*.csv"), recursive=True):
        found_graph_files.append(os.path.relpath(g_file, root))

    found_split_files = []
    for s_file in glob.glob(os.path.join(root, "data", "processed", "**", "train.csv"), recursive=True):
        found_split_files.append(os.path.relpath(s_file, root))

    found_auc_files = []
    for a_file in glob.glob(os.path.join(root, "runs", "**", "*runs*.csv"), recursive=True) + glob.glob(os.path.join(root, "results", "**", "*auc*.csv"), recursive=True):
        found_auc_files.append(os.path.relpath(a_file, root))

    # Print scan report for master log redirection
    print("[REPO_SCAN]")
    print(f"root={root}")
    print(f"python_version={python_version}")
    print(f"git_commit={git_commit}")
    print(f"found_data_dirs={found_data_dirs}")
    print(f"found_result_dirs={found_result_dirs}")
    print(f"found_training_scripts={found_training_scripts[:15]}")  # limit length
    print(f"found_graph_files={found_graph_files[:15]}")
    print(f"found_split_files={found_split_files[:15]}")
    print(f"found_auc_files={found_auc_files[:15]}")

if __name__ == "__main__":
    main()
