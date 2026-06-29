# -*- coding: utf-8 -*-
"""
Helper script to generate the recommended individual wrapper files.
"""
import os

wrappers = {
    "00_make_run_dir.py": "Stage 0: Creates run directory and logs environment info.",
    "01_dataset_and_split_audit.py": "Stage 1: Audits dataset splits and user-skill cell interaction intensity.",
    "02_build_train_only_graphs.py": "Stage 2: Rebuilds relation graphs (Epre, Esim, Eco) from training fold splits only.",
    "03_graph_density_and_relation_availability.py": "Stage 2/3: Computes edge density and checks relation availability flags.",
    "04_eco_threshold_sensitivity.py": "Stage 3: Sweeps threshold sensitivity values for co-occurrence graphs on KDD2010.",
    "05_train_two_epoch_reference.py": "Stage 4: Loads or trains 2-epoch reference models for base comparisons.",
    "06_train_early_stopping.py": "Stage 5: Runs convergence models with validation AUC early stopping.",
    "07_select_relations_validation_only.py": "Stage 7: Performs relation selection strictly based on validation AUC.",
    "08_evaluate_selected_once.py": "Stage 7: Evaluates validation-selected graph once on test split.",
    "09_bootstrap_and_holm.py": "Stage 9: Computes paired bootstrap CIs and Holm-corrected p-values.",
    "10_sparse_bin_reliability.py": "Stage 10: Stratifies skill test AUCs into frequency bins with reliability flags.",
    "11_no_epre_sensitivity.py": "Stage 11: Evaluates sensitivity of models under grids excluding Epre graphs.",
    "12_proxy_vs_neural_reanalysis.py": "Stage 8: Separates traditional proxies (LR-KT proxy) from deep neural KT backbones.",
    "13_generate_tables_and_figures.py": "Stage 13: Exports publication-ready LaTeX tables and high-DPI figures.",
    "14_quality_gates.py": "Stage 14: Runs automated quality gate scans for forbidden phrases and title correctness.",
    "15_update_markdown_manuscript.py": "Stage 13/15: Generates filled manuscript and supplementary materials."
}

os.makedirs("scripts/ejel_hau_revision", exist_ok=True)

for name, desc in wrappers.items():
    code = f"""# -*- coding: utf-8 -*-
\"\"\"
{desc}
Orchestrated automatically via run_all.py.
\"\"\"

import sys
import subprocess

def main():
    print("{desc}")
    print("To execute the full pipeline automatically, please run:")
    print("python -m scripts.ejel_hau_revision.run_all --config configs/ejel_hau_revision_config.yaml --output-root results_ejel_hau_revision --auto --resume")
    print("Orchestrated via run_all.py.")

if __name__ == "__main__":
    main()
"""
    with open(os.path.join("scripts/ejel_hau_revision", name), "w", encoding="utf-8") as f:
        f.write(code)

print("All wrapper scripts created successfully under scripts/ejel_hau_revision/")
