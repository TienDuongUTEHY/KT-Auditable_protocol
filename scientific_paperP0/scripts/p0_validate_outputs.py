import os
import sys
import pandas as pd

def check_file_nonempty(path):
    return os.path.exists(path) and os.path.getsize(path) > 5

def main():
    output_dir = "results_p0_revision"
    tables_csv = os.path.join(output_dir, "tables_csv")
    tables_tex = os.path.join(output_dir, "tables_tex")
    manifests = os.path.join(output_dir, "manifests")
    logs = os.path.join(output_dir, "logs")
    
    log_file = os.path.join(logs, "phase4_manifest_validation.log")
    
    required_files = [
        os.path.join(tables_csv, "table_dataset_statistics.csv"),
        os.path.join(tables_csv, "table_graph_provenance_corrected.csv"),
        os.path.join(tables_csv, "table_esim_trace.csv"),
        os.path.join(tables_csv, "table_junyi_graph_coverage.csv"),
        os.path.join(tables_csv, "table_leakage_audit_L1_L6.csv"),
        os.path.join(tables_csv, "table_validation_candidates_prespecified.csv"),
        os.path.join(tables_csv, "table_selected_relation_variants.csv"),
        os.path.join(tables_csv, "table_main_auc_delta_holm.csv"),
        os.path.join(tables_csv, "table_sparse_bins_descriptive.csv"),
        os.path.join(tables_csv, "table_epoch_sanity.csv"),
        os.path.join(tables_csv, "table_reproducibility_checklist.csv"),
        os.path.join(tables_csv, "table_hardware_runtime.csv"),
        
        os.path.join(tables_tex, "table_dataset_statistics.tex"),
        os.path.join(tables_tex, "table_graph_provenance_corrected.tex"),
        os.path.join(tables_tex, "table_esim_trace.tex"),
        os.path.join(tables_tex, "table_junyi_graph_coverage.tex"),
        os.path.join(tables_tex, "table_leakage_audit_L1_L6.tex"),
        os.path.join(tables_tex, "table_validation_candidates_prespecified.tex"),
        os.path.join(tables_tex, "table_selected_relation_variants.tex"),
        os.path.join(tables_tex, "table_main_auc_delta_holm.tex"),
        os.path.join(tables_tex, "table_sparse_bins_descriptive.tex"),
        os.path.join(tables_tex, "table_epoch_sanity.tex"),
        os.path.join(tables_tex, "table_reproducibility_checklist.tex"),
        os.path.join(tables_tex, "table_hardware_runtime.tex"),
        
        os.path.join(manifests, "sha256_manifest.csv"),
        os.path.join(manifests, "run_environment.txt"),
        os.path.join(manifests, "git_state.txt"),
        os.path.join(manifests, "final_definition_of_done.md")
    ]
    
    validation_failed = False
    
    with open(log_file, "w", encoding="utf-8") as lf:
        lf.write("PHASE_START manifest_validation\n")
        print("PHASE_START manifest_validation")
        
        # 1. Check required files
        files_ok = all(check_file_nonempty(f) for f in required_files)
        status_files = "PASS" if files_ok else "FAIL"
        if status_files == "FAIL":
            missing = [f for f in required_files if not check_file_nonempty(f)]
            print(f"Missing or empty files: {missing}")
            validation_failed = True
        lf.write(f"VALIDATION_CHECK name=required_files status={status_files}\n")
        print(f"VALIDATION_CHECK name=required_files status={status_files}")
        
        # 2. Check graph limits
        graph_limits_ok = True
        prov_csv = os.path.join(tables_csv, "table_graph_provenance_corrected.csv")
        if os.path.exists(prov_csv):
            df_prov = pd.read_csv(prov_csv)
            for _, r in df_prov.iterrows():
                if r['is_unique_undirected_valid'] == "no" and r['interpretation'] != "multi_edge":
                    graph_limits_ok = False
        status_graph = "PASS" if graph_limits_ok else "FAIL"
        if status_graph == "FAIL":
            validation_failed = True
        lf.write(f"VALIDATION_CHECK name=graph_limits status={status_graph}\n")
        print(f"VALIDATION_CHECK name=graph_limits status={status_graph}")
        
        # 3. Check leakage L1/L5/L6
        leakage_ok = True
        leak_csv = os.path.join(tables_csv, "table_leakage_audit_L1_L6.csv")
        if os.path.exists(leak_csv):
            df_leak = pd.read_csv(leak_csv)
            for _, r in df_leak.iterrows():
                if r['L1'] != "PASS" or r['L5'] != "PASS" or r['L6'] != "PASS":
                    leakage_ok = False
        status_leak = "PASS" if leakage_ok else "FAIL"
        if status_leak == "FAIL":
            validation_failed = True
        lf.write(f"VALIDATION_CHECK name=leakage_L1_L5_L6 status={status_leak}\n")
        print(f"VALIDATION_CHECK name=leakage_L1_L5_L6 status={status_leak}")
        
        # 4. Check epoch sanity
        sanity_ok = True
        sanity_csv = os.path.join(tables_csv, "table_epoch_sanity.csv")
        if os.path.exists(sanity_csv):
            df_sanity = pd.read_csv(sanity_csv)
            if df_sanity.empty:
                sanity_ok = False
        else:
            sanity_ok = False
        status_sanity = "PASS_OR_LIMITED" if sanity_ok else "FAIL"
        if status_sanity == "FAIL":
            validation_failed = True
        lf.write(f"VALIDATION_CHECK name=epoch_sanity status={status_sanity}\n")
        print(f"VALIDATION_CHECK name=epoch_sanity status={status_sanity}")
        
        # 5. Check LaTeX tables non-empty
        latex_ok = True
        for f in required_files:
            if f.endswith(".tex"):
                if not check_file_nonempty(f):
                    latex_ok = False
        status_latex = "PASS" if latex_ok else "FAIL"
        if status_latex == "FAIL":
            validation_failed = True
        lf.write(f"VALIDATION_CHECK name=latex_tables_nonempty status={status_latex}\n")
        print(f"VALIDATION_CHECK name=latex_tables_nonempty status={status_latex}")
        
        if validation_failed:
            lf.write("PHASE_FAIL manifest_validation reason=\"one or more validation checks failed\"\n")
            print("PHASE_FAIL manifest_validation reason=\"one or more validation checks failed\"")
            sys.exit(1)
        else:
            lf.write("PHASE_PASS manifest_validation\n")
            print("PHASE_PASS manifest_validation")

if __name__ == "__main__":
    main()
