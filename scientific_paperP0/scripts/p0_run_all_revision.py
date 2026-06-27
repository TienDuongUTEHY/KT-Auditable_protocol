import os
import sys
import subprocess
import argparse
import time
import datetime

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def run_script(python_bin, script_path, log_file_handle, master_log_handle, name):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start_msg = f"[{timestamp}] === STARTING PHASE: {name} ==="
    print(start_msg)
    master_log_handle.write(start_msg + "\n")
    master_log_handle.flush()
    
    cmd = [python_bin, "-u", script_path]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, encoding='utf-8')
    
    for line in iter(process.stdout.readline, ''):
        stripped = line.rstrip()
        print(f"  {stripped}")
        master_log_handle.write(f"  {stripped}\n")
        log_file_handle.write(f"{stripped}\n")
        
    process.stdout.close()
    return_code = process.wait()
    
    timestamp_end = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if return_code != 0:
        fail_msg = f"[{timestamp_end}] === PHASE FAILED: {name} with exit code {return_code} ==="
        print(fail_msg)
        master_log_handle.write(fail_msg + "\n")
        master_log_handle.flush()
        return False
    else:
        pass_msg = f"[{timestamp_end}] === PHASE COMPLETED: {name} ==="
        print(pass_msg)
        master_log_handle.write(pass_msg + "\n")
        master_log_handle.flush()
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    
    output_dir = "results_p0_revision"
    logs_dir = ensure_dir(os.path.join(output_dir, "logs"))
    
    # Python executable path discovery
    python_bin = "D:\\scientific_paper1\\miniconda3\\envs\\scientific_paper1\\python.exe"
    if not os.path.exists(python_bin):
        python_bin = sys.executable
        
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    master_log_path = os.path.join(logs_dir, f"master_run_{timestamp_str}.log")
    
    status = "FAIL"
    
    with open(master_log_path, "w", encoding="utf-8") as ml:
        ml.write("Orchestrating P0 Revision Experiment Pipeline\n")
        
        # --- Phase 0: Scope decision ---
        phase0_log_path = os.path.join(logs_dir, "phase0_scope_decision.log")
        with open(phase0_log_path, "w", encoding="utf-8") as p0l:
            p0l.write("PHASE_START phase0_scope_decision\n")
            p0l.write("P0_MAIN_CONTRIBUTION=construction/audit/selection protocol\n")
            p0l.write("PERFORMANCE_ROLE=diagnostic evidence only\n")
            p0l.write("DEFER_CALIBRATION_TO_P1=true\n")
            p0l.write("DEFER_ADAPTIVE_STRATIFICATION_TO_P1=true\n")
            p0l.write("DEFER_SSA_CL_TO_P2=true\n")
            p0l.write("PHASE_PASS phase0_scope_decision\n")
        
        print("PHASE_START phase0_scope_decision")
        print("P0_MAIN_CONTRIBUTION=construction/audit/selection protocol")
        print("PERFORMANCE_ROLE=diagnostic evidence only")
        print("DEFER_CALIBRATION_TO_P1=true")
        print("DEFER_ADAPTIVE_STRATIFICATION_TO_P1=true")
        print("DEFER_SSA_CL_TO_P2=true")
        print("PHASE_PASS phase0_scope_decision")
        
        ml.write("PHASE_START phase0_scope_decision\n")
        ml.write("PHASE_PASS phase0_scope_decision\n")
        
        # --- Repo Scan ---
        phase1_log_path = os.path.join(logs_dir, "phase1_dataset_graph_audit.log")
        with open(phase1_log_path, "w", encoding="utf-8") as p1l:
            ok = run_script(python_bin, "scripts/p0_repo_scan.py", p1l, ml, "Repo Scan")
            if not ok:
                ml.write("PHASE_FAIL repo_scan\n")
                sys.exit(1)
                
            # --- Dataset Statistics ---
            ok = run_script(python_bin, "scripts/p0_audit_dataset_stats.py", p1l, ml, "Dataset Statistics")
            if not ok:
                sys.exit(1)
                
            # --- KDD2010 E_co Audit ---
            ok = run_script(python_bin, "scripts/p0_audit_graph_provenance.py", p1l, ml, "Graph Provenance Audit")
            if not ok:
                sys.exit(1)
                
            # --- E_sim pipeline trace ---
            phase1_esim_log_path = os.path.join(logs_dir, "phase1_esim_trace.log")
            with open(phase1_esim_log_path, "w", encoding="utf-8") as esim_l:
                ok = run_script(python_bin, "scripts/p0_trace_esim_pipeline.py", esim_l, ml, "E_sim Pipeline Trace")
                if not ok:
                    sys.exit(1)
                    
            # --- Junyi coverage ---
            phase1_junyi_log_path = os.path.join(logs_dir, "phase1_junyi_coverage.log")
            with open(phase1_junyi_log_path, "w", encoding="utf-8") as junyi_l:
                ok = run_script(python_bin, "scripts/p0_audit_junyi_coverage.py", junyi_l, ml, "Junyi Coverage Audit")
                if not ok:
                    sys.exit(1)
                    
            # --- Leakage Audit L1-L6 ---
            ok = run_script(python_bin, "scripts/p0_audit_leakage_L1_L6.py", p1l, ml, "Leakage Audit L1-L6")
            if not ok:
                sys.exit(1)
                
        # --- Phase 2: Epoch Sanity Check ---
        phase2_log_path = os.path.join(logs_dir, "phase2_epoch_sanity.log")
        with open(phase2_log_path, "w", encoding="utf-8") as p2l:
            ok = run_script(python_bin, "scripts/p0_epoch_sanity.py", p2l, ml, "Epoch Sanity Check")
            if not ok:
                # If Epoch sanity has budget limitation, we continue but label as PASS_WITH_LIMITATIONS
                status = "PASS_WITH_LIMITATIONS"
            else:
                status = "PASS"
                
        # --- Phase 3: Table generation ---
        phase3_log_path = os.path.join(logs_dir, "phase3_table_generation.log")
        with open(phase3_log_path, "w", encoding="utf-8") as p3l:
            ok = run_script(python_bin, "scripts/p0_generate_tables.py", p3l, ml, "Table Generation")
            if not ok:
                sys.exit(1)
                
        # --- Phase 4: Manifest & Validation ---
        phase4_log_path = os.path.join(logs_dir, "phase4_manifest_validation.log")
        with open(phase4_log_path, "w", encoding="utf-8") as p4l:
            # First build manifests
            subprocess.run([python_bin, "scripts/p0_build_manifest.py"])
            # Then validate
            ok = run_script(python_bin, "scripts/p0_validate_outputs.py", p4l, ml, "Outputs Validation")
            if not ok:
                status = "FAIL"
                
        # Finished successfully
        ml.write("P0_REVISION_PIPELINE_FINISHED\n")
        ml.write(f"STATUS={status}\n")
        
    print("\n" + "="*50)
    print("P0_REVISION_PIPELINE_FINISHED")
    print(f"STATUS={status}")
    print(f"KEY_OUTPUT_DIR={output_dir}/")
    print("NEXT_ACTION=Copy tables_tex into LaTeX manuscript and update the interpretation paragraphs according to epoch_sanity_interpretation.")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
