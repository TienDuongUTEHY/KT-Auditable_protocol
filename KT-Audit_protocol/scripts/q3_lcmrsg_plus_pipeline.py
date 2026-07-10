import os
import sys
import subprocess
import time

python_path = "D:\\scientific_paper1\\miniconda3\\envs\\scientific_paper1\\python.exe"
config_path = "configs/q3_lcmrsg_plus.yaml"
output_dir = "runs/q3_lcmrsg_plus_20260528_234100"

log_file_path = "full_q3_run.log"

def log_message(f, msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    f.write(formatted + "\n")
    f.flush()

def run_step(f, cmd, name):
    log_message(f, f"=== STARTING STEP: {name} ===")
    log_message(f, f"Command: {' '.join(cmd)}")
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, encoding='utf-8')
    
    for line in iter(process.stdout.readline, ''):
        stripped = line.rstrip()
        log_message(f, f"  {stripped}")
        
    process.stdout.close()
    return_code = process.wait()
    
    if return_code != 0:
        log_message(f, f"=== STEP FAILED: {name} with exit code {return_code} ===")
        sys.exit(return_code)
    else:
        log_message(f, f"=== STEP COMPLETED: {name} ===")

def main():
    with open(log_file_path, "w", encoding="utf-8") as f:
        log_message(f, "Orchestrating Q3 LC-MRSG++ Experiment Pipeline")
        
        # 1. Run Experiments
        cmd_exp = [python_path, "-u", "scripts/q3_lcmrsg_plus_run_experiments.py", "--config", config_path, "--output_dir", output_dir]
        run_step(f, cmd_exp, "Model Training & Validation-Guided Search")
        
        # 2. Run Analyze
        cmd_ana = [python_path, "-u", "scripts/q3_lcmrsg_plus_analyze.py", "--run_dir", output_dir]
        run_step(f, cmd_ana, "Statistical Analysis & Stratum Summaries")
        
        # 3. Render Tables
        cmd_tab = [python_path, "-u", "scripts/q3_lcmrsg_plus_render_tables.py", "--run_dir", output_dir]
        run_step(f, cmd_tab, "LaTeX and CSV Tables Rendering")
        
        # 4. Export Appendix Figures
        cmd_fig = [python_path, "-u", "scripts/q3_lcmrsg_plus_export_appendix.py", "--run_dir", output_dir]
        run_step(f, cmd_fig, "Publication Figures Export")
        
        log_message(f, "All steps in Q3 LC-MRSG++ Experiment Pipeline completed successfully!")

if __name__ == "__main__":
    main()
