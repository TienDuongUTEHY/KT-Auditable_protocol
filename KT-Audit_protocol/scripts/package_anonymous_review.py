# -*- coding: utf-8 -*-
"""
Package Anonymous Review Code Package for LC-MRSG
===================================================
This script packages all relevant code, configs, tests, and documentation
into a clean, anonymized ZIP file suitable for Zenodo/OSF/GitHub anonymous upload.
It excludes any dataset files, checkpoints, local run folders, logs, or backups.
"""

import os
import zipfile
import shutil

def main():
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    zip_path = os.path.join(workspace, "anonymous_review_code_package.zip")
    
    print(f"Workspace root: {workspace}")
    print(f"Creating anonymous package: {zip_path}")
    
    # Items to include in the ZIP archive
    includes = {
        "configs": "configs",
        "src": "src",
        "tests": "tests",
        "scripts/ejel_hau_revision": "scripts/ejel_hau_revision",
    }
    
    # Root level files to include with their mapping name
    root_files = {
        "README_Q3_LCMRSG_PLUS.md": "README.md",
        "ANTIGRAVITY_EJEL_HAU_AUTOMATED_PIPELINE.md": "AUTOMATED_PIPELINE.md",
    }
    
    # Add tables/figures from the latest run to show as artifact results
    latest_run_tables = "results_ejel_hau_revision_20260624_225226/tables"
    latest_run_figures = "results_ejel_hau_revision_20260624_225226/figures"
    latest_status = "results_ejel_hau_revision_20260624_225226/RUN_STATUS.md"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 1. Package directories
        for rel_src, rel_dst in includes.items():
            full_src = os.path.join(workspace, rel_src)
            if not os.path.exists(full_src):
                print(f"Warning: {full_src} does not exist. Skipping.")
                continue
            for root, dirs, files in os.walk(full_src):
                # Ignore __pycache__
                if "__pycache__" in root:
                    continue
                for file in files:
                    # Omit temp files
                    if file.endswith('.pyc') or file.endswith('.pyo') or file.endswith('.log'):
                        continue
                    full_file = os.path.join(root, file)
                    rel_to_workspace = os.path.relpath(full_file, workspace)
                    # Compute destination path inside the zip
                    rel_to_src = os.path.relpath(full_file, full_src)
                    dst_zip_path = os.path.join(rel_dst, rel_to_src).replace('\\', '/')
                    zipf.write(full_file, dst_zip_path)
                    print(f"Added: {dst_zip_path}")
                    
        # 2. Package other main scripts at the script root level
        scripts_dir = os.path.join(workspace, "scripts")
        for file in os.listdir(scripts_dir):
            full_file = os.path.join(scripts_dir, file)
            if os.path.isfile(full_file) and file.endswith('.py'):
                # Avoid adding packaging script itself
                if file == "package_anonymous_review.py":
                    continue
                dst_zip_path = f"scripts/{file}"
                zipf.write(full_file, dst_zip_path)
                print(f"Added: {dst_zip_path}")

        # 3. Package root files
        for file, mapped_name in root_files.items():
            full_file = os.path.join(workspace, file)
            if os.path.exists(full_file):
                zipf.write(full_file, mapped_name)
                print(f"Added: {mapped_name}")

        # 4. Package latest run statistics & tables (anonymized results)
        for root, dirs, files in os.walk(os.path.join(workspace, latest_run_tables)):
            for file in files:
                full_file = os.path.join(root, file)
                rel_path = os.path.relpath(full_file, os.path.join(workspace, latest_run_tables))
                dst_zip_path = f"results/tables/{rel_path}".replace('\\', '/')
                zipf.write(full_file, dst_zip_path)
                
        for root, dirs, files in os.walk(os.path.join(workspace, latest_run_figures)):
            for file in files:
                full_file = os.path.join(root, file)
                rel_path = os.path.relpath(full_file, os.path.join(workspace, latest_run_figures))
                dst_zip_path = f"results/figures/{rel_path}".replace('\\', '/')
                zipf.write(full_file, dst_zip_path)
                
        # Package RUN_STATUS.md
        full_status = os.path.join(workspace, latest_status)
        if os.path.exists(full_status):
            zipf.write(full_status, "results/RUN_STATUS.md")
            print("Added: results/RUN_STATUS.md")
            
    print(f"Packaging complete! Created anonymous_review_code_package.zip successfully.")

if __name__ == "__main__":
    main()
