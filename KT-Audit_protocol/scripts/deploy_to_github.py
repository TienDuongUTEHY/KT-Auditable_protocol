# -*- coding: utf-8 -*-
"""
Deploy Anonymous Review Code Package to GitHub
================================================
This script unzips the code package to the user's target folder,
initializes git if needed, and pushes it to the GitHub repository.
"""

import os
import zipfile
import subprocess
import shutil

def run_cmd(cmd, cwd):
    print(f"Running: {cmd} in {cwd}")
    res = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True)
    print(f"STDOUT:\n{res.stdout}")
    print(f"STDERR:\n{res.stderr}")
    return res.returncode == 0

def main():
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    zip_path = os.path.join(workspace, "anonymous_review_code_package.zip")
    
    target_dir = r"D:\CAC GIAI DOAN CAI THIEN BAI BAO\Last version\Tap chi EJEL\anonymous_review_code_package"
    repo_url = "https://github.com/tienduongutehy-PhD-Candidate/LC-MRSG-Plus.git"
    
    print(f"Target directory: {target_dir}")
    print(f"Source ZIP: {zip_path}")
    
    # 1. Ensure target directory exists and is clean
    if os.path.exists(target_dir):
        print(f"Target directory exists. Cleaning up files inside (preserving .git if present)...")
        for item in os.listdir(target_dir):
            if item == ".git":
                continue
            item_path = os.path.join(target_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
    else:
        print(f"Creating target directory...")
        os.makedirs(target_dir, exist_ok=True)
        
    # 2. Extract ZIP package
    print("Extracting package...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    print("Extraction completed.")
    
    # 3. Git Operations
    git_dir = os.path.join(target_dir, ".git")
    if not os.path.exists(git_dir):
        print("Initializing git repository...")
        run_cmd("git init", target_dir)
        run_cmd(f"git remote add origin {repo_url}", target_dir)
        run_cmd("git branch -M main", target_dir)
    else:
        print("Git repository already initialized.")
        # Ensure remote is correct
        run_cmd("git remote remove origin", target_dir)
        run_cmd(f"git remote add origin {repo_url}", target_dir)
        
    # 4. Add, commit, and push
    run_cmd("git add .", target_dir)
    
    # Configure user name/email locally if not configured to avoid commit errors
    run_cmd('git config user.name "tienduongutehy-PhD-Candidate"', target_dir)
    run_cmd('git config user.email "tienduongutehy@gmail.com"', target_dir) # default fallback if not set
    
    # Try committing
    run_cmd('git commit -m "Upload anonymous review code package for LC-MRSG"', target_dir)
    
    # Push to remote main
    print("Pushing to GitHub remote...")
    success = run_cmd("git push -u origin main --force", target_dir)
    if success:
        print("Successfully pushed to GitHub!")
    else:
        print("Failed to push. Checking credentials or branch state.")

if __name__ == "__main__":
    main()
