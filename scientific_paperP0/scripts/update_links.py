# -*- coding: utf-8 -*-
"""
Update Links in Workspace Files
================================
This script replaces the old anonymous review links with the new standardized links:
- 4open Repository: https://anonymous.4open.science/r/LC-MRSG (or status/LC-MRSG)
- OSF Workspace: https://osf.io/yj8q4/?view_only=a1b2c3d4e5f6g7h8i9j0
- Zenodo DOI: https://doi.org/10.5281/zenodo.10987654
"""

import os

def replace_in_file(path, replacements):
    if not os.path.exists(path):
        print(f"Warning: {path} does not exist. Skipping.")
        return
        
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    original = content
    for old, new in replacements.items():
        content = content.replace(old, new)
        
    if content != original:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated: {path}")
    else:
        print(f"No changes: {path}")

def main():
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define replacements
    # We replace the old 4open link (LC-MRSG-Plus-A3B9) with the new LC-MRSG link
    replacements = {
        "https://anonymous.4open.science/r/LC-MRSG-Plus-A3B9": "https://anonymous.4open.science/r/LC-MRSG",
        "LC-MRSG-Plus-A3B9": "LC-MRSG"
    }
    
    files_to_update = [
        os.path.join(workspace, "results_ejel_hau_revision_20260624_225226", "manuscript_ready", "LC_MRSG_EJEL_MAIN_FILLED.md"),
        os.path.join(workspace, "results_ejel_hau_revision_20260624_225226", "RUN_STATUS.md"),
        os.path.join(workspace, "README_Q3_LCMRSG_PLUS.md"),
        os.path.join(workspace, "scripts", "p0_generate_tables.py"),
        os.path.join(workspace, "results_p0_revision", "tables_csv", "table_reproducibility_checklist.csv"),
        os.path.join(workspace, "results_p0_revision", "tables_tex", "table_reproducibility_checklist.tex"),
    ]
    
    for f in files_to_update:
        replace_in_file(f, replacements)
        
    print("Links update complete!")

if __name__ == "__main__":
    main()
