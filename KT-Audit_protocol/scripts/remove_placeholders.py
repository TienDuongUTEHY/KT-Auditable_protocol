# -*- coding: utf-8 -*-
"""
Remove Placeholder links (OSF and Zenodo) from all project files
================================================================
This script removes the placeholders of Anonymous OSF Workspace and Anonymous Zenodo Archive.
It keeps only the active and verified Anonymous Code Repository via 4open.
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
    
    # 1. Update LC_MRSG_EJEL_MAIN_FILLED.md
    main_manuscript = os.path.join(workspace, "results_ejel_hau_revision_20260624_225226", "manuscript_ready", "LC_MRSG_EJEL_MAIN_FILLED.md")
    main_replacements = {
        """## Data and Code Availability Statement
For double-blind peer review, the reproducibility code package, processed datasets, and metadata are shared anonymously via the following platforms:
- **Anonymous Code Repository (4open)**: [https://anonymous.4open.science/r/LC-MRSG](https://anonymous.4open.science/r/LC-MRSG)
- **Anonymous OSF Workspace**: [https://osf.io/yj8q4/?view_only=a1b2c3d4e5f6g7h8i9j0](https://osf.io/yj8q4/?view_only=a1b2c3d4e5f6g7h8i9j0)
- **Anonymous Zenodo Archive (Dataset and Metadata)**: [https://doi.org/10.5281/zenodo.10987654](https://doi.org/10.5281/zenodo.10987654) (placeholder DOI for peer review)""":
        """## Data and Code Availability Statement
For double-blind peer review, the complete reproducibility code package, configuration files, and processed datasets are shared anonymously via the following platform:
- **Anonymous Code Repository (via 4open)**: [https://anonymous.4open.science/r/LC-MRSG](https://anonymous.4open.science/r/LC-MRSG)"""
    }
    replace_in_file(main_manuscript, main_replacements)
    
    # 2. Update RUN_STATUS.md
    run_status = os.path.join(workspace, "results_ejel_hau_revision_20260624_225226", "RUN_STATUS.md")
    status_replacements = {
        "- Repository/data availability link: Shared via 4open (https://anonymous.4open.science/r/LC-MRSG) and OSF (https://osf.io/yj8q4/?view_only=a1b2c3d4e5f6g7h8i9j0)":
        "- Repository/data availability link: Shared via 4open (https://anonymous.4open.science/r/LC-MRSG)"
    }
    replace_in_file(run_status, status_replacements)
    
    # 3. Update README_Q3_LCMRSG_PLUS.md
    readme_file = os.path.join(workspace, "README_Q3_LCMRSG_PLUS.md")
    readme_replacements = {
        """## Reproducibility Package Links (Anonymous)

For double-blind peer review, the code repository, structured datasets, and metadata are shared via the following anonymous links:

- **Anonymous Code Repository (via 4open)**: [https://anonymous.4open.science/r/LC-MRSG](https://anonymous.4open.science/r/LC-MRSG)
- **Anonymous OSF Workspace**: [https://osf.io/yj8q4/?view_only=a1b2c3d4e5f6g7h8i9j0](https://osf.io/yj8q4/?view_only=a1b2c3d4e5f6g7h8i9j0)
- **Anonymous Zenodo Archive (Dataset and Metadata)**: [https://doi.org/10.5281/zenodo.10987654](https://doi.org/10.5281/zenodo.10987654) (placeholder DOI for peer review)""":
        """## Reproducibility Package Links (Anonymous)

For double-blind peer review, the code repository, configuration files, and processed datasets are shared anonymously via the following link:

- **Anonymous Code Repository (via 4open)**: [https://anonymous.4open.science/r/LC-MRSG](https://anonymous.4open.science/r/LC-MRSG)"""
    }
    replace_in_file(readme_file, readme_replacements)
    
    # 4. Update scripts/p0_generate_tables.py
    p0_tables = os.path.join(workspace, "scripts", "p0_generate_tables.py")
    # Let's inspect scripts/p0_generate_tables.py lines 412-419:
    #             {
    #                 "artifact": "Anonymous OSF Project",
    #                 "path": "https://osf.io/yj8q4/?view_only=a1b2c3d4e5f6g7h8i9j0",
    #                 "status": "available",
    #                 "sha256": "N/A",
    #                 "purpose": "Anonymous repository sharing and project workspace.",
    #                 "used_in_main_text_yes_no": "yes"
    #             }
    p0_replacements = {
        """,
            {
                "artifact": "Anonymous OSF Project",
                "path": "https://osf.io/yj8q4/?view_only=a1b2c3d4e5f6g7h8i9j0",
                "status": "available",
                "sha256": "N/A",
                "purpose": "Anonymous repository sharing and project workspace.",
                "used_in_main_text_yes_no": "yes"
            }""": ""
    }
    replace_in_file(p0_tables, p0_replacements)
    
    # 5. Update table_reproducibility_checklist.csv
    csv_checklist = os.path.join(workspace, "results_p0_revision", "tables_csv", "table_reproducibility_checklist.csv")
    csv_replacements = {
        "\nAnonymous OSF Project,https://osf.io/yj8q4/?view_only=a1b2c3d4e5f6g7h8i9j0,available,,Anonymous repository sharing and project workspace.,yes": ""
    }
    replace_in_file(csv_checklist, csv_replacements)
    
    # 6. Update table_reproducibility_checklist.tex
    tex_checklist = os.path.join(workspace, "results_p0_revision", "tables_tex", "table_reproducibility_checklist.tex")
    tex_replacements = {
        "\nAnonymous OSF Project & ?view_only=a1b2c3d4e5f6g7h8i9j0 & available & Anonymous repository sharing and project workspace. \\\\": ""
    }
    replace_in_file(tex_checklist, tex_replacements)

    print("Placeholders removal complete!")

if __name__ == "__main__":
    main()
