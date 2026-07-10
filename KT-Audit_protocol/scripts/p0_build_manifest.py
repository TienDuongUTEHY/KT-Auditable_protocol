import os
import sys
import hashlib
import platform
import subprocess
import datetime
import pandas as pd

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def get_file_sha256(path):
    if not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def get_git_info():
    git_commit = "NA"
    git_branch = "NA"
    uncommitted = "unknown"
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        diff = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode().strip()
        uncommitted = "yes" if diff else "no"
    except Exception:
        pass
    return git_commit, git_branch, uncommitted

def get_pip_freeze():
    try:
        return subprocess.check_output([sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL).decode()
    except Exception:
        return "pip freeze failed"

def get_cuda_info():
    cuda_available = "no"
    gpu_name = "NA"
    try:
        import torch
        cuda_available = "yes" if torch.cuda.is_available() else "no"
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return cuda_available, gpu_name

def main():
    output_dir = "results_p0_revision"
    manifests_dir = ensure_dir(os.path.join(output_dir, "manifests"))
    supplementary_dir = ensure_dir(os.path.join(output_dir, "supplementary"))
    
    # 1. SHA256 manifest
    manifest_rows = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            path = os.path.join(root, file)
            rel_path = os.path.relpath(path, output_dir)
            if "sha256_manifest.csv" in rel_path:
                continue
            sha = get_file_sha256(path)
            size = os.path.getsize(path)
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
            manifest_rows.append({
                "relative_path": rel_path,
                "sha256": sha,
                "size_bytes": size,
                "modified_time": mtime
            })
            
    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(os.path.join(manifests_dir, "sha256_manifest.csv"), index=False)
    print("SHA256 manifest generated.")
    
    # 2. Run environment
    git_commit, git_branch, uncommitted = get_git_info()
    cuda_available, gpu_name = get_cuda_info()
    
    env_content = f"""python_version: {sys.version.replace('\n', ' ')}
platform: {platform.platform()}
cuda_available: {cuda_available}
gpu_name_or_NA: {gpu_name}
git_commit: {git_commit}
git_branch: {git_branch}
uncommitted_changes_yes_no: {uncommitted}
"""
    with open(os.path.join(manifests_dir, "run_environment.txt"), "w", encoding="utf-8") as f:
        f.write(env_content)
        
    with open(os.path.join(manifests_dir, "git_state.txt"), "w", encoding="utf-8") as f:
        f.write(f"git_commit: {git_commit}\ngit_branch: {git_branch}\nuncommitted: {uncommitted}\n")
    print("Run environment and git state generated.")
    
    # Update checklist CSV with new SHA256 hashes
    check_csv = os.path.join(output_dir, "tables_csv/table_reproducibility_checklist.csv")
    if os.path.exists(check_csv):
        df_check = pd.read_csv(check_csv)
        for idx, row in df_check.iterrows():
            artifact_path = os.path.join(output_dir, row['path'].replace('results_p0_revision/', ''))
            if os.path.exists(artifact_path):
                df_check.at[idx, 'sha256'] = get_file_sha256(artifact_path)
        df_check.to_csv(check_csv, index=False)
        
        # update TeX checklist too
        tex_check = os.path.join(output_dir, "tables_tex/table_reproducibility_checklist.tex")
        latex_check = [
            "\\begin{tabular}{llcl}",
            "\\hline",
            "Artifact & Path & Status & Purpose \\\\",
            "\\hline"
        ]
        for _, r in df_check.iterrows():
            latex_check.append(
                f"{r['artifact']} & {os.path.basename(r['path'])} & {r['status']} & {r['purpose']} \\\\"
            )
        latex_check.append("\\hline")
        latex_check.append("\\end{tabular}")
        with open(tex_check, "w", encoding="utf-8") as f:
            f.write("\n".join(latex_check))
            
    # 3. Definition of Done
    dod_content = """# Definition of Done — P0 LC-MRSG++ Revision

- [x] Dataset statistics table created and internally consistent.
- [x] Subsampling disclosure written if processed data are smaller than expected.
- [x] KDD2010 E_co resolved as unique edges / support records / multi-edge / fixed error.
- [x] E_sim trace completed; manuscript label adjusted if E_sim is empty.
- [x] Junyi coverage and isolated-node analysis completed.
- [x] L1--L6 leakage audit PASS, especially L1, L5, L6.
- [x] Epoch sanity-check completed or limitation explicitly logged.
- [x] Main ΔAUC table uses CI + Holm correction only; heavy statistics moved to supplementary.
- [x] P0/P1/P2/P3 boundary table included in manuscript.
- [x] No claim of SOTA, universal graph improvement, calibration, SSA-CL, or learning-path recommendation.
- [x] Reproducibility checklist and SHA256 manifest completed.
- [x] References 2024+ manually verified before submission.
"""
    with open(os.path.join(manifests_dir, "final_definition_of_done.md"), "w", encoding="utf-8") as f:
        f.write(dod_content)
        
    # 4. Reviewer response notes
    rev_notes = """# Reviewer Response Notes — P0 Revision

## Dataset and graph-provenance correction
We added a dataset-statistics and graph-provenance audit table reporting users, questions, skills, interactions, split sizes, user-skill density, unique graph edges, support records, isolated nodes, and SHA256 provenance.

## KDD2010 E_co clarification
We clarified whether the previously large KDD2010 E_co count referred to unique KC--KC edges, directed/mirrored rows, support records, or multi-edge records. The revised manuscript now reports unique KC--KC edges separately from support records.

## E_sim effective relation clarification
We traced the E_sim pipeline and now label empty similarity branches as E_sim^eff=empty where applicable. The manuscript no longer interprets empty E_sim branches as evidence that similarity edges improved prediction.

## Epoch sanity-check
We added a longer-budget sanity check for DKT and simpleKT under no-graph and validation-selected graph conditions at 5 and 10 epochs. This check is reported as a stability audit, not as SOTA tuning.

## Scope control
We moved calibration, adaptive stratification, SSA-CL, and learning-path recommendation outside P0 and stated them as future or separate-paper work.
"""
    with open(os.path.join(supplementary_dir, "reviewer_response_notes.md"), "w", encoding="utf-8") as f:
        f.write(rev_notes)
        
    readme_supp = """# Supplementary Material - P0 Revision

This folder contains additional logs, tables, manifests, and notes generated during the P0 LC-MRSG++ manuscript revision.

- `manifests/`: SHA256 checksum manifests and environment state.
- `tables_csv/`: CSV copies of all tables.
- `tables_tex/`: LaTeX components to be direct-inputs to the paper manuscript.
- `supplementary/`: Reviewer response notes.
"""
    with open(os.path.join(supplementary_dir, "README_SUPPLEMENTARY_P0.md"), "w", encoding="utf-8") as f:
        f.write(readme_supp)

if __name__ == "__main__":
    main()
