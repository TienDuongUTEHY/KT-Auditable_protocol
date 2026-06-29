import os
import shutil
from pathlib import Path

def copy_tree(src, dst):
    if not src.exists():
        print(f"Source does not exist: {src}")
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.glob("**/*"):
        if item.is_file():
            rel_path = item.relative_to(src)
            target = dst / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
            # print(f"Copied: {rel_path}")

def main():
    src_results = Path(r"d:\Paper P0 Nguyen Tien Duong\SCIE_P0\KT-Auditable_protocol\KT-Auditable_protocol\results")
    dst_results = Path(r"D:\Paper P0 Nguyen Tien Duong\Paper_P0\KT-Auditable_protocol\results")
    
    print(f"Syncing results from:\n  {src_results}\nto:\n  {dst_results}")
    copy_tree(src_results, dst_results)
    print("Syncing complete successfully!")

if __name__ == "__main__":
    main()
