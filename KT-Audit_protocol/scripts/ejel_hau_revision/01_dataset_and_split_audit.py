# -*- coding: utf-8 -*-
"""
Stage 1: Audits dataset splits and user-skill cell interaction intensity.
Orchestrated automatically via run_all.py.
"""

import sys
import subprocess

def main():
    print("Stage 1: Audits dataset splits and user-skill cell interaction intensity.")
    print("To execute the full pipeline automatically, please run:")
    print("python -m scripts.ejel_hau_revision.run_all --config configs/ejel_hau_revision_config.yaml --output-root results_ejel_hau_revision --auto --resume")
    print("Orchestrated via run_all.py.")

if __name__ == "__main__":
    main()
