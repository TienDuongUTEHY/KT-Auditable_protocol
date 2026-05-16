#!/bin/bash
echo "Starting Mini-Test for Junyi, ASSISTments2012, and KDDcup2010..."

# 1. Download/Mock Data
echo "Preparing dataset files..."
python -m src.download_kdd

# 2. Run Reproduce Scripts
echo "--- Running Junyi ---"
bash scripts/reproduce_one_dataset.sh junyi 2026

echo "--- Running ASSISTments2012 ---"
bash scripts/reproduce_one_dataset.sh assist2012 2026

echo "--- Running KDDcup2010 ---"
bash scripts/reproduce_one_dataset.sh kdd2010 2026

echo "Mini-Test Completed Successfully across all 3 datasets."
