#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/q3_fix_config.yaml}
python scripts/lc_mrsg_q3_fix_pipeline.py --config "$CONFIG" --task scale
python scripts/lc_mrsg_q3_fix_pipeline.py --config "$CONFIG" --task aggregate
python scripts/lc_mrsg_q3_fix_pipeline.py --config "$CONFIG" --task zero
python scripts/lc_mrsg_q3_fix_pipeline.py --config "$CONFIG" --task epre
python scripts/lc_mrsg_q3_fix_pipeline.py --config "$CONFIG" --task eco
python scripts/lc_mrsg_q3_fix_pipeline.py --config "$CONFIG" --task sparse
python scripts/lc_mrsg_q3_fix_pipeline.py --config "$CONFIG" --task consistency

echo "Done. Check results/q3_fix/tables and results/q3_fix/reports."
