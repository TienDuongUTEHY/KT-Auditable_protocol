#!/bin/bash
DATASET=$1
SEED=$2
echo "Running full reproduction for ${DATASET} with seed ${SEED}..."
CONFIG="configs/${DATASET}.yaml"

python -m src.preprocess --config ${CONFIG}
python -m src.split_checker --config ${CONFIG} --seed ${SEED}
python -m src.qmatrix_provenance --config ${CONFIG} --fold 0
python -m src.tri_relation_graph_builder --config ${CONFIG} --fold 0
python -m src.dag_audit --config ${CONFIG} --fold 0
python -m src.eco_audit --config ${CONFIG} --fold 0
python -m src.leakage_audit --config ${CONFIG} --fold 0
python -m src.graph_statistics --config ${CONFIG} --fold 0
python -m src.sparse_skill_profile --config ${CONFIG} --fold 0
python -m src.baseline_probe --config ${CONFIG} --fold 0 --model BKT
python -m src.baseline_probe --config ${CONFIG} --fold 0 --model DKT
python -m src.make_figures --config ${CONFIG} --fold 0
python -m src.report_generator --config ${CONFIG} --fold 0
echo "Done ${DATASET} reproduction!"
