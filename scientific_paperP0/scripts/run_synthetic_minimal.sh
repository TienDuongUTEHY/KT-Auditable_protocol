#!/bin/bash
python -m src.synthetic_data --config configs/synthetic.yaml
python -m src.preprocess --config configs/synthetic.yaml
python -m src.split_checker --config configs/synthetic.yaml --seed 2026
python -m src.qmatrix_provenance --config configs/synthetic.yaml --fold 0
python -m src.tri_relation_graph_builder --config configs/synthetic.yaml --fold 0
python -m src.dag_audit --config configs/synthetic.yaml --fold 0
python -m src.eco_audit --config configs/synthetic.yaml --fold 0
python -m src.leakage_audit --config configs/synthetic.yaml --fold 0
python -m src.graph_statistics --config configs/synthetic.yaml --fold 0
python -m src.sparse_skill_profile --config configs/synthetic.yaml --fold 0
python -m src.baseline_probe --config configs/synthetic.yaml --fold 0 --model BKT
python -m src.baseline_probe --config configs/synthetic.yaml --fold 0 --model DKT
python -m src.make_figures --config configs/synthetic.yaml --fold 0
python -m src.report_generator --config configs/synthetic.yaml --fold 0
