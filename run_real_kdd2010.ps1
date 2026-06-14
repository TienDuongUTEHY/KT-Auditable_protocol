$PythonPath = "D:\scientific_paper1\miniconda3\envs\scientific_paper1\python.exe"

Write-Host "Starting FULL replacement run for KDD Cup 2010 using real Algebra 2005-2006 dataset..."

$Dataset = "kdd2010"
$Seed = 2026
$Config = "configs/${Dataset}.yaml"

Write-Host "[1/12] Running PREPROCESS..."
& $PythonPath -m src.preprocess --config $Config

Write-Host "[2/12] Running SPLIT CHECKER..."
& $PythonPath -m src.split_checker --config $Config --seed $Seed

Write-Host "[3/12] Running Q-MATRIX PROVENANCE..."
& $PythonPath -m src.qmatrix_provenance --config $Config --fold 0

Write-Host "[4/12] Running TRI-RELATION GRAPH BUILDER..."
& $PythonPath -m src.tri_relation_graph_builder --config $Config --fold 0

Write-Host "[5/12] Running DAG AUDIT..."
& $PythonPath -m src.dag_audit --config $Config --fold 0

Write-Host "[6/12] Running ECO AUDIT..."
& $PythonPath -m src.eco_audit --config $Config --fold 0

Write-Host "[7/12] Running LEAKAGE AUDIT..."
& $PythonPath -m src.leakage_audit --config $Config --fold 0

Write-Host "[8/12] Running GRAPH STATISTICS..."
& $PythonPath -m src.graph_statistics --config $Config --fold 0

Write-Host "[9/12] Running SPARSE SKILL PROFILE..."
& $PythonPath -m src.sparse_skill_profile --config $Config --fold 0

Write-Host "[10/12] Running BASELINE PROBES (BKT & DKT)..."
& $PythonPath -m src.baseline_probe --config $Config --fold 0 --model BKT
& $PythonPath -m src.baseline_probe --config $Config --fold 0 --model DKT

Write-Host "[11/13] Running DAG DISRUPTION (DDR)..."
& $PythonPath -m src.dag_disruption --config $Config --fold 0 --seed $Seed

Write-Host "[12/13] Running MAKE FIGURES..."
& $PythonPath -m src.make_figures --config $Config --fold 0

Write-Host "[13/13] Running REPORT GENERATOR..."
& $PythonPath -m src.report_generator --config $Config --fold 0

Write-Host "ALL KDD2010 RESULTS SUCCESSFULLY OVERWRITTEN WITH ALGEBRA 2005-2006 DATA."
