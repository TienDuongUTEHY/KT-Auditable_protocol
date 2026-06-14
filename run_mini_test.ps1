$PythonPath = "D:\scientific_paper1\miniconda3\envs\scientific_paper1\python.exe"

$Datasets = @("junyi", "assist2012", "kdd2010")
$Seed = 2026

foreach ($Dataset in $Datasets) {
    Write-Host "Running full reproduction for $Dataset with seed $Seed..."
    $Config = "configs/${Dataset}.yaml"

    & $PythonPath -m src.preprocess --config $Config
    & $PythonPath -m src.split_checker --config $Config --seed $Seed
    & $PythonPath -m src.qmatrix_provenance --config $Config --fold 0
    & $PythonPath -m src.tri_relation_graph_builder --config $Config --fold 0
    & $PythonPath -m src.dag_audit --config $Config --fold 0
    & $PythonPath -m src.eco_audit --config $Config --fold 0
    & $PythonPath -m src.leakage_audit --config $Config --fold 0
    & $PythonPath -m src.graph_statistics --config $Config --fold 0
    & $PythonPath -m src.sparse_skill_profile --config $Config --fold 0
    & $PythonPath -m src.dag_disruption --config $Config --fold 0 --seed $Seed
    & $PythonPath -m src.baseline_probe --config $Config --fold 0 --model BKT
    & $PythonPath -m src.baseline_probe --config $Config --fold 0 --model DKT
    & $PythonPath -m src.make_figures --config $Config --fold 0
    & $PythonPath -m src.report_generator --config $Config --fold 0
    Write-Host "Done $Dataset reproduction!"
}

Write-Host "Generating Final Publication Tables..."
& $PythonPath scripts/generate_publication_tables.py

Write-Host "Mini-Test Completed Successfully across all 3 datasets."
