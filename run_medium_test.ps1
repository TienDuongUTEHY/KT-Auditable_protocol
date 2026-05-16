$PythonPath = "D:\scientific_paper1\miniconda3\envs\scientific_paper1\python.exe"

Write-Host "Starting Medium-Test for Junyi, ASSISTments2012, and KDDcup2010..."

# 1. Download/Mock Data (Medium size with cycles)
echo "Preparing dataset files (Medium Size)..."
& $PythonPath -m src.download_kdd

$Datasets = @("junyi", "assist2012", "kdd2010")
$Seed = 2026

foreach ($Dataset in $Datasets) {
    Write-Host "--- Running full core logic reproduction for $Dataset with seed $Seed ---"
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
    & $PythonPath -m src.baseline_probe --config $Config --fold 0 --model BKT
    & $PythonPath -m src.baseline_probe --config $Config --fold 0 --model DKT
    & $PythonPath -m src.make_figures --config $Config --fold 0
    & $PythonPath -m src.report_generator --config $Config --fold 0
    Write-Host "Done $Dataset reproduction!"
}

Write-Host "Medium-Test Completed Successfully across all 3 datasets."
