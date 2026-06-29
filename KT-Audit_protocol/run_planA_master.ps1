$PythonPath = "D:\scientific_paper1\miniconda3\envs\scientific_paper1\python.exe"
$Datasets = @("kdd2010", "assist2012", "junyi")
$Seeds = @(2022, 2023, 2024, 2025, 2026)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "MASTER AUTOMATION RUN - PLAN A (5 SEEDS, 3 DATASETS)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Ensuring common python dependencies are available..."
& $PythonPath -m pip install --quiet matplotlib pandas numpy seaborn tqdm openml tabulate pyyaml

# First, clear out any leftover baseline results files so each run is fresh
foreach ($D in $Datasets) {
    $ResFile = "results/tables/$D/fold_0/baseline_results.csv"
    if (Test-Path $ResFile) {
        Remove-Item $ResFile -Force
        Write-Host "Cleaned previous baseline for $D"
    }
}

foreach ($Dataset in $Datasets) {
    Write-Host "`n==========================================================" -ForegroundColor Green
    Write-Host "PROCESSING DATASET: $Dataset" -ForegroundColor Green
    Write-Host "==========================================================" -ForegroundColor Green
    $Config = "configs/${Dataset}.yaml"
    
    Write-Host "[CORE] Normalizing full dataset cache..."
    & $PythonPath -m src.preprocess --config $Config

    foreach ($Seed in $Seeds) {
        Write-Host "  --> Executing Full Pipeline for Seed: $Seed" -ForegroundColor Yellow
        
        # Phase 1: Split Data by Seed
        & $PythonPath -m src.split_checker --config $Config --seed $Seed
        
        # Phase 2: Construction
        & $PythonPath -m src.qmatrix_provenance --config $Config --fold 0
        & $PythonPath -m src.tri_relation_graph_builder --config $Config --fold 0
        
        # Phase 3: Audit
        & $PythonPath -m src.dag_audit --config $Config --fold 0
        & $PythonPath -m src.eco_audit --config $Config --fold 0
        & $PythonPath -m src.leakage_audit --config $Config --fold 0
        
        # Phase 4: Profiles & Baseline Probes (5 Seeds aggregated here)
        & $PythonPath -m src.graph_statistics --config $Config --fold 0
        & $PythonPath -m src.sparse_skill_profile --config $Config --fold 0
        
        & $PythonPath -m src.baseline_probe --config $Config --fold 0 --model BKT --seed $Seed
        & $PythonPath -m src.baseline_probe --config $Config --fold 0 --model DKT --seed $Seed
        
        Write-Host "  [✓] Completed iteration for seed $Seed." -ForegroundColor Gray
    }
    
    # Post-process Dataset-wide artifacts
    Write-Host "[POST] Generating Visualizations and Master Report for $Dataset..."
    & $PythonPath -m src.make_figures --config $Config --fold 0
    & $PythonPath -m src.report_generator --config $Config --fold 0
    Write-Host "FINISHED DATASET: $Dataset`n" -ForegroundColor Green
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "ALL 3 DATASETS & 5 SEEDS EXECUTED SUCCESSFULLY!" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
