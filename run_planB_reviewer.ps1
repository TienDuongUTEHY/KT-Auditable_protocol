$PythonPath = "D:\scientific_paper1\miniconda3\envs\scientific_paper1\python.exe"
$Datasets = @("kdd2010", "assist2012", "junyi")
$Seeds = @(2024, 2025, 2026) # 3 seeds
$Models = @("BKT", "DKT", "simpleKT", "GIKT", "SKT")

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "MASTER AUTOMATION RUN - PLAN B (REVIEWER RESPONSES)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

foreach ($Dataset in $Datasets) {
    Write-Host "`n==========================================================" -ForegroundColor Green
    Write-Host "PROCESSING DATASET: $Dataset" -ForegroundColor Green
    Write-Host "==========================================================" -ForegroundColor Green
    $Config = "configs/${Dataset}.yaml"
    
    # 10. Temporal 80/20 deployment-style split
    Write-Host "[CORE] Running Preprocess for Deployment 80/20 Split..."
    & $PythonPath -m src.preprocess --config $Config

    foreach ($Seed in $Seeds) {
        Write-Host "  --> Executing Pipeline for Seed: $Seed" -ForegroundColor Yellow
        
        # Phase 1: Split
        & $PythonPath -m src.split_checker --config $Config --seed $Seed
        
        # Phase 2: Graph Construction (includes Top-K Fix)
        & $PythonPath -m src.qmatrix_provenance --config $Config --fold 0
        & $PythonPath -m src.tri_relation_graph_builder --config $Config --fold 0
        
        # Phase 3: Audits
        & $PythonPath -m src.dag_audit --config $Config --fold 0
        & $PythonPath -m src.eco_audit --config $Config --fold 0
        & $PythonPath -m src.leakage_audit --config $Config --fold 0
        
        # Phase 4: Stats & Profiles (Strata coverage)
        & $PythonPath -m src.graph_statistics --config $Config --fold 0
        & $PythonPath -m src.sparse_skill_profile --config $Config --fold 0
        & $PythonPath -m src.dag_disruption --config $Config --fold 0 --seed $Seed
        
        # Phase 5: Run Models (BKT, DKT, SimpleKT, GIKT, SKT)
        foreach ($Mod in $Models) {
            & $PythonPath -m src.baseline_probe --config $Config --fold 0 --model $Mod --seed $Seed
        }
        
        # Phase 6: Noise Robustness (10% and 20%)
        Write-Host "  --> Running Noise Robustness (10% and 20%)..." -ForegroundColor Magenta
        & $PythonPath src/noise_robustness.py --config $Config --fold 0 --seed $Seed --noise_ratio 0.10
        & $PythonPath src/noise_robustness.py --config $Config --fold 0 --seed $Seed --noise_ratio 0.20
        
        $ProcDir = "data/processed/$Dataset/fold_0"
        & $PythonPath -m src.baseline_probe --config $Config --fold 0 --model DKT --seed $Seed --data_dir "$ProcDir/noisy_0.1"
        & $PythonPath -m src.baseline_probe --config $Config --fold 0 --model DKT --seed $Seed --data_dir "$ProcDir/noisy_0.2"
        
        Write-Host "  [v] Completed iteration for seed $Seed." -ForegroundColor Gray
    }
    
    # Visualizations
    & $PythonPath -m src.make_figures --config $Config --fold 0
    & $PythonPath -m src.report_generator --config $Config --fold 0
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "ALL 3 DATASETS AND 3 SEEDS EXECUTED SUCCESSFULLY!" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

Write-Host "[POST] Generating Final Publication Tables (144-run export, Taxonomy)..." -ForegroundColor Cyan
& $PythonPath scripts/generate_publication_tables.py
