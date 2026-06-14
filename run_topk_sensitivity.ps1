$PythonPath = "D:\scientific_paper1\miniconda3\envs\scientific_paper1\python.exe"
$Datasets = @("assist2012", "junyi", "kdd2010")
$TopKValues = @(3, 5, 10)
$Seed = 2026

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "TOP-K SENSITIVITY ANALYSIS FOR SIMILARITY GRAPH (E_sim)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

foreach ($Dataset in $Datasets) {
    Write-Host "`nPROCESSING DATASET: $Dataset" -ForegroundColor Green
    $Config = "configs/${Dataset}.yaml"
    
    foreach ($TopK in $TopKValues) {
        Write-Host "  --> Generating E_sim with Top-$TopK..." -ForegroundColor Magenta
        # Re-run graph builder with top_k override
        & $PythonPath -m src.tri_relation_graph_builder --config $Config --fold 0 --top_k $TopK
        
        Write-Host "  --> Evaluating DKT with Top-$TopK Graph..." -ForegroundColor Yellow
        & $PythonPath -m src.baseline_probe --config $Config --fold 0 --model DKT --seed $Seed --notes "Top-K=$TopK"
        & $PythonPath -m src.baseline_probe --config $Config --fold 0 --model BKT --seed $Seed --notes "Top-K=$TopK"
    }
}

Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host "TOP-K SENSITIVITY ANALYSIS COMPLETED!" -ForegroundColor Cyan
Write-Host "Results appended to results/tables/<dataset>/fold_0/baseline_results.csv" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
