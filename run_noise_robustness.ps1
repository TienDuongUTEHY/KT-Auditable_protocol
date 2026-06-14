$PythonPath = "D:\scientific_paper1\miniconda3\envs\scientific_paper1\python.exe"
$Datasets = @("assist2012", "junyi", "kdd2010")
$NoiseLevels = @(0.10, 0.20, 0.50, 1.00)
$Seed = 2026

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "NOISE ROBUSTNESS EVALUATION (10%, 20%, 50%, 100%) FOR DKT" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

foreach ($Dataset in $Datasets) {
    Write-Host "`nPROCESSING DATASET: $Dataset" -ForegroundColor Green
    $Config = "configs/${Dataset}.yaml"
    $ProcDir = "data/processed/$Dataset/fold_0"
    
    foreach ($Noise in $NoiseLevels) {
        Write-Host "  --> Generating $Noise Noise Data..." -ForegroundColor Magenta
        & $PythonPath src/noise_robustness.py --config $Config --fold 0 --seed $Seed --noise_ratio $Noise
        
        Write-Host "  --> Evaluating DKT on $Noise Noise Data..." -ForegroundColor Yellow
        $NoisyDir = "$ProcDir/noisy_$Noise"
        & $PythonPath -m src.baseline_probe --config $Config --fold 0 --model DKT --seed $Seed --data_dir $NoisyDir --noise_ratio $Noise
    }
}

Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host "NOISE ROBUSTNESS EVALUATION COMPLETED!" -ForegroundColor Cyan
Write-Host "Results appended to results/tables/<dataset>/fold_0/baseline_results.csv" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
