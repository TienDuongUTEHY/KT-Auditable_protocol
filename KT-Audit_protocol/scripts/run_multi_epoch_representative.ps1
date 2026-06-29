$PythonPath = "D:\scientific_paper1\miniconda3\envs\scientific_paper1\python.exe"
$Dataset = "assist2012"
$Fold = 0
$Seed = 42
$Config = "configs/${Dataset}.yaml"

$ModelsToRun = @("bkt", "dkt", "simplekt", "gikt", "skt")

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host " CHAY THUC NGHIEM NHIEU EPOCH (DAI DIEN) DE LAY DUONG CONG HOC TAP" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

foreach ($m in $ModelsToRun) {
    Write-Host "  -> Dang thuc thi mo hinh $m voi 50 epochs..." -ForegroundColor Yellow
    Invoke-Expression "& `"$PythonPath`" -m src.baseline_probe --config $Config --fold $Fold --model $m --seed $Seed --epochs 50"
}

Write-Host "  -> Tong hop log va ve bieu do..." -ForegroundColor Yellow
Invoke-Expression "& `"$PythonPath`" scripts/export_epoch_logs.py"
Invoke-Expression "& `"$PythonPath`" scripts/audit_training_integrity.py"
Invoke-Expression "& `"$PythonPath`" scripts/plot_training_curves.py"

Write-Host "`n================================================================" -ForegroundColor Cyan
Write-Host " HOAN THANH!" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
