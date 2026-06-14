$PythonPath = "D:\scientific_paper1\miniconda3\envs\scientific_paper1\python.exe"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Q3 EXPERIMENTAL UPGRADE - FULL PIPELINE (LIMITED COMPUTE)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# Ensure ResultBS exists
New-Item -Path "ResultBS" -ItemType Directory -Force | Out-Null

# Run Orchestrator
Write-Host "Starting python orchestrator..." -ForegroundColor Green
& $PythonPath -m src.q3_orchestrator

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "ALL PROCESSES COMPLETED. WAKE UP, PROFESSOR!" -ForegroundColor Cyan
Write-Host "Results are in ResultBS/" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
