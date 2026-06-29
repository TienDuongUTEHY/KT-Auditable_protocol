# run_confirmatory_A.ps1
# =====================================================================
# Option-A Confirmatory Experiment Runner (Updated for 5 Seeds)
# 3 datasets x 3 folds x 5 seeds x 5 models x 4 graph variants
# Fully automated - no user input required
# =====================================================================

$ProjectRoot = "d:\Paper P0 Nguyen Tien Duong\SCIE_P0\KT-Auditable_protocol"
Set-Location $ProjectRoot

$PythonPath = "D:\scientific_paper1\miniconda3\envs\scientific_paper1\python.exe"
$LogFile  = "$ProjectRoot\ResultBS\confirmatory\run_log.txt"
$ErrorLog = "$ProjectRoot\ResultBS\confirmatory\error_log.txt"

New-Item -ItemType Directory -Force -Path "$ProjectRoot\ResultBS\confirmatory" | Out-Null

function Write-Log {
    param([string]$msg)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $msg"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

Write-Log "=== Confirmatory Option-A started ==="
Write-Log "Running on local machine"

# Step 0: Install dependencies if missing
Write-Log "Checking dependencies..."
& $PythonPath -c "import psutil" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Log "Installing psutil..."
    & $PythonPath -m pip install psutil --quiet
}
& $PythonPath -c "import scipy" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Log "Installing scipy..."
    & $PythonPath -m pip install scipy --quiet
}

# Step 1: Quick smoke test (fold 0, seed 42, 1 model, 1 dataset)
Write-Log "Running smoke test (assist2012, fold=0, seed=42, BKT only)..."
& $PythonPath -m src.confirmatory_runner `
    --datasets assist2012 `
    --seeds 42 `
    --folds 0 2>&1 | Tee-Object -Append -FilePath $LogFile

if ($LASTEXITCODE -ne 0) {
    Write-Log "ERROR: Smoke test failed. Check $ErrorLog"
    exit 1
}
Write-Log "Smoke test passed."

# Step 2: Full run - all datasets, 5 seeds, 3 folds
Write-Log "=== Starting full confirmatory run (including Seeds 2025, 2026) ==="
Write-Log "Datasets : assist2012  junyi  kdd2010"
Write-Log "Seeds    : 42  43  44  2025  2026"
Write-Log "Folds    : 0  1  2"
Write-Log "Models   : BKT  DKT  simpleKT  GIKT  SKT"
Write-Log "Variants : no_graph  E_pre  E_pre_E_sim  E_pre_E_sim_E_co"

$StartTime = Get-Date

& $PythonPath -m src.confirmatory_runner `
    --datasets assist2012 junyi kdd2010 `
    --seeds 42 43 44 2025 2026 `
    --folds 0 1 2 2>&1 | Tee-Object -Append -FilePath $LogFile

$ExitCode = $LASTEXITCODE
$Duration = (Get-Date) - $StartTime
Write-Log "Full run finished. Exit=$ExitCode Duration=$Duration"

if ($ExitCode -ne 0) {
    Write-Log "ERROR in full run - partial results still saved in ResultBS/confirmatory/"
    exit 1
}

# Step 3: Generate summary report
Write-Log "Generating summary tables..."
$SummaryPyPath = "$ProjectRoot\ResultBS\confirmatory\generate_summary.py"
& $PythonPath $SummaryPyPath

Write-Log "=== All done. Results in ResultBS/confirmatory/ ==="
Write-Log "Key files:"
Write-Log "  confirmatory_results.csv  - raw per-run results"
Write-Log "  multifold_summary.csv     - mean/std across seeds/folds"
Write-Log "  run_log.txt               - full execution log"
