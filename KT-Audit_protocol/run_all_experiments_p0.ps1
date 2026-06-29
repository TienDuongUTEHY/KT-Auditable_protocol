param(
    [switch]$ForceRun = $false,
    [string]$PythonPath = "D:\scientific_paper1\miniconda3\envs\scientific_paper1\python.exe"
)

# Thu muc luu log
if (-not (Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" | Out-Null }

$Datasets = @("kdd2010", "junyi", "assist2012")
$Seed = 2026

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host " BAT DAU CHUOI THUC NGHIEM TU DONG CHO BAI BAO P0" -ForegroundColor Cyan
Write-Host " Danh sach Dataset: $($Datasets -join ', ')" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

foreach ($Dataset in $Datasets) {
    $Config = "configs/${Dataset}.yaml"
    $LogFile = "logs/run_log_${Dataset}_full.txt"
    
    # Bat dau ghi log cho dataset nay
    Start-Transcript -Path $LogFile -Append
    
    Write-Host "`n--- DANG XU LY DATASET: $Dataset ---" -ForegroundColor Yellow
    Write-Host "Cau hinh: $Config" -ForegroundColor DarkGray

    # Ham tien ich kiem tra va chay
    function Run-Step {
        param (
            [string]$StepName,
            [string]$CommandArgs,
            [string]$ExpectedOutputFile
        )
        
        Write-Host "`n[*] $StepName" -ForegroundColor Green
        
        if (-not $ForceRun -and $ExpectedOutputFile -ne "" -and (Test-Path $ExpectedOutputFile)) {
            Write-Host "  -> Da tim thay ket qua cu: $ExpectedOutputFile. BO QUA de tiet kiem thoi gian!" -ForegroundColor DarkGray
        } else {
            Write-Host "  -> Dang thuc thi: python -m $CommandArgs" -ForegroundColor DarkGray
            
            $FullCommand = "& `"$PythonPath`" -m $CommandArgs"
            Invoke-Expression $FullCommand
            
            if ($LASTEXITCODE -ne 0) {
                Write-Host "  [LOI] Tien trinh that bai tai buoc: $StepName. Dung thuc thi cho $Dataset." -ForegroundColor Red
                Stop-Transcript
                return $false
            }
            
            # Ep he dieu hanh don rac (GC) sau moi buoc
            Write-Host "  -> Hoan thanh. Tam nghi 3 giay de giai phong RAM (Garbage Collection)..." -ForegroundColor DarkGray
            Start-Sleep -Seconds 3
        }
        return $true
    }

    # BUOC 1: Preprocess
    $Expected1 = "data/processed/${Dataset}/interactions.csv"
    if (-not (Run-Step -StepName "[1/12] Chuan hoa du lieu tho (Preprocess)" -CommandArgs "src.preprocess --config $Config" -ExpectedOutputFile $Expected1)) { continue }

    # BUOC 2: Split Checker
    $Expected2 = "data/processed/${Dataset}/fold_0/train.csv"
    if (-not (Run-Step -StepName "[2/12] Chia du lieu va kiem tra ro ri (Split Checker)" -CommandArgs "src.split_checker --config $Config --seed $Seed" -ExpectedOutputFile $Expected2)) { continue }

    # BUOC 3: Q-Matrix Provenance
    $Expected3 = "results/tables/${Dataset}/fold_0/qmatrix_audit.csv"
    if (-not (Run-Step -StepName "[3/12] Truy xuat nguon goc Q-Matrix (Q-Matrix Provenance)" -CommandArgs "src.qmatrix_provenance --config $Config --fold 0" -ExpectedOutputFile $Expected3)) { continue }

    # BUOC 4: Tri-relation Graph Builder
    $Expected4 = "results/tables/${Dataset}/fold_0/E_pre_train.csv"
    if (-not (Run-Step -StepName "[4/12] Xay dung do thi 3 quan he (Tri-relation Graph Builder)" -CommandArgs "src.tri_relation_graph_builder --config $Config --fold 0" -ExpectedOutputFile $Expected4)) { continue }

    # BUOC 5: DAG Audit
    $Expected5 = "results/tables/${Dataset}/fold_0/dag_audit.csv"
    if (-not (Run-Step -StepName "[5/12] Kiem toan Do thi co huong (DAG Audit)" -CommandArgs "src.dag_audit --config $Config --fold 0" -ExpectedOutputFile $Expected5)) { continue }

    # BUOC 6: ECO Audit
    $Expected6 = "results/tables/${Dataset}/fold_0/eco_audit.csv"
    if (-not (Run-Step -StepName "[6/12] Kiem toan quan he Dong xuat hien (ECO Audit)" -CommandArgs "src.eco_audit --config $Config --fold 0" -ExpectedOutputFile $Expected6)) { continue }

    # BUOC 7: Leakage Audit
    $Expected7 = "results/tables/${Dataset}/fold_0/leakage_audit_log.csv"
    if (-not (Run-Step -StepName "[7/12] Kiem toan ro ri 5 cap do (Leakage Audit)" -CommandArgs "src.leakage_audit --config $Config --fold 0" -ExpectedOutputFile $Expected7)) { continue }

    # BUOC 8: Graph Statistics
    $Expected8 = "results/tables/${Dataset}/fold_0/graph_stats.csv"
    if (-not (Run-Step -StepName "[8/12] Thong ke Do thi (Graph Statistics)" -CommandArgs "src.graph_statistics --config $Config --fold 0" -ExpectedOutputFile $Expected8)) { continue }

    # BUOC 9: Sparse Skill Profile
    $Expected9 = "results/tables/${Dataset}/fold_0/sparse_skill_profile.csv"
    if (-not (Run-Step -StepName "[9/12] Ho so Ky nang thua thot (Sparse Skill Profile)" -CommandArgs "src.sparse_skill_profile --config $Config --fold 0" -ExpectedOutputFile $Expected9)) { continue }

    # BUOC 10: Baseline Probes
    $Expected10 = "results/tables/${Dataset}/fold_0/baseline_results.csv"
    Write-Host "`n[*] [10/12] Chay cac mo hinh co so danh gia chan doan (Baseline Probes)" -ForegroundColor Green
    if (-not $ForceRun -and (Test-Path $Expected10)) {
        Write-Host "  -> Da tim thay ket qua cu: $Expected10. BO QUA de tiet kiem thoi gian!" -ForegroundColor DarkGray
    } else {
        $SeedsToRun = @(2024, 2025, 2026)
        $ModelsToRun = @("DKT", "simpleKT", "GKT", "GIKT")
        foreach ($s in $SeedsToRun) {
            foreach ($m in $ModelsToRun) {
                Write-Host "  -> Dang thuc thi mo hinh $m voi Seed $s..." -ForegroundColor DarkGray
                Invoke-Expression "& `"$PythonPath`" -m src.baseline_probe --config $Config --fold 0 --model $m --seed $s"
                if ($LASTEXITCODE -ne 0) { Write-Host "  [CANH BAO] Loi khi chay $m voi seed $s."; }
            }
        }
        Start-Sleep -Seconds 3
    }

    # BUOC 11: Make Figures
    $Expected11 = "results/figures/${Dataset}/fold_0/fig4_relation_ablation.pdf"
    if (-not (Run-Step -StepName "[11/12] Ve Bieu do Bao cao (Make Figures)" -CommandArgs "src.make_figures --config $Config --fold 0" -ExpectedOutputFile $Expected11)) { continue }

    # BUOC 12: Report Generator
    $Expected12 = "results/reports/${Dataset}/fold_0/p0_diagnostic_report.md"
    if (-not (Run-Step -StepName "[12/12] Tong hop Bao cao chan doan cuoi cung (Report Generator)" -CommandArgs "src.report_generator --config $Config --fold 0" -ExpectedOutputFile $Expected12)) { continue }

    Write-Host "`n--- HOAN THANH XUAT SAC DATASET: $Dataset ---" -ForegroundColor Cyan
    Stop-Transcript
}

Write-Host "`n================================================================" -ForegroundColor Cyan
Write-Host " HOAN THANH TOAN BO CHUOI THUC NGHIEM!" -ForegroundColor Cyan
Write-Host " Log files duoc luu tai thu muc: logs/" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
