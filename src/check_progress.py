"""check_progress.py — kiểm tra tiến độ chạy xác nhận."""
import pandas as pd, os, sys
from datetime import datetime

f = 'ResultBS/confirmatory/confirmatory_results.csv'
if not os.path.exists(f):
    print('KHONG TIM THAY FILE KET QUA: ResultBS/confirmatory/confirmatory_results.csv')
    sys.exit(0)

df = pd.read_csv(f)
DATASETS = ['assist2012','junyi','kdd2010']
MODELS   = ['BKT','DKT','simpleKT','GIKT','SKT']
VARIANTS = ['no_graph','E_pre','E_pre_E_sim','E_pre_E_sim_E_co']
SEEDS    = [42,43,44]
TOTAL    = len(DATASETS)*len(MODELS)*len(VARIANTS)*len(SEEDS)   # 180

now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('='*65)
print(f'PROGRESS REPORT  {now_str}')
print('='*65)
print(f'  Tong runs hoan thanh : {len(df)} / {TOTAL}')
print(f'  Con lai              : {TOTAL - len(df)}')
pct_done = len(df)/TOTAL*100
bar = '#'*int(pct_done/2) + '-'*(50-int(pct_done/2))
print(f'  [{bar}] {pct_done:.1f}%')
print()

# Per-dataset breakdown
print(f"  {'Dataset':15s} {'Done':>6} {'Expected':>8} {'%':>6}")
print(f"  {'-'*40}")
for ds in DATASETS:
    sub = df[df['dataset']==ds]
    exp = len(MODELS)*len(VARIANTS)*len(SEEDS)   # 60
    print(f"  {ds:15s} {len(sub):6d} {exp:8d} {len(sub)/exp*100:6.1f}%")
    for model in MODELS:
        m = sub[sub['model']==model]
        exp_m = len(VARIANTS)*len(SEEDS)   # 12
        done_m = len(m)
        bar2 = '#'*done_m + '.'*(exp_m-done_m)
        print(f"    {model:10s} [{bar2}] {done_m}/{exp_m}")
print()

# ETA
if 'created_at' in df.columns and len(df) > 1:
    df['ts'] = pd.to_datetime(df['created_at'])
    first_ts = df['ts'].min()
    last_ts  = df['ts'].max()
    elapsed_min = (last_ts - first_ts).total_seconds() / 60
    remaining   = TOTAL - len(df)
    rate = len(df) / elapsed_min if elapsed_min > 0 else 0
    eta_min  = remaining / rate if rate > 0 else float('inf')
    eta_h    = eta_min / 60

    print(f"  Run dau tien  : {first_ts.strftime('%H:%M:%S %d/%m')}")
    print(f"  Run cuoi      : {last_ts.strftime('%H:%M:%S %d/%m')}")
    print(f"  Da chay       : {elapsed_min:.0f} phut ({elapsed_min/60:.1f} gio)")
    print(f"  Toc do        : {rate:.2f} runs/phut ({rate*60:.0f} runs/gio)")
    print(f"  Con lai       : {remaining} runs")
    if eta_h < 99:
        print(f"  Du kien xong  : {eta_min:.0f} phut nua (~{eta_h:.1f} gio)")
        finish_ts = last_ts + pd.Timedelta(minutes=eta_min)
        print(f"  Gio xong ~    : {(finish_ts + pd.Timedelta(hours=7)).strftime('%H:%M  %d/%m/%Y')} (ICT)")
    else:
        print(f"  Du kien xong  : Khong du lieu de tinh")

print()
print("  5 run cuoi:")
cols = ['dataset','model','graph_variant','seed','auc','acc','elapsed_s','created_at']
tail_cols = [c for c in cols if c in df.columns]
print(df.tail(5)[tail_cols].to_string(index=False))

# Check if already complete
if len(df) >= TOTAL:
    print()
    print('  *** DA HOAN THANH TOAN BO! ***')
    print('  Chay: python src/generate_q3_stats.py  de tao bang thong ke')
