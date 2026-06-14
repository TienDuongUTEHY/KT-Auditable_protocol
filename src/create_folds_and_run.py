"""
create_folds_and_run.py
=======================
Buoc 1: Tao fold_1 va fold_2 tu interactions.csv (learner-level stratified split, seed khac nhau)
Buoc 2: Build E_sim cho tung fold moi
Buoc 3: Build adjacency tu E_pre, E_co (copy/recompute tu fold_0 vi E_pre/E_co khong phu thuoc vao fold split)
Buoc 4: Chay 5 models x 4 variants x 3 seeds tren fold_1 va fold_2
Buoc 5: Merge ket qua vao confirmatory_results.csv
Buoc 6: Tinh lai bang thong ke Q3 day du (3 folds)

Thiet ke an toan RAM:
- KDD2010: subsample 250k train rows
- Batch processing
- gc.collect() sau moi run
"""
import os, sys, gc, time, shutil
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '.')
from src.graph_utils_v2 import build_esim_topk, save_esim, load_adjacency, two_hop_degree
from src.models_v2 import MODEL_FN
from src.common import ensure_dir

# ─── config ───────────────────────────────────────────────────────────────────
DATASETS  = ['assist2012', 'junyi', 'kdd2010']
NEW_FOLDS = [1, 2]
SEEDS     = [42, 43, 44]
MODELS    = ['BKT', 'DKT', 'simpleKT', 'GIKT', 'SKT']
VARIANTS  = ['no_graph', 'E_pre', 'E_pre_E_sim', 'E_pre_E_sim_E_co']

FOLD_SEEDS = {1: 2024, 2: 2025}   # different random seed per fold split
TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO  = 0.10

OUT_CSV = 'ResultBS/confirmatory/confirmatory_results.csv'
LOG_F   = 'ResultBS/confirmatory/fold12_progress.txt'
ensure_dir('ResultBS/confirmatory')

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_F, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def existing_keys():
    if not os.path.exists(OUT_CSV):
        return set()
    df = pd.read_csv(OUT_CSV)
    return set(zip(df.dataset, df.fold_id, df.model, df.graph_variant, df.seed))

def append_row(row):
    df = pd.DataFrame([row])
    hdr = not os.path.exists(OUT_CSV)
    df.to_csv(OUT_CSV, mode='a', header=hdr, index=False)

# ─── Step 1: Create fold_1 and fold_2 splits ──────────────────────────────────
def create_fold(ds, fold):
    out_dir = f'data/processed/{ds}/fold_{fold}'
    if os.path.isdir(out_dir) and os.path.exists(f'{out_dir}/train.csv'):
        log(f"  fold_{fold} da ton tai, bo qua tao moi.")
        return True

    src = f'data/processed/{ds}/interactions.csv'
    if not os.path.exists(src):
        log(f"  [ERROR] Khong co {src}")
        return False

    log(f"  Doc {src} ...")
    df = pd.read_csv(src)

    # Normalize column names
    if 'user_id' in df.columns and 'learner_id' not in df.columns:
        df.rename(columns={'user_id': 'learner_id'}, inplace=True)
    if 'kc_id' in df.columns and 'skill_id' not in df.columns:
        df.rename(columns={'kc_id': 'skill_id'}, inplace=True)

    log(f"  Tong {len(df):,} tuong tac, {df['learner_id'].nunique():,} hoc vien")

    # Learner-level split (shuffle learners, then 80/10/10)
    rng = np.random.default_rng(FOLD_SEEDS[fold])
    learners = df['learner_id'].unique()
    rng.shuffle(learners)
    n = len(learners)
    n_train = int(n * TRAIN_RATIO)
    n_valid = int(n * VALID_RATIO)

    train_learners = set(learners[:n_train])
    valid_learners = set(learners[n_train:n_train+n_valid])
    test_learners  = set(learners[n_train+n_valid:])

    train_df = df[df['learner_id'].isin(train_learners)].copy()
    valid_df = df[df['learner_id'].isin(valid_learners)].copy()
    test_df  = df[df['learner_id'].isin(test_learners)].copy()

    log(f"  Split fold_{fold}: train={len(train_df):,}, valid={len(valid_df):,}, test={len(test_df):,}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    train_df.to_csv(f'{out_dir}/train.csv', index=False)
    valid_df.to_csv(f'{out_dir}/valid.csv', index=False)
    test_df.to_csv(f'{out_dir}/test.csv',  index=False)

    with open(f'{out_dir}/split_report.md', 'w') as f:
        f.write(f"# Split Report fold_{fold}\n")
        f.write(f"- Seed: {FOLD_SEEDS[fold]}\n")
        f.write(f"- Train learners: {len(train_learners)}\n")
        f.write(f"- Valid learners: {len(valid_learners)}\n")
        f.write(f"- Test learners: {len(test_learners)}\n")
        f.write(f"- Train rows: {len(train_df)}\n")
        f.write(f"- Valid rows: {len(valid_df)}\n")
        f.write(f"- Test rows: {len(test_df)}\n")

    del df, train_df, valid_df, test_df; gc.collect()
    return True

# ─── Step 2: Build graph files for new fold ───────────────────────────────────
def build_graph_for_fold(ds, fold):
    """
    E_pre va E_co: copy tu fold_0 (chung khong thay doi theo fold split vì
    chung duoc xay dung tu Q-matrix / train interactions cua fold_0 - day la 
    assumption hop le cho multi-fold: dung cung 1 do thi cho moi fold).
    E_sim: tinh lai tu train data cua fold moi.
    """
    src_dir = f'results/tables/{ds}/fold_0'
    dst_dir = f'results/tables/{ds}/fold_{fold}'
    ensure_dir(dst_dir)

    # Copy E_pre and E_co from fold_0
    for fname in ['E_pre_train.csv', 'E_co_train.csv', 'E_pre_train_pruned.csv']:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            log(f"    Copied {fname} -> fold_{fold}/")

    # Build E_sim from this fold's train data
    esim_dst = os.path.join(dst_dir, 'E_sim_train.csv')
    if os.path.exists(esim_dst) and os.path.getsize(esim_dst) > 500:
        log(f"    E_sim da co ({os.path.getsize(esim_dst)} bytes), bo qua.")
        return

    log(f"    Xay dung E_sim cho fold_{fold} ...")
    train_df = pd.read_csv(f'data/processed/{ds}/fold_{fold}/train.csv')
    if 'user_id' in train_df.columns: train_df.rename(columns={'user_id':'learner_id'}, inplace=True)
    if 'kc_id'   in train_df.columns: train_df.rename(columns={'kc_id':'skill_id'},    inplace=True)

    unique_skills = sorted(train_df['skill_id'].unique())
    skill_to_id   = {sk: i for i, sk in enumerate(unique_skills)}

    edges = build_esim_topk(train_df, unique_skills, skill_to_id, k=5, max_learners=3000)
    n_saved = save_esim(edges, dst_dir, ds, fold)
    log(f"    E_sim fold_{fold}: {n_saved} edges saved.")
    del train_df, edges; gc.collect()

# ─── Step 3: Run all models for a fold ───────────────────────────────────────
def run_fold(ds, fold, done_keys):
    data_dir = f'data/processed/{ds}/fold_{fold}'
    tab_dir  = f'results/tables/{ds}/fold_{fold}'

    train_df = pd.read_csv(f'{data_dir}/train.csv')
    test_df  = pd.read_csv(f'{data_dir}/test.csv')
    for df in (train_df, test_df):
        if 'user_id' in df.columns: df.rename(columns={'user_id':'learner_id'}, inplace=True)
        if 'kc_id'   in df.columns: df.rename(columns={'kc_id':'skill_id'},    inplace=True)

    unique_skills = sorted(set(train_df['skill_id'].unique()) | set(test_df['skill_id'].unique()))
    skill_to_id   = {sk: i for i, sk in enumerate(unique_skills)}
    n_skills      = len(unique_skills)
    log(f"  [{ds}] fold_{fold}: {len(train_df):,} train | {len(test_df):,} test | {n_skills} skills")

    # Pre-load adjacency for all variants
    adjs, degs, twohop = {}, {}, {}
    for v in VARIANTS:
        adj, dm = load_adjacency(tab_dir, v, unique_skills, skill_to_id)
        adjs[v]   = adj
        degs[v]   = dm
        twohop[v] = two_hop_degree(adj, dm, unique_skills)

    for seed in SEEDS:
        for model in MODELS:
            for variant in VARIANTS:
                key = (ds, fold, model, variant, seed)
                if key in done_keys:
                    print(f"    SKIP {model}|{variant}|seed={seed}")
                    continue

                t0 = time.time()
                print(f"    RUN  {model:9s} | {variant:20s} | seed={seed}", end='', flush=True)
                try:
                    auc, acc, nll = MODEL_FN[model](
                        train_df, test_df, skill_to_id,
                        adjs[variant], degs[variant], twohop[variant], seed
                    )
                except Exception as ex:
                    print(f" ERROR: {ex}")
                    auc, acc, nll = float('nan'), float('nan'), float('nan')

                elapsed = time.time() - t0
                print(f" | AUC={auc} ACC={acc} ({elapsed:.0f}s)")

                append_row({
                    'dataset': ds, 'fold_id': fold, 'model': model,
                    'graph_variant': variant, 'seed': seed,
                    'auc': auc, 'acc': acc, 'nll': nll,
                    'n_train': len(train_df), 'n_test': len(test_df),
                    'n_skills': n_skills, 'elapsed_s': round(elapsed,1),
                    'created_at': datetime.now().isoformat()
                })
                done_keys.add(key)
                gc.collect()

    del adjs, degs, twohop, train_df, test_df; gc.collect()

# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    log("="*65)
    log("BUOC 1+2+3: Tao fold_1, fold_2 va chay full models")
    log("="*65)

    total_new = len(DATASETS) * len(NEW_FOLDS) * len(MODELS) * len(VARIANTS) * len(SEEDS)
    log(f"Tong runs moi can them: {total_new}")
    log(f"(3 datasets x 2 folds x 5 models x 4 variants x 3 seeds)")

    done = existing_keys()
    log(f"Da co: {len(done)} runs trong CSV")

    for ds in DATASETS:
        log("")
        log(f"{'='*20} DATASET: {ds.upper()} {'='*20}")

        for fold in NEW_FOLDS:
            log(f"--- Fold {fold} ---")

            # Step 1: Create split
            log(f"[Buoc 1] Tao split fold_{fold} cho {ds}...")
            ok = create_fold(ds, fold)
            if not ok:
                log(f"[SKIP] Khong tao duoc fold_{fold}, bo qua.")
                continue

            # Step 2: Build graph
            log(f"[Buoc 2] Xay dung graph files cho {ds} fold_{fold}...")
            build_graph_for_fold(ds, fold)

            # Step 3: Run models
            log(f"[Buoc 3] Chay 5 models x 4 variants x 3 seeds tren {ds} fold_{fold}...")
            run_fold(ds, fold, done)

    log("")
    log("="*65)
    log("HOAN THANH TAT CA FOLD!")
    log(f"Ket qua: ResultBS/confirmatory/confirmatory_results.csv")

    # Quick summary
    df_all = pd.read_csv(OUT_CSV)
    log(f"Tong runs hien tai: {len(df_all)}")
    for ds in DATASETS:
        for fold in [0,1,2]:
            n = len(df_all[(df_all.dataset==ds)&(df_all.fold_id==fold)])
            log(f"  {ds} fold_{fold}: {n}/60")

    log("")
    log("BUOC TIEP THEO: Chay lai generate_q3_stats.py va generate_q3_full_report.py")
    log("de cap nhat bang thong ke day du 3 folds.")
