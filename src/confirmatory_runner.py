"""
confirmatory_runner.py
======================
Option-A multi-fold confirmatory runner.

Features:
- Fully automated; writes results incrementally (resume-safe)
- RAM guard: skips if psutil detects > 85% memory used
- 3 datasets × available_folds × 3 seeds × 5 models × 4 graph variants
- Rebuilds real E_sim (top-K cosine) before first run of each fold
- Saves to: ResultBS/confirmatory/confirmatory_results.csv
            ResultBS/confirmatory/multifold_summary.csv

Usage:
    python -m src.confirmatory_runner [--datasets assist2012 junyi kdd2010]
                                      [--seeds 42 43 44]
                                      [--folds 0 1 2]
"""

import argparse, gc, os, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

from src.graph_utils_v2 import (
    build_esim_topk, save_esim,
    load_adjacency, two_hop_degree
)
from src.models_v2 import MODEL_FN
from src.common import load_config, ensure_dir

# ─── constants ────────────────────────────────────────────────────────────────
GRAPH_VARIANTS = ['no_graph', 'E_pre', 'E_pre_E_sim', 'E_pre_E_sim_E_co']
MODELS         = ['BKT', 'DKT', 'simpleKT', 'GIKT', 'SKT']
OUT_DIR        = 'ResultBS/confirmatory'
OUT_CSV        = f'{OUT_DIR}/confirmatory_results.csv'
OUT_SUM        = f'{OUT_DIR}/multifold_summary.csv'
RAM_LIMIT_PCT  = 85          # % — pause if exceeded


# ─── helpers ─────────────────────────────────────────────────────────────────
def _ram_ok():
    if not _HAS_PSUTIL:
        return True
    return psutil.virtual_memory().percent < RAM_LIMIT_PCT


def _wait_for_ram(tag=''):
    if _HAS_PSUTIL:
        pct = psutil.virtual_memory().percent
        while pct >= RAM_LIMIT_PCT:
            print(f'  [RAM] {pct:.0f}% used — waiting 30s {tag}')
            time.sleep(30)
            pct = psutil.virtual_memory().percent


def _existing_keys(csv_path):
    """Return set of (dataset, fold_id, model, graph_variant, seed) already done."""
    if not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path)
    return set(zip(df.dataset, df.fold_id, df.model, df.graph_variant, df.seed))


def _append_row(row: dict, csv_path: str):
    df = pd.DataFrame([row])
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode='a', header=header, index=False)


def _find_folds(dataset, requested_folds):
    available = []
    for f in requested_folds:
        p = f'data/processed/{dataset}/fold_{f}/train.csv'
        if os.path.exists(p):
            available.append(f)
    return available


# ─── main runner ─────────────────────────────────────────────────────────────
def run_confirmatory(datasets, seeds, folds):
    ensure_dir(OUT_DIR)
    done = _existing_keys(OUT_CSV)
    print(f'Resuming: {len(done)} runs already completed.')

    for dataset in datasets:
        cfg_path = f'configs/{dataset}.yaml'
        if not os.path.exists(cfg_path):
            print(f'[WARN] No config for {dataset}, skipping.')
            continue

        available_folds = _find_folds(dataset, folds)
        if not available_folds:
            print(f'[WARN] No fold data found for {dataset}, skipping.')
            continue
        print(f'\n=== Dataset: {dataset} | Folds: {available_folds} ===')

        for fold in available_folds:
            data_dir = f'data/processed/{dataset}/fold_{fold}'
            tab_dir  = f'results/tables/{dataset}/fold_{fold}'
            ensure_dir(tab_dir)

            # ── Load split data ──────────────────────────────────────────
            try:
                train_df = pd.read_csv(f'{data_dir}/train.csv')
                test_df  = pd.read_csv(f'{data_dir}/test.csv')
            except Exception as e:
                print(f'  [ERROR] Cannot load split: {e}')
                continue

            # Normalize column names (some datasets use user_id / kc_id)
            for df in (train_df, test_df):
                if 'user_id' in df.columns and 'learner_id' not in df.columns:
                    df.rename(columns={'user_id': 'learner_id'}, inplace=True)
                if 'kc_id' in df.columns and 'skill_id' not in df.columns:
                    df.rename(columns={'kc_id': 'skill_id'}, inplace=True)

            unique_skills = sorted(set(train_df['skill_id'].unique()) |
                                   set(test_df['skill_id'].unique()))
            skill_to_id   = {sk: i for i, sk in enumerate(unique_skills)}
            n_skills      = len(unique_skills)
            print(f'  Fold {fold}: {len(train_df):,} train | {len(test_df):,} test | {n_skills} skills')

            # ── Rebuild E_sim once per fold ──────────────────────────────
            esim_path = os.path.join(tab_dir, 'E_sim_train.csv')
            esim_size = os.path.getsize(esim_path) if os.path.exists(esim_path) else 0
            if esim_size < 500:        # placeholder / empty
                print(f'  Building E_sim top-K=5 for {dataset} fold {fold}...')
                _wait_for_ram('(E_sim build)')
                edges = build_esim_topk(train_df, unique_skills, skill_to_id, k=5, max_learners=3000)
                n_saved = save_esim(edges, tab_dir, dataset, fold)
                print(f'  E_sim: {n_saved} edges saved.')
                del edges; gc.collect()
            else:
                print(f'  E_sim already exists ({esim_size} bytes), skipping rebuild.')

            # ── Pre-load adjacency matrices for all variants ─────────────
            print(f'  Pre-loading graph adjacency matrices...')
            adjs   = {}
            degs   = {}
            twohop = {}
            for v in GRAPH_VARIANTS:
                adj, dm = load_adjacency(tab_dir, v, unique_skills, skill_to_id)
                adjs[v] = adj
                degs[v] = dm
                twohop[v] = two_hop_degree(adj, dm, unique_skills)
            print(f'  Adjacency loaded.')

            # ── Run models × seeds × variants ───────────────────────────
            for seed in seeds:
                for model_name in MODELS:
                    for variant in GRAPH_VARIANTS:
                        key = (dataset, fold, model_name, variant, seed)
                        if key in done:
                            print(f'    SKIP {model_name}|{variant}|seed={seed} (already done)')
                            continue

                        _wait_for_ram(f'{model_name}|{variant}|seed={seed}')
                        t0 = time.time()
                        print(f'    RUN  {model_name:9s} | {variant:20s} | seed={seed}', end='', flush=True)

                        try:
                            fn = MODEL_FN[model_name]
                            auc, acc, nll = fn(
                                train_df, test_df, skill_to_id,
                                adjs[variant], degs[variant], twohop[variant],
                                seed
                            )
                        except Exception as ex:
                            print(f' ERROR: {ex}')
                            auc, acc, nll = float('nan'), float('nan'), float('nan')

                        elapsed = time.time() - t0
                        print(f' | AUC={auc} ACC={acc} NLL={nll} ({elapsed:.0f}s)')

                        row = {
                            'dataset':    dataset,
                            'fold_id':    fold,
                            'model':      model_name,
                            'graph_variant': variant,
                            'seed':       seed,
                            'auc':        auc,
                            'acc':        acc,
                            'nll':        nll,
                            'n_train':    len(train_df),
                            'n_test':     len(test_df),
                            'n_skills':   n_skills,
                            'elapsed_s':  round(elapsed, 1),
                            'created_at': datetime.now().isoformat()
                        }
                        _append_row(row, OUT_CSV)
                        done.add(key)
                        gc.collect()

            # Free adjacency from RAM before next fold
            del adjs, degs, twohop, train_df, test_df
            gc.collect()

    # ── Compute summary table ─────────────────────────────────────────────
    if os.path.exists(OUT_CSV):
        df = pd.read_csv(OUT_CSV)
        summ = (df.groupby(['dataset', 'model', 'graph_variant'])
                  .agg(auc_mean=('auc', 'mean'), auc_std=('auc', 'std'),
                       acc_mean=('acc', 'mean'), acc_std=('acc', 'std'),
                       nll_mean=('nll', 'mean'), n_runs=('auc', 'count'))
                  .reset_index())
        summ.to_csv(OUT_SUM, index=False)
        print(f'\n[OK] Summary saved to {OUT_SUM}')
        print(summ.to_string(index=False))


# ─── entry point ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+',
                        default=['assist2012', 'junyi', 'kdd2010'])
    parser.add_argument('--seeds',    nargs='+', type=int,
                        default=[42, 43, 44])
    parser.add_argument('--folds',    nargs='+', type=int,
                        default=[0, 1, 2])
    args = parser.parse_args()

    run_confirmatory(
        datasets=args.datasets,
        seeds=args.seeds,
        folds=args.folds
    )
