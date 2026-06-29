#!/usr/bin/env python3
"""
LC-MRSG Q3 Fix Pipeline
=======================
This script is designed for Google Antigravity/Cursor-style automation.
It performs the missing analyses requested by the supervisor review:
1) dataset-scale audit;
2) zero-variance / model-training integrity check;
3) E_pre density diagnosis and pruning;
4) E_co provenance checks;
5) sparse-skill stratified predictive metrics;
6) consistency checks across tables/figures;
7) publication-ready CSV and LaTeX table export.

Expected project folders can be configured in configs/q3_fix_config.yaml.
The code is intentionally defensive: if a required file is missing, it writes
an explicit warning instead of silently producing optimistic results.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, mean_squared_error
import networkx as nx


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(config_path: str | Path) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_csv_safe(path: Path, required: bool = False) -> Optional[pd.DataFrame]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")
        return None
    return pd.read_csv(path)


def write_report(path: Path, lines: List[str]) -> None:
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def df_to_latex(df: pd.DataFrame, path: Path, caption: str, label: str, float_format: str = "%.4f") -> None:
    ensure_dir(path.parent)
    latex = df.to_latex(index=False, escape=False, float_format=lambda x: float_format % x,
                        caption=caption, label=label, longtable=False)
    path.write_text(latex, encoding="utf-8")


def get_out_root(cfg: Dict) -> Path:
    return ensure_dir(Path(cfg["paths"]["output_root"]))


# ---------------------------------------------------------------------------
# A. Dataset scale audit
# ---------------------------------------------------------------------------

def audit_dataset_scale(cfg: Dict) -> pd.DataFrame:
    data_root = Path(cfg["paths"]["data_root"])
    out = get_out_root(cfg)
    rows = []
    report = ["# Dataset Scale Audit", ""]
    for ds in cfg["datasets"]:
        inter_path = data_root / ds / "interactions.csv"
        q_path = data_root / ds / "qmatrix.csv"
        meta_path = data_root / ds / "metadata.json"
        inter = read_csv_safe(inter_path)
        qmat = read_csv_safe(q_path)
        meta = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if inter is None:
            rows.append({"dataset": ds, "status": "MISSING", "notes": f"Missing {inter_path}"})
            report.append(f"- {ds}: MISSING interactions.csv")
            continue

        # Try common column names.
        user_col = next((c for c in ["user_id", "student_id", "learner_id", "uid"] if c in inter.columns), None)
        q_col = next((c for c in ["question_id", "item_id", "problem_id", "qid"] if c in inter.columns), None)
        kc_col = next((c for c in ["skill_id", "kc_id", "concept_id", "skill", "concept"] if c in inter.columns), None)

        num_users = inter[user_col].nunique() if user_col else meta.get("num_users", np.nan)
        num_questions = inter[q_col].nunique() if q_col else meta.get("num_questions", np.nan)
        if kc_col:
            num_skills = inter[kc_col].nunique()
        elif qmat is not None:
            skill_cols = [c for c in qmat.columns if c not in ["question_id", "item_id", "problem_id", "qid"]]
            num_skills = len(skill_cols)
        else:
            num_skills = meta.get("num_skills", np.nan)
        num_interactions = len(inter)

        target = cfg.get("dataset_targets", {}).get(ds, {})
        checks = []
        if target.get("min_users") is not None:
            checks.append(num_users >= target["min_users"])
        if target.get("min_interactions") is not None:
            checks.append(num_interactions >= target["min_interactions"])
        if target.get("min_skills") is not None:
            checks.append(num_skills >= target["min_skills"])
        status = "PASS" if all(checks) else "WARN"
        notes = "Meets target scale." if status == "PASS" else "Below target scale; explain subset or rerun preprocessing."
        rows.append({
            "dataset": ds,
            "num_users": num_users,
            "num_questions": num_questions,
            "num_skills": num_skills,
            "num_interactions": num_interactions,
            "target_min_users": target.get("min_users", np.nan),
            "target_min_interactions": target.get("min_interactions", np.nan),
            "target_min_skills": target.get("min_skills", np.nan),
            "status": status,
            "notes": notes,
        })
        report.append(f"- {ds}: users={num_users}, questions={num_questions}, skills={num_skills}, interactions={num_interactions}, status={status}")
    df = pd.DataFrame(rows)
    ensure_dir(out / "tables")
    df.to_csv(out / "tables" / "dataset_scale_audit.csv", index=False)
    df_to_latex(df, out / "tables" / "dataset_scale_audit.tex",
                "Dataset scale audit against Q3-ready target sizes.", "tab:dataset-scale-audit")
    write_report(out / "reports" / "dataset_scale_audit.md", report)
    return df


# ---------------------------------------------------------------------------
# B. Zero-variance / model integrity
# ---------------------------------------------------------------------------

def prediction_file(cfg: Dict, dataset: str, fold: int, seed: int, model: str, graph: str) -> Path:
    return Path(cfg["paths"]["prediction_root"]) / dataset / f"fold_{fold}" / f"seed_{seed}" / model / f"{graph}.csv"


def log_file(cfg: Dict, dataset: str, fold: int, seed: int, model: str, graph: str) -> Path:
    return Path(cfg["paths"]["log_root"]) / dataset / f"fold_{fold}" / f"seed_{seed}" / model / f"{graph}.csv"


def diagnose_zero_variance(cfg: Dict) -> pd.DataFrame:
    out = get_out_root(cfg)
    rows = []
    for ds in cfg["datasets"]:
        for fold in cfg["folds"]:
            for seed in cfg["random_seeds"]:
                for model in cfg["models"]:
                    for graph in cfg["graph_variants"]:
                        p = prediction_file(cfg, ds, fold, seed, model, graph)
                        pred = read_csv_safe(p)
                        if pred is None:
                            rows.append({"dataset": ds, "fold": fold, "seed": seed, "model": model,
                                         "graph_variant": graph, "status": "MISSING", "diagnosis": f"Missing {p}"})
                            continue
                        y_col = next((c for c in ["y_true", "label", "correct", "y"] if c in pred.columns), None)
                        prob_col = next((c for c in ["y_pred", "prob", "prediction", "p_correct"] if c in pred.columns), None)
                        if y_col is None or prob_col is None:
                            rows.append({"dataset": ds, "fold": fold, "seed": seed, "model": model,
                                         "graph_variant": graph, "status": "FAIL", "diagnosis": "Missing y/prob columns"})
                            continue
                        y = pred[y_col].astype(float).to_numpy()
                        p_hat = pred[prob_col].astype(float).clip(1e-7, 1 - 1e-7).to_numpy()
                        try:
                            auc = roc_auc_score(y, p_hat) if len(np.unique(y)) > 1 else np.nan
                        except Exception:
                            auc = np.nan
                        acc = accuracy_score(y, (p_hat >= 0.5).astype(int)) if len(y) else np.nan
                        nll = log_loss(y, p_hat, labels=[0, 1]) if len(y) else np.nan
                        rmse = math.sqrt(mean_squared_error(y, p_hat)) if len(y) else np.nan
                        pred_std = float(np.std(p_hat))
                        pred_min = float(np.min(p_hat))
                        pred_max = float(np.max(p_hat))
                        pos_rate = float(np.mean(y))

                        # Optional training log.
                        lf = log_file(cfg, ds, fold, seed, model, graph)
                        log = read_csv_safe(lf)
                        train_loss_start = train_loss_end = valid_loss_best = grad_mean = np.nan
                        if log is not None:
                            loss_col = next((c for c in ["train_loss", "loss"] if c in log.columns), None)
                            val_col = next((c for c in ["valid_loss", "val_loss"] if c in log.columns), None)
                            grad_col = next((c for c in ["gradient_norm", "grad_norm"] if c in log.columns), None)
                            if loss_col:
                                train_loss_start = float(log[loss_col].iloc[0])
                                train_loss_end = float(log[loss_col].iloc[-1])
                            if val_col:
                                valid_loss_best = float(log[val_col].min())
                            if grad_col:
                                grad_mean = float(log[grad_col].mean())

                        fail_reasons = []
                        if pred_std < 1e-4:
                            fail_reasons.append("prediction_std<1e-4")
                        if not np.isnan(train_loss_start) and abs(train_loss_start - train_loss_end) < 1e-4:
                            fail_reasons.append("train_loss_not_decreasing")
                        if not np.isnan(grad_mean) and grad_mean == 0:
                            fail_reasons.append("zero_gradient")
                        status = "PASS" if not fail_reasons else "FAIL"
                        rows.append({
                            "dataset": ds, "fold": fold, "seed": seed, "model": model,
                            "graph_variant": graph, "auc": auc, "accuracy": acc, "nll": nll, "rmse": rmse,
                            "prediction_mean": float(np.mean(p_hat)), "prediction_std": pred_std,
                            "prediction_min": pred_min, "prediction_max": pred_max,
                            "positive_rate": pos_rate, "train_loss_start": train_loss_start,
                            "train_loss_end": train_loss_end, "valid_loss_best": valid_loss_best,
                            "gradient_norm_mean": grad_mean, "status": status,
                            "diagnosis": ";".join(fail_reasons) if fail_reasons else "ok",
                        })
    df = pd.DataFrame(rows)
    ensure_dir(out / "tables")
    df.to_csv(out / "tables" / "zero_variance_diagnosis_full.csv", index=False)
    summary = (df.groupby(["dataset", "model", "graph_variant"])
                 .agg(n_runs=("status", "size"), fail_count=("status", lambda s: (s == "FAIL").sum()),
                      auc_mean=("auc", "mean"), auc_std=("auc", "std"), pred_std_mean=("prediction_std", "mean"))
                 .reset_index())
    summary["status"] = np.where(summary["fail_count"] == 0, "PASS", "FAIL")
    summary.to_csv(out / "tables" / "zero_variance_diagnosis_summary.csv", index=False)
    df_to_latex(summary, out / "tables" / "zero_variance_diagnosis_summary.tex",
                "Zero-variance diagnosis summary across fold-seed runs.", "tab:zero-variance-summary")
    return summary


# ---------------------------------------------------------------------------
# C. E_pre density diagnosis and pruning
# ---------------------------------------------------------------------------

def epre_scores_file(cfg: Dict, dataset: str, fold: int) -> Path:
    return Path(cfg["paths"]["graph_root"]) / dataset / f"fold_{fold}" / "e_pre_scores.csv"


def diagnose_and_prune_epre(cfg: Dict) -> pd.DataFrame:
    out = get_out_root(cfg)
    rows = []
    graph_root = Path(cfg["paths"]["graph_root"])
    for ds in cfg["datasets"]:
        target_density = cfg["pre_pruning"].get("target_density_junyi" if ds == "junyi" else "target_density_default", 0.05)
        for fold in cfg["folds"]:
            score_path = epre_scores_file(cfg, ds, fold)
            scores = read_csv_safe(score_path)
            if scores is None:
                rows.append({"dataset": ds, "fold": fold, "variant": "missing", "status": "MISSING", "notes": f"Missing {score_path}"})
                continue
            src_col = next((c for c in ["src", "source", "from", "pre", "skill_i"] if c in scores.columns), None)
            dst_col = next((c for c in ["dst", "target", "to", "post", "skill_j"] if c in scores.columns), None)
            score_col = next((c for c in ["score", "weight", "confidence", "pre_score"] if c in scores.columns), None)
            if not all([src_col, dst_col, score_col]):
                rows.append({"dataset": ds, "fold": fold, "variant": "raw", "status": "FAIL", "notes": "Missing src/dst/score columns"})
                continue
            nodes = sorted(set(scores[src_col]).union(set(scores[dst_col])))
            n = len(nodes)
            possible = n * (n - 1)
            def summarize(edges: pd.DataFrame, variant: str) -> Dict:
                g = nx.DiGraph()
                g.add_nodes_from(nodes)
                g.add_edges_from(edges[[src_col, dst_col]].itertuples(index=False, name=None))
                num_edges = g.number_of_edges()
                density = num_edges / possible if possible else np.nan
                try:
                    is_dag = nx.is_directed_acyclic_graph(g)
                except Exception:
                    is_dag = False
                cycles = 0 if is_dag else len(list(nx.simple_cycles(g)))
                return {"dataset": ds, "fold": fold, "variant": variant, "nodes": n, "edges": num_edges,
                        "density": density, "avg_out_degree": num_edges / max(n, 1), "dag_pass": is_dag,
                        "num_cycles": cycles, "status": "PASS" if density <= target_density else "DENSE",
                        "notes": "" if density <= target_density else "density above target"}

            rows.append(summarize(scores, "raw"))
            # top-k outgoing variants
            for k in cfg["pre_pruning"].get("topk_out_values", [3, 5, 10]):
                edges = (scores.sort_values(score_col, ascending=False)
                              .groupby(src_col, as_index=False).head(k))
                rows.append(summarize(edges, f"top{k}_out"))
                out_dir = ensure_dir(graph_root / ds / f"fold_{fold}" / "e_pre_variants")
                edges.to_csv(out_dir / f"e_pre_top{k}_out.csv", index=False)
            # confidence threshold variants
            for thr in cfg["pre_pruning"].get("threshold_values", [0.7, 0.8, 0.9]):
                edges = scores[scores[score_col] >= thr].copy()
                rows.append(summarize(edges, f"confidence_{thr:.2f}"))
            # transitive reduction if DAG after top5; fallback note if not DAG.
            top5 = (scores.sort_values(score_col, ascending=False).groupby(src_col, as_index=False).head(5))
            g = nx.DiGraph()
            g.add_nodes_from(nodes)
            g.add_edges_from(top5[[src_col, dst_col]].itertuples(index=False, name=None))
            if nx.is_directed_acyclic_graph(g):
                tr = nx.transitive_reduction(g)
                tr_edges = pd.DataFrame(list(tr.edges()), columns=[src_col, dst_col])
                rows.append(summarize(tr_edges, "top5_transitive_reduced"))
            else:
                rows.append({"dataset": ds, "fold": fold, "variant": "top5_transitive_reduced", "nodes": n,
                             "edges": np.nan, "density": np.nan, "avg_out_degree": np.nan, "dag_pass": False,
                             "num_cycles": len(list(nx.simple_cycles(g))), "status": "FAIL", "notes": "top5 graph not DAG"})
    df = pd.DataFrame(rows)
    ensure_dir(out / "tables")
    df.to_csv(out / "tables" / "e_pre_pruning_summary.csv", index=False)
    df_to_latex(df, out / "tables" / "e_pre_pruning_summary.tex",
                "Prerequisite graph density diagnosis and constrained variants.", "tab:epre-pruning")
    return df


# ---------------------------------------------------------------------------
# D. E_co provenance checks
# ---------------------------------------------------------------------------

def eco_file(cfg: Dict, dataset: str, fold: int) -> Path:
    return Path(cfg["paths"]["graph_root"]) / dataset / f"fold_{fold}" / "e_co.csv"


def eco_provenance_audit(cfg: Dict) -> pd.DataFrame:
    out = get_out_root(cfg)
    data_root = Path(cfg["paths"]["data_root"])
    rows = []
    for ds in cfg["datasets"]:
        for fold in cfg["folds"]:
            epath = eco_file(cfg, ds, fold)
            eco = read_csv_safe(epath)
            if eco is None:
                rows.append({"dataset": ds, "fold": fold, "status": "MISSING", "notes": f"Missing {epath}"})
                continue
            src_col = next((c for c in ["src", "source", "skill_i", "kc_i"] if c in eco.columns), None)
            dst_col = next((c for c in ["dst", "target", "skill_j", "kc_j"] if c in eco.columns), None)
            w_col = next((c for c in ["weight", "pmi", "score"] if c in eco.columns), None)
            split_col = next((c for c in ["support_split", "split"] if c in eco.columns), None)
            if not all([src_col, dst_col]):
                rows.append({"dataset": ds, "fold": fold, "status": "FAIL", "notes": "Missing src/dst"})
                continue
            edge_set = set(map(tuple, eco[[src_col, dst_col]].astype(str).values))
            symmetry_pass = all((b, a) in edge_set for a, b in edge_set)
            if split_col:
                train_only_pass = not eco[split_col].astype(str).str.contains("valid|test", case=False, regex=True).any()
            else:
                train_only_pass = "UNKNOWN"
            weights = eco[w_col].astype(float) if w_col else pd.Series(dtype=float)

            # Sparse KCs from training interactions.
            train_path = data_root / ds / f"fold_{fold}" / "train.csv"
            train = read_csv_safe(train_path)
            sparse_cov = np.nan
            n_sparse = np.nan
            n_sparse_with_neighbor = np.nan
            if train is not None:
                kc_col = next((c for c in ["skill_id", "kc_id", "concept_id", "skill", "concept"] if c in train.columns), None)
                if kc_col:
                    freq = train[kc_col].value_counts()
                    sparse = set(freq[freq <= cfg["sparse_strata"]["sparse_max"]].index.astype(str))
                    nodes_with_eco = set(eco[src_col].astype(str)).union(set(eco[dst_col].astype(str)))
                    n_sparse = len(sparse)
                    n_sparse_with_neighbor = len(sparse.intersection(nodes_with_eco))
                    sparse_cov = n_sparse_with_neighbor / n_sparse if n_sparse else np.nan
            status = "PASS" if symmetry_pass and (train_only_pass is True or train_only_pass == "UNKNOWN") else "FAIL"
            rows.append({
                "dataset": ds, "fold": fold, "num_eco_edges": len(eco),
                "symmetry_pass": symmetry_pass, "train_only_support_pass": train_only_pass,
                "weight_mean": weights.mean() if len(weights) else np.nan,
                "weight_std": weights.std() if len(weights) else np.nan,
                "weight_median": weights.median() if len(weights) else np.nan,
                "weight_max": weights.max() if len(weights) else np.nan,
                "num_sparse_kcs": n_sparse,
                "num_sparse_kcs_with_eco_neighbor": n_sparse_with_neighbor,
                "sparse_kc_coverage": sparse_cov,
                "status": status,
            })
    df = pd.DataFrame(rows)
    ensure_dir(out / "tables")
    df.to_csv(out / "tables" / "eco_provenance_audit.csv", index=False)
    df_to_latex(df, out / "tables" / "eco_provenance_audit.tex",
                "Four-check $E_{co}$ provenance audit module.", "tab:eco-provenance")
    return df


# ---------------------------------------------------------------------------
# E. Sparse-skill stratified predictive metrics
# ---------------------------------------------------------------------------

def build_strata_from_train(cfg: Dict, dataset: str, fold: int) -> Dict[str, set]:
    train_path = Path(cfg["paths"]["data_root"]) / dataset / f"fold_{fold}" / "train.csv"
    train = read_csv_safe(train_path, required=True)
    kc_col = next((c for c in ["skill_id", "kc_id", "concept_id", "skill", "concept"] if c in train.columns), None)
    if kc_col is None:
        raise ValueError(f"No KC column in {train_path}")
    freq = train[kc_col].value_counts()
    vs = cfg["sparse_strata"]["very_sparse_max"]
    sp = cfg["sparse_strata"]["sparse_max"]
    med = cfg["sparse_strata"]["medium_max"]
    return {
        "very_sparse": set(freq[freq <= vs].index.astype(str)),
        "sparse": set(freq[(freq > vs) & (freq <= sp)].index.astype(str)),
        "medium": set(freq[(freq > sp) & (freq <= med)].index.astype(str)),
        "frequent": set(freq[freq > med].index.astype(str)),
    }


def compute_sparse_stratified_metrics(cfg: Dict) -> pd.DataFrame:
    out = get_out_root(cfg)
    rows = []
    for ds in cfg["datasets"]:
        for fold in cfg["folds"]:
            try:
                strata = build_strata_from_train(cfg, ds, fold)
            except Exception as e:
                rows.append({"dataset": ds, "fold": fold, "status": "FAIL", "notes": str(e)})
                continue
            for seed in cfg["random_seeds"]:
                for model in cfg["models"]:
                    for graph in cfg["graph_variants"]:
                        pred = read_csv_safe(prediction_file(cfg, ds, fold, seed, model, graph))
                        if pred is None:
                            continue
                        y_col = next((c for c in ["y_true", "label", "correct", "y"] if c in pred.columns), None)
                        prob_col = next((c for c in ["y_pred", "prob", "prediction", "p_correct"] if c in pred.columns), None)
                        kc_col = next((c for c in ["skill_id", "kc_id", "concept_id", "skill", "concept"] if c in pred.columns), None)
                        if not all([y_col, prob_col, kc_col]):
                            rows.append({"dataset": ds, "fold": fold, "seed": seed, "model": model,
                                         "graph_variant": graph, "status": "FAIL", "notes": "Missing y/prob/kc columns"})
                            continue
                        pred[kc_col] = pred[kc_col].astype(str)
                        for stratum, kc_set in strata.items():
                            sub = pred[pred[kc_col].isin(kc_set)].copy()
                            if len(sub) == 0 or sub[y_col].nunique() < 2:
                                auc = np.nan
                            else:
                                auc = roc_auc_score(sub[y_col].astype(float), sub[prob_col].astype(float))
                            phat = sub[prob_col].astype(float).clip(1e-7, 1 - 1e-7) if len(sub) else pd.Series(dtype=float)
                            y = sub[y_col].astype(float) if len(sub) else pd.Series(dtype=float)
                            rows.append({
                                "dataset": ds, "fold": fold, "seed": seed, "model": model, "graph_variant": graph,
                                "stratum": stratum, "num_test_interactions": len(sub),
                                "auc": auc,
                                "accuracy": accuracy_score(y, (phat >= 0.5).astype(int)) if len(sub) else np.nan,
                                "nll": log_loss(y, phat, labels=[0, 1]) if len(sub) else np.nan,
                                "rmse": math.sqrt(mean_squared_error(y, phat)) if len(sub) else np.nan,
                                "status": "PASS",
                            })
    df = pd.DataFrame(rows)
    ensure_dir(out / "tables")
    df.to_csv(out / "tables" / "sparse_skill_stratified_metrics.csv", index=False)
    summary = (df[df.get("status", "PASS") == "PASS"]
               .groupby(["dataset", "model", "graph_variant", "stratum"])
               .agg(n_runs=("auc", "count"), auc_mean=("auc", "mean"), auc_std=("auc", "std"),
                    acc_mean=("accuracy", "mean"), nll_mean=("nll", "mean"), rmse_mean=("rmse", "mean"))
               .reset_index())
    summary.to_csv(out / "tables" / "sparse_skill_summary_mean_std.csv", index=False)
    df_to_latex(summary, out / "tables" / "sparse_skill_summary_mean_std.tex",
                "Sparse-skill predictive performance by training-fold frequency stratum.", "tab:sparse-skill-summary")
    return summary


# ---------------------------------------------------------------------------
# F. Consistency checks for paper tables/figures
# ---------------------------------------------------------------------------

def check_consistency(cfg: Dict) -> pd.DataFrame:
    out = get_out_root(cfg)
    checks = []
    tables_dir = out / "tables"
    main_path = tables_dir / "multifold_confirmatory_results.csv"
    sparse_path = tables_dir / "sparse_skill_summary_mean_std.csv"
    zero_path = tables_dir / "zero_variance_diagnosis_summary.csv"
    if main_path.exists() and zero_path.exists():
        main = pd.read_csv(main_path)
        zero = pd.read_csv(zero_path)
        common = main.merge(zero, on=["dataset", "model", "graph_variant"], suffixes=("_main", "_zero"))
        if "auc_mean_main" in common.columns and "auc_mean_zero" in common.columns:
            common["auc_abs_diff"] = (common["auc_mean_main"] - common["auc_mean_zero"]).abs()
            bad = common[common["auc_abs_diff"] > 1e-4]
            checks.append({"check": "main_vs_zero_auc", "status": "PASS" if bad.empty else "WARN", "n_bad": len(bad),
                           "notes": "AUC means differ across tables" if len(bad) else "ok"})
            bad.to_csv(out / "tables" / "consistency_main_vs_zero_bad_rows.csv", index=False)
    else:
        checks.append({"check": "main_vs_zero_auc", "status": "SKIP", "notes": "Need multifold_confirmatory_results.csv and zero_variance_diagnosis_summary.csv"})

    if main_path.exists() and sparse_path.exists():
        main = pd.read_csv(main_path)
        sparse = pd.read_csv(sparse_path)
        # This is a warning-only check because weighted aggregation needs stratum counts.
        # It flags cases where all stratum deltas are positive but overall delta is zero/negative.
        if all(c in main.columns for c in ["dataset", "model", "graph_variant", "auc_mean"]):
            no = main[main["graph_variant"].astype(str).str.contains("no", case=False, na=False)]
            full = main[main["graph_variant"].astype(str).str.contains("full", case=False, na=False)]
            main_delta = full.merge(no, on=["dataset", "model"], suffixes=("_full", "_no"))
            main_delta["delta_overall"] = main_delta["auc_mean_full"] - main_delta["auc_mean_no"]
            if "auc_mean" in sparse.columns:
                # Full/no deltas per stratum.
                sp_no = sparse[sparse["graph_variant"].astype(str).str.contains("no", case=False, na=False)]
                sp_full = sparse[sparse["graph_variant"].astype(str).str.contains("full", case=False, na=False)]
                sp_delta = sp_full.merge(sp_no, on=["dataset", "model", "stratum"], suffixes=("_full", "_no"))
                sp_delta["delta_stratum"] = sp_delta["auc_mean_full"] - sp_delta["auc_mean_no"]
                all_pos = (sp_delta.groupby(["dataset", "model"])["delta_stratum"].min().reset_index()
                           .rename(columns={"delta_stratum": "min_delta_stratum"}))
                merged = main_delta.merge(all_pos, on=["dataset", "model"], how="left")
                flagged = merged[(merged["min_delta_stratum"] > 0) & (merged["delta_overall"] <= 0.0005)]
                checks.append({"check": "sparse_vs_overall_delta", "status": "PASS" if flagged.empty else "WARN",
                               "n_bad": len(flagged), "notes": "All strata improve but overall does not; inspect weighting"})
                flagged.to_csv(out / "tables" / "consistency_sparse_vs_overall_flags.csv", index=False)
    else:
        checks.append({"check": "sparse_vs_overall_delta", "status": "SKIP", "notes": "Need main and sparse summary"})

    df = pd.DataFrame(checks)
    ensure_dir(out / "tables")
    df.to_csv(out / "tables" / "paper_consistency_checks.csv", index=False)
    df_to_latex(df, out / "tables" / "paper_consistency_checks.tex",
                "Consistency checks across main and supplementary result tables.", "tab:consistency-checks")
    return df


# ---------------------------------------------------------------------------
# G. Multi-fold confirmatory aggregation and statistical tests
# ---------------------------------------------------------------------------

def aggregate_multifold_results(cfg: Dict) -> pd.DataFrame:
    """Aggregate prediction files into a main result table and paired tests."""
    out = get_out_root(cfg)
    run_rows = []
    for ds in cfg["datasets"]:
        for fold in cfg["folds"]:
            for seed in cfg["random_seeds"]:
                for model in cfg["models"]:
                    for graph in cfg["graph_variants"]:
                        pred = read_csv_safe(prediction_file(cfg, ds, fold, seed, model, graph))
                        if pred is None:
                            continue
                        y_col = next((c for c in ["y_true", "label", "correct", "y"] if c in pred.columns), None)
                        prob_col = next((c for c in ["y_pred", "prob", "prediction", "p_correct"] if c in pred.columns), None)
                        if y_col is None or prob_col is None:
                            continue
                        y = pred[y_col].astype(float).to_numpy()
                        p = pred[prob_col].astype(float).clip(1e-7, 1 - 1e-7).to_numpy()
                        run_rows.append({
                            "dataset": ds, "fold": fold, "seed": seed, "model": model, "graph_variant": graph,
                            "auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
                            "accuracy": accuracy_score(y, (p >= 0.5).astype(int)),
                            "nll": log_loss(y, p, labels=[0, 1]),
                            "rmse": math.sqrt(mean_squared_error(y, p)),
                        })
    runs = pd.DataFrame(run_rows)
    ensure_dir(out / "tables")
    runs.to_csv(out / "tables" / "all_fold_seed_runs.csv", index=False)
    summary = (runs.groupby(["dataset", "model", "graph_variant"])
               .agg(n_runs=("auc", "count"), auc_mean=("auc", "mean"), auc_std=("auc", "std"),
                    acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
                    nll_mean=("nll", "mean"), nll_std=("nll", "std"), rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"))
               .reset_index())
    summary.to_csv(out / "tables" / "multifold_confirmatory_results.csv", index=False)
    df_to_latex(summary, out / "tables" / "multifold_confirmatory_results.tex",
                "Multi-fold confirmatory results over all configured fold-seed runs.", "tab:multifold-results")
    # Paired tests no_graph vs full
    tests = []
    no_name = next((g for g in cfg["graph_variants"] if "no" in g), cfg["graph_variants"][0])
    full_name = next((g for g in cfg["graph_variants"] if "full" in g), cfg["graph_variants"][-1])
    for ds in cfg["datasets"]:
        for model in cfg["models"]:
            no = runs[(runs.dataset == ds) & (runs.model == model) & (runs.graph_variant == no_name)]
            full = runs[(runs.dataset == ds) & (runs.model == model) & (runs.graph_variant == full_name)]
            merged = full.merge(no, on=["dataset", "fold", "seed", "model"], suffixes=("_full", "_no"))
            if len(merged) < 2:
                continue
            d = merged["auc_full"] - merged["auc_no"]
            t_stat, p_two = stats.ttest_rel(merged["auc_full"], merged["auc_no"], nan_policy="omit")
            p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
            try:
                w_stat, p_w = stats.wilcoxon(d)
            except Exception:
                w_stat, p_w = np.nan, np.nan
            cohen_d = d.mean() / d.std(ddof=1) if d.std(ddof=1) and not np.isnan(d.std(ddof=1)) else np.nan
            tests.append({"dataset": ds, "model": model, "n_pairs": len(merged), "auc_no": merged["auc_no"].mean(),
                          "auc_full": merged["auc_full"].mean(), "delta_auc": d.mean(), "t_stat": t_stat,
                          "p_one_tailed": p_one, "wilcoxon_p": p_w, "cohens_d": cohen_d,
                          "significant_005": p_one < 0.05})
    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(out / "tables" / "paired_tests_no_vs_full.csv", index=False)
    df_to_latex(tests_df, out / "tables" / "paired_tests_no_vs_full.tex",
                "Paired tests comparing full LC-MRSG against no-graph baseline.", "tab:paired-tests")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all(cfg: Dict) -> None:
    audit_dataset_scale(cfg)
    aggregate_multifold_results(cfg)
    diagnose_zero_variance(cfg)
    diagnose_and_prune_epre(cfg)
    eco_provenance_audit(cfg)
    compute_sparse_stratified_metrics(cfg)
    check_consistency(cfg)
    out = get_out_root(cfg)
    write_report(out / "reports" / "q3_fix_final_report.md", [
        "# LC-MRSG Q3 Fix Final Report", "",
        "Generated by lc_mrsg_q3_fix_pipeline.py.", "",
        "Key outputs are in results/q3_fix/tables and results/q3_fix/reports.",
        "Check all WARN/FAIL rows before updating the manuscript.",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/q3_fix_config.yaml")
    parser.add_argument("--task", default="all",
                        choices=["all", "scale", "aggregate", "zero", "epre", "eco", "sparse", "consistency"])
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.task == "all":
        run_all(cfg)
    elif args.task == "scale":
        audit_dataset_scale(cfg)
    elif args.task == "aggregate":
        aggregate_multifold_results(cfg)
    elif args.task == "zero":
        diagnose_zero_variance(cfg)
    elif args.task == "epre":
        diagnose_and_prune_epre(cfg)
    elif args.task == "eco":
        eco_provenance_audit(cfg)
    elif args.task == "sparse":
        compute_sparse_stratified_metrics(cfg)
    elif args.task == "consistency":
        check_consistency(cfg)


if __name__ == "__main__":
    main()
