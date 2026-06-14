"""
generate_q3_stats.py
====================
Sinh bảng thống kê chuẩn Q3 từ confirmatory_results.csv:
  1. mean ± std (3 seeds) per model / dataset / graph_variant
  2. 95% CI (t-distribution, df=2)
  3. Wilcoxon signed-rank test: no_graph vs E_pre_E_sim_E_co
  4. LaTeX tables  (Table 3 & Table 4 kiểu bài báo)
  5. Summary report Markdown
Output → ResultBS/q3_stats/
"""

import os, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────────
SRC  = "ResultBS/confirmatory/confirmatory_results.csv"
OUT  = "ResultBS/q3_stats"
Path(OUT).mkdir(parents=True, exist_ok=True)

# ── load ───────────────────────────────────────────────────────────────────
df = pd.read_csv(SRC)
print(f"Loaded {len(df)} rows from {SRC}")
print(f"Datasets : {sorted(df['dataset'].unique())}")
print(f"Models   : {sorted(df['model'].unique())}")
print(f"Variants : {sorted(df['graph_variant'].unique())}")
print(f"Seeds    : {sorted(df['seed'].unique())}")

VARIANTS = ['no_graph', 'E_pre', 'E_pre_E_sim', 'E_pre_E_sim_E_co']
MODELS   = ['BKT', 'DKT', 'simpleKT', 'GIKT', 'SKT']
DATASETS = ['assist2012', 'junyi', 'kdd2010']
DS_LABEL = {'assist2012': 'ASSIST2012', 'junyi': 'Junyi', 'kdd2010': 'KDD2010'}
VAR_LABEL = {
    'no_graph':         r'No Graph',
    'E_pre':            r'$E_{pre}$',
    'E_pre_E_sim':      r'$E_{pre}+E_{sim}$',
    'E_pre_E_sim_E_co': r'LC-MRSG (Full)',
}

# ── 1. Per-cell stats: mean, std, 95%-CI ──────────────────────────────────
def ci95(vals):
    """95% CI half-width using t-distribution (df = n-1)."""
    n = len(vals)
    if n < 2: return float('nan')
    se = np.std(vals, ddof=1) / np.sqrt(n)
    t  = stats.t.ppf(0.975, df=n-1)
    return t * se

records = []
grp = df.groupby(['dataset', 'model', 'graph_variant'])
for (ds, model, var), sub in grp:
    aucs = sub['auc'].dropna().values
    accs = sub['acc'].dropna().values
    nlls = sub['nll'].dropna().values
    records.append({
        'dataset':       ds,
        'model':         model,
        'graph_variant': var,
        'n_runs':        len(aucs),
        'auc_mean':      round(np.mean(aucs), 4),
        'auc_std':       round(np.std(aucs, ddof=1), 4) if len(aucs)>1 else 0,
        'auc_ci95':      round(ci95(aucs), 4),
        'acc_mean':      round(np.mean(accs), 4),
        'acc_std':       round(np.std(accs, ddof=1), 4) if len(accs)>1 else 0,
        'acc_ci95':      round(ci95(accs), 4),
        'nll_mean':      round(np.mean(nlls), 4),
        'nll_std':       round(np.std(nlls, ddof=1), 4) if len(nlls)>1 else 0,
    })

stats_df = pd.DataFrame(records)
stats_df.to_csv(f"{OUT}/cell_stats.csv", index=False)
print(f"\nCell stats saved ({len(stats_df)} rows)")

# ── 2. Wilcoxon test: no_graph vs E_pre_E_sim_E_co per (dataset, model) ──
wilcox_rows = []
for ds in DATASETS:
    for model in MODELS:
        ng  = df[(df.dataset==ds)&(df.model==model)&(df.graph_variant=='no_graph')]['auc'].dropna().values
        full= df[(df.dataset==ds)&(df.model==model)&(df.graph_variant=='E_pre_E_sim_E_co')]['auc'].dropna().values
        if len(ng) < 3 or len(full) < 3:
            p, stat, sig = float('nan'), float('nan'), 'N/A'
        else:
            try:
                stat, p = stats.wilcoxon(ng, full, alternative='less')
                sig = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns'))
            except Exception:
                # Wilcoxon fails when all differences=0
                p, stat, sig = 1.0, 0.0, 'ns'
        delta = round(np.mean(full)-np.mean(ng), 4) if len(ng)>0 and len(full)>0 else float('nan')
        wilcox_rows.append({
            'dataset': ds, 'model': model,
            'auc_no_graph':  round(np.mean(ng), 4)   if len(ng)>0   else float('nan'),
            'auc_full':      round(np.mean(full), 4)  if len(full)>0 else float('nan'),
            'delta_auc':     delta,
            'wilcoxon_stat': round(stat, 4) if not np.isnan(stat) else float('nan'),
            'p_value':       round(p, 4)    if not np.isnan(p)    else float('nan'),
            'significance':  sig,
        })

wilcox_df = pd.DataFrame(wilcox_rows)
wilcox_df.to_csv(f"{OUT}/wilcoxon_no_vs_full.csv", index=False)
print(f"Wilcoxon tests saved ({len(wilcox_df)} rows)")
print(wilcox_df[['dataset','model','delta_auc','p_value','significance']].to_string(index=False))

# ── 3. LaTeX Table: mean±std AUC per model × graph_variant (one table per dataset) ──
def fmt_cell(mean, std, bold=False):
    s = f"{mean:.4f}$\\pm${std:.4f}"
    return f"\\textbf{{{s}}}" if bold else s

def make_latex_auc_table(ds):
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{AUC (mean $\pm$ std, 3 seeds) on " + DS_LABEL[ds] + r" --- Graph Variant Ablation}")
    lines.append(r"\label{tab:auc_" + ds + r"}")
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    cols = "l" + "c"*len(VARIANTS)
    lines.append(r"\begin{tabular}{" + cols + r"}")
    lines.append(r"\hline")
    header = "Model & " + " & ".join(VAR_LABEL[v] for v in VARIANTS) + r" \\"
    lines.append(header)
    lines.append(r"\hline")
    for model in MODELS:
        cells = [model]
        row_means = []
        for v in VARIANTS:
            sub = stats_df[(stats_df.dataset==ds)&(stats_df.model==model)&(stats_df.graph_variant==v)]
            if len(sub)==0:
                cells.append('--')
                row_means.append(-1)
            else:
                row_means.append(sub.iloc[0]['auc_mean'])
                cells.append((sub.iloc[0]['auc_mean'], sub.iloc[0]['auc_std']))
        best_idx = int(np.argmax([m for m in row_means]))
        formatted = []
        for i, c in enumerate(cells[1:]):
            if isinstance(c, tuple):
                formatted.append(fmt_cell(c[0], c[1], bold=(i==best_idx)))
            else:
                formatted.append(c)
        lines.append(model + " & " + " & ".join(formatted) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

for ds in DATASETS:
    tex = make_latex_auc_table(ds)
    fpath = f"{OUT}/table_auc_{ds}.tex"
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(tex)
    print(f"LaTeX AUC table: {fpath}")

# ── 4. LaTeX Table: Wilcoxon summary (all datasets combined) ──────────────
def make_wilcoxon_table():
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Statistical significance (Wilcoxon signed-rank, one-tailed): No Graph vs.\ LC-MRSG (Full Graph). $\Delta$AUC = AUC$_\text{full}$ - AUC$_\text{no-graph}$.}")
    lines.append(r"\label{tab:wilcoxon}")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\hline")
    lines.append(r"Dataset & Model & AUC (No Graph) & AUC (Full) & $\Delta$AUC & $p$-value \\")
    lines.append(r"\hline")
    for ds in DATASETS:
        sub = wilcox_df[wilcox_df.dataset==ds]
        for i, row in sub.iterrows():
            sig = row['significance']
            sig_str = f"${sig}$" if sig not in ('ns','N/A') else sig
            p_str = f"{row['p_value']:.4f}{sig_str}" if not np.isnan(row['p_value']) else '--'
            d_str = f"+{row['delta_auc']:.4f}" if row['delta_auc']>0 else f"{row['delta_auc']:.4f}"
            lines.append(
                f"{DS_LABEL[ds]} & {row['model']} & "
                f"{row['auc_no_graph']:.4f} & {row['auc_full']:.4f} & "
                f"{d_str} & {p_str} \\\\"
            )
    lines.append(r"\hline")
    lines.append(r"\multicolumn{6}{l}{\footnotesize $^{***}p<0.001$, $^{**}p<0.01$, $^{*}p<0.05$, ns: not significant}")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

tex_w = make_wilcoxon_table()
with open(f"{OUT}/table_wilcoxon.tex", 'w', encoding='utf-8') as f:
    f.write(tex_w)
print(f"LaTeX Wilcoxon table saved.")

# ── 5. Multi-fold pivot: mean±CI per (dataset, model, variant) ────────────
pivot_rows = []
for ds in DATASETS:
    for model in MODELS:
        row = {'dataset': DS_LABEL[ds], 'model': model}
        for v in VARIANTS:
            sub = stats_df[(stats_df.dataset==ds)&(stats_df.model==model)&(stats_df.graph_variant==v)]
            if len(sub)==0:
                row[f'auc_{v}'] = '--'
                row[f'acc_{v}'] = '--'
            else:
                r = sub.iloc[0]
                row[f'auc_{v}'] = f"{r['auc_mean']:.4f}±{r['auc_ci95']:.4f}"
                row[f'acc_{v}'] = f"{r['acc_mean']:.4f}±{r['acc_ci95']:.4f}"
        pivot_rows.append(row)

pivot_df = pd.DataFrame(pivot_rows)
pivot_df.to_csv(f"{OUT}/multifold_pivot_ci95.csv", index=False)

# ── 6. Zero-variance diagnosis ────────────────────────────────────────────
zv_rows = []
for ds in DATASETS:
    for model in MODELS:
        for v in VARIANTS:
            sub = stats_df[(stats_df.dataset==ds)&(stats_df.model==model)&(stats_df.graph_variant==v)]
            if len(sub)==0: continue
            r = sub.iloc[0]
            status = 'FAIL' if r['auc_std'] < 1e-4 else 'PASS'
            zv_rows.append({
                'dataset':ds,'model':model,'graph_variant':v,
                'auc_mean':r['auc_mean'],'auc_std':r['auc_std'],
                'n_runs':r['n_runs'],'status':status
            })
zv_df = pd.DataFrame(zv_rows)
zv_df.to_csv(f"{OUT}/zero_variance_diagnosis.csv", index=False)
n_fail = (zv_df.status=='FAIL').sum()
n_pass = (zv_df.status=='PASS').sum()
print(f"\nZero-variance: PASS={n_pass}, FAIL={n_fail}")
if n_fail > 0:
    print("FAIL rows:")
    print(zv_df[zv_df.status=='FAIL'][['dataset','model','graph_variant','auc_std']].to_string(index=False))

# ── 7. Summary Markdown report ────────────────────────────────────────────
md = []
md.append("# Q3 Confirmatory Statistical Report\n")
md.append(f"- **Total runs**: {len(df)}")
md.append(f"- **Datasets**: {', '.join(DATASETS)}")
md.append(f"- **Models**: {', '.join(MODELS)}")
md.append(f"- **Seeds**: {sorted(df['seed'].unique())}")
md.append(f"- **Graph variants**: {', '.join(VARIANTS)}\n")

md.append("## Zero-Variance Diagnosis\n")
md.append(f"- PASS: {n_pass} / {len(zv_df)} cells")
md.append(f"- FAIL (auc_std < 1e-4): {n_fail}\n")
if n_fail > 0:
    md.append("**FAIL cells:**\n```")
    md.append(zv_df[zv_df.status=='FAIL'][['dataset','model','graph_variant','auc_std']].to_string(index=False))
    md.append("```\n")

md.append("## Wilcoxon Test: No Graph vs LC-MRSG Full\n")
md.append("| Dataset | Model | AUC no_graph | AUC full | ΔAUC | p-value | Sig |")
md.append("|---------|-------|-------------|----------|------|---------|-----|")
for _, row in wilcox_df.iterrows():
    d = f"+{row['delta_auc']:.4f}" if row['delta_auc']>0 else f"{row['delta_auc']:.4f}"
    md.append(f"| {DS_LABEL[row['dataset']]} | {row['model']} | {row['auc_no_graph']:.4f} | {row['auc_full']:.4f} | {d} | {row['p_value']:.4f} | {row['significance']} |")

md.append("\n## AUC Summary (mean±std) per Dataset\n")
for ds in DATASETS:
    md.append(f"### {DS_LABEL[ds]}\n")
    md.append("| Model | no_graph | E_pre | E_pre+E_sim | LC-MRSG (Full) |")
    md.append("|-------|----------|-------|-------------|----------------|")
    for model in MODELS:
        cells = []
        for v in VARIANTS:
            sub = stats_df[(stats_df.dataset==ds)&(stats_df.model==model)&(stats_df.graph_variant==v)]
            if len(sub)==0: cells.append('--')
            else: cells.append(f"{sub.iloc[0]['auc_mean']:.4f}±{sub.iloc[0]['auc_std']:.4f}")
        md.append(f"| {model} | " + " | ".join(cells) + " |")
    md.append("")

md.append("## Output Files\n")
md.append(f"- `cell_stats.csv` — per-cell mean/std/CI95")
md.append(f"- `wilcoxon_no_vs_full.csv` — Wilcoxon test results")
md.append(f"- `table_auc_assist2012.tex` / `..._junyi.tex` / `..._kdd2010.tex` — LaTeX AUC tables")
md.append(f"- `table_wilcoxon.tex` — LaTeX significance table")
md.append(f"- `multifold_pivot_ci95.csv` — Pivot with 95% CI")
md.append(f"- `zero_variance_diagnosis.csv` — Zero-variance check")

with open(f"{OUT}/q3_stats_report.md", 'w', encoding='utf-8') as f:
    f.write("\n".join(md))

print(f"\n[DONE] All Q3 stats saved to {OUT}/")
print("  cell_stats.csv")
print("  wilcoxon_no_vs_full.csv")
print("  table_auc_assist2012.tex / table_auc_junyi.tex / table_auc_kdd2010.tex")
print("  table_wilcoxon.tex")
print("  multifold_pivot_ci95.csv")
print("  zero_variance_diagnosis.csv")
print("  q3_stats_report.md")
