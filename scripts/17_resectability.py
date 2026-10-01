#!/usr/bin/env python3
"""Test whether biopsy morphology predicts resection: per-cluster Mann-Whitney tests, then cross-validated classifiers.
Inputs: --survival-csv (cluster proportions, clinical data, outcome), optional --gnn-embeddings (step 15) and --fold-indices (fold_indices.pkl).
Outputs: cluster_mwu.csv, volcano and box plots, per-fold AUCs, comparison plot and comparison_summary.json under --output-dir."""

import json
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

import xgboost as xgb
import optuna

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
np.random.seed(SEED)

DPI = 300
plt.rcParams.update({
    'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'savefig.dpi': DPI, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
    'font.family': 'sans-serif', 'pdf.fonttype': 42, 'axes.linewidth': 0.6,
})
sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': ':',
                              'grid.alpha': 0.4})

COLORS = ['#636363', '#FF7F00', '#2166AC', '#B2182B', '#4DAF4A']

CLINICAL_VARS = [
    'AGE_AT_DIAGNOSIS', 'SEX', 'PRETX_SIZE',
    'PRETX_LARGE_VASCULAR', 'NEO_FOLFORINOX', 'NEO_GEMCITABINE',
]


def load_data(survival_csv, gnn_embeddings_csv=None):
    df = pd.read_csv(survival_csv)
    df = df.dropna(subset=['SURGERY_SPECIMEN']).copy()
    df['IS_RESECTED'] = (df['SURGERY_SPECIMEN'] == 1.0).astype(int)

    cluster_cols = sorted([c for c in df.columns if c.startswith('Cluster_')],
                           key=lambda x: int(x.split('_')[1]))
    emb_cols = []

    if gnn_embeddings_csv:
        emb_df = pd.read_csv(gnn_embeddings_csv)
        emb_cols = [c for c in emb_df.columns if c.startswith('emb_')]
        df = df.merge(emb_df[['slide_name'] + emb_cols], on='slide_name', how='left')
        matched = df[emb_cols[0]].notna().sum()
        print(f"  GNN embeddings: {matched}/{len(df)} slides ({len(emb_cols)} dims)")

    df[cluster_cols] = df[cluster_cols].fillna(0)
    if emb_cols:
        df[emb_cols] = df[emb_cols].fillna(0)
    for col in CLINICAL_VARS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())

    clinical_cols = [c for c in CLINICAL_VARS if c in df.columns]

    resected = df['IS_RESECTED'].sum()
    print(f"  Patients  : {len(df)}  (resected: {resected}, "
          f"non-resected: {len(df)-resected}  ratio 1:{(len(df)-resected)/max(resected,1):.1f})")

    return df, cluster_cols, emb_cols, clinical_cols


def descriptive_analysis(df, cluster_cols, output_dir):
    print(f"\n{'=' * 70}")
    print("SECTION A — DESCRIPTIVE ANALYSIS")
    print(f"{'=' * 70}")

    resected     = df[df['IS_RESECTED'] == 1]
    not_resected = df[df['IS_RESECTED'] == 0]

    results = []
    for col in cluster_cols:
        a = resected[col].dropna().values
        b = not_resected[col].dropna().values
        if len(a) < 3 or len(b) < 3:
            continue
        stat, p = stats.mannwhitneyu(a, b, alternative='two-sided')
        log2fc  = np.log2((a.mean() + 1e-9) / (b.mean() + 1e-9))
        results.append({
            'cluster':       col,
            'cluster_id':    int(col.split('_')[1]),
            'mean_resected': float(a.mean()),
            'mean_not':      float(b.mean()),
            'log2fc':        float(log2fc),
            'mw_statistic':  float(stat),
            'pvalue':        float(p),
        })

    res_df = pd.DataFrame(results).sort_values('pvalue')
    _, fdr, _, _ = multipletests(res_df['pvalue'].values, method='fdr_bh')
    res_df['fdr'] = fdr
    res_df['significant'] = res_df['fdr'] < 0.05

    sig   = res_df[res_df['fdr'] < 0.05]
    trend = res_df[res_df['fdr'] < 0.10]

    print(f"  Tested: {len(res_df)} clusters")
    print(f"  Significant FDR<0.05 : {len(sig)}")
    print(f"  Trend FDR<0.10       : {len(trend)}")
    print(f"  Enriched in resected     : "
          f"{sig[sig['log2fc']>0]['cluster'].tolist()}")
    print(f"  Enriched in non-resected : "
          f"{sig[sig['log2fc']<0]['cluster'].tolist()}")

    res_df.to_csv(Path(output_dir) / 'cluster_mwu.csv', index=False)

    _plot_volcano(res_df, output_dir)

    _plot_boxplots(df, res_df, output_dir, n_top=6)

    return res_df


def _plot_volcano(res_df, output_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    x = res_df['log2fc'].values
    y = -np.log10(res_df['fdr'].values + 1e-300)

    colors = []
    for i in range(len(res_df)):
        fdr = res_df['fdr'].iloc[i]
        lfc = res_df['log2fc'].iloc[i]
        if fdr < 0.05 and lfc > 0:
            colors.append('#B2182B')
        elif fdr < 0.05 and lfc < 0:
            colors.append('#2166AC')
        else:
            colors.append('#AAAAAA')

    ax.scatter(x, y, c=colors, s=20, alpha=0.7,
               edgecolors='white', linewidths=0.3)

    if (res_df['fdr'] < 0.05).any():
        ax.axhline(-np.log10(0.05), ls='--', color='grey', lw=0.8, alpha=0.5)

    ax.axvline(0, ls='-', color='grey', lw=0.5, alpha=0.3)

    for _, r in res_df.head(8).iterrows():
        xi = r['log2fc']
        yi = -np.log10(r['fdr'] + 1e-300)
        ax.annotate(r['cluster'].replace('Cluster_', 'C'), (xi, yi),
                    fontsize=5, ha='center', va='bottom',
                    xytext=(0, 4), textcoords='offset points')

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#B2182B',
               label='↑ Resected (FDR<0.05)', markersize=5),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#2166AC',
               label='↑ Non-resected (FDR<0.05)', markersize=5),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#AAAAAA',
               label='Not significant', markersize=5),
    ], loc='upper left', fontsize=6, framealpha=0.8)

    ax.set_xlabel('log₂FC (resected / non-resected)')
    ax.set_ylabel('−log₁₀(FDR)')
    ax.set_title('Resectability — Cluster Enrichment (Mann-Whitney U)')
    plt.tight_layout()
    fig.savefig(Path(output_dir) / 'volcano_plot.png', dpi=DPI)
    plt.close(fig)
    print("  Saved volcano plot")


def _plot_boxplots(df, res_df, output_dir, n_top=6):
    top = res_df.head(n_top)
    ncols = 3
    nrows = (len(top) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7, 2.5 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    for idx, (_, row) in enumerate(top.iterrows()):
        ax = axes[idx // ncols, idx % ncols]
        col = row['cluster']
        plot_df = pd.DataFrame({
            'proportion': df[col].values,
            'group': ['Resected' if r == 1 else 'Non-resected'
                      for r in df['IS_RESECTED'].values]
        })
        sns.boxplot(data=plot_df, x='group', y='proportion',
                    palette=['#B2182B', '#2166AC'], width=0.5,
                    fliersize=2, linewidth=0.8, ax=ax)
        ax.set_title(col.replace('Cluster_', 'C'), fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Proportion')
        p_txt = f"FDR={row['fdr']:.2e}  log2FC={row['log2fc']:+.2f}"
        ax.text(0.5, 0.98, p_txt, transform=ax.transAxes,
                ha='center', va='top', fontsize=5)

    for idx in range(len(top), nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    plt.tight_layout()
    fig.savefig(Path(output_dir) / 'boxplots_top_clusters.png', dpi=DPI)
    plt.close(fig)
    print("  Saved boxplots")


def select_clusters_mwu(mwu_df, n_select):
    selected = mwu_df.sort_values('pvalue').head(n_select)['cluster'].tolist()
    print(f"\n  MWU top {n_select} selected for prediction: {selected}")
    return selected


def _patient_grouped_splits(X, y, groups, n_folds, n_repeats, fold_indices_pkl=None):
    """Yield (fold_idx, tr, te) splits grouped by patient, so all slides of a patient stay on one side."""
    if fold_indices_pkl is not None:
        import pickle as _pickle
        with open(fold_indices_pkl, "rb") as fh:
            payload = _pickle.load(fh)
        # the pkl holds slide names, which cannot be mapped to rows from arrays alone
        raise RuntimeError(
            "_patient_grouped_splits called with fold_indices_pkl but no slide "
            "lookup. Use _canonical_splits_from_pkl instead.")
    fold_idx = 0
    for rep in range(n_repeats):
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True,
                                    random_state=SEED + rep)
        for tr, te in sgkf.split(X, y, groups=groups):
            yield fold_idx, tr, te
            fold_idx += 1


def _canonical_splits_from_pkl(df, fold_indices_pkl):
    """Yield (fold_idx, tr, te) from a fold_indices.pkl, mapped to rows by slide name."""
    import pickle as _pickle
    with open(fold_indices_pkl, "rb") as fh:
        payload = _pickle.load(fh)
    slide_to_idx = {s: i for i, s in enumerate(df["slide_name"].astype(str))}
    for fold_idx, (key, entry) in enumerate(sorted(payload.items())):
        tr_names = entry["train"] if isinstance(entry, dict) else entry[0]
        te_names = entry["test"]  if isinstance(entry, dict) else entry[1]
        tr = np.array([slide_to_idx[n] for n in tr_names if n in slide_to_idx], dtype=int)
        te = np.array([slide_to_idx[n] for n in te_names if n in slide_to_idx], dtype=int)
        yield fold_idx, tr, te


def evaluate_elasticnet_cv(df, feature_cols, n_folds=5, n_repeats=5, fold_indices_pkl=None):
    X = df[feature_cols].values.astype(np.float64)
    y = df['IS_RESECTED'].values.astype(int)
    groups = df['PATHID'].values  # patient-grouped CV

    outer = (_canonical_splits_from_pkl(df, fold_indices_pkl)
             if fold_indices_pkl is not None
             else _patient_grouped_splits(X, y, groups, n_folds, n_repeats))

    results = []
    for fold_idx, tr, te in outer:
        best_c, best_auc = 0.01, 0.0
        inner = StratifiedGroupKFold(n_splits=3, shuffle=True,
                                      random_state=SEED + fold_idx)
        for c in [0.001, 0.01, 0.1, 1.0, 10.0]:
            aucs = []
            for i_tr, i_val in inner.split(X[tr], y[tr], groups=groups[tr]):
                pipe = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=c, penalty='elasticnet', solver='saga',
                                        l1_ratio=0.5, max_iter=2000,
                                        # 'balanced' offsets the ~1-in-5 positive rate; helps AUC, not calibration
                                        class_weight='balanced', random_state=SEED))
                pipe.fit(X[tr[i_tr]], y[tr[i_tr]])
                if len(np.unique(y[tr[i_val]])) > 1:
                    aucs.append(roc_auc_score(y[tr[i_val]],
                                               pipe.predict_proba(X[tr[i_val]])[:,1]))
            if aucs and np.mean(aucs) > best_auc:
                best_auc, best_c = np.mean(aucs), c

        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=best_c, penalty='elasticnet', solver='saga',
                                l1_ratio=0.5, max_iter=2000,
                                class_weight='balanced', random_state=SEED))
        pipe.fit(X[tr], y[tr])
        auc = roc_auc_score(y[te], pipe.predict_proba(X[te])[:,1]) \
              if len(np.unique(y[te])) > 1 else 0.5

        results.append({
            'repeat': fold_idx // n_folds + 1,
            'fold':   fold_idx % n_folds + 1,
            'auc':    float(auc),
            'best_c': best_c,
        })
    return results


def evaluate_xgb_cv(df, feature_cols, n_folds=5, n_repeats=5, n_trials=100,
                    fold_indices_pkl=None):
    X = df[feature_cols].values.astype(np.float32)
    y = df['IS_RESECTED'].values.astype(int)
    groups = df['PATHID'].values  # patient-grouped CV
    spw = int((y == 0).sum()) / max(int((y == 1).sum()), 1)

    outer = (_canonical_splits_from_pkl(df, fold_indices_pkl)
             if fold_indices_pkl is not None
             else _patient_grouped_splits(X, y, groups, n_folds, n_repeats))

    results = []
    for fold_idx, tr, te in outer:
        def objective(trial):
            params = {
                'n_estimators':     trial.suggest_int('n_estimators', 50, 300),
                'max_depth':        trial.suggest_int('max_depth', 2, 5),
                'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample':        trial.suggest_float('subsample', 0.5, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),
                'min_child_weight': trial.suggest_int('min_child_weight', 3, 20),
                'reg_alpha':  trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                'scale_pos_weight': spw,
                'objective': 'binary:logistic', 'eval_metric': 'auc',
                'tree_method': 'hist', 'device': 'cuda',
                'random_state': SEED, 'verbosity': 0,
            }
            inner = StratifiedGroupKFold(n_splits=3, shuffle=True,
                                          random_state=SEED + fold_idx)
            aucs = []
            for i_tr, i_val in inner.split(X[tr], y[tr], groups=groups[tr]):
                m = xgb.XGBClassifier(**params)
                m.fit(X[tr[i_tr]], y[tr[i_tr]], verbose=False)
                if len(np.unique(y[tr[i_val]])) > 1:
                    aucs.append(roc_auc_score(y[tr[i_val]],
                                               m.predict_proba(X[tr[i_val]])[:,1]))
            return np.mean(aucs) if aucs else 0.5

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=SEED + fold_idx))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        bp = study.best_params.copy()
        final = {**bp, 'scale_pos_weight': spw,
                 'objective': 'binary:logistic', 'eval_metric': 'auc',
                 'tree_method': 'hist', 'device': 'cuda',
                 'random_state': SEED, 'verbosity': 0}
        m = xgb.XGBClassifier(**final)
        m.fit(X[tr], y[tr], verbose=False)
        auc = roc_auc_score(y[te], m.predict_proba(X[te])[:,1]) \
              if len(np.unique(y[te])) > 1 else 0.5

        rep = fold_idx // n_folds + 1
        f   = fold_idx % n_folds + 1
        print(f"    Rep {rep} Fold {f}: AUC={auc:.4f}  "
              f"(inner best={study.best_value:.4f})")
        results.append({'repeat': rep, 'fold': f, 'auc': float(auc),
                        'best_inner_auc': study.best_value})
    return results


def plot_comparison(all_results, output_dir):
    rows = []
    for name, folds in all_results.items():
        for f in folds:
            rows.append({'Model': name, 'AUC': f['auc']})
    df_plot = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(max(6, len(all_results) * 1.4), 4))
    palette = COLORS[:len(all_results)]
    sns.boxplot(data=df_plot, x='Model', y='AUC', palette=palette,
                width=0.5, fliersize=2, linewidth=0.8, ax=ax)
    sns.stripplot(data=df_plot, x='Model', y='AUC', color='black',
                  size=2, alpha=0.4, jitter=True, ax=ax)
    ax.axhline(0.5, ls='--', color='grey', alpha=0.5, lw=0.8)
    ylim = ax.get_ylim()
    for i, (name, folds) in enumerate(all_results.items()):
        aucs = [f['auc'] for f in folds]
        ax.text(i, ylim[1] - 0.01,
                f'{np.mean(aucs):.3f}±{np.std(aucs):.3f}',
                ha='center', va='top', fontsize=6, fontweight='bold')
    ax.set_ylabel('AUC (outer fold)')
    ax.set_title('Resectability Prediction — Repeated Stratified CV (5×5)')
    ax.set_ylim(0.3, ylim[1])
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    fig.savefig(Path(output_dir) / 'model_comparison_boxplot.png', dpi=DPI)
    plt.close(fig)
    print("  Saved comparison boxplot")


def parse_args():
    p = argparse.ArgumentParser(
        description="Resectability descriptive + predictive (anti-overfit CV)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--survival-csv',   required=True)
    p.add_argument('--output-dir',     required=True)
    p.add_argument('--gnn-embeddings', default=None)
    p.add_argument('--n-select',       type=int,   default=10)
    p.add_argument('--n-folds',        type=int,   default=5)
    p.add_argument('--n-repeats',      type=int,   default=5)
    p.add_argument('--n-optuna-trials', type=int,  default=100)
    p.add_argument('--fold-indices',   default=None,
                   help='fold_indices.pkl (from the script 16 dry run with '
                        '--fold-indices-pkl). If provided, the outer folds match the '
                        'folds of script 16 exactly.')
    p.add_argument('--skip-xgb',       action='store_true')
    p.add_argument('--skip-predictive', action='store_true',
                   help="Run descriptive only")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("C3PRO — RESECTABILITY ANALYSIS (Script 17)")
    print("=" * 70)
    print(f"  Survival CSV    : {args.survival_csv}")
    print(f"  GNN embeddings  : {args.gnn_embeddings or 'None (Model 4 disabled)'}")
    print(f"  Output          : {out_dir}")
    print(f"  Feature sel     : top {args.n_select} clusters (MWU)")
    print(f"  CV              : {args.n_folds}-fold × {args.n_repeats} repeats")
    print()

    df, cluster_cols, emb_cols, clinical_cols = load_data(
        args.survival_csv, args.gnn_embeddings)
    print(f"  Clinical features: {len(clinical_cols)} — {clinical_cols}")

    mwu_df = descriptive_analysis(df, cluster_cols, out_dir)

    if args.skip_predictive:
        print(f"\n  Done (descriptive only).  Time: {time.time()-t0:.1f}s")
        return

    print(f"\n{'=' * 70}")
    print("SECTION B — PREDICTIVE MODELS")
    print(f"{'=' * 70}")

    # selection runs once on all slides, before the folds: models using `selected` are optimistic
    selected = select_clusters_mwu(mwu_df, args.n_select)

    # model letters are local to this script
    models = {
        '1. ElasticNet\n(clinical)':      clinical_cols,
        '2. ElasticNet\n(clusters)':      selected,
        '3. ElasticNet\n(clin+clusters)': clinical_cols + selected,
    }
    if emb_cols:
        models['4. ElasticNet\n(clin+GNN)'] = clinical_cols + emb_cols

    all_results = {}

    for name, features in models.items():
        print(f"\n{'─' * 60}")
        print(f"MODEL: {name.replace(chr(10),' ')}  ({len(features)} features)")
        print(f"{'─' * 60}")
        folds = evaluate_elasticnet_cv(df, features,
                                        n_folds=args.n_folds,
                                        n_repeats=args.n_repeats,
                                        fold_indices_pkl=args.fold_indices)
        aucs = [f['auc'] for f in folds]
        print(f"  Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        all_results[name] = folds

    if not args.skip_xgb:
        name = '5. XGBoost\n(clin+clusters)'
        features = clinical_cols + selected
        print(f"\n{'─' * 60}")
        print(f"MODEL: {name.replace(chr(10),' ')}  ({len(features)} features)")
        print(f"{'─' * 60}")
        folds = evaluate_xgb_cv(df, features,
                                  n_folds=args.n_folds,
                                  n_repeats=args.n_repeats,
                                  n_trials=args.n_optuna_trials,
                                  fold_indices_pkl=args.fold_indices)
        aucs = [f['auc'] for f in folds]
        print(f"  Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        all_results[name] = folds

    plot_comparison(all_results, out_dir)

    summary = {}
    for name, folds in all_results.items():
        aucs = [f['auc'] for f in folds]
        clean = name.replace('\n', ' ')
        summary[clean] = {
            'mean_auc':   round(float(np.mean(aucs)), 4),
            'std_auc':    round(float(np.std(aucs)),  4),
            'n_folds':    len(aucs),
            'per_fold':   [round(a, 4) for a in aucs],
        }
        pd.DataFrame(folds).to_csv(
            out_dir / f"cv_folds_{clean.split('.')[0].strip()}.csv", index=False)

    with open(out_dir / 'comparison_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print("FINAL COMPARISON")
    print(f"{'=' * 70}")
    print(f"{'Model':<35}  {'Mean AUC':>9}  {'Std':>6}")
    print('─' * 55)
    for name, s in summary.items():
        print(f"{name:<35}  {s['mean_auc']:>9.4f}  {s['std_auc']:>6.4f}")
    print(f"\n  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)  Output: {out_dir}")
    print("Done.")


if __name__ == '__main__':
    main()
