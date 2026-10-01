#!/usr/bin/env python3
"""Diagnosis (cancer vs normal) from slide-level cluster proportions, with univariate tests and XGBoost + SHAP.
Inputs: per-slide projected parquets (global_cluster column) and a slide split CSV; outputs go to --output-dir.
Patient grouping comes only from the split CSV: the inner CV splits are stratified, not patient-grouped."""

import numpy as np
import pandas as pd
import sys
import time
import json
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm
import gc

warnings.filterwarnings("ignore")

SCIPY_AVAILABLE = SKLEARN_AVAILABLE = SHAP_AVAILABLE = OPTUNA_AVAILABLE = False

try:
    from scipy.stats import fisher_exact, chi2_contingency, spearmanr
    from statsmodels.stats.multitest import multipletests
    SCIPY_AVAILABLE = True
except ImportError:
    print("Error: scipy and statsmodels required")
    sys.exit(1)

try:
    from sklearn.metrics import (roc_auc_score, roc_curve,
                                  precision_score, recall_score, f1_score, confusion_matrix)
    from sklearn.model_selection import train_test_split, StratifiedKFold
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import shap
    import xgboost
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def check_gpu_availability():
    gpu_info = {'cuda_available': False, 'xgboost_gpu_support': False}
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        gpu_info['cuda_available'] = result.returncode == 0
    except:
        pass
    if SHAP_AVAILABLE and gpu_info['cuda_available']:
        try:
            test_model = xgboost.XGBClassifier(tree_method='hist', device='cuda', n_estimators=1)
            X, y = np.random.rand(10, 5), np.random.randint(0, 2, 10)
            test_model.fit(X, y)
            gpu_info['xgboost_gpu_support'] = True
        except:
            pass
    return gpu_info


def load_faiss_projections(projected_dir, slide_split_csv, ignore_split=False):
    """Build slide-level cluster proportions, labels and splits from projected parquets (no FAISS involved)."""
    projected_dir = Path(projected_dir)

    print(f"\nLoading slide split: {slide_split_csv}")
    split_df = pd.read_csv(slide_split_csv)
    split_df['slide_name'] = split_df['slide_name'].str.strip()
    split_df['diagnostic_category'] = split_df['label'].map({
        'Cancer': 'cancer', 'NoCancer': 'no_cancer'
    })

    slide_to_label = dict(zip(split_df['slide_name'], split_df['diagnostic_category']))

    if ignore_split:
        slide_to_split = {s: 'all' for s in split_df['slide_name']}
        print(f"  {len(split_df)} slides (CV mode, split column ignored)")
    else:
        slide_to_split = dict(zip(split_df['slide_name'], split_df['split']))
        n_train = (split_df['split'] == 'train').sum()
        n_test = (split_df['split'] == 'test').sum()
        print(f"  {len(split_df)} slides: {n_train} train, {n_test} test")

    parquet_files = sorted(projected_dir.glob("*.parquet"))
    print(f"\nLoading {len(parquet_files)} projected parquets...")

    slide_cluster_counts = {}
    slide_cell_counts = {}
    all_cluster_ids = set()
    total_cells = 0

    for pf in tqdm(parquet_files, desc="Reading parquets"):
        slide_name = pf.stem.replace('_features', '').replace('_projected', '').strip()

        if slide_name not in slide_to_label:
            continue

        try:
            df = pd.read_parquet(pf, columns=['global_cluster'])
            clusters = df['global_cluster'].values

            unique, counts = np.unique(clusters, return_counts=True)
            slide_cluster_counts[slide_name] = dict(zip(unique.astype(int), counts.astype(int)))
            slide_cell_counts[slide_name] = len(df)
            all_cluster_ids.update(unique.astype(int))
            total_cells += len(df)

            del df
        except Exception as e:
            print(f"  Warning: Failed to read {pf.name}: {e}")

    gc.collect()

    all_cluster_ids = sorted(all_cluster_ids)
    n_clusters = len(all_cluster_ids)
    print(f"\n  {len(slide_cluster_counts)} slides loaded, {total_cells:,} total cells, {n_clusters} clusters")

    print("\nBuilding slide-level proportion matrix...")
    proportion_data = []
    slide_names = []

    for slide_name in sorted(slide_cluster_counts.keys()):
        counts = slide_cluster_counts[slide_name]
        total = slide_cell_counts[slide_name]

        row = {}
        for cid in all_cluster_ids:
            row[f"Cluster_{cid}"] = counts.get(cid, 0) / total

        proportion_data.append(row)
        slide_names.append(slide_name)

    slide_proportions = pd.DataFrame(proportion_data, index=slide_names)
    slide_labels = pd.Series({s: slide_to_label[s] for s in slide_names})
    slide_splits = pd.Series({s: slide_to_split[s] for s in slide_names})

    train_mask = slide_splits == 'train'
    test_mask = slide_splits == 'test'

    print(f"  Proportions matrix: {slide_proportions.shape}")
    print(f"  Train: {train_mask.sum()} slides ({(slide_labels[train_mask] == 'cancer').sum()} cancer, "
          f"{(slide_labels[train_mask] == 'no_cancer').sum()} no_cancer)")
    print(f"  Test:  {test_mask.sum()} slides ({(slide_labels[test_mask] == 'cancer').sum()} cancer, "
          f"{(slide_labels[test_mask] == 'no_cancer').sum()} no_cancer)")

    print("\nBuilding cell-level cluster counts for univariate analysis...")
    cell_data = []
    for slide_name, counts in slide_cluster_counts.items():
        label = slide_to_label[slide_name]
        split = slide_to_split[slide_name]
        for cid, count in counts.items():
            cell_data.append({
                'slide_name': slide_name,
                'global_cluster_id': int(cid),
                'n_cells': int(count),
                'diagnostic_category': label,
                'split': split
            })

    cell_counts_df = pd.DataFrame(cell_data)

    return cell_counts_df, slide_proportions, slide_labels, slide_splits


def perform_univariate_analysis(cell_counts_df, alpha=0.05, min_cluster_size=100, fisher_threshold=500):
    """Univariate test per cluster (Fisher/Chi²) on all slides."""
    print(f"\n{'='*60}")
    print("UNIVARIATE ANALYSIS")
    print(f"{'='*60}")
    print(f"Min cluster size: {min_cluster_size}, Fisher threshold: {fisher_threshold}")

    cluster_diag = cell_counts_df.groupby(['global_cluster_id', 'diagnostic_category'])['n_cells'].sum().unstack(fill_value=0)

    if 'cancer' not in cluster_diag.columns:
        cluster_diag['cancer'] = 0
    if 'no_cancer' not in cluster_diag.columns:
        cluster_diag['no_cancer'] = 0

    cluster_totals = cluster_diag.sum(axis=1)
    valid_clusters = cluster_totals[cluster_totals >= min_cluster_size].index.tolist()
    print(f"Testing {len(valid_clusters)} clusters (>= {min_cluster_size} cells)")

    total_cancer = cluster_diag['cancer'].sum()
    total_no_cancer = cluster_diag['no_cancer'].sum()
    total_all = total_cancer + total_no_cancer
    overall_cancer_prop = total_cancer / total_all

    test_results = []
    n_fisher = n_chi2 = 0

    for cid in valid_clusters:
        cc = int(cluster_diag.loc[cid, 'cancer'])
        cnc = int(cluster_diag.loc[cid, 'no_cancer'])
        oc = int(total_cancer - cc)
        onc = int(total_no_cancer - cnc)
        ct = cc + cnc

        if ct == 0 or (oc + onc) == 0:
            continue

        table = [[cc, cnc], [oc, onc]]

        try:
            if ct < fisher_threshold:
                odds_ratio, p_value = fisher_exact(table)
                test_used = 'fisher'
                n_fisher += 1
            else:
                chi2_stat, p_value, dof, expected = chi2_contingency(table)
                odds_ratio = (cc * onc) / (cnc * oc) if cnc > 0 and oc > 0 else np.inf
                test_used = 'chi2'
                n_chi2 += 1

            cancer_prop = cc / ct

            # 95% CI (Woolf method), with a 0.5 continuity correction when a cell is zero
            a, b, c, d = cc, cnc, oc, onc
            if a == 0 or b == 0 or c == 0 or d == 0:
                a_, b_, c_, d_ = a+0.5, b+0.5, c+0.5, d+0.5
                or_ci = (a_*d_) / (b_*c_)
                log_or = np.log(or_ci)
                se = np.sqrt(1/a_ + 1/b_ + 1/c_ + 1/d_)
            else:
                log_or = np.log(odds_ratio)
                se = np.sqrt(1/a + 1/b + 1/c + 1/d)

            ci_lower = np.exp(log_or - 1.96 * se)
            ci_upper = np.exp(log_or + 1.96 * se)

            test_results.append({
                'global_cluster_id': cid,
                'cluster_total': ct,
                'cluster_cancer': cc,
                'cluster_no_cancer': cnc,
                'cluster_cancer_prop': cancer_prop,
                'odds_ratio': odds_ratio,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'p_value': p_value,
                'test_used': test_used,
                'enriched_in': 'cancer' if cancer_prop > overall_cancer_prop else 'no_cancer'
            })
        except Exception as e:
            print(f"  Warning: cluster {cid}: {e}")

    if not test_results:
        print("No results!")
        return None

    test_df = pd.DataFrame(test_results)

    _, p_bonf, _, _ = multipletests(test_df['p_value'], alpha=alpha, method='bonferroni')
    test_df['p_bonferroni'] = p_bonf
    _, p_fdr, _, _ = multipletests(test_df['p_value'], alpha=alpha, method='fdr_bh')
    test_df['p_fdr'] = p_fdr

    n_sig = (test_df['p_value'] < alpha).sum()
    n_sig_fdr = (test_df['p_fdr'] < alpha).sum()

    print(f"\nTests: {n_fisher} Fisher, {n_chi2} Chi²")
    print(f"Significant: {n_sig} raw (p<{alpha}), {n_sig_fdr} FDR-corrected")

    sig_df = test_df[test_df['p_fdr'] < alpha].sort_values('odds_ratio', ascending=False)
    cancer_enriched = sig_df[sig_df['enriched_in'] == 'cancer']
    nocancer_enriched = sig_df[sig_df['enriched_in'] == 'no_cancer']

    if len(cancer_enriched) > 0:
        print(f"\nTop 5 cancer-enriched clusters (FDR < {alpha}):")
        for _, row in cancer_enriched.head(5).iterrows():
            print(f"  Cluster {int(row['global_cluster_id'])}: OR={row['odds_ratio']:.2f} "
                  f"({row['cluster_cancer']}/{row['cluster_total']} cancer), p_FDR={row['p_fdr']:.2e}")

    if len(nocancer_enriched) > 0:
        print(f"\nTop 5 no-cancer-enriched clusters (FDR < {alpha}):")
        for _, row in nocancer_enriched.head(5).iterrows():
            print(f"  Cluster {int(row['global_cluster_id'])}: OR={row['odds_ratio']:.2f} "
                  f"({row['cluster_no_cancer']}/{row['cluster_total']} no_cancer), p_FDR={row['p_fdr']:.2e}")

    return test_df


def optuna_objective(trial, X_train, y_train, X_val, y_val, use_gpu=False):
    """Optuna objective for XGBoost hyperparameter optimization."""
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42,
        'tree_method': 'hist',
        'device': 'cuda' if use_gpu else 'cpu', 'base_score': 0.5,
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True)
    }
    try:
        model = xgboost.XGBClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_pred)
    except:
        return 0.0


def monte_carlo_cv(X_train_all, y_train_all, output_dir,
                   training_ratios=[1.0, 0.75, 0.5],
                   n_iterations=10, n_trials_per_ratio=20):
    """Monte Carlo CV with Optuna tuning inside the training set, at several training fractions."""
    print(f"\n{'='*60}")
    print("MONTE CARLO CROSS-VALIDATION (within train set)")
    print(f"{'='*60}")
    print(f"Ratios: {training_ratios}, Iterations: {n_iterations}, Trials/ratio: {n_trials_per_ratio}")

    gpu_info = check_gpu_availability()
    use_gpu = gpu_info['xgboost_gpu_support']
    print(f"Acceleration: {'GPU' if use_gpu else 'CPU'}")

    y_numpy = y_train_all.values if hasattr(y_train_all, 'values') else y_train_all

    all_results = {}
    all_trials_data = []
    best_model_global = None
    best_auc_global = 0.0
    colors = ['#DC143C', '#4169E1', '#2E8B57']

    plt.figure(figsize=(12, 10))
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.7, label='Random (AUC = 0.500)')
    base_fpr = np.linspace(0, 1, 101)

    for ratio, color in zip(training_ratios, colors):
        label = f'{ratio*100:.0f}% training'
        print(f"\n  {label}...")

        ratio_aucs, ratio_precisions, ratio_recalls, ratio_f1s = [], [], [], []
        all_tprs = []

        for iteration in range(n_iterations):
            try:
                if ratio < 1.0:
                    indices = np.arange(len(X_train_all))
                    subset_idx, _ = train_test_split(indices, test_size=1-ratio,
                                                      stratify=y_numpy, random_state=42+iteration)
                    X_subset = X_train_all.iloc[subset_idx]
                    y_subset = y_train_all.iloc[subset_idx]
                else:
                    X_subset = X_train_all
                    y_subset = y_train_all

                y_sub_np = y_subset.values if hasattr(y_subset, 'values') else y_subset
                train_idx, val_idx = train_test_split(np.arange(len(X_subset)), test_size=0.2,
                                                       stratify=y_sub_np, random_state=42+iteration)

                X_tr = X_subset.iloc[train_idx]
                X_val = X_subset.iloc[val_idx]
                y_tr = y_subset.iloc[train_idx]
                y_val = y_subset.iloc[val_idx]

                study = optuna.create_study(direction='maximize',
                                            sampler=optuna.samplers.TPESampler(seed=42+iteration))
                study.optimize(lambda trial: optuna_objective(trial, X_tr, y_tr, X_val, y_val, use_gpu),
                               n_trials=n_trials_per_ratio, show_progress_bar=False)

                for trial in study.trials:
                    all_trials_data.append({
                        'training_ratio': ratio, 'iteration': iteration,
                        'trial_number': trial.number,
                        'auc': trial.value if trial.value else 0.0,
                        'state': trial.state.name, **trial.params
                    })

                best_params = study.best_params.copy()
                best_params.update({
                    'objective': 'binary:logistic', 'eval_metric': 'logloss',
                    'random_state': 42, 'tree_method': 'hist',
                    'device': 'cuda' if use_gpu else 'cpu', 'base_score': 0.5
                })

                model = xgboost.XGBClassifier(**best_params)
                model.fit(X_tr, y_tr)

                y_pred_proba = model.predict_proba(X_val)[:, 1]
                y_val_np = y_val.values if hasattr(y_val, 'values') else y_val

                auc = roc_auc_score(y_val_np, y_pred_proba)
                y_pred = (y_pred_proba > 0.5).astype(int)

                ratio_aucs.append(auc)
                ratio_precisions.append(precision_score(y_val_np, y_pred, zero_division=0))
                ratio_recalls.append(recall_score(y_val_np, y_pred, zero_division=0))
                ratio_f1s.append(f1_score(y_val_np, y_pred, zero_division=0))

                if auc > best_auc_global:
                    best_auc_global = auc
                    best_model_global = {
                        'model': model, 'auc_score': auc, 'best_params': study.best_params,
                        'training_ratio': ratio, 'iteration': iteration
                    }

                fpr, tpr, _ = roc_curve(y_val_np, y_pred_proba)
                tpr_interp = np.interp(base_fpr, fpr, tpr)
                tpr_interp[0] = 0.0
                all_tprs.append(tpr_interp)

            except Exception as e:
                print(f"    Iteration {iteration} failed: {e}")
                continue

        if ratio_aucs:
            mean_auc = np.mean(ratio_aucs)
            std_auc = np.std(ratio_aucs)

            all_results[label] = {
                'mean_auc': mean_auc, 'std_auc': std_auc,
                'mean_precision': np.mean(ratio_precisions), 'std_precision': np.std(ratio_precisions),
                'mean_recall': np.mean(ratio_recalls), 'std_recall': np.std(ratio_recalls),
                'mean_f1': np.mean(ratio_f1s), 'std_f1': np.std(ratio_f1s),
                'auc_scores': ratio_aucs
            }

            mean_tpr = np.mean(all_tprs, axis=0)
            std_tpr = np.std(all_tprs, axis=0)
            plt.plot(base_fpr, mean_tpr, color=color, linewidth=2,
                     label=f'{label} (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
            plt.fill_between(base_fpr, np.maximum(mean_tpr - std_tpr, 0),
                             np.minimum(mean_tpr + std_tpr, 1), color=color, alpha=0.2)

            print(f"    AUC = {mean_auc:.3f} ± {std_auc:.3f}")
            print(f"    Precision = {np.mean(ratio_precisions):.3f} ± {np.std(ratio_precisions):.3f}")
            print(f"    Recall = {np.mean(ratio_recalls):.3f} ± {np.std(ratio_recalls):.3f}")
            print(f"    F1 = {np.mean(ratio_f1s):.3f} ± {np.std(ratio_f1s):.3f}")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Monte Carlo CV with Optuna (Train Set Only)', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'monte_carlo_cv_roc.png', dpi=600, bbox_inches='tight')
    plt.close()

    trials_df = pd.DataFrame(all_trials_data)
    trials_df.to_csv(Path(output_dir) / 'all_optuna_trials.csv', index=False)

    summary_data = []
    for label, res in all_results.items():
        summary_data.append({
            'Training Ratio': label,
            'AUC': f"{res['mean_auc']:.3f} ± {res['std_auc']:.3f}",
            'Precision': f"{res['mean_precision']:.3f} ± {res['std_precision']:.3f}",
            'Recall': f"{res['mean_recall']:.3f} ± {res['std_recall']:.3f}",
            'F1-Score': f"{res['mean_f1']:.3f} ± {res['std_f1']:.3f}"
        })
    pd.DataFrame(summary_data).to_csv(Path(output_dir) / 'performance_summary_table.csv', index=False)

    return {
        'all_results': all_results, 'best_model': best_model_global, 'all_trials': trials_df
    }


def occams_razor_select_params(trials_df, top_percentile=90):
    """Pick the simplest trial among the top-scoring ones rather than the single best trial."""
    successful = trials_df[trials_df['state'] == 'COMPLETE'].copy()
    auc_threshold = np.percentile(successful['auc'], top_percentile)
    elite = successful[successful['auc'] >= auc_threshold].copy()

    good_enough_auc = elite['auc'].median()
    good_enough = elite[elite['auc'] >= good_enough_auc].copy()

    # AUC gaps among elite trials are mostly noise at this sample size, so prefer the least complex model
    simplest = good_enough.sort_values(by=['max_depth', 'n_estimators'], ascending=[True, True]).iloc[0]

    param_names = ['n_estimators', 'learning_rate', 'max_depth', 'subsample',
                   'colsample_bytree', 'reg_alpha', 'reg_lambda', 'gamma']

    recommended = {p: simplest[p] for p in param_names}
    for p in ['n_estimators', 'max_depth']:
        recommended[p] = int(recommended[p])

    selection_info = {
        'recommended_params': recommended,
        'selected_auc': float(simplest['auc']),
        'elite_threshold': float(auc_threshold),
        'good_enough_threshold': float(good_enough_auc)
    }
    return recommended, selection_info


def occams_razor_selection(trials_df, output_dir, top_percentile=90):
    """Occam's-razor selection that also saves recommended_params.json."""
    print(f"\n{'='*60}")
    print("OCCAM'S RAZOR: Simplest High-Performing Model")
    print(f"{'='*60}")

    recommended, selection_info = occams_razor_select_params(trials_df, top_percentile)

    print(f"  Elite threshold: AUC >= {selection_info['elite_threshold']:.4f}")
    print(f"  Selected AUC: {selection_info['selected_auc']:.5f}")
    print(f"  max_depth={recommended['max_depth']}, n_estimators={recommended['n_estimators']}")

    output_path = Path(output_dir) / 'plateau_analysis'
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / 'recommended_params.json', 'w') as f:
        json.dump(selection_info, f, indent=2)

    return recommended


def run_cv_fold_optuna(X_train_fold, y_train_fold, n_trials, use_gpu, fold_idx):
    """Run Optuna tuning within a single CV fold (single 80/20 inner split)."""
    y_np = y_train_fold.values if hasattr(y_train_fold, 'values') else y_train_fold
    train_idx, val_idx = train_test_split(
        np.arange(len(X_train_fold)), test_size=0.2,
        stratify=y_np, random_state=42 + fold_idx
    )

    X_tr = X_train_fold.iloc[train_idx]
    X_val = X_train_fold.iloc[val_idx]
    y_tr = y_train_fold.iloc[train_idx]
    y_val = y_train_fold.iloc[val_idx]

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42 + fold_idx))
    study.optimize(lambda trial: optuna_objective(trial, X_tr, y_tr, X_val, y_val, use_gpu),
                   n_trials=n_trials, show_progress_bar=False)

    trials_data = []
    for trial in study.trials:
        trials_data.append({
            'fold': fold_idx, 'trial_number': trial.number,
            'auc': trial.value if trial.value else 0.0,
            'state': trial.state.name, **trial.params
        })
    trials_df = pd.DataFrame(trials_data)

    recommended, _ = occams_razor_select_params(trials_df)
    return recommended, trials_df


def perform_cv_multivariate_analysis(slide_proportions, slide_labels, output_dir,
                                      n_folds=5, n_trials_per_fold=50):
    """Stratified k-fold CV with per-fold Optuna tuning, then SHAP on a model refit on all slides."""
    if not SHAP_AVAILABLE:
        print("SHAP/XGBoost not available!")
        return None

    print(f"\n{'='*60}")
    print(f"{n_folds}-FOLD STRATIFIED CROSS-VALIDATION")
    print(f"{'='*60}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    gpu_info = check_gpu_availability()
    use_gpu = gpu_info['xgboost_gpu_support']
    print(f"Acceleration: {'GPU' if use_gpu else 'CPU'}")

    y_binary = (slide_labels == 'cancer').astype(int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    print(f"\nTotal: {len(slide_proportions)} slides "
          f"({y_binary.sum()} cancer, {(y_binary == 0).sum()} no_cancer)")
    print(f"Optuna trials per fold: {n_trials_per_fold}")

    fold_metrics = []
    fold_params_list = []
    all_predictions = []
    all_tprs = []
    base_fpr = np.linspace(0, 1, 101)
    all_trials_data = []

    colors = ['#DC143C', '#4169E1', '#2E8B57', '#FF8C00', '#8B008B']

    plt.figure(figsize=(12, 10))
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.7, label='Random (AUC = 0.500)')

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(slide_proportions, y_binary)):
        print(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")

        X_train_fold = slide_proportions.iloc[train_idx]
        X_test_fold = slide_proportions.iloc[test_idx]
        y_train_fold = y_binary.iloc[train_idx]
        y_test_fold = y_binary.iloc[test_idx]

        print(f"  Train: {len(X_train_fold)} ({y_train_fold.sum()} cancer, {(y_train_fold == 0).sum()} no_cancer)")
        print(f"  Test:  {len(X_test_fold)} ({y_test_fold.sum()} cancer, {(y_test_fold == 0).sum()} no_cancer)")

        print(f"  Optuna tuning ({n_trials_per_fold} trials)...")
        fold_params, fold_trials = run_cv_fold_optuna(
            X_train_fold, y_train_fold, n_trials_per_fold, use_gpu, fold_idx
        )
        fold_params_list.append(fold_params)
        all_trials_data.append(fold_trials)

        fold_dir = output_path / f'cv_fold_{fold_idx}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_trials.to_csv(fold_dir / 'optuna_trials.csv', index=False)

        final_params = fold_params.copy()
        final_params.update({
            'objective': 'binary:logistic', 'eval_metric': 'logloss',
            'random_state': 42, 'tree_method': 'hist',
            'device': 'cpu', 'base_score': 0.5
        })

        model = xgboost.XGBClassifier(**final_params)
        X_tr_np = X_train_fold.values.astype(np.float32)
        y_tr_np = y_train_fold.values
        model.fit(X_tr_np, y_tr_np)

        X_te_np = X_test_fold.values.astype(np.float32)
        y_te_np = y_test_fold.values
        y_pred_proba = model.predict_proba(X_te_np)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        fold_auc = roc_auc_score(y_te_np, y_pred_proba)
        fold_precision = precision_score(y_te_np, y_pred, zero_division=0)
        fold_recall = recall_score(y_te_np, y_pred, zero_division=0)
        fold_f1 = f1_score(y_te_np, y_pred, zero_division=0)
        fold_cm = confusion_matrix(y_te_np, y_pred)

        # train AUC, to report the train/test gap
        y_train_pred_proba = model.predict_proba(X_tr_np)[:, 1]
        train_auc = roc_auc_score(y_tr_np, y_train_pred_proba)

        metrics = {
            'fold': fold_idx, 'test_auc': float(fold_auc),
            'test_precision': float(fold_precision), 'test_recall': float(fold_recall),
            'test_f1': float(fold_f1), 'train_auc': float(train_auc),
            'n_train': len(X_train_fold), 'n_test': len(X_test_fold),
            'confusion_matrix': fold_cm.tolist(), 'params': fold_params
        }
        fold_metrics.append(metrics)

        with open(fold_dir / 'fold_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"  AUC: {fold_auc:.3f} (train: {train_auc:.3f}, gap: {train_auc - fold_auc:.3f})")
        print(f"  Precision: {fold_precision:.3f}, Recall: {fold_recall:.3f}, F1: {fold_f1:.3f}")

        for i, idx in enumerate(test_idx):
            all_predictions.append({
                'slide_name': slide_proportions.index[idx],
                'y_true': int(y_te_np[i]),
                'y_pred_proba': float(y_pred_proba[i]),
                'y_pred': int(y_pred[i]),
                'fold': fold_idx
            })

        fpr, tpr, _ = roc_curve(y_te_np, y_pred_proba)
        color = colors[fold_idx % len(colors)]
        plt.plot(fpr, tpr, color=color, linewidth=1, alpha=0.5,
                 label=f'Fold {fold_idx + 1} (AUC = {fold_auc:.3f})')

        tpr_interp = np.interp(base_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        all_tprs.append(tpr_interp)

    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION RESULTS ({n_folds} folds)")
    print(f"{'='*60}")

    aucs = [m['test_auc'] for m in fold_metrics]
    precisions = [m['test_precision'] for m in fold_metrics]
    recalls = [m['test_recall'] for m in fold_metrics]
    f1s = [m['test_f1'] for m in fold_metrics]
    train_aucs = [m['train_auc'] for m in fold_metrics]

    print(f"  AUC:       {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"  Precision: {np.mean(precisions):.3f} ± {np.std(precisions):.3f}")
    print(f"  Recall:    {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
    print(f"  F1-Score:  {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    print(f"  Train AUC: {np.mean(train_aucs):.3f} ± {np.std(train_aucs):.3f}")
    print(f"  Mean gap:  {np.mean(np.array(train_aucs) - np.array(aucs)):.3f}")

    # pooled AUC over all out-of-fold predictions
    pred_df = pd.DataFrame(all_predictions)
    pooled_auc = roc_auc_score(pred_df['y_true'], pred_df['y_pred_proba'])
    print(f"  Pooled AUC: {pooled_auc:.3f}")

    mean_tpr = np.mean(all_tprs, axis=0)
    std_tpr = np.std(all_tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    plt.plot(base_fpr, mean_tpr, color='black', linewidth=2.5,
             label=f'Mean (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
    plt.fill_between(base_fpr, np.maximum(mean_tpr - std_tpr, 0),
                     np.minimum(mean_tpr + std_tpr, 1), color='gray', alpha=0.2)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{n_folds}-Fold Stratified CV — ROC Curves', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'cv_roc_curves.png', dpi=600, bbox_inches='tight')
    plt.close()

    pred_df.to_csv(output_path / 'cv_all_predictions.csv', index=False)

    cv_summary = {
        'n_folds': n_folds,
        'mean_auc': float(mean_auc), 'std_auc': float(std_auc),
        'mean_precision': float(np.mean(precisions)), 'std_precision': float(np.std(precisions)),
        'mean_recall': float(np.mean(recalls)), 'std_recall': float(np.std(recalls)),
        'mean_f1': float(np.mean(f1s)), 'std_f1': float(np.std(f1s)),
        'pooled_auc': float(pooled_auc),
        'per_fold': fold_metrics
    }
    with open(output_path / 'cv_results_summary.json', 'w') as f:
        json.dump(cv_summary, f, indent=2)

    # same table format as the Monte Carlo CV mode
    summary_data = [{
        'Training Ratio': f'{n_folds}-fold CV',
        'AUC': f"{mean_auc:.3f} ± {std_auc:.3f}",
        'Precision': f"{np.mean(precisions):.3f} ± {np.std(precisions):.3f}",
        'Recall': f"{np.mean(recalls):.3f} ± {np.std(recalls):.3f}",
        'F1-Score': f"{np.mean(f1s):.3f} ± {np.std(f1s):.3f}"
    }]
    pd.DataFrame(summary_data).to_csv(output_path / 'performance_summary_table.csv', index=False)

    print(f"\n{'='*60}")
    print("FINAL MODEL: Retrain on all data with consensus params (for SHAP)")
    print(f"{'='*60}")

    # consensus hyperparameters: per-parameter median across folds
    param_names = ['n_estimators', 'learning_rate', 'max_depth', 'subsample',
                   'colsample_bytree', 'reg_alpha', 'reg_lambda', 'gamma']
    consensus_params = {}
    for p in param_names:
        values = [fp[p] for fp in fold_params_list]
        if p in ['n_estimators', 'max_depth']:
            consensus_params[p] = int(np.median(values))
        else:
            consensus_params[p] = float(np.median(values))

    print(f"  Consensus params: max_depth={consensus_params['max_depth']}, "
          f"n_estimators={consensus_params['n_estimators']}")

    with open(output_path / 'cv_consensus_params.json', 'w') as f:
        json.dump(consensus_params, f, indent=2)

    final_params = consensus_params.copy()
    final_params.update({
        'objective': 'binary:logistic', 'eval_metric': 'logloss',
        'random_state': 42, 'tree_method': 'hist',
        'device': 'cpu', 'base_score': 0.5
    })

    final_model = xgboost.XGBClassifier(**final_params)
    X_all_np = slide_proportions.values.astype(np.float32)
    y_all_np = y_binary.values
    final_model.fit(X_all_np, y_all_np)

    print("\nComputing SHAP values (full data)...")
    # newer XGBoost stores base_score as "[5E-1]", which shap's TreeExplainer cannot parse
    import json as json_module
    booster = final_model.get_booster()
    config = json_module.loads(booster.save_config())
    bs = config["learner"]["learner_model_param"]["base_score"]
    if bs.startswith("["):
        config["learner"]["learner_model_param"]["base_score"] = bs.strip("[]")
        booster.load_config(json_module.dumps(config))

    # SHAP values are additive (they sum to the gap from the mean prediction), so contributions
    # can be compared in magnitude, not just ranked
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer(X_all_np)
    shap_values.feature_names = list(slide_proportions.columns)

    plt.figure(figsize=(10, 8))
    mean_shap = np.abs(shap_values.values).mean(axis=0)
    top_idx = np.argsort(mean_shap)[::-1][:20]
    plt.barh(range(len(top_idx)), mean_shap[top_idx], color='steelblue', alpha=0.7)
    plt.yticks(range(len(top_idx)), [slide_proportions.columns[i] for i in top_idx])
    plt.xlabel('Mean |SHAP Value|')
    plt.title('Top 20 Feature Importance (SHAP) — Full Data')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(output_path / 'shap_feature_importance.png', dpi=600, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_all_np, feature_names=list(slide_proportions.columns),
                      show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_path / 'shap_summary_beeswarm.png', dpi=600, bbox_inches='tight')
    plt.close()

    y_pred_full = final_model.predict_proba(X_all_np)[:, 1]
    cancer_idx = np.where(y_all_np == 1)[0]
    nocancer_idx = np.where(y_all_np == 0)[0]

    if len(cancer_idx) > 0:
        extreme_cancer = cancer_idx[np.argmax(y_pred_full[cancer_idx])]
        slide_name = slide_proportions.index[extreme_cancer]
        shap.plots.waterfall(shap_values[extreme_cancer], max_display=15, show=False)
        plt.title(f'SHAP: {slide_name} (Cancer, P={y_pred_full[extreme_cancer]:.3f})')
        plt.tight_layout()
        plt.savefig(output_path / 'shap_waterfall_cancer.png', dpi=600, bbox_inches='tight')
        plt.close()

    if len(nocancer_idx) > 0:
        extreme_nc = nocancer_idx[np.argmin(y_pred_full[nocancer_idx])]
        slide_name = slide_proportions.index[extreme_nc]
        shap.plots.waterfall(shap_values[extreme_nc], max_display=15, show=False)
        plt.title(f'SHAP: {slide_name} (NoCancer, P={y_pred_full[extreme_nc]:.3f})')
        plt.tight_layout()
        plt.savefig(output_path / 'shap_waterfall_nocancer.png', dpi=600, bbox_inches='tight')
        plt.close()

    model_data = {
        'model': final_model,
        'feature_names': list(slide_proportions.columns),
        'training_info': {
            'n_slides': len(slide_proportions),
            'cv_mean_auc': float(mean_auc), 'cv_std_auc': float(std_auc),
            'pooled_auc': float(pooled_auc),
            'consensus_params': consensus_params,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    joblib.dump(model_data, output_path / 'slide_level_model_cv.joblib')

    print("\nAnalyzing cluster redundancy...")
    cluster_corr = slide_proportions.corr()
    redundant = []
    for i in range(len(cluster_corr.columns)):
        for j in range(i + 1, len(cluster_corr.columns)):
            if abs(cluster_corr.iloc[i, j]) > 0.7:
                redundant.append({
                    'cluster_1': cluster_corr.columns[i],
                    'cluster_2': cluster_corr.columns[j],
                    'correlation': cluster_corr.iloc[i, j]
                })

    if redundant:
        pd.DataFrame(redundant).to_csv(output_path / 'redundant_cluster_pairs.csv', index=False)
        print(f"  {len(redundant)} redundant pairs (|r| > 0.7)")

    top30_idx = np.argsort(mean_shap)[::-1][:min(30, len(mean_shap))]
    corr_subset = cluster_corr.iloc[top30_idx, top30_idx]

    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_subset, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, cbar_kws={'label': 'Correlation'})
    plt.title(f'Cluster Correlation Matrix (Top 30 by SHAP)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path / 'cluster_correlation_matrix.png', dpi=600, bbox_inches='tight')
    plt.close()

    return {
        'shap_values': shap_values,
        'X_train': slide_proportions,  # full data for univariate comparison
        'test_auc': mean_auc,
        'train_auc': np.mean(train_aucs),
        'cv_summary': cv_summary
    }


def perform_multivariate_analysis(slide_proportions, slide_labels, slide_splits,
                                   output_dir, n_trials=150, monte_carlo_iterations=100,
                                   resume_shap=False):
    """Single-split pipeline: Monte Carlo CV, Occam's-razor selection, final fit, test evaluation, SHAP."""
    if not SHAP_AVAILABLE:
        print("SHAP/XGBoost not available!")
        return None

    print(f"\n{'='*60}")
    print("MULTIVARIATE ANALYSIS: XGBoost + SHAP")
    print(f"{'='*60}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_mask = slide_splits == 'train'
    test_mask = slide_splits == 'test'

    X_train = slide_proportions[train_mask]
    X_test = slide_proportions[test_mask]
    y_train = (slide_labels[train_mask] == 'cancer').astype(int)
    y_test = (slide_labels[test_mask] == 'cancer').astype(int)

    print(f"\nTrain: {len(X_train)} slides ({y_train.sum()} cancer, {(y_train == 0).sum()} no_cancer)")
    print(f"Test:  {len(X_test)} slides ({y_test.sum()} cancer, {(y_test == 0).sum()} no_cancer)")
    print(f"Features: {X_train.shape[1]} cluster proportions")

    if resume_shap:
        print("\n>>> RESUME MODE: loading saved Occam's Razor params, skipping MC CV...")
        rec_path = output_path / 'plateau_analysis' / 'recommended_params.json'
        with open(rec_path) as f:
            rec = json.load(f)
        recommended_params = rec['recommended_params']
        print(f"  Loaded params: {recommended_params}")
    else:
        mc_results = monte_carlo_cv(
            X_train, y_train, str(output_path),
            training_ratios=[1.0, 0.75, 0.5],
            n_iterations=monte_carlo_iterations,
            n_trials_per_ratio=max(1, n_trials // 3)
        )

        recommended_params = occams_razor_selection(mc_results['all_trials'], str(output_path))

    print(f"\n{'='*60}")
    print("FINAL MODEL: Training on full train set")
    print(f"{'='*60}")

    gpu_info = check_gpu_availability()
    use_gpu = gpu_info['xgboost_gpu_support']

    final_params = recommended_params.copy()
    final_params.update({
        'objective': 'binary:logistic', 'eval_metric': 'logloss',
        'random_state': 42, 'tree_method': 'hist',
        'device': 'cpu', 'base_score': 0.5  # CPU avoids CUDA/pandas compatibility issues
    })

    final_model = xgboost.XGBClassifier(**final_params)
    # numpy input avoids XGBoost/pandas/CUDA compatibility issues
    X_train_np = X_train.values.astype(np.float32) if hasattr(X_train, 'values') else np.asarray(X_train, dtype=np.float32)
    y_train_np = y_train.values if hasattr(y_train, 'values') else np.asarray(y_train)
    final_model.fit(X_train_np, y_train_np)

    X_test_np = X_test.values.astype(np.float32) if hasattr(X_test, 'values') else np.asarray(X_test, dtype=np.float32)
    y_test_pred_proba = final_model.predict_proba(X_test_np)[:, 1]
    y_test_pred = (y_test_pred_proba > 0.5).astype(int)
    y_test_np = y_test.values if hasattr(y_test, 'values') else np.asarray(y_test)

    test_auc = roc_auc_score(y_test_np, y_test_pred_proba)
    test_precision = precision_score(y_test_np, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test_np, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test_np, y_test_pred, zero_division=0)
    test_cm = confusion_matrix(y_test_np, y_test_pred)

    print(f"\n  ** HELD-OUT TEST SET RESULTS **")
    print(f"  AUC:       {test_auc:.3f}")
    print(f"  Precision: {test_precision:.3f}")
    print(f"  Recall:    {test_recall:.3f}")
    print(f"  F1-Score:  {test_f1:.3f}")
    print(f"  Confusion matrix:\n{test_cm}")

    y_train_pred_proba = final_model.predict_proba(X_train_np)[:, 1]
    train_auc = roc_auc_score(y_train_np, y_train_pred_proba)
    print(f"\n  Train AUC: {train_auc:.3f} (for reference)")
    print(f"  Gap: {train_auc - test_auc:.3f} ({'overfitting risk' if train_auc - test_auc > 0.1 else 'acceptable'})")

    fpr, tpr, _ = roc_curve(y_test_np, y_test_pred_proba)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='#DC143C', linewidth=2, label=f'Test (AUC = {test_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve — Held-Out Test Set ({len(X_test)} slides)', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'test_roc_curve.png', dpi=600, bbox_inches='tight')
    plt.close()

    print("\nComputing SHAP values...")
    # newer XGBoost stores base_score as "[5E-1]", which shap's TreeExplainer cannot parse
    import json as json_module
    booster = final_model.get_booster()
    config = json_module.loads(booster.save_config())
    bs = config["learner"]["learner_model_param"]["base_score"]
    if bs.startswith("["):
        config["learner"]["learner_model_param"]["base_score"] = bs.strip("[]")
        booster.load_config(json_module.dumps(config))
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer(X_train_np)
    # numpy input dropped the column names
    shap_values.feature_names = list(X_train.columns)

    plt.figure(figsize=(10, 8))
    mean_shap = np.abs(shap_values.values).mean(axis=0)
    top_idx = np.argsort(mean_shap)[::-1][:20]
    plt.barh(range(len(top_idx)), mean_shap[top_idx], color='steelblue', alpha=0.7)
    plt.yticks(range(len(top_idx)), [X_train.columns[i] for i in top_idx])
    plt.xlabel('Mean |SHAP Value|')
    plt.title('Top 20 Feature Importance (SHAP)')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(output_path / 'shap_feature_importance.png', dpi=600, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_train_np, feature_names=list(X_train.columns), show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_path / 'shap_summary_beeswarm.png', dpi=600, bbox_inches='tight')
    plt.close()

    y_train_pred_proba_full = final_model.predict_proba(X_train_np)[:, 1]
    cancer_idx = np.where(y_train_np == 1)[0]
    nocancer_idx = np.where(y_train_np == 0)[0]

    if len(cancer_idx) > 0:
        extreme_cancer = cancer_idx[np.argmax(y_train_pred_proba_full[cancer_idx])]
        slide_name = X_train.index[extreme_cancer]
        shap.plots.waterfall(shap_values[extreme_cancer], max_display=15, show=False)
        plt.title(f'SHAP: {slide_name} (Cancer, P={y_train_pred_proba_full[extreme_cancer]:.3f})')
        plt.tight_layout()
        plt.savefig(output_path / 'shap_waterfall_cancer.png', dpi=600, bbox_inches='tight')
        plt.close()

    if len(nocancer_idx) > 0:
        extreme_nc = nocancer_idx[np.argmin(y_train_pred_proba_full[nocancer_idx])]
        slide_name = X_train.index[extreme_nc]
        shap.plots.waterfall(shap_values[extreme_nc], max_display=15, show=False)
        plt.title(f'SHAP: {slide_name} (NoCancer, P={y_train_pred_proba_full[extreme_nc]:.3f})')
        plt.tight_layout()
        plt.savefig(output_path / 'shap_waterfall_nocancer.png', dpi=600, bbox_inches='tight')
        plt.close()

    model_data = {
        'model': final_model,
        'feature_names': list(X_train.columns),
        'training_info': {
            'n_train': len(X_train), 'n_test': len(X_test),
            'train_auc': float(train_auc), 'test_auc': float(test_auc),
            'test_precision': float(test_precision), 'test_recall': float(test_recall),
            'test_f1': float(test_f1),
            'params': final_params,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    joblib.dump(model_data, output_path / 'slide_level_model.joblib')

    print("\nAnalyzing cluster redundancy...")
    cluster_corr = X_train.corr()
    redundant = []
    for i in range(len(cluster_corr.columns)):
        for j in range(i+1, len(cluster_corr.columns)):
            if abs(cluster_corr.iloc[i, j]) > 0.7:
                redundant.append({
                    'cluster_1': cluster_corr.columns[i],
                    'cluster_2': cluster_corr.columns[j],
                    'correlation': cluster_corr.iloc[i, j]
                })

    if redundant:
        pd.DataFrame(redundant).to_csv(output_path / 'redundant_cluster_pairs.csv', index=False)
        print(f"  {len(redundant)} redundant pairs (|r| > 0.7)")

    top30_idx = np.argsort(mean_shap)[::-1][:min(30, len(mean_shap))]
    corr_subset = cluster_corr.iloc[top30_idx, top30_idx]

    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_subset, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, cbar_kws={'label': 'Correlation'})
    plt.title('Cluster Correlation Matrix (Top 30 by SHAP)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path / 'cluster_correlation_matrix.png', dpi=600, bbox_inches='tight')
    plt.close()

    test_results = {
        'test_auc': float(test_auc),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_f1': float(test_f1),
        'train_auc': float(train_auc),
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'confusion_matrix': test_cm.tolist(),
        'params': {k: v for k, v in final_params.items() if k not in ['objective', 'eval_metric', 'tree_method', 'device']}
    }
    with open(output_path / 'test_evaluation.json', 'w') as f:
        json.dump(test_results, f, indent=2)

    return {
        'model': final_model, 'shap_values': shap_values,
        'X_train': X_train, 'X_test': X_test,
        'test_auc': test_auc, 'train_auc': train_auc,
        'test_results': test_results
    }


def compare_univariate_shap(univariate_results, shap_values, X_train, output_dir):
    """Compare univariate OR with SHAP importance."""
    if univariate_results is None or shap_values is None:
        return

    print(f"\n{'='*60}")
    print("UNIVARIATE vs SHAP COMPARISON")
    print(f"{'='*60}")

    output_path = Path(output_dir)

    mean_abs_shap = pd.DataFrame({
        'global_cluster_id': [int(col.split('_')[1]) for col in X_train.columns],
        'mean_abs_shap': np.abs(shap_values.values).mean(axis=0),
        'mean_shap_signed': shap_values.values.mean(axis=0)
    })

    comparison = univariate_results.merge(mean_abs_shap, on='global_cluster_id', how='inner')

    corr, p = spearmanr(np.log10(comparison['odds_ratio'].clip(lower=1e-10)), comparison['mean_abs_shap'])
    print(f"  Spearman correlation: rho = {corr:.3f}, p = {p:.2e}")

    comparison['or_rank'] = comparison['odds_ratio'].rank(ascending=False)
    comparison['shap_rank'] = comparison['mean_abs_shap'].rank(ascending=False)
    comparison['rank_diff'] = np.abs(comparison['or_rank'] - comparison['shap_rank'])
    comparison = comparison.sort_values('rank_diff', ascending=False)

    comparison.to_csv(output_path / 'univariate_vs_shap_comparison.csv', index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    colors = comparison['enriched_in'].map({'cancer': '#e74c3c', 'no_cancer': '#3498db'})

    ax1.scatter(np.log10(comparison['odds_ratio'].clip(lower=1e-10)),
                comparison['mean_abs_shap'], c=colors, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('log₁₀(Odds Ratio)', fontsize=12)
    ax1.set_ylabel('Mean |SHAP Value|', fontsize=12)
    ax1.set_title(f'Univariate OR vs SHAP\nSpearman ρ = {corr:.3f}', fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.scatter(comparison['or_rank'], comparison['shap_rank'], c=colors, alpha=0.6, s=50)
    max_rank = max(comparison['or_rank'].max(), comparison['shap_rank'].max())
    ax2.plot([0, max_rank], [0, max_rank], 'k--', alpha=0.5, linewidth=2)
    ax2.set_xlabel('Odds Ratio Rank', fontsize=12)
    ax2.set_ylabel('SHAP Rank', fontsize=12)
    ax2.set_title('Rank Comparison', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'univariate_vs_shap.png', dpi=600, bbox_inches='tight')
    plt.close()


def calculate_slide_diversity(cell_counts_df, output_dir):
    """Calculate and visualize slide diversity per cluster."""
    print(f"\n{'='*60}")
    print("SLIDE DIVERSITY ANALYSIS")
    print(f"{'='*60}")

    diversity = cell_counts_df.groupby('global_cluster_id').agg(
        n_slides=('slide_name', 'nunique'),
        n_cells=('n_cells', 'sum')
    ).reset_index()
    diversity['cells_per_slide'] = diversity['n_cells'] / diversity['n_slides']

    print(f"  {len(diversity)} clusters")
    print(f"  Mean slides/cluster: {diversity['n_slides'].mean():.1f}")
    print(f"  Median slides/cluster: {diversity['n_slides'].median():.1f}")
    print(f"  Range: {diversity['n_slides'].min()}-{diversity['n_slides'].max()}")

    diversity.to_csv(Path(output_dir) / 'slide_diversity.csv', index=False)

    top50 = diversity.nlargest(50, 'n_cells')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

    ax1.bar(range(len(top50)), top50['n_slides'], color='steelblue', alpha=0.7)
    ax1.set_xticks(range(len(top50)))
    ax1.set_xticklabels([f"C{int(c)}" for c in top50['global_cluster_id']], rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Number of Slides')
    ax1.set_title('Slide Diversity: Unique Slides per Cluster (Top 50 by Cell Count)')
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.bar(range(len(top50)), top50['n_cells'], color='#e74c3c', alpha=0.7)
    ax2.set_xticks(range(len(top50)))
    ax2.set_xticklabels([f"C{int(c)}" for c in top50['global_cluster_id']], rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Number of Cells')
    ax2.set_title('Cell Count per Cluster (Top 50)')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'slide_diversity_histogram.png', dpi=600, bbox_inches='tight')
    plt.close()

    return diversity


def main():
    parser = argparse.ArgumentParser(description='C3PRO step 10 — diagnosis from cluster composition')
    parser.add_argument('--projected-dir', required=True, help='Path to the projected_slides directory written by step 05')
    parser.add_argument('--slide-split', required=True, help='Path to slide_split.csv')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--univariate', action='store_true', help='Run univariate analysis')
    parser.add_argument('--multivariate', action='store_true', help='Run multivariate XGBoost + SHAP')
    parser.add_argument('--min-cluster-size', type=int, default=100, help='Min cluster size for univariate')
    parser.add_argument('--fisher-threshold', type=int, default=500, help='Fisher vs Chi² threshold')
    parser.add_argument('--optuna-trials', type=int, default=150, help='Total Optuna trials')
    parser.add_argument('--monte-carlo-iterations', type=int, default=100, help='Monte Carlo iterations')
    parser.add_argument("--resume-shap", action="store_true", help="Resume from SHAP step: skip parquet loading and Monte Carlo CV, reload saved proportions and Occam params")
    parser.add_argument('--cv-folds', type=int, default=0, help='Number of CV folds (0=use slide_split.csv, 5=5-fold CV)')
    parser.add_argument('--cv-optuna-trials', type=int, default=50, help='Optuna trials per CV fold')

    args = parser.parse_args()

    start_time = time.time()

    print("=" * 60)
    print("DIAGNOSIS STATISTICS ANALYSIS v2")
    print("=" * 60)
    print(f"Projected dir: {args.projected_dir}")
    print(f"Slide split: {args.slide_split}")
    print(f"Output: {args.output_dir}")
    print(f"Univariate: {args.univariate}")
    print(f"Multivariate: {args.multivariate}")
    if args.cv_folds > 0:
        print(f"Mode: {args.cv_folds}-fold cross-validation")
        print(f"Optuna trials per fold: {args.cv_optuna_trials}")
    else:
        print(f"Mode: single train/test split")
        print(f"Optuna trials: {args.optuna_trials}")
        print(f"Monte Carlo iterations: {args.monte_carlo_iterations}")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.cv_folds > 0:
        cell_counts_df, slide_proportions, slide_labels, _ = load_faiss_projections(
            args.projected_dir, args.slide_split, ignore_split=True
        )

        props_with_labels = slide_proportions.copy()
        props_with_labels['diagnostic'] = slide_labels
        props_with_labels.to_csv(output_path / 'slide_proportions_all.csv')

        diversity = calculate_slide_diversity(cell_counts_df, args.output_dir)

        univariate_results = None
        if args.univariate:
            univariate_results = perform_univariate_analysis(
                cell_counts_df, min_cluster_size=args.min_cluster_size,
                fisher_threshold=args.fisher_threshold
            )
            if univariate_results is not None:
                univariate_results.to_csv(output_path / 'univariate_results.csv', index=False)

        mv_results = None
        if args.multivariate:
            mv_results = perform_cv_multivariate_analysis(
                slide_proportions, slide_labels, args.output_dir,
                n_folds=args.cv_folds, n_trials_per_fold=args.cv_optuna_trials
            )

            if univariate_results is not None and mv_results is not None:
                compare_univariate_shap(
                    univariate_results, mv_results['shap_values'],
                    mv_results['X_train'], args.output_dir
                )

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"ANALYSIS COMPLETE ({elapsed/60:.1f} min)")
        print(f"{'='*60}")
        print(f"Output: {args.output_dir}")

        if mv_results and 'cv_summary' in mv_results:
            s = mv_results['cv_summary']
            print(f"\n  ** {args.cv_folds}-FOLD CV RESULTS **")
            print(f"  AUC:       {s['mean_auc']:.3f} ± {s['std_auc']:.3f}")
            print(f"  Pooled AUC: {s['pooled_auc']:.3f}")

        if univariate_results is not None:
            n_sig = (univariate_results['p_fdr'] < 0.05).sum()
            print(f"  Significant clusters (FDR < 0.05): {n_sig}")

        print(f"\nTotal time: {elapsed/60:.1f} min")

    else:
        if args.resume_shap:
            print("\n>>> RESUME MODE: skipping parquet loading, univariate, and Monte Carlo CV")
            print(">>> Loading saved slide proportions and Occam's Razor params...\n")

            saved_props = pd.read_csv(output_path / 'slide_proportions_all.csv', index_col=0)
            feature_cols = [c for c in saved_props.columns if c.startswith('Cluster_') or c.startswith('cluster_')]
            slide_proportions = saved_props[feature_cols]
            slide_labels = saved_props['diagnostic'].str.lower()
            slide_splits = saved_props['split']

            univariate_results = None
            if (output_path / 'univariate_results.csv').exists():
                univariate_results = pd.read_csv(output_path / 'univariate_results.csv')
        else:
            cell_counts_df, slide_proportions, slide_labels, slide_splits = load_faiss_projections(
                args.projected_dir, args.slide_split
            )

            train_mask = slide_splits == 'train'
            test_mask = slide_splits == 'test'

            props_with_labels = slide_proportions.copy()
            props_with_labels['diagnostic'] = slide_labels
            props_with_labels['split'] = slide_splits
            props_with_labels.to_csv(output_path / 'slide_proportions_all.csv')

            diversity = calculate_slide_diversity(cell_counts_df, args.output_dir)

            univariate_results = None
            if args.univariate:
                univariate_results = perform_univariate_analysis(
                    cell_counts_df, min_cluster_size=args.min_cluster_size,
                    fisher_threshold=args.fisher_threshold
                )
                if univariate_results is not None:
                    univariate_results.to_csv(output_path / 'univariate_results.csv', index=False)

        mv_results = None
        if args.multivariate or args.resume_shap:
            mv_results = perform_multivariate_analysis(
                slide_proportions, slide_labels, slide_splits,
                args.output_dir, n_trials=args.optuna_trials,
                monte_carlo_iterations=args.monte_carlo_iterations,
                resume_shap=args.resume_shap
            )

            if univariate_results is not None and mv_results is not None:
                compare_univariate_shap(
                    univariate_results, mv_results['shap_values'],
                    mv_results['X_train'], args.output_dir
                )

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"ANALYSIS COMPLETE ({elapsed/60:.1f} min)")
        print(f"{'='*60}")
        print(f"Output: {args.output_dir}")

        if mv_results:
            print(f"\n  ** KEY RESULTS **")
            print(f"  Train AUC: {mv_results['train_auc']:.3f}")
            print(f"  Test AUC:  {mv_results['test_auc']:.3f}")

        if univariate_results is not None:
            n_sig = (univariate_results['p_fdr'] < 0.05).sum()
            print(f"  Significant clusters (FDR < 0.05): {n_sig}")

        print(f"\nTotal time: {elapsed/60:.1f} min")


if __name__ == '__main__':
    main()
