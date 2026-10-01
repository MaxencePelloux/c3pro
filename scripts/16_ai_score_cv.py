"""Score every prognosis (C-index) and resectability (AUC) model on shared patient-grouped folds and write out-of-fold predictions.
Inputs: --experiment-dir (cluster proportions, per-fold GNN embeddings from step 15), --clinical-xlsx, --slide-split, --fold-indices-pkl (from step 14).
Outputs: out-of-fold scores, evaluation report, figures and final model under --output-dir; --dry-run only writes cancer_dataset.csv and fold_indices.pkl."""

import argparse
import json
import pickle
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.pathid import extract_pathid

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
DPI = 300
FONT_SIZE = 7

plt.rcParams.update({'font.size': FONT_SIZE, 'font.family': 'Arial'})


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _inject_fold_gnn(work_df, gnn_per_fold, gnn_cols, repeat, fold):
    """Return a copy of work_df with gnn_cols taken from the GNN of fold (repeat, fold); missing slides get 0."""
    fold_emb = gnn_per_fold[
        (gnn_per_fold["repeat"] == repeat) & (gnn_per_fold["fold"] == fold)
    ]
    emb_lookup = fold_emb.drop_duplicates(subset="slide_name").set_index("slide_name")[gnn_cols]
    aligned = emb_lookup.reindex(work_df["slide_name"]).fillna(0.0)
    aligned.index = work_df.index
    out = work_df.copy()
    for col in gnn_cols:
        out[col] = aligned[col].values
    return out


def phase0_merge(experiment_dir, clinical_xlsx, slide_split_csv, output_dir):
    _log("PHASE 0 — DATA LOADING & MERGE")

    props = pd.read_csv(experiment_dir / "slide_proportions_kmeans.csv")
    props["slide_name"] = props["slide_name"].str.replace("_features", "").str.replace("_projected", "").str.replace(" ", "")
    cluster_cols = [c for c in props.columns if c.startswith("Cluster_")]
    _log(f"  Proportions: {props.shape}  ({len(cluster_cols)} clusters)")

    clin = pd.read_excel(clinical_xlsx)
    clin["PATHID"] = clin["PATHID"].astype(str)
    clin = clin.rename(columns={"MONTHS_FOLLOWEDUP": "MONTHS_FU"})
    clin_cols = ["PATHID", "DEATH", "MONTHS_FU", "AGE_AT_DIAGNOSIS", "SEX", "PATH_GRADE",
                 "SURGERY_SPECIMEN", "LOCALISATION", "PRETX_LARGE_VASCULAR", "PRETX_SIZE",
                 "NEO_FOLFORINOX", "NEO_GEMCITABINE", "TX_GEMCITABINE", "TX_FOLFORINOX", "TX_NONE"]
    clin_sub = clin[clin_cols].drop_duplicates(subset="PATHID")

    # drop duplicate slide rows, which would otherwise inflate the merged dataset
    split = pd.read_csv(slide_split_csv)
    split = split.drop_duplicates(subset="slide_name", keep="first").reset_index(drop=True)

    # prefer per-fold embeddings; the fold-agnostic file leaks test slides into the GNN
    gnn_per_fold_path = experiment_dir / "gnn_survival" / "gnn_embeddings_per_fold.csv"
    gnn_all_path = experiment_dir / "gnn_survival" / "gnn_embeddings_all.csv"

    gnn_per_fold = None
    gnn = None
    gnn_cols = []

    if gnn_per_fold_path.exists():
        gnn_per_fold = pd.read_csv(gnn_per_fold_path)
        gnn_cols = [c for c in gnn_per_fold.columns if c.startswith("emb")]
        n_folds_present = gnn_per_fold[["repeat", "fold"]].drop_duplicates().shape[0]
        _log(f"  GNN embeddings (per-fold, leakage-safe): {gnn_per_fold.shape}  "
             f"({len(gnn_cols)} dims, {n_folds_present} folds)")
    elif gnn_all_path.exists():
        gnn = pd.read_csv(gnn_all_path)
        gnn_cols = [c for c in gnn.columns if c.startswith("emb")]
        _log(f"  GNN embeddings (fold-agnostic): {gnn.shape}  ({len(gnn_cols)} dims)")
        _log("  WARNING: gnn_embeddings_all.csv has leakage — rerun 15_gnn_survival_nested_cv.py")
    else:
        _log("  GNN embeddings: NOT FOUND — GNN models will be skipped")

    # extract_pathid accepts both '-' and ' ' between patient ID and block ID
    split["PATHID"] = split["slide_name"].apply(extract_pathid)
    split["year"] = split["slide_name"].str[:2].astype(int)
    split["scanner_batch"] = split["year"].apply(lambda y: "pre" if y <= 21 else "post")

    merged = split.merge(clin_sub, on="PATHID", how="left")

    # join on slide names without separators, which differ between files
    props_join = props.copy()
    props_join["slide_name_join"] = props_join["slide_name"].str.replace("-", "").str.replace(" ", "")
    merged["slide_name_join"] = merged["slide_name"].str.replace("-", "").str.replace(" ", "")
    merged = merged.merge(props_join.drop(columns=["slide_name"]), on="slide_name_join", how="left")
    merged = merged.drop(columns=["slide_name_join"])

    # per-fold embeddings are injected inside the phase 2/3 fold loops instead
    if gnn is not None:
        merged = merged.merge(gnn[["slide_name"] + gnn_cols], on="slide_name", how="left")
        merged[gnn_cols] = merged[gnn_cols].fillna(0)

    clinical_features = ["AGE_AT_DIAGNOSIS", "SEX", "PATH_GRADE", "PRETX_SIZE",
                         "LOCALISATION", "PRETX_LARGE_VASCULAR", "SURGERY_SPECIMEN"]
    for col in clinical_features:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
            merged[col] = merged[col].fillna(merged[col].median())

    cancer = merged[merged["label"] == "Cancer"].copy()

    surv_mask = cancer["DEATH"].notna() & cancer["MONTHS_FU"].notna() & (cancer["MONTHS_FU"] > 0)
    resect_mask = cancer["SURGERY_SPECIMEN"].notna()

    _log(f"  Total slides: {len(merged)}")
    _log(f"  Cancer slides: {len(cancer)}")
    _log(f"  With survival data: {surv_mask.sum()}")
    _log(f"  With resectability data: {resect_mask.sum()}")
    _log(f"  Pre-scanner: {(cancer['scanner_batch'] == 'pre').sum()}  Post-scanner: {(cancer['scanner_batch'] == 'post').sum()}")
    _log(f"  Year distribution:")
    for y in sorted(cancer["year"].unique()):
        _log(f"    20{y}: {(cancer['year'] == y).sum()} slides")

    merged.to_csv(output_dir / "merged_dataset.csv", index=False)
    cancer.to_csv(output_dir / "cancer_dataset.csv", index=False)

    return cancer, cluster_cols, gnn_cols, clinical_features, gnn_per_fold


def phase1_folds(df, n_folds, n_repeats, output_dir, external_pkl=None):
    _log("PHASE 1 — FOLD ASSIGNMENT (year-stratified, patient-grouped)")

    # use the shared partition from step 14 so paired comparisons with the MIL baselines are valid;
    # folds are re-mapped by slide_name, so row order does not matter
    if external_pkl is not None:
        _log(f"PHASE 1 — LOADING EXTERNAL (benchmark-matched) FOLDS from {external_pkl}")
        with open(external_pkl, "rb") as fh:
            ext = pickle.load(fh)
        pos = {n: i for i, n in enumerate(df["slide_name"])}
        fold_indices = {}
        recs = []
        for (repeat, fold), p in sorted(ext.items()):
            tr = np.array([pos[n] for n in p["train"] if n in pos], dtype=int)
            te = np.array([pos[n] for n in p["test"]  if n in pos], dtype=int)
            fold_indices[(repeat, fold)] = (tr, te)
            for n in p["test"]:
                if n in pos:
                    recs.append({"slide_name": n, "repeat": repeat, "fold": fold})
        pd.DataFrame(recs).to_csv(output_dir / "fold_assignments.csv", index=False)
        # rewrite the pkl with indices aligned to this df, for step 15
        payload = {}
        for (repeat, fold), (tr, te) in fold_indices.items():
            payload[(repeat, fold)] = {
                "train": df.iloc[tr]["slide_name"].tolist(),
                "test":  df.iloc[te]["slide_name"].tolist(),
                "train_idx": tr, "test_idx": te,
            }
        with open(output_dir / "fold_indices.pkl", "wb") as fh:
            pickle.dump(payload, fh)
        n_test_pat = pd.DataFrame(recs).assign(
            pid=lambda d: d.slide_name.str.extract(r"(\d{2}[A-Z]{2}\d{6})")[0]
        ).groupby("repeat").pid.nunique().to_dict()
        _log(f"  Loaded {len(fold_indices)} external folds; test patients/repeat={n_test_pat}")
        return fold_indices

    from sklearn.model_selection import StratifiedGroupKFold

    if "PATHID" not in df.columns:
        raise ValueError("phase1_folds requires a PATHID column on df. "
                         "Run phase0_merge first (it populates PATHID via "
                         "utils.pathid.extract_pathid).")

    years = df["year"].values
    groups = df["PATHID"].values

    fold_records = []
    fold_indices = {}  # (repeat, fold) -> (train_idx, test_idx)

    # StratifiedGroupKFold has no repeats, so reseed per repeat; year stratification is
    # approximate because patient grouping takes priority
    for repeat in range(n_repeats):
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True,
                                    random_state=SEED + repeat)
        for fold, (train_idx, test_idx) in enumerate(sgkf.split(df, years, groups=groups)):
            fold_indices[(repeat, fold)] = (train_idx, test_idx)
            for idx in test_idx:
                fold_records.append({
                    "slide_name": df.iloc[idx]["slide_name"],
                    "repeat": repeat,
                    "fold": fold,
                })

    fold_df = pd.DataFrame(fold_records)
    fold_df.to_csv(output_dir / "fold_assignments.csv", index=False)

    # slide names are saved alongside indices for step 15
    fold_payload = {}
    for (repeat, fold), (train_idx, test_idx) in fold_indices.items():
        fold_payload[(repeat, fold)] = {
            "train": df.iloc[train_idx]["slide_name"].tolist(),
            "test": df.iloc[test_idx]["slide_name"].tolist(),
            "train_idx": np.asarray(train_idx),
            "test_idx": np.asarray(test_idx),
        }
    with open(output_dir / "fold_indices.pkl", "wb") as f:
        pickle.dump(fold_payload, f)
    _log(f"  Saved fold_indices.pkl ({len(fold_payload)} folds)")

    _log("  Fold balance check (Repeat 0):")
    for f in range(n_folds):
        _, te = fold_indices[(0, f)]
        test_df = df.iloc[te]
        pre = (test_df["scanner_batch"] == "pre").sum()
        post = (test_df["scanner_batch"] == "post").sum()
        years_str = ", ".join(f"{y}:{(test_df['year']==y).sum()}" for y in sorted(test_df["year"].unique()))
        _log(f"    Fold {f}: {len(te)} slides (pre={pre}, post={post})  years=[{years_str}]")

    return fold_indices


def phase2_prognosis(df, fold_indices, cluster_cols, gnn_cols, clinical_features, output_dir,
                     gnn_per_fold=None):
    _log("PHASE 2 — PROGNOSIS MODELS")
    if gnn_per_fold is not None:
        _log("  Using per-fold OOF GNN embeddings (leakage-safe)")

    from sklearn.preprocessing import StandardScaler
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored

    mask = df["DEATH"].notna() & df["MONTHS_FU"].notna() & (df["MONTHS_FU"] > 0)
    surv_df = df[mask].reset_index(drop=True)
    _log(f"  Survival subset: {len(surv_df)} patients, {int(surv_df['DEATH'].sum())} events")

    y_all = np.array([(bool(e), t) for e, t in zip(surv_df["DEATH"], surv_df["MONTHS_FU"])],
                     dtype=[("event", bool), ("time", float)])

    clinical_cols_avail = [c for c in clinical_features if c in surv_df.columns]
    # model letters are local to this script
    models = {
        "A": ("Clinical only", clinical_cols_avail),
        "B": ("Clusters only", cluster_cols),
        "C": ("GNN only", gnn_cols) if gnn_cols else None,
        "D": ("Clin+clusters", clinical_cols_avail + cluster_cols),
        "E": ("Clin+GNN", clinical_cols_avail + gnn_cols) if gnn_cols else None,
        "F": ("XGBoost clin+clusters", clinical_cols_avail + cluster_cols),
    }
    models = {k: v for k, v in models.items() if v is not None}

    all_scores = []

    for model_id, (model_name, feat_cols) in models.items():
        _log(f"  Model {model_id}: {model_name} ({len(feat_cols)} features)")
        fold_cindices = []

        uses_gnn = bool(gnn_cols) and any(c in feat_cols for c in gnn_cols)

        for (repeat, fold), (train_all, test_all) in fold_indices.items():
            # use embeddings from the GNN trained on this fold only
            if gnn_per_fold is not None and uses_gnn:
                fold_surv_df = _inject_fold_gnn(surv_df, gnn_per_fold, gnn_cols, repeat, fold)
            else:
                fold_surv_df = surv_df

            # fold indices refer to df, so map them to the survival subset by slide_name
            train_slides = set(df.iloc[train_all]["slide_name"])
            test_slides = set(df.iloc[test_all]["slide_name"])

            train_mask = fold_surv_df["slide_name"].isin(train_slides)
            test_mask = fold_surv_df["slide_name"].isin(test_slides)

            if test_mask.sum() < 5:
                continue

            X_train = fold_surv_df.loc[train_mask, feat_cols].values.astype(np.float64)
            X_test = fold_surv_df.loc[test_mask, feat_cols].values.astype(np.float64)
            y_train = y_all[train_mask.values]
            y_test = y_all[test_mask.values]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            X_train_s = np.nan_to_num(X_train_s, 0)
            X_test_s = np.nan_to_num(X_test_s, 0)

            if model_id == "F":
                try:
                    import xgboost as xgb
                    times_tr = fold_surv_df.loc[train_mask, "MONTHS_FU"].values
                    events_tr = fold_surv_df.loc[train_mask, "DEATH"].values
                    dtrain = xgb.DMatrix(X_train_s)
                    dtrain.set_float_info("label_lower_bound", times_tr.astype(np.float32))
                    dtrain.set_float_info("label_upper_bound",
                        np.where(events_tr == 1, times_tr, np.inf).astype(np.float32))
                    params = {
                        "objective": "survival:aft", "eval_metric": "aft-nloglik",
                        "aft_loss_distribution": "extreme",
                        "learning_rate": 0.01, "max_depth": 2,
                        "subsample": 0.8, "colsample_bytree": 0.5,
                        "reg_alpha": 1.0, "reg_lambda": 0.01,
                        "tree_method": "hist", "random_state": SEED, "verbosity": 0,
                    }
                    bst = xgb.train(params, dtrain, num_boost_round=500, verbose_eval=False)
                    dtest = xgb.DMatrix(X_test_s)
                    scores = -bst.predict(dtest)  # AFT predicts time; negate so higher means higher risk
                except Exception:
                    continue
            else:
                try:
                    # l1_ratio=0.9 favours a sparse model, given many correlated cluster proportions
                    cox = CoxnetSurvivalAnalysis(
                        l1_ratio=0.9, alpha_min_ratio=0.01, n_alphas=100,
                        max_iter=100000, fit_baseline_model=True,
                    )
                    cox.fit(X_train_s, y_train)
                    scores = cox.predict(X_test_s)
                except Exception:
                    continue

            try:
                ci = concordance_index_censored(y_test["event"], y_test["time"], scores)[0]
            except Exception:
                ci = 0.5

            fold_cindices.append(ci)

            test_slides_list = fold_surv_df.loc[test_mask, "slide_name"].values
            test_events = fold_surv_df.loc[test_mask, "DEATH"].values
            test_times = fold_surv_df.loc[test_mask, "MONTHS_FU"].values
            test_batches = fold_surv_df.loc[test_mask, "scanner_batch"].values
            for i in range(len(scores)):
                all_scores.append({
                    "slide_name": test_slides_list[i],
                    "repeat": repeat, "fold": fold,
                    "model_id": model_id, "model_name": model_name,
                    "ai_score": float(scores[i]),
                    "os_event": int(test_events[i]),
                    "os_months": float(test_times[i]),
                    "scanner_batch": test_batches[i],
                })

        if fold_cindices:
            mean_ci = np.mean(fold_cindices)
            std_ci = np.std(fold_cindices)
            _log(f"    C-index: {mean_ci:.4f} +/- {std_ci:.4f}  ({len(fold_cindices)} folds)")

    scores_df = pd.DataFrame(all_scores)
    scores_df.to_csv(output_dir / "ai_scores_prognosis.csv", index=False)
    _log(f"  Saved {len(scores_df)} OOF scores → ai_scores_prognosis.csv")
    return scores_df


def phase3_resectability(df, fold_indices, cluster_cols, gnn_cols, clinical_features, output_dir,
                         gnn_per_fold=None):
    _log("PHASE 3 — RESECTABILITY MODELS")
    if gnn_per_fold is not None:
        _log("  Using per-fold OOF GNN embeddings (leakage-safe)")

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import roc_auc_score

    mask = df["SURGERY_SPECIMEN"].notna()
    resect_df = df[mask].reset_index(drop=True)
    resect_df["IS_RESECTED"] = (resect_df["SURGERY_SPECIMEN"] == 1.0).astype(int)
    _log(f"  Resectability subset: {len(resect_df)} patients, {resect_df['IS_RESECTED'].sum()} resected")

    # SURGERY_SPECIMEN is the target, so it is excluded from the features
    clinical_cols_avail = [c for c in clinical_features if c in resect_df.columns and c != "SURGERY_SPECIMEN"]
    models = {
        "1": ("Clinical only", clinical_cols_avail),
        "2": ("Clusters only", cluster_cols),
        "3": ("GNN only", gnn_cols) if gnn_cols else None,
        "4": ("Clin+clusters", clinical_cols_avail + cluster_cols),
        "5": ("Clin+GNN", clinical_cols_avail + gnn_cols) if gnn_cols else None,
        "6": ("XGBoost clin+clusters", clinical_cols_avail + cluster_cols),
    }
    models = {k: v for k, v in models.items() if v is not None}

    all_scores = []

    for model_id, (model_name, feat_cols) in models.items():
        _log(f"  Model {model_id}: {model_name} ({len(feat_cols)} features)")
        fold_aucs = []

        uses_gnn = bool(gnn_cols) and any(c in feat_cols for c in gnn_cols)

        for (repeat, fold), (train_all, test_all) in fold_indices.items():
            # use embeddings from the GNN trained on this fold only
            if gnn_per_fold is not None and uses_gnn:
                fold_resect_df = _inject_fold_gnn(resect_df, gnn_per_fold, gnn_cols, repeat, fold)
            else:
                fold_resect_df = resect_df

            train_slides = set(df.iloc[train_all]["slide_name"])
            test_slides = set(df.iloc[test_all]["slide_name"])
            train_mask = fold_resect_df["slide_name"].isin(train_slides)
            test_mask = fold_resect_df["slide_name"].isin(test_slides)

            if test_mask.sum() < 5:
                continue

            X_train = fold_resect_df.loc[train_mask, feat_cols].values.astype(np.float64)
            X_test = fold_resect_df.loc[test_mask, feat_cols].values.astype(np.float64)
            y_train = fold_resect_df.loc[train_mask, "IS_RESECTED"].values
            y_test = fold_resect_df.loc[test_mask, "IS_RESECTED"].values

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            X_train_s = np.nan_to_num(X_train_s, 0)
            X_test_s = np.nan_to_num(X_test_s, 0)

            if model_id == "6":
                try:
                    import xgboost as xgb
                    spw = int((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1)
                    clf = xgb.XGBClassifier(
                        n_estimators=300, max_depth=3, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.5,
                        scale_pos_weight=spw, eval_metric="auc",
                        tree_method="hist", random_state=SEED, verbosity=0,
                    )
                    clf.fit(X_train_s, y_train, verbose=False)
                    scores = clf.predict_proba(X_test_s)[:, 1]
                except Exception:
                    continue
            else:
                try:
                    lr = LogisticRegressionCV(
                        penalty="elasticnet", solver="saga",
                        l1_ratios=[0.1, 0.5, 0.9], Cs=20, cv=3,
                        max_iter=10000, random_state=SEED,
                        scoring="roc_auc", class_weight="balanced",
                    )
                    lr.fit(X_train_s, y_train)
                    scores = lr.predict_proba(X_test_s)[:, 1]
                except Exception:
                    continue

            if len(np.unique(y_test)) > 1:
                auc = roc_auc_score(y_test, scores)
            else:
                auc = 0.5
            fold_aucs.append(auc)

            test_slides_list = fold_resect_df.loc[test_mask, "slide_name"].values
            test_batches = fold_resect_df.loc[test_mask, "scanner_batch"].values
            for i in range(len(scores)):
                all_scores.append({
                    "slide_name": test_slides_list[i],
                    "repeat": repeat, "fold": fold,
                    "model_id": model_id, "model_name": model_name,
                    "ai_score": float(scores[i]),
                    "label": int(y_test[i]),
                    "scanner_batch": test_batches[i],
                })

        if fold_aucs:
            _log(f"    AUC: {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}  ({len(fold_aucs)} folds)")

    scores_df = pd.DataFrame(all_scores)
    scores_df.to_csv(output_dir / "ai_scores_resectability.csv", index=False)
    _log(f"  Saved {len(scores_df)} OOF scores → ai_scores_resectability.csv")
    return scores_df


def phase4_evaluation(prog_scores, resect_scores, output_dir):
    _log("PHASE 4 — EVALUATION & FIGURES")

    from sksurv.metrics import concordance_index_censored
    from sklearn.metrics import roc_auc_score

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    report = {"prognosis": {}, "resectability": {}}

    _log("  Prognosis metrics:")
    for model_id in sorted(prog_scores["model_id"].unique()):
        model_name = prog_scores[prog_scores["model_id"] == model_id]["model_name"].iloc[0]
        msub = prog_scores[prog_scores["model_id"] == model_id]

        fold_cis = []
        for (rep, fold), grp in msub.groupby(["repeat", "fold"]):
            try:
                ci = concordance_index_censored(
                    grp["os_event"].astype(bool), grp["os_months"], grp["ai_score"]
                )[0]
                fold_cis.append(ci)
            except Exception:
                pass

        # pooled C-index on repeat 0 only, so each slide is counted once
        rep0 = msub[msub["repeat"] == 0]
        try:
            global_ci = concordance_index_censored(
                rep0["os_event"].astype(bool), rep0["os_months"], rep0["ai_score"]
            )[0]
        except Exception:
            global_ci = np.mean(fold_cis) if fold_cis else 0.5

        subcohort_ci = {}
        for batch in ["pre", "post"]:
            batch_data = rep0[rep0["scanner_batch"] == batch]
            if len(batch_data) > 10:
                try:
                    subcohort_ci[batch] = concordance_index_censored(
                        batch_data["os_event"].astype(bool),
                        batch_data["os_months"],
                        batch_data["ai_score"]
                    )[0]
                except Exception:
                    subcohort_ci[batch] = None

        report["prognosis"][f"{model_id}. {model_name}"] = {
            "global_cindex": round(global_ci, 4),
            "per_fold_mean": round(np.mean(fold_cis), 4) if fold_cis else None,
            "per_fold_std": round(np.std(fold_cis), 4) if fold_cis else None,
            "n_folds": len(fold_cis),
            "per_fold_values": [round(c, 4) for c in fold_cis],
            "subcohort_cindex": {k: round(v, 4) for k, v in subcohort_ci.items() if v},
        }
        _log(f"    {model_id}. {model_name}: global={global_ci:.4f}  mean={np.mean(fold_cis):.4f}+/-{np.std(fold_cis):.4f}  pre={subcohort_ci.get('pre', 'N/A')}  post={subcohort_ci.get('post', 'N/A')}")

    _log("  Resectability metrics:")
    for model_id in sorted(resect_scores["model_id"].unique()):
        model_name = resect_scores[resect_scores["model_id"] == model_id]["model_name"].iloc[0]
        msub = resect_scores[resect_scores["model_id"] == model_id]

        fold_aucs = []
        for (rep, fold), grp in msub.groupby(["repeat", "fold"]):
            if len(np.unique(grp["label"])) > 1:
                fold_aucs.append(roc_auc_score(grp["label"], grp["ai_score"]))

        rep0 = msub[msub["repeat"] == 0]
        try:
            global_auc = roc_auc_score(rep0["label"], rep0["ai_score"])
        except Exception:
            global_auc = np.mean(fold_aucs) if fold_aucs else 0.5

        subcohort_auc = {}
        for batch in ["pre", "post"]:
            batch_data = rep0[rep0["scanner_batch"] == batch]
            if len(batch_data) > 10 and len(np.unique(batch_data["label"])) > 1:
                subcohort_auc[batch] = roc_auc_score(batch_data["label"], batch_data["ai_score"])

        report["resectability"][f"{model_id}. {model_name}"] = {
            "global_auc": round(global_auc, 4),
            "per_fold_mean": round(np.mean(fold_aucs), 4) if fold_aucs else None,
            "per_fold_std": round(np.std(fold_aucs), 4) if fold_aucs else None,
            "n_folds": len(fold_aucs),
            "per_fold_values": [round(a, 4) for a in fold_aucs],
            "subcohort_auc": {k: round(v, 4) for k, v in subcohort_auc.items() if v},
        }
        _log(f"    {model_id}. {model_name}: global={global_auc:.4f}  mean={np.mean(fold_aucs):.4f}+/-{np.std(fold_aucs):.4f}  pre={subcohort_auc.get('pre', 'N/A')}  post={subcohort_auc.get('post', 'N/A')}")

    _log("  Batch robustness check:")
    batch_report = {}
    best_prog = prog_scores[prog_scores["model_id"] == "E"] if "E" in prog_scores["model_id"].values else prog_scores[prog_scores["model_id"] == "D"]
    if len(best_prog) > 0:
        rep0 = best_prog[best_prog["repeat"] == 0]
        try:
            import statsmodels.api as sm
            rep0_clean = rep0.dropna(subset=["ai_score"])
            X_adj = pd.get_dummies(rep0_clean[["scanner_batch"]], drop_first=True).astype(float)
            X_adj = sm.add_constant(X_adj)
            ols = sm.OLS(rep0_clean["ai_score"].astype(float), X_adj).fit()
            batch_pval = ols.pvalues.iloc[1] if len(ols.pvalues) > 1 else 1.0
            batch_report["batch_pval_unadjusted"] = round(float(batch_pval), 6)
            _log(f"    Batch effect (unadjusted OLS): p={batch_pval:.4f}")
        except Exception as e:
            _log(f"    Batch regression failed: {e}")

    batch_report_path = output_dir / "batch_robustness.json"
    json.dump(batch_report, open(batch_report_path, "w"), indent=2)

    _log("  Generating figures...")

    fig, ax = plt.subplots(figsize=(8, 5))
    model_data = []
    model_labels = []
    for model_key in sorted(report["prognosis"].keys()):
        vals = report["prognosis"][model_key]["per_fold_values"]
        if vals:
            model_data.append(vals)
            model_labels.append(model_key.split(". ")[1] if ". " in model_key else model_key)
    if model_data:
        bp = ax.boxplot(model_data, labels=model_labels, patch_artist=True)
        colors = plt.cm.Set2(np.linspace(0, 1, len(model_data)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_ylabel("C-index")
        ax.set_title("Prognosis — Model Comparison (year-stratified 5x5 CV)")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(fig_dir / "prognosis_cindex_boxplot.png", dpi=DPI)
        plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    model_data = []
    model_labels = []
    for model_key in sorted(report["resectability"].keys()):
        vals = report["resectability"][model_key]["per_fold_values"]
        if vals:
            model_data.append(vals)
            model_labels.append(model_key.split(". ")[1] if ". " in model_key else model_key)
    if model_data:
        bp = ax.boxplot(model_data, labels=model_labels, patch_artist=True)
        colors = plt.cm.Set2(np.linspace(0, 1, len(model_data)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_ylabel("AUC")
        ax.set_title("Resectability — Model Comparison (year-stratified 5x5 CV)")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(fig_dir / "resectability_auc_boxplot.png", dpi=DPI)
        plt.close()

    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test

        best_model_id = "E" if "E" in prog_scores["model_id"].values else "D"
        rep0 = prog_scores[(prog_scores["model_id"] == best_model_id) & (prog_scores["repeat"] == 0)]

        if len(rep0) > 30:
            tertiles = pd.qcut(rep0["ai_score"], 3, labels=["Low risk", "Medium risk", "High risk"])
            fig, ax = plt.subplots(figsize=(7, 5))
            kmf = KaplanMeierFitter()
            colors_km = ["#2ca02c", "#ff7f0e", "#d62728"]
            for i, (label, color) in enumerate(zip(["Low risk", "Medium risk", "High risk"], colors_km)):
                mask = tertiles == label
                kmf.fit(rep0.loc[mask, "os_months"], rep0.loc[mask, "os_event"].astype(bool), label=label)
                kmf.plot_survival_function(ax=ax, color=color, ci_show=True)

            low = tertiles == "Low risk"
            high = tertiles == "High risk"
            lr = logrank_test(rep0.loc[low, "os_months"], rep0.loc[high, "os_months"],
                              rep0.loc[low, "os_event"], rep0.loc[high, "os_event"])
            ax.set_title(f"KM by AI Score Tertiles (Model {best_model_id}, log-rank p={lr.p_value:.2e})")
            ax.set_xlabel("Months"); ax.set_ylabel("Survival Probability")
            fig.tight_layout()
            fig.savefig(fig_dir / "km_ai_score_tertiles.png", dpi=DPI)
            plt.close()
    except Exception as e:
        _log(f"    KM plot failed: {e}")

    try:
        best_model_id = "E" if "E" in prog_scores["model_id"].values else "D"
        rep0 = prog_scores[(prog_scores["model_id"] == best_model_id) & (prog_scores["repeat"] == 0)]
        fig, ax = plt.subplots(figsize=(6, 4))
        for batch, color in [("pre", "steelblue"), ("post", "coral")]:
            data = rep0[rep0["scanner_batch"] == batch]["ai_score"]
            ax.hist(data, bins=30, alpha=0.6, color=color, label=f"{batch}-scanner (n={len(data)})")
        ax.legend()
        ax.set_xlabel("AI Score"); ax.set_ylabel("Count")
        ax.set_title(f"AI Score Distribution by Scanner Batch (Model {best_model_id})")
        fig.tight_layout()
        fig.savefig(fig_dir / "ai_score_distribution_by_batch.png", dpi=DPI)
        plt.close()
    except Exception as e:
        _log(f"    Batch distribution plot failed: {e}")

    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        model_names = []
        pre_cis = []
        post_cis = []
        global_cis = []
        for model_key in sorted(report["prognosis"].keys()):
            r = report["prognosis"][model_key]
            sc = r.get("subcohort_cindex", {})
            if "pre" in sc and "post" in sc:
                model_names.append(model_key.split(". ")[1] if ". " in model_key else model_key)
                pre_cis.append(sc["pre"])
                post_cis.append(sc["post"])
                global_cis.append(r["global_cindex"])

        if model_names:
            x = np.arange(len(model_names))
            w = 0.25
            ax.bar(x - w, pre_cis, w, label="Pre-scanner", color="steelblue")
            ax.bar(x, global_cis, w, label="Global", color="grey")
            ax.bar(x + w, post_cis, w, label="Post-scanner", color="coral")
            ax.set_xticks(x); ax.set_xticklabels(model_names, rotation=30)
            ax.set_ylabel("C-index"); ax.legend()
            ax.set_title("C-index by Sub-cohort")
            fig.tight_layout()
            fig.savefig(fig_dir / "cindex_by_subcohort.png", dpi=DPI)
        plt.close()
    except Exception as e:
        _log(f"    Subcohort plot failed: {e}")

    json.dump(report, open(output_dir / "evaluation_report.json", "w"), indent=2)
    _log(f"  Report saved → evaluation_report.json")

    return report


def phase5_final_model(df, cluster_cols, gnn_cols, clinical_features, output_dir,
                       gnn_per_fold=None):
    _log("PHASE 5 — FINAL MODEL & CLUSTER WEIGHTS")

    from sklearn.preprocessing import StandardScaler
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    import joblib

    clinical_cols_avail = [c for c in clinical_features if c in df.columns]

    # per-fold mode: average each slide's out-of-fold embeddings across repeats to get
    # one vector per slide for the final cohort-wide fit
    use_gnn = bool(gnn_cols)
    if use_gnn and gnn_per_fold is not None:
        oof = gnn_per_fold[gnn_per_fold["split"] == "test"]
        if len(oof) == 0:
            _log("  WARNING: no OOF rows in gnn_per_fold — falling back to clin+clusters")
            use_gnn = False
        else:
            consensus = (oof.groupby("slide_name", as_index=False)[gnn_cols]
                            .mean())
            n_with_oof = consensus.shape[0]
            n_total = df["slide_name"].nunique()
            df = df.merge(consensus, on="slide_name", how="left")
            df[gnn_cols] = df[gnn_cols].fillna(0.0)
            _log(f"  Consensus GNN embeddings merged: {n_with_oof}/{n_total} slides "
                 f"have OOF predictions (the rest get 0-vector — typically slides "
                 "without survival data filtered out anyway).")

    surv_mask = df["DEATH"].notna() & df["MONTHS_FU"].notna() & (df["MONTHS_FU"] > 0)
    surv_df = df[surv_mask]

    if use_gnn and all(c in surv_df.columns for c in gnn_cols):
        feat_cols = clinical_cols_avail + gnn_cols
        model_name = "clin+GNN (consensus OOF)" if gnn_per_fold is not None else "clin+GNN"
    else:
        feat_cols = clinical_cols_avail + cluster_cols
        model_name = "clin+clusters"

    X = surv_df[feat_cols].values.astype(np.float64)
    y = np.array([(bool(e), t) for e, t in zip(surv_df["DEATH"], surv_df["MONTHS_FU"])],
                 dtype=[("event", bool), ("time", float)])

    scaler = StandardScaler()
    X_s = scaler.fit_transform(np.nan_to_num(X, 0))

    cox = CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01, n_alphas=100,
                                  max_iter=100000, fit_baseline_model=True)
    cox.fit(X_s, y)

    coefs = cox.coef_[:, -1]  # last alpha (least regularized)
    weights_df = pd.DataFrame({"feature": feat_cols, "coefficient": coefs})
    weights_df = weights_df.sort_values("coefficient", key=abs, ascending=False)
    weights_df.to_csv(output_dir / "cluster_weights_prognosis.csv", index=False)

    joblib.dump(cox, output_dir / "final_model_prognosis.pkl")
    joblib.dump(scaler, output_dir / "final_scaler_prognosis.pkl")
    _log(f"  Final prognosis model ({model_name}): {len(feat_cols)} features")
    _log(f"  Top 5 features: {weights_df.head(5)['feature'].tolist()}")

    cluster_weights = weights_df[weights_df["feature"].str.startswith("Cluster_") | weights_df["feature"].str.startswith("emb")]
    if len(cluster_weights) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        top_n = min(20, len(cluster_weights))
        top = cluster_weights.head(top_n)
        colors = ["salmon" if c > 0 else "steelblue" for c in top["coefficient"]]
        ax.barh(range(top_n), top["coefficient"].values, color=colors)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top["feature"].values, fontsize=7)
        ax.set_xlabel("Coefficient")
        ax.set_title(f"Top Feature Weights — Prognosis ({model_name})")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(output_dir / "figures" / "cluster_weights_barplot.png", dpi=DPI)
        plt.close()

    _log("  Phase 5 done.")


def main():
    parser = argparse.ArgumentParser(description="AI Score CV Pipeline — year-stratified 5-fold")
    parser.add_argument("--experiment-dir", type=Path, required=True,
                        help="Directory with clustering results (slide_proportions, gnn_survival/)")
    parser.add_argument("--clinical-xlsx", type=Path, required=True)
    parser.add_argument("--slide-split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Run only phase0 (merge) + phase1 (folds) and exit. "
                             "Use this to emit cancer_dataset.csv + fold_indices.pkl "
                             "for 15_gnn_survival_nested_cv.py before the full run.")
    parser.add_argument("--fold-indices-pkl", type=Path, default=None,
                        help="If set, load CV folds from this pickle (e.g. produced by "
                             "14_benchmark_fold_indices.py from a fold_assignments.csv) instead "
                             "of generating StratifiedGroupKFold. Folds are re-mapped by "
                             "slide_name against the cancer cohort, so row order is irrelevant.")
    args = parser.parse_args()

    global SEED
    SEED = args.seed
    t0 = time.perf_counter()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    print(f"AI SCORE CV PIPELINE — YEAR-STRATIFIED {args.n_folds}-FOLD x {args.n_repeats} REPEATS")
    print(f"  experiment : {args.experiment_dir}")
    print(f"  output     : {args.output_dir}")
    print(f"  seed       : {args.seed}")
    print(f"{'='*70}\n")

    cancer_df, cluster_cols, gnn_cols, clinical_features, gnn_per_fold = phase0_merge(
        args.experiment_dir, args.clinical_xlsx, args.slide_split, args.output_dir
    )

    fold_indices = phase1_folds(cancer_df, args.n_folds, args.n_repeats, args.output_dir,
                                external_pkl=args.fold_indices_pkl)

    if args.dry_run:
        elapsed = (time.perf_counter() - t0) / 60
        _log(f"\n{'='*70}")
        _log(f"DRY RUN — stopped after phase1 ({elapsed:.1f} min)")
        _log(f"  Emitted: cancer_dataset.csv, fold_assignments.csv, fold_indices.pkl")
        _log(f"  Next:    python scripts/15_gnn_survival_nested_cv.py --ai-score-dir {args.output_dir} ...")
        _log(f"{'='*70}")
        return

    prog_scores = phase2_prognosis(
        cancer_df, fold_indices, cluster_cols, gnn_cols, clinical_features, args.output_dir,
        gnn_per_fold=gnn_per_fold,
    )

    resect_scores = phase3_resectability(
        cancer_df, fold_indices, cluster_cols, gnn_cols, clinical_features, args.output_dir,
        gnn_per_fold=gnn_per_fold,
    )

    report = phase4_evaluation(prog_scores, resect_scores, args.output_dir)

    phase5_final_model(cancer_df, cluster_cols, gnn_cols, clinical_features, args.output_dir,
                       gnn_per_fold=gnn_per_fold)

    elapsed = (time.perf_counter() - t0) / 60
    _log(f"\n{'='*70}")
    _log(f"DONE — {elapsed:.1f} min total")
    _log(f"  Output: {args.output_dir}")
    _log(f"{'='*70}")


if __name__ == "__main__":
    main()
