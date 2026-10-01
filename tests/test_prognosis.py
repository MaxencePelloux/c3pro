import json

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import run, slide_name, write_projected

K = 6
N = 40


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cohort")
    rng = np.random.default_rng(2)
    names = [slide_name(18 if i % 2 else 22, i) for i in range(N)]
    props = rng.dirichlet(np.ones(K), N)
    exp = tmp / "experiment"
    exp.mkdir()
    pd.DataFrame(props, columns=[f"Cluster_{j}" for j in range(K)]).assign(slide_name=names) \
        .to_csv(exp / "slide_proportions_kmeans.csv", index=False)
    risk = props[:, 0] - props[:, 1]
    clin = pd.DataFrame({
        "PATHID": [n[:10] for n in names],
        "DEATH": rng.binomial(1, 0.8, N),
        "MONTHS_FU": np.round(rng.exponential(20 * np.exp(-2 * risk)) + 1, 1),
        "AGE_AT_DIAGNOSIS": rng.normal(66, 9, N).round(),
        "SEX": rng.integers(0, 2, N),
        "PATH_GRADE": rng.integers(1, 4, N),
        "SURGERY_SPECIMEN": np.tile([0, 1], N // 2),
        "LOCALISATION": rng.integers(0, 2, N),
        "PRETX_LARGE_VASCULAR": rng.integers(0, 2, N),
        "PRETX_SIZE": rng.normal(30, 8, N).round(),
        **{c: 0 for c in ("NEO_FOLFORINOX", "NEO_GEMCITABINE", "TX_GEMCITABINE", "TX_FOLFORINOX", "TX_NONE")},
    })
    clin.to_excel(tmp / "clinical.xlsx", index=False)
    pd.DataFrame({"slide_name": names, "label": "Cancer"}).to_csv(tmp / "slide_split.csv", index=False)
    write_projected(tmp / "projected", names, K, rng, n_cells=80)
    return tmp, names


def score_args(tmp):
    return ["--experiment-dir", tmp / "experiment", "--clinical-xlsx", tmp / "clinical.xlsx",
            "--slide-split", tmp / "slide_split.csv", "--output-dir", tmp / "ai_score",
            "--n-folds", 2, "--n-repeats", 1]


def test_folds_graphs_gnn_and_scores(cohort):
    tmp, names = cohort
    run("16", *score_args(tmp), "--dry-run")
    assert (tmp / "ai_score" / "fold_indices.pkl").exists()

    run("13", "--projected-dir", tmp / "projected", "--output-dir", tmp / "graphs",
        "--n-clusters", K, "--skip-qc")
    graphs = sorted((tmp / "graphs").glob("*.pt"))
    assert len(graphs) == N
    g = torch.load(graphs[0], weights_only=False)
    assert g.x.shape == (80, K) and g.edge_index.shape[0] == 2

    gnn = tmp / "experiment" / "gnn_survival"
    run("15", "--ai-score-dir", tmp / "ai_score", "--graph-dir", tmp / "graphs", "--output-dir", gnn,
        "--epochs", 2, "--patience", 2, "--inner-val-frac", 0.3, "--device", "cpu")
    emb = pd.read_csv(gnn / "gnn_embeddings_per_fold.csv")
    assert {"repeat", "fold", "split", "slide_name"} <= set(emb.columns)
    assert sum(c.startswith("emb") for c in emb.columns) == 32

    run("16", *score_args(tmp))
    report = json.loads((tmp / "ai_score" / "evaluation_report.json").read_text())
    assert report["prognosis"] and report["resectability"]
    for model in report["prognosis"].values():
        assert 0 <= model["global_cindex"] <= 1


def test_resectability(cohort):
    tmp, _ = cohort
    if not (tmp / "ai_score" / "cancer_dataset.csv").exists():
        run("16", *score_args(tmp), "--dry-run")
    out = tmp / "resectability"
    run("17", "--survival-csv", tmp / "ai_score" / "cancer_dataset.csv", "--output-dir", out,
        "--n-folds", 2, "--n-repeats", 1, "--n-optuna-trials", 2, "--n-select", 3)
    summary = json.loads((out / "comparison_summary.json").read_text())
    assert summary
    for model in summary.values():
        assert 0 <= model["mean_auc"] <= 1
