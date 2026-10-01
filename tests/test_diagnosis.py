import json

import numpy as np
import pandas as pd

from conftest import run, slide_name, write_projected


def test_diagnosis_cv_with_shap(tmp_path):
    rng = np.random.default_rng(1)
    names = [slide_name(22, i) for i in range(30)]
    cancer = set(names[:15])
    write_projected(tmp_path / "projected", names, 8, rng, enriched=lambda n: n in cancer)
    pd.DataFrame({"slide_name": names,
                  "label": ["Cancer" if n in cancer else "NoCancer" for n in names],
                  "split": "train"}).to_csv(tmp_path / "slide_split.csv", index=False)
    out = tmp_path / "out"
    run("10", "--projected-dir", tmp_path / "projected", "--slide-split", tmp_path / "slide_split.csv",
        "--output-dir", out, "--multivariate", "--cv-folds", 3, "--cv-optuna-trials", 3)
    summary = json.loads((out / "cv_results_summary.json").read_text())
    assert 0 <= summary["pooled_auc"] <= 1
    assert (out / "slide_level_model_cv.joblib").exists()
    assert (out / "shap_summary_beeswarm.png").exists()
