import sys
import types

import numpy as np
import pandas as pd

from conftest import load, slide_name


def fake_cupy():
    # enough of the cupy API for the streaming k-means to run on numpy
    cp = types.ModuleType("cupy")
    for name in ("asarray", "zeros", "eye", "argmax", "bincount", "maximum", "linalg", "float32", "int32"):
        setattr(cp, name, getattr(np, name))
    cp.asnumpy = np.asarray
    return cp


def test_streaming_spherical_kmeans(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy())
    m = load("04")
    rng = np.random.default_rng(0)
    d, k = 16, 3
    directions = np.linalg.qr(rng.normal(size=(d, d)))[0][:k]
    feat_cols = [f"f{i}" for i in range(d)]
    src = tmp_path / "features"
    src.mkdir()
    for i, year in enumerate([18, 19, 22, 23]):
        labels = rng.integers(0, k, 200)
        feats = directions[labels] + 0.05 * rng.normal(size=(200, d))
        df = pd.DataFrame(feats, columns=feat_cols)
        df.insert(0, "slide", slide_name(year, i))
        df.insert(1, "x", rng.uniform(0, 1000, 200))
        df.insert(2, "y", rng.uniform(0, 1000, 200))
        df.to_parquet(src / f"{slide_name(year, i)}.parquet", index=False)
    parquets = sorted(src.glob("*.parquet"))
    out = tmp_path / "out"
    out.mkdir()

    grand_mean, batch_means = m.phase1_compute_means(parquets, feat_cols)
    init = m.phase2_init_centroids(parquets, feat_cols, grand_mean, batch_means, k, 50, out)
    centroids = m.phase3_lloyd_streaming(parquets, feat_cols, grand_mean, batch_means, init, 3, out)
    obs = m.phase4_assign_export(parquets, feat_cols, grand_mean, batch_means, centroids, out)
    props = m.phase5_proportions(obs, out)
    m.phase6_qc(obs, props, out, "mean_center")

    assert centroids.shape == (k, d)
    assert np.allclose(np.linalg.norm(centroids, axis=1), 1, atol=1e-5)
    assert np.abs(centroids @ directions.T).max(axis=0).min() > 0.9
    assert len(obs) == 800 and set(obs.global_cluster) <= set(range(k))
    assert props.shape[0] == 4 and np.allclose(props.sum(axis=1), 1)
    assert len(list((out / "projected_slides").glob("*.parquet"))) == 4
    assert (out / "comparison_metrics.json").exists() and (out / "batch_effect_scanner_batch.csv").exists()
