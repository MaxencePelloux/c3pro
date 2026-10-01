import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def script(prefix):
    return next(SCRIPTS.glob(f"{prefix}_*.py"))


def run(prefix, *args, cwd=None):
    r = subprocess.run([sys.executable, str(script(prefix)), *map(str, args)],
                       capture_output=True, text=True, cwd=cwd)
    assert r.returncode == 0, f"{script(prefix).name} failed\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    return r


def load(prefix):
    spec = importlib.util.spec_from_file_location(f"script_{prefix}", script(prefix))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def slide_name(year, i, block="A01-1"):
    # fictitious names in the expected format: two-digit year, two letters, six digits, block
    return f"{year:02d}ZZ{i:06d}-{block}"


def write_projected(directory, names, n_clusters, rng, enriched=None, n_cells=150):
    """Per-slide parquets as written by steps 04/05: x, y, global_cluster."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        p = np.ones(n_clusters)
        if enriched is not None and enriched(name):
            p[0] = 6
        labels = rng.choice(n_clusters, size=n_cells, p=p / p.sum())
        pd.DataFrame({"x": rng.uniform(0, 2000, n_cells), "y": rng.uniform(0, 2000, n_cells),
                      "global_cluster": labels}).to_parquet(directory / f"{name}.parquet", index=False)
