#!/usr/bin/env python3
"""Streaming spherical K-means over per-cell UNI2-h embeddings, with per-scanner-era mean-centering.
Input: per-slide parquets (slide, x, y, f0..f1535), optionally restricted with --slide-list.
Output: centroids, per-cell labels, per-slide labelled parquets and per-slide cluster proportions."""

import os
os.environ['PYTHONHASHSEED'] = '42'

if os.environ.get('C3PRO_RMM_MANAGED', '0') == '1':
    try:
        import rmm
        rmm.mr.set_current_device_resource(rmm.mr.ManagedMemoryResource())
        print("RMM managed memory enabled", flush=True)
    except Exception as _e:
        print(f"RMM not available: {_e}", flush=True)

import argparse
import gc
import json
import sys
import time
import warnings
from pathlib import Path as _Path
# shared PATHID extractor: slide names use either a dash or a space after the PATHID
sys.path.insert(0, str(_Path(__file__).resolve().parent))
try:
    from utils.pathid import extract_pathid as _extract_pathid
except Exception:
    import re as _re
    _PATHID_RE = _re.compile(r"^(\d{2}[A-Z]{2}\d{6})")
    def _extract_pathid(s):
        m = _PATHID_RE.match(str(s).strip())
        if not m:
            raise ValueError(f"Cannot extract PATHID from {s!r}")
        return m.group(1)
warnings.filterwarnings("ignore")

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import psutil
from pathlib import Path
from tqdm import tqdm
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests


SEED = 42
PRE_SCANNER_YEARS  = {17, 18, 19, 20, 21}
POST_SCANNER_YEARS = {22, 23, 24}


def _mem() -> str:
    vm = psutil.virtual_memory()
    return f"{vm.percent:.0f}% ({vm.used/1e9:.1f}/{vm.total/1e9:.1f} GB)"

def _elapsed(t0: float) -> str:
    return f"{(time.perf_counter()-t0)/60:.1f} min"

def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def _year_prefix(stem: str) -> int:
    try: return int(stem[:2])
    except ValueError: return -1

def _batch(year: int) -> str:
    if year in PRE_SCANNER_YEARS:  return 'pre_scanner'
    if year in POST_SCANNER_YEARS: return 'post_scanner'
    return 'unknown'

def _slide_name(p: Path) -> str:
    return p.stem.strip().replace(' ', '')

def _load_slide_feats(p: Path, feat_cols: list[str]) -> tuple[str, NDArray, NDArray, NDArray] | None:
    """Load one slide as (slide_name, float32 features, xs, ys); None if unreadable or empty."""
    try:
        df = pd.read_parquet(p, columns=['slide', 'x', 'y'] + feat_cols)
    except Exception as e:
        _log(f"WARNING: skipping corrupted file {p.name}: {e}")
        return None
    name = str(df['slide'].iloc[0]).strip().replace(' ', '')
    feats = df[feat_cols].values.astype(np.float32)
    nan_mask = np.isnan(feats).any(axis=1)
    if nan_mask.any():
        n_bad = int(nan_mask.sum())
        _log(f"WARNING: {p.name}: dropping {n_bad}/{len(feats)} cells with NaN features")
        good = ~nan_mask
        feats = feats[good]
        df = df.loc[good]
    if len(feats) == 0:
        _log(f"WARNING: {p.name}: no valid cells — skipping")
        return None
    return name, feats, df['x'].values, df['y'].values


def phase0_setup(
    source_dir: Path,
    n_slides_test: int | None,
    slide_list: Path | None = None,
) -> tuple[list[Path], list[str]]:
    print(f"\nPHASE 0 — SETUP", flush=True)

    all_parquets_raw = sorted(source_dir.glob("*.parquet"))
    if not all_parquets_raw:
        raise FileNotFoundError(f"No .parquet files in {source_dir}")
    _log(f"Found {len(all_parquets_raw)} parquet files in source dir")

    # --slide-list: a directory of split CSVs (union of train/val/test), a CSV with a
    # slide_name column, or a text file with one slide name or PATHID per line
    pathid_expand = True
    if slide_list is not None:
        slide_list = Path(slide_list)
        if slide_list.is_dir():
            csvs = sorted(slide_list.glob("splits_R*_F*.csv"))
            if not csvs:
                raise FileNotFoundError(f"No splits_R*_F*.csv in {slide_list}")
            keep_names = set()
            for csv_path in csvs:
                sp = pd.read_csv(csv_path)
                for col in ("train", "val", "test"):
                    if col in sp.columns:
                        keep_names.update(
                            str(x).strip() for x in sp[col].dropna().tolist()
                            if str(x).strip()
                        )
            _log(f"Canonical cohort from {len(csvs)} split CSVs: "
                 f"{len(keep_names)} unique PATHIDs/slide_names")
        elif slide_list.suffix.lower() == ".csv":
            sp = pd.read_csv(slide_list)
            if "slide_name" not in sp.columns:
                raise ValueError(
                    f"{slide_list} has no 'slide_name' column "
                    f"(found {list(sp.columns)})")
            keep_names = set(
                str(x).strip() for x in sp["slide_name"].dropna().tolist()
                if str(x).strip()
            )
            _log(f"Canonical cohort from {slide_list.name}: "
                 f"{len(keep_names)} unique slide_names")
            # exact slide-name matching: a PATHID fallback would pull in the other
            # slides of patients who have several
            pathid_expand = False
        else:
            keep_names = set(l.strip() for l in slide_list.read_text().splitlines() if l.strip())
        keep_normalized = set()
        for n in keep_names:
            keep_normalized.add(n)
            keep_normalized.add(n.replace(' ', ''))
            keep_normalized.add(n.replace('-', ' '))  # dash/space separator variants
            if pathid_expand:
                # entries with block/section suffixes must still match parquets named by bare PATHID
                try:
                    keep_normalized.add(_extract_pathid(n))
                except ValueError:
                    pass
        _log(f"Slide list: {len(keep_names)} slides to keep "
             f"(pathid_expand={pathid_expand})")

        def _matches(p):
            stem = p.stem.strip()
            candidates = [
                _slide_name(p),
                _slide_name(p).replace('_features', '').replace('_projected', ''),
                stem,
                stem.replace(' _features', '').replace(' _projected', ''),
                stem.replace(' ', '').replace('_features', '').replace('_projected', ''),
            ]
            if pathid_expand:
                pathid_with_suffix = stem.split(' ')[0].split('_features')[0].split('_projected')[0]
                candidates.append(pathid_with_suffix)
                try:
                    candidates.append(_extract_pathid(stem))
                except ValueError:
                    pass
            return any(c in keep_names or c in keep_normalized for c in candidates)

        all_parquets_raw = [p for p in all_parquets_raw if _matches(p)]
        _log(f"After filter: {len(all_parquets_raw)} parquets matched")

    _log("Validating parquet files...")
    import pyarrow.parquet as pq
    all_parquets, bad = [], []
    for p in all_parquets_raw:
        try:
            pq.read_schema(p)
            all_parquets.append(p)
        except Exception:
            bad.append(p.name)
    if bad:
        _log(f"WARNING: skipping {len(bad)} corrupted/incomplete parquets: {bad}")
    _log(f"Valid parquets: {len(all_parquets)}")

    # smoke test: keep pre- and post-scanner slides balanced
    if n_slides_test and n_slides_test < len(all_parquets):
        pre  = [p for p in all_parquets if _year_prefix(_slide_name(p)) in PRE_SCANNER_YEARS]
        post = [p for p in all_parquets if _year_prefix(_slide_name(p)) in POST_SCANNER_YEARS]
        rng  = np.random.default_rng(SEED)
        n_pre  = min(n_slides_test // 2, len(pre))
        n_post = min(n_slides_test - n_pre, len(post))
        all_parquets = list(rng.choice(pre,  n_pre,  replace=False)) + \
                       list(rng.choice(post, n_post, replace=False))
        _log(f"[smoke-test] {len(all_parquets)} slides ({n_pre} pre + {n_post} post)")

    sample = pd.read_parquet(all_parquets[0], columns=None)
    feat_cols = sorted([c for c in sample.columns if c.startswith('f') and c[1:].isdigit()],
                       key=lambda c: int(c[1:]))
    assert len(feat_cols) == 1536, f"Expected 1536 feature dims, got {len(feat_cols)}"
    del sample

    batch_dist = {}
    for p in all_parquets:
        b = _batch(_year_prefix(_slide_name(p)))
        batch_dist[b] = batch_dist.get(b, 0) + 1
    _log(f"Batch distribution: {batch_dist}")
    return all_parquets, feat_cols


def phase1_compute_means(
    all_parquets: list[Path],
    feat_cols: list[str],
) -> tuple[NDArray, dict[str, NDArray]]:
    """Exact grand mean and per-scanner-era means, accumulated one slide at a time."""
    print(f"\nPHASE 1 — COMPUTE BATCH MEANS", flush=True)
    t0 = time.perf_counter()
    D  = len(feat_cols)

    total_sum    = np.zeros(D, dtype=np.float64)
    total_count  = 0
    batch_sums   = {'pre_scanner': np.zeros(D, np.float64), 'post_scanner': np.zeros(D, np.float64)}
    batch_counts = {'pre_scanner': 0, 'post_scanner': 0}

    for p in tqdm(all_parquets, desc="Computing means"):
        result = _load_slide_feats(p, feat_cols)
        if result is None:
            continue
        _, feats, _, _ = result
        b = _batch(_year_prefix(_slide_name(p)))
        total_sum   += feats.sum(axis=0).astype(np.float64)
        total_count += len(feats)
        if b in batch_sums:
            batch_sums[b]   += feats.sum(axis=0).astype(np.float64)
            batch_counts[b] += len(feats)

    grand_mean  = (total_sum / total_count).astype(np.float32)
    batch_means = {
        b: (batch_sums[b] / max(batch_counts[b], 1)).astype(np.float32)
        for b in batch_sums
    }
    _log(f"Grand mean norm: {np.linalg.norm(grand_mean):.4f}")
    for b, m in batch_means.items():
        _log(f"  {b} mean norm: {np.linalg.norm(m):.4f}  (n={batch_counts[b]:,})")
    _log(f"Phase 1 done in {_elapsed(t0)}")
    return grand_mean, batch_means


def _apply_correction(feats: NDArray, batch: str,
                      grand_mean: NDArray, batch_means: dict) -> NDArray:
    """Remove the scanner-era offset by a pure translation (no distortion), then L2-normalise."""
    if batch in batch_means:
        feats = feats + (grand_mean - batch_means[batch])
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    feats /= norms
    return feats


def _apply_harmony_correction(feats: NDArray, slide_offset: int,
                              pca_mean: NDArray, pca_components: NDArray,
                              harmony_pca: NDArray, original_pca: NDArray) -> NDArray:
    """Add the Harmony shift, mapped back from PCA to the full feature space, then L2-normalise."""
    n = len(feats)
    delta_pca = harmony_pca[slide_offset:slide_offset + n] - \
                original_pca[slide_offset:slide_offset + n]
    delta_full = delta_pca @ pca_components
    feats = feats + delta_full.astype(np.float32)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    feats /= norms
    return feats


def phase1b_harmony(
    all_parquets: list[Path],
    feat_cols: list[str],
    output_dir: Path,
    n_pca: int = 50,
) -> tuple[NDArray, NDArray, NDArray, NDArray, list[int]]:
    """Streaming PCA, then Harmony on the PCA embeddings, with checkpoints in between."""
    print(f"\nPHASE 1b — HARMONY BATCH CORRECTION", flush=True)
    t0 = time.perf_counter()
    D = len(feat_cols)

    _log(f"Step 1/3: IncrementalPCA (n_components={n_pca}) ...")
    from sklearn.decomposition import IncrementalPCA
    ipca = IncrementalPCA(n_components=n_pca, batch_size=50_000)

    for p in tqdm(all_parquets, desc="PCA fit"):
        result = _load_slide_feats(p, feat_cols)
        if result is None:
            continue
        _, feats, _, _ = result
        # partial_fit needs at least n_components rows per chunk
        for start in range(0, len(feats), 50_000):
            chunk = feats[start:start + 50_000]
            if len(chunk) >= n_pca:
                ipca.partial_fit(chunk)
    _log(f"  PCA explained variance (top 5): {ipca.explained_variance_ratio_[:5]}")
    _log(f"  Total explained: {ipca.explained_variance_ratio_.sum():.4f}")

    pca_mean = ipca.mean_.astype(np.float32)
    pca_components = ipca.components_.astype(np.float32)

    _log("Step 2/3: Transform all cells to PCA space ...")
    pca_parts = []
    batch_labels = []
    slide_offsets = [0]

    for p in tqdm(all_parquets, desc="PCA transform"):
        result = _load_slide_feats(p, feat_cols)
        if result is None:
            slide_offsets.append(slide_offsets[-1])
            continue
        _, feats, _, _ = result
        X_pca = (feats - pca_mean) @ pca_components.T
        pca_parts.append(X_pca)
        b = _batch(_year_prefix(_slide_name(p)))
        batch_labels.extend([b] * len(feats))
        slide_offsets.append(slide_offsets[-1] + len(feats))

    original_pca = np.vstack(pca_parts).astype(np.float32)
    batch_labels = np.array(batch_labels)
    del pca_parts
    gc.collect()
    _log(f"  PCA embeddings: {original_pca.shape}  ({original_pca.nbytes / 1e9:.1f} GB)")
    _log(f"  Batches: {np.unique(batch_labels, return_counts=True)}")

    # checkpoint before Harmony, which can run out of memory
    np.save(output_dir / "checkpoint_pca_mean.npy", pca_mean)
    np.save(output_dir / "checkpoint_pca_components.npy", pca_components)
    np.save(output_dir / "checkpoint_original_pca.npy", original_pca)
    np.save(output_dir / "checkpoint_slide_offsets.npy", np.array(slide_offsets))
    np.save(output_dir / "checkpoint_batch_labels.npy", batch_labels)
    _log("  PCA checkpoints saved (before Harmony)")

    # Harmony is forced onto the CPU: on GPU it runs out of memory at this cell count
    _log("Step 3/3: Running Harmony (CPU mode) ...")
    import torch
    _orig_device = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    import harmonypy as hm
    ho = hm.run_harmony(original_pca, pd.DataFrame({'batch': batch_labels}), 'batch',
                        max_iter_harmony=20, verbose=True)
    torch.cuda.is_available = _orig_device
    Z = ho.Z_corr
    # Z_corr orientation depends on the harmonypy version
    if Z.shape[0] == n_pca and Z.shape[1] != n_pca:
        Z = Z.T
    harmony_pca = Z.astype(np.float32)
    assert harmony_pca.shape == original_pca.shape, \
        f"Shape mismatch: harmony {harmony_pca.shape} vs original {original_pca.shape}"
    _log(f"  Harmony done — corrected shape: {harmony_pca.shape}")

    np.save(output_dir / "checkpoint_harmony_pca.npy", harmony_pca)
    _log(f"  All checkpoints saved  |  Phase 1b done in {_elapsed(t0)}")

    return pca_mean, pca_components, harmony_pca, original_pca, slide_offsets


def _correct_feats(feats, batch, grand_mean, batch_means,
                   harmony_data=None, slide_offset=0):
    """Apply mean-centering or Harmony correction, then L2-normalise."""
    if harmony_data is not None:
        pca_mean, pca_comp, harmony_pca, original_pca, _ = harmony_data
        return _apply_harmony_correction(feats, slide_offset,
                                         pca_mean, pca_comp,
                                         harmony_pca, original_pca)
    return _apply_correction(feats, batch, grand_mean, batch_means)


def phase2_init_centroids(
    all_parquets: list[Path],
    feat_cols: list[str],
    grand_mean: NDArray,
    batch_means: dict,
    n_clusters: int,
    n_sample_per_slide: int = 100,
    output_dir: Path | None = None,
    harmony_data=None,
) -> NDArray:
    """K-means++ on a per-slide random sample of corrected cells; returns L2-normalised centroids."""
    print(f"\nPHASE 2 — K-MEANS++ INIT", flush=True)
    t0  = time.perf_counter()
    rng = np.random.default_rng(SEED)
    sample_parts = []
    cell_offset = 0

    for p in tqdm(all_parquets, desc="Sampling"):
        b = _batch(_year_prefix(_slide_name(p)))
        result = _load_slide_feats(p, feat_cols)
        if result is None:
            continue
        _, feats, _, _ = result
        feats = _correct_feats(feats, b, grand_mean, batch_means,
                               harmony_data, cell_offset)
        cell_offset += len(feats)
        idx = rng.choice(len(feats), size=min(n_sample_per_slide, len(feats)), replace=False)
        sample_parts.append(feats[idx])

    X_sample = np.vstack(sample_parts).astype(np.float32)
    _log(f"Sample: {len(X_sample):,} cells  |  RAM: {_mem()}")

    from sklearn.cluster import KMeans as skKMeans
    km = skKMeans(n_clusters=n_clusters, init='k-means++', n_init=1,
                  max_iter=5, random_state=SEED)
    km.fit(X_sample)

    centroids = km.cluster_centers_.astype(np.float32)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids /= np.maximum(norms, 1e-12)
    np.save(output_dir / "checkpoint_init_centroids.npy", centroids)
    _log(f"Centroids initialised + saved  |  Phase 2 done in {_elapsed(t0)}")
    return centroids


def phase3_lloyd_streaming(
    all_parquets: list[Path],
    feat_cols: list[str],
    grand_mean: NDArray,
    batch_means: dict,
    centroids: NDArray,
    n_epochs: int,
    output_dir: Path,
    start_epoch: int = 0,
    harmony_data=None,
) -> NDArray:
    """Exact spherical Lloyd iterations, streaming one slide at a time through the GPU."""
    remaining = n_epochs - start_epoch
    print(f"\nPHASE 3 — LLOYD'S K-MEANS STREAMING "
          f"(epochs {start_epoch+1}-{n_epochs}/{n_epochs})",
          flush=True)
    t0 = time.perf_counter()

    import cupy as cp
    K, D = centroids.shape

    centroids_gpu = cp.asarray(centroids)

    for epoch in range(start_epoch, n_epochs):
        t_ep = time.perf_counter()
        centroid_sums_gpu   = cp.zeros((K, D), dtype=cp.float32)
        centroid_counts_gpu = cp.zeros(K,      dtype=cp.float32)

        cell_offset = 0
        for p in tqdm(all_parquets, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False):
            b = _batch(_year_prefix(_slide_name(p)))
            result = _load_slide_feats(p, feat_cols)
            if result is None:
                continue
            _, feats, _, _ = result
            feats = _correct_feats(feats, b, grand_mean, batch_means,
                                   harmony_data, cell_offset)
            cell_offset += len(feats)

            feats_gpu  = cp.asarray(feats)
            sims       = feats_gpu @ centroids_gpu.T             # cosine similarity: rows are unit vectors
            labels_gpu = cp.argmax(sims, axis=1)

            # one-hot matmul instead of a scatter-add: works on every CuPy version and runs on BLAS
            one_hot = cp.eye(K, dtype=cp.float32)[labels_gpu]
            centroid_sums_gpu  += one_hot.T @ feats_gpu
            centroid_counts_gpu += cp.bincount(labels_gpu, minlength=K).astype(cp.float32)

            del feats_gpu, sims, labels_gpu, one_hot
            gc.collect()

        # sums cover the whole pass, so this is exact batch k-means, not mini-batch
        safe_counts = cp.maximum(centroid_counts_gpu[:, None], 1.0)
        centroids_gpu = centroid_sums_gpu / safe_counts
        # back onto the unit sphere: this is what makes it spherical k-means
        norms = cp.linalg.norm(centroids_gpu, axis=1, keepdims=True)
        centroids_gpu /= cp.maximum(norms, 1e-12)

        n_empty = int((centroid_counts_gpu == 0).sum())
        _log(f"  Epoch {epoch+1}/{n_epochs}  |  {_elapsed(t_ep)}  |  "
             f"empty clusters: {n_empty}  |  RAM: {_mem()}")

        centroids_np = cp.asnumpy(centroids_gpu)
        np.save(output_dir / "kmeans_centroids.npy", centroids_np)
        np.save(output_dir / "checkpoint_epoch.npy", np.array(epoch + 1))
        del centroid_sums_gpu, centroid_counts_gpu, safe_counts, norms

    centroids = cp.asnumpy(centroids_gpu)
    np.save(output_dir / "kmeans_centroids.npy", centroids)
    _log(f"Centroids saved  |  Phase 3 done in {_elapsed(t0)}")
    return centroids


def phase4_assign_export(
    all_parquets: list[Path],
    feat_cols: list[str],
    grand_mean: NDArray,
    batch_means: dict,
    centroids: NDArray,
    output_dir: Path,
    harmony_data=None,
) -> pd.DataFrame:
    """Assign every cell to its nearest centroid and write one (x, y, global_cluster) parquet per slide."""
    print(f"\nPHASE 4 — ASSIGN ALL CELLS + EXPORT", flush=True)
    t0 = time.perf_counter()

    import cupy as cp
    projected_dir = output_dir / "projected_slides"
    projected_dir.mkdir(exist_ok=True)
    centroids_gpu = cp.asarray(centroids)
    already_done  = {f.stem for f in projected_dir.glob("*.parquet")}

    obs_parts = []
    labels_all = []

    cell_offset = 0
    for p in tqdm(all_parquets, desc="Assigning"):
        slide = _slide_name(p)
        b     = _batch(_year_prefix(p.stem.split()[0] if ' ' in p.stem else p.stem))
        year  = _year_prefix(p.stem.split()[0] if ' ' in p.stem else p.stem)

        result = _load_slide_feats(p, feat_cols)
        if result is None:
            continue
        _, feats, xs, ys = result
        feats = _correct_feats(feats, b, grand_mean, batch_means,
                               harmony_data, cell_offset)
        cell_offset += len(feats)
        feats_gpu = cp.asarray(feats)
        sims      = feats_gpu @ centroids_gpu.T
        labs_gpu  = cp.argmax(sims, axis=1).astype(cp.int32)
        labs      = cp.asnumpy(labs_gpu)

        del feats_gpu, sims, labs_gpu
        gc.collect()

        if slide not in already_done:
            out = pd.DataFrame({'x': xs, 'y': ys, 'global_cluster': labs})
            out.to_parquet(projected_dir / f"{slide}.parquet", index=False)

        obs_parts.append(pd.DataFrame({
            'slide_name':    slide,
            'x':             xs,
            'y':             ys,
            'year':          year,
            'batch':         b,
            'global_cluster': labs,
        }))
        labels_all.append(labs)

    obs_df = pd.concat(obs_parts, ignore_index=True)
    obs_df.to_parquet(output_dir / "obs_meta.parquet", index=False)

    labels = np.concatenate(labels_all)
    np.save(output_dir / "kmeans_labels.npy", labels)

    n_clusters = int(labels.max()) + 1
    _log(f"Exported {len(all_parquets)} slides  |  clusters used: {n_clusters}")
    _log(f"Phase 4 done in {_elapsed(t0)}")
    return obs_df


def phase5_proportions(obs_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    print(f"\nPHASE 5 — SLIDE PROPORTIONS", flush=True)
    t0 = time.perf_counter()

    counts = (obs_df.groupby(['slide_name', 'global_cluster'])
                    .size().unstack(fill_value=0))
    props  = counts.div(counts.sum(axis=1), axis=0)
    props.columns = [f"Cluster_{c}" for c in props.columns]
    out_path = output_dir / "slide_proportions_kmeans.csv"
    props.to_csv(out_path)
    _log(f"Proportions saved → {out_path}  {props.shape}")
    _log(f"Phase 5 done in {_elapsed(t0)}")
    return props


def phase6_qc(
    obs_df: pd.DataFrame,
    props: pd.DataFrame,
    output_dir: Path,
    batch_correction: str,
) -> None:
    print(f"\nPHASE 6 — QC", flush=True)
    t0 = time.perf_counter()
    N  = len(obs_df)

    for group_col, label in [('year', 'year'), ('batch', 'scanner_batch')]:
        if group_col not in obs_df.columns:
            continue
        group_map = obs_df.groupby('slide_name')[group_col].first().to_dict()
        slides    = props.index
        groups    = np.array([str(group_map.get(s, 'unknown')) for s in slides])
        unique_g  = [g for g in sorted(set(groups)) if g != 'unknown']

        if len(unique_g) < 2:
            _log(f"  Skipping KW test for {label}: only {len(unique_g)} group(s)")
            continue

        rows = []
        for col in props.columns:
            grp_vals = [props.loc[groups == g, col].values for g in unique_g]
            if all(len(v) >= 3 for v in grp_vals):
                stat, pval = kruskal(*grp_vals)
            else:
                stat, pval = np.nan, np.nan
            rows.append({'cluster': col, 'kruskal_stat': stat, 'p_value': pval})

        kw_df = pd.DataFrame(rows)
        valid = kw_df['p_value'].notna()
        if valid.sum() > 0:
            _, fdr, _, _ = multipletests(kw_df.loc[valid, 'p_value'], method='fdr_bh')
            kw_df.loc[valid, 'fdr'] = fdr

        n_sig = int((kw_df.get('fdr', pd.Series(dtype=float)) < 0.05).sum())
        _log(f"  Batch-sig clusters ({label}): {n_sig}/{len(kw_df)}")
        kw_df.to_csv(output_dir / f"batch_effect_{label}.csv", index=False)

    labels = obs_df['global_cluster'].values
    metrics = {
        'n_clusters':     int(labels.max()) + 1,
        'batch_correction': batch_correction,
        'n_slides':       int(obs_df['slide_name'].nunique()),
        'n_cells':        int(N),
    }
    with open(output_dir / "comparison_metrics.json", 'w') as fh:
        json.dump(metrics, fh, indent=2)

    _log(f"Phase 6 done in {_elapsed(t0)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="C3PRO Step 4 — streaming spherical K-Means over per-slide cell features"
    )
    parser.add_argument('--source-dir',       type=Path, required=True)
    parser.add_argument('--output-dir',       type=Path, required=True)
    parser.add_argument('--n-clusters',       type=int,  default=50)
    parser.add_argument('--n-epochs',         type=int,  default=15,
                        help='Lloyd iterations (default: 15)')
    parser.add_argument('--n-sample-per-slide', type=int, default=100,
                        help='Cells per slide for k-means++ init (default: 100)')
    parser.add_argument('--batch-correction', choices=['mean_center', 'harmony', 'none'],
                        default='mean_center')
    parser.add_argument('--harmony-pca-dims', type=int, default=50,
                        help='PCA dims for Harmony (default: 50)')
    parser.add_argument('--skip-phase6',      action='store_true',
                        help='Skip QC (batch-effect tests)')
    parser.add_argument('--n-slides-test',    type=int, default=None,
                        help='Smoke test: N slides balanced pre/post')
    parser.add_argument('--slide-list',       type=Path, default=None,
                        help='Text file with slide names to include (one per line)')
    parser.add_argument('--resume',           action='store_true',
                        help='Auto-resume from last checkpoint')
    args = parser.parse_args()

    t_total = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output_dir

    print(f"\nC3PRO — STREAMING SPHERICAL K-MEANS")
    print(f"  source     : {args.source_dir}")
    print(f"  output     : {args.output_dir}")
    print(f"  k          : {args.n_clusters}")
    print(f"  epochs     : {args.n_epochs}")
    print(f"  correction : {args.batch_correction}")
    if args.slide_list:
        print(f"  slide_list : {args.slide_list}")
    if args.n_slides_test:
        print(f"  [SMOKE TEST] {args.n_slides_test} slides")
    if args.resume:
        print(f"  [RESUME MODE]")
    print(flush=True)

    all_parquets, feat_cols = phase0_setup(args.source_dir, args.n_slides_test,
                                             slide_list=args.slide_list)

    has_means     = (ckpt_dir / "batch_means.npz").exists()
    has_init_c    = (ckpt_dir / "checkpoint_init_centroids.npy").exists()
    has_centroids = (ckpt_dir / "kmeans_centroids.npy").exists()
    has_epoch_ck  = (ckpt_dir / "checkpoint_epoch.npy").exists()

    if args.resume and has_means:
        _log("RESUME: loading saved batch means from batch_means.npz")
        means_data  = np.load(ckpt_dir / "batch_means.npz")
        grand_mean  = means_data['grand_mean']
        batch_means = {k: means_data[k] for k in means_data if k != 'grand_mean'}
    else:
        grand_mean, batch_means = phase1_compute_means(all_parquets, feat_cols)
        np.savez(ckpt_dir / "batch_means.npz",
                 grand_mean=grand_mean, **batch_means)

    harmony_data = None  # (pca_mean, pca_components, harmony_pca, original_pca, slide_offsets)

    if args.batch_correction == 'harmony':
        has_harmony = (ckpt_dir / "checkpoint_harmony_pca.npy").exists()
        has_pca_ckpt = (ckpt_dir / "checkpoint_original_pca.npy").exists()
        if args.resume and has_harmony:
            _log("RESUME: loading full Harmony checkpoints (PCA + Harmony)")
            h_pca_mean   = np.load(ckpt_dir / "checkpoint_pca_mean.npy")
            h_pca_comp   = np.load(ckpt_dir / "checkpoint_pca_components.npy")
            h_harmony    = np.load(ckpt_dir / "checkpoint_harmony_pca.npy")
            h_original   = np.load(ckpt_dir / "checkpoint_original_pca.npy")
            h_offsets    = np.load(ckpt_dir / "checkpoint_slide_offsets.npy").tolist()
            harmony_data = (h_pca_mean, h_pca_comp, h_harmony, h_original, h_offsets)
        elif args.resume and has_pca_ckpt:
            _log("RESUME: loading PCA checkpoints — running Harmony only (Step 3)")
            pca_mean     = np.load(ckpt_dir / "checkpoint_pca_mean.npy")
            pca_comp     = np.load(ckpt_dir / "checkpoint_pca_components.npy")
            original_pca = np.load(ckpt_dir / "checkpoint_original_pca.npy")
            slide_offsets= np.load(ckpt_dir / "checkpoint_slide_offsets.npy").tolist()
            batch_labels = np.load(ckpt_dir / "checkpoint_batch_labels.npy")
            _log(f"  PCA shape: {original_pca.shape}  ({original_pca.nbytes/1e9:.1f} GB)")
            import torch
            _orig = torch.cuda.is_available
            torch.cuda.is_available = lambda: False
            import harmonypy as hm
            _log("  Running Harmony (CPU) ...")
            ho = hm.run_harmony(original_pca, pd.DataFrame({'batch': batch_labels}), 'batch',
                                max_iter_harmony=20, verbose=True)
            torch.cuda.is_available = _orig
            Z = ho.Z_corr
            if Z.shape[0] == args.harmony_pca_dims and Z.shape[1] != args.harmony_pca_dims:
                Z = Z.T
            harmony_pca = Z.astype(np.float32)
            np.save(ckpt_dir / "checkpoint_harmony_pca.npy", harmony_pca)
            _log(f"  Harmony done — {harmony_pca.shape}")
            harmony_data = (pca_mean, pca_comp, harmony_pca, original_pca, slide_offsets)
        else:
            harmony_data = phase1b_harmony(
                all_parquets, feat_cols, args.output_dir,
                n_pca=args.harmony_pca_dims,
            )
        # mean-centering is disabled: the Harmony shift is applied through harmony_data
        grand_mean  = np.zeros(len(feat_cols), dtype=np.float32)
        batch_means = {}

    elif args.batch_correction == 'none':
        grand_mean  = np.zeros_like(grand_mean)
        batch_means = {k: np.zeros_like(v) for k, v in batch_means.items()}

    if args.resume and has_init_c:
        _log("RESUME: loading saved init centroids from checkpoint_init_centroids.npy")
        centroids = np.load(ckpt_dir / "checkpoint_init_centroids.npy")
    else:
        centroids = phase2_init_centroids(
            all_parquets, feat_cols, grand_mean, batch_means,
            args.n_clusters, args.n_sample_per_slide, ckpt_dir,
            harmony_data=harmony_data,
        )

    start_epoch = 0
    if args.resume and has_centroids and has_epoch_ck:
        done_epochs = int(np.load(ckpt_dir / "checkpoint_epoch.npy"))
        if done_epochs >= args.n_epochs:
            _log(f"RESUME: all {args.n_epochs} epochs already done — loading centroids")
            centroids = np.load(ckpt_dir / "kmeans_centroids.npy")
            start_epoch = args.n_epochs
        else:
            _log(f"RESUME: resuming Lloyd's from epoch {done_epochs+1}/{args.n_epochs}")
            centroids = np.load(ckpt_dir / "kmeans_centroids.npy")
            start_epoch = done_epochs

    if start_epoch < args.n_epochs:
        centroids = phase3_lloyd_streaming(
            all_parquets, feat_cols, grand_mean, batch_means,
            centroids, args.n_epochs, args.output_dir,
            start_epoch=start_epoch,
            harmony_data=harmony_data,
        )

    obs_meta_path = args.output_dir / "obs_meta.parquet"
    props_path = args.output_dir / "slide_proportions_kmeans.csv"
    if args.resume and obs_meta_path.exists() and props_path.exists():
        _log(f"RESUME: loading saved Phase 4-5 outputs (obs_meta + proportions)")
        obs_df = pd.read_parquet(obs_meta_path)
        props = pd.read_csv(props_path, index_col='slide_name')
    else:
        obs_df = phase4_assign_export(
            all_parquets, feat_cols, grand_mean, batch_means,
            centroids, args.output_dir,
            harmony_data=harmony_data,
        )
        props = phase5_proportions(obs_df, args.output_dir)

    if not args.skip_phase6:
        phase6_qc(obs_df, props, args.output_dir,
                  args.batch_correction)

    elapsed = (time.perf_counter() - t_total) / 60
    n_cl    = int(np.load(args.output_dir / "kmeans_labels.npy").max()) + 1
    _log(f"DONE  —  {elapsed:.1f} min total")
    _log(f"  n_clusters : {n_cl}")
    _log(f"  outputs    : {args.output_dir}")
    _log(f"  → projected_slides/  (scripts 10, 09, 10, GNN)")
    _log(f"  → slide_proportions_kmeans.csv")
    _log(f"  → kmeans_centroids.npy  (for projecting new slides)")


if __name__ == '__main__':
    main()
