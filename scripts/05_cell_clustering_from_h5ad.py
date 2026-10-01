#!/usr/bin/env python3
"""Spherical K-Means (cuML, GPU) on per-cell UNI2-h embeddings held in memory; cluster IDs do not match step 4's.

Input: --adata-path, an .h5ad with the embeddings and obs (slide_name, x, y, year).
Output in --output-dir: centroids, labels, per-slide parquets (x, y, global_cluster), slide proportions and QC.
"""

import os
os.environ['PYTHONHASHSEED'] = '42'

# optional RMM managed memory lets GPU allocations spill into host RAM; it must be set up
# before any CUDA/CuPy context is created
if os.environ.get('C3PRO_RMM_MANAGED', '0') == '1':
    try:
        import rmm
        rmm.mr.set_current_device_resource(rmm.mr.ManagedMemoryResource())
        print("RMM managed memory enabled (GH200 unified memory mode)", flush=True)
    except Exception as _e:
        print(f"RMM not available: {_e} — using default allocator", flush=True)

import argparse
import gc
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import h5py
import psutil
from pathlib import Path
from tqdm import tqdm
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
DPI = 150


def _mem_pct() -> str:
    vm = psutil.virtual_memory()
    return f"{vm.percent:.1f}%  ({vm.used / 1e9:.1f}/{vm.total / 1e9:.1f} GB)"


def _elapsed(t0: float) -> str:
    return f"{(time.perf_counter() - t0) / 60:.1f} min"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def phase0_load(
    adata_path: Path,
    output_dir: Path,
    n_cells_test: int | None = None,
) -> tuple[NDArray[np.float32], pd.DataFrame]:
    """Read embeddings (N, D float32) and obs from the .h5ad in chunks, optionally subsampled."""
    print(f"\n{'='*70}\nPHASE 0 — LOAD EMBEDDINGS\n{'='*70}", flush=True)
    t0 = time.perf_counter()
    _log(f"RAM before load: {_mem_pct()}")

    with h5py.File(adata_path, 'r') as f:
        x_node = f['X']

        if isinstance(x_node, h5py.Dataset):
            N_total, D = x_node.shape
            is_sparse = False
            _log(f"X is dense: {N_total:,} × {D}")
        else:
            # Sparse CSR group: /X/data, /X/indices, /X/indptr
            N_total = f['X/indptr'].shape[0] - 1
            D = int(f['X'].attrs.get('shape', [0, 0])[1])
            is_sparse = True
            _log(f"X is sparse CSR: {N_total:,} × {D}")

        if n_cells_test is not None and n_cells_test < N_total:
            rng = np.random.default_rng(SEED)
            idx = np.sort(rng.choice(N_total, size=n_cells_test, replace=False))
            _log(f"[smoke-test] Subsampling {n_cells_test:,} / {N_total:,} cells")
            # saved so a --skip-phase0 rerun reloads the same cells for phase 4
            np.save(output_dir / "cell_indices.npy", idx)
        else:
            idx = None  # use all

        N = n_cells_test if (idx is not None) else N_total

        _log(f"Allocating {N:,} × {D} float32  ({N * D * 4 / 1e9:.1f} GB) ...")
        X: NDArray[np.float32] = np.empty((N, D), dtype=np.float32)
        _log(f"RAM after allocation: {_mem_pct()}")

        CHUNK = 200_000
        if not is_sparse:
            if idx is None:
                _log("Reading dense X in chunks ...")
                for start in tqdm(range(0, N_total, CHUNK), desc="Loading"):
                    end = min(start + CHUNK, N_total)
                    out_start = start
                    X[out_start:out_start + (end - start)] = x_node[start:end]
            else:
                _log("Reading dense X (indexed, chunked by selection) ...")
                out_pos = 0
                for i in tqdm(range(0, len(idx), CHUNK), desc="Loading"):
                    batch_idx = idx[i: i + CHUNK]
                    rows = x_node[batch_idx.tolist()]
                    rows = rows.astype(np.float32, copy=False)
                    X[out_pos: out_pos + len(batch_idx)] = rows
                    out_pos += len(batch_idx)
        else:
            _log("Sparse X detected — loading via scanpy ...")
            import scanpy as sc
            adata_tmp = sc.read(adata_path)
            if idx is not None:
                adata_tmp = adata_tmp[idx]
            raw = adata_tmp.X
            if hasattr(raw, 'toarray'):
                X[:] = raw.toarray().astype(np.float32)
            else:
                X[:] = np.asarray(raw, dtype=np.float32)
            del adata_tmp
            gc.collect()

        _log("Reading obs metadata ...")
        obs_cols = {}
        obs_group = f['obs']
        for col in ['slide_name', 'x', 'y', 'year']:
            if col not in obs_group:
                continue
            node = obs_group[col]
            if isinstance(node, h5py.Dataset):
                raw_vals = node[:]
            else:
                # categorical columns are stored as categories + codes
                cats = node['categories'][:]
                codes = node['codes'][:]
                raw_vals = cats[codes]

            # h5py returns strings as bytes
            if raw_vals.dtype.kind in ('S', 'O'):
                raw_vals = np.array(
                    [v.decode() if isinstance(v, bytes) else str(v) for v in raw_vals],
                    dtype=object
                )
            if idx is not None:
                raw_vals = raw_vals[idx]
            obs_cols[col] = raw_vals

        obs_df = pd.DataFrame(obs_cols).reset_index(drop=True)

    assert X.flags['C_CONTIGUOUS'], "X must be C-contiguous after load"
    assert X.dtype == np.float32, "cuML K-Means requires float32"

    _log(f"RAM after full load: {_mem_pct()}")
    _log(f"Phase 0 done in {_elapsed(t0)}")
    return X, obs_df


def phase1_kmeans(
    X: NDArray[np.float32],
    output_dir: Path,
    n_clusters: int,
    n_iter: int = 20,
    n_init: int = 1,
) -> NDArray[np.int64]:
    """L2-normalise X in place, then run cuML K-Means; on unit vectors this is cosine (spherical) K-Means."""
    print(f"\n{'='*70}\nPHASE 1 — SPHERICAL K-MEANS (k={n_clusters})\n{'='*70}", flush=True)
    t0 = time.perf_counter()
    N, D = X.shape

    assert X.flags['C_CONTIGUOUS'], "X must be C-contiguous"
    assert X.dtype == np.float32, "cuML requires float32"

    import cupy as cp
    from cuml.cluster import KMeans

    _log(f"Moving {N * D * 4 / 1e9:.1f} GB to GPU (unified memory) ...")
    t_gpu = time.perf_counter()
    X_gpu = cp.asarray(X)
    _log(f"  Transfer done in {time.perf_counter() - t_gpu:.1f}s")

    _log("Normalising L2 in-place (CuPy) ...")
    t_norm = time.perf_counter()
    norms = cp.linalg.norm(X_gpu, axis=1, keepdims=True)
    norms = cp.maximum(norms, 1e-12)   # avoid div-by-zero
    X_gpu /= norms
    del norms
    gc.collect()
    _log(f"  Normalisation done in {time.perf_counter() - t_norm:.1f}s")

    # write the normalised vectors back to X, which phase 4 reuses for the UMAP
    X[:] = cp.asnumpy(X_gpu)

    _log(f"Training cuML KMeans (k={n_clusters}, max_iter={n_iter}, n_init={n_init}) [GPU] ...")
    t_km = time.perf_counter()

    km = KMeans(
        n_clusters=n_clusters,
        max_iter=n_iter,
        n_init=n_init,
        random_state=SEED,
        init='k-means++',
        tol=1e-4,
        verbose=1,
    )
    km.fit(X_gpu)

    t_km_done = time.perf_counter() - t_km
    _log(f"  K-Means training done in {t_km_done:.1f}s  ({t_km_done/60:.1f} min)")

    labels: NDArray[np.int64] = cp.asnumpy(km.labels_).astype(np.int64)
    n_actual = int(np.unique(labels).size)
    _log(f"  Clusters used: {n_actual}/{n_clusters}")

    # centroids and labels are saved so later phases can be rerun with --skip-phase0
    centroids_path = output_dir / "kmeans_centroids.npy"
    np.save(centroids_path, cp.asnumpy(km.cluster_centers_))
    _log(f"  Centroids saved → {centroids_path}")

    labels_path = output_dir / "kmeans_labels.npy"
    np.save(labels_path, labels)
    _log(f"  Labels saved → {labels_path}")

    del X_gpu, km
    gc.collect()

    _log(f"Phase 1 done in {_elapsed(t0)}")
    return labels


def phase2_export_slides(
    obs_df: pd.DataFrame,
    labels: NDArray[np.int64],
    output_dir: Path,
) -> None:
    print(f"\n{'='*70}\nPHASE 2 — EXPORT PER-SLIDE PARQUETS\n{'='*70}", flush=True)
    t0 = time.perf_counter()

    projected_dir = output_dir / "projected_slides"
    projected_dir.mkdir(exist_ok=True)

    df = obs_df[['slide_name', 'x', 'y']].copy()
    df['global_cluster'] = labels.astype(np.int32)

    slides = sorted(df['slide_name'].unique())
    already_done = {f.stem for f in projected_dir.glob("*.parquet")}
    n_clusters = int(df['global_cluster'].nunique())

    _log(f"Clusters: {n_clusters}  |  Slides: {len(slides)}")

    for slide in tqdm(slides, desc="Exporting slides"):
        if slide in already_done:
            continue
        out = df.loc[df['slide_name'] == slide, ['x', 'y', 'global_cluster']].copy()
        out.reset_index(drop=True, inplace=True)
        out.to_parquet(projected_dir / f"{slide}.parquet", index=False)

    _log(f"Done — {len(slides)} slides exported to {projected_dir}")
    _log(f"n_clusters={n_clusters}  (use --in-channels {n_clusters} for scripts 13/15)")
    _log(f"Phase 2 done in {_elapsed(t0)}")


def phase3_proportions(
    obs_df: pd.DataFrame,
    labels: NDArray[np.int64],
    output_dir: Path,
) -> pd.DataFrame:
    print(f"\n{'='*70}\nPHASE 3 — SLIDE PROPORTIONS\n{'='*70}", flush=True)
    t0 = time.perf_counter()

    df = obs_df[['slide_name']].copy()
    df['global_cluster'] = labels

    counts = (
        df.groupby(['slide_name', 'global_cluster'])
        .size()
        .unstack(fill_value=0)
    )
    props = counts.div(counts.sum(axis=1), axis=0)
    props.columns = [f"Cluster_{c}" for c in props.columns]

    out_path = output_dir / "slide_proportions_kmeans.csv"
    props.to_csv(out_path)
    _log(f"Proportions saved → {out_path}  {props.shape}")
    _log(f"Phase 3 done in {_elapsed(t0)}")
    return props


def phase4_qc(
    X: NDArray[np.float32],
    obs_df: pd.DataFrame,
    labels: NDArray[np.int64],
    props: pd.DataFrame,
    output_dir: Path,
) -> None:
    print(f"\n{'='*70}\nPHASE 4 — QC FIGURES\n{'='*70}", flush=True)
    t0 = time.perf_counter()
    N = len(X)

    # batch effect: Kruskal-Wallis test of each cluster's slide proportions across years
    if 'year' in obs_df.columns:
        _log("Computing batch effect (Kruskal-Wallis) ...")
        year_map = obs_df.groupby('slide_name')['year'].first().to_dict()
        slides = props.index
        years = np.array([str(year_map.get(s, 'unknown')) for s in slides])
        unique_years = [y for y in sorted(set(years)) if y != 'unknown']

        batch_rows = []
        for col in props.columns:
            groups = [props.loc[years == y, col].values for y in unique_years]
            if all(len(g) >= 3 for g in groups):
                stat, pval = kruskal(*groups)
            else:
                stat, pval = np.nan, np.nan
            batch_rows.append({'cluster': col, 'kruskal_stat': stat, 'p_value': pval})

        batch_df = pd.DataFrame(batch_rows)
        valid = batch_df['p_value'].notna()
        if valid.sum() > 0:
            _, fdr, _, _ = multipletests(batch_df.loc[valid, 'p_value'], method='fdr_bh')
            batch_df.loc[valid, 'fdr'] = fdr

        batch_sig = int((batch_df.get('fdr', pd.Series(dtype=float)) < 0.05).sum())
        batch_total = int(len(batch_df))
        batch_pct = round(batch_sig / batch_total * 100, 1)

        batch_df.to_csv(output_dir / "batch_effect_kmeans.csv", index=False)
        _log(f"Batch-significant clusters: {batch_sig}/{batch_total} ({batch_pct}%)")

        metrics = {
            'n_clusters': int(labels.max()) + 1,
            'batch_sig': batch_sig,
            'batch_total': batch_total,
            'batch_pct': batch_pct,
        }
        with open(output_dir / "comparison_metrics.json", 'w') as fh:
            json.dump(metrics, fh, indent=2)

    _log(f"UMAP on ALL {N:,} cells [cuML GPU] ...")
    t_umap = time.perf_counter()

    import cupy as cp
    from cuml.manifold import UMAP as cuUMAP

    X_gpu = cp.asarray(X)
    reducer = cuUMAP(
        n_neighbors=15,
        n_components=2,
        n_epochs=200,
        random_state=SEED,
        verbose=True,
    )
    umap_coords = reducer.fit_transform(X_gpu)
    if hasattr(umap_coords, 'to_numpy'):
        umap_coords = umap_coords.to_numpy()
    elif hasattr(umap_coords, 'get'):
        umap_coords = umap_coords.get()

    del X_gpu, reducer
    gc.collect()
    _log(f"UMAP done in {time.perf_counter() - t_umap:.1f}s")

    np.save(output_dir / "umap_coords.npy", umap_coords.astype(np.float32))
    _log("UMAP coords saved → umap_coords.npy")

    scatter_kw = dict(s=0.5, alpha=0.3, linewidths=0, rasterized=True)
    n_cl = int(labels.max()) + 1

    fig, ax = plt.subplots(figsize=(12, 10))
    cmap_cl = plt.cm.get_cmap('tab20', n_cl)
    sc = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                    c=labels, cmap=cmap_cl, vmin=0, vmax=n_cl - 1, **scatter_kw)
    plt.colorbar(sc, ax=ax, label='Cluster', shrink=0.6)
    ax.set_title(f'UMAP — Spherical K-Means (k={n_cl}, all {N:,} cells)')
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    fig.tight_layout()
    fig.savefig(output_dir / 'umap_kmeans_clusters.png', dpi=DPI, bbox_inches='tight')
    fig.savefig(output_dir / 'umap_kmeans_clusters.pdf', dpi=DPI, bbox_inches='tight')
    plt.close(fig)

    if 'year' in obs_df.columns:
        years_all = obs_df['year'].astype(str).values
        unique_y = sorted(set(years_all))
        cmap_y = plt.cm.get_cmap('Set1', len(unique_y))
        fig, ax = plt.subplots(figsize=(12, 10))
        for i, yr in enumerate(unique_y):
            mask = years_all == yr
            ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                       c=[cmap_y(i)], label=yr, **scatter_kw)
        ax.legend(markerscale=8, title='Year', loc='best')
        ax.set_title(f'UMAP — Year batch (all {N:,} cells)')
        ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
        fig.tight_layout()
        fig.savefig(output_dir / 'umap_year_kmeans.png', dpi=DPI, bbox_inches='tight')
        fig.savefig(output_dir / 'umap_year_kmeans.pdf', dpi=DPI, bbox_inches='tight')
        plt.close(fig)

    _log(f"Figures saved to {output_dir}")
    _log(f"Phase 4 done in {_elapsed(t0)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spherical K-Means clustering via cuML GPU (GH200/ARM64)"
    )
    parser.add_argument('--adata-path', type=Path, required=True,
                        help='Path to adata_raw.h5ad')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--n-clusters', type=int, default=50,
                        help='K — number of clusters (default: 50)')
    parser.add_argument('--n-iter', type=int, default=20,
                        help='K-Means max iterations (default: 20)')
    parser.add_argument('--n-init', type=int, default=1,
                        help='K-Means random restarts (default: 1)')
    parser.add_argument('--skip-phase0', action='store_true',
                        help='Skip load; requires kmeans_labels.npy + obs_meta.parquet in output-dir')
    parser.add_argument('--skip-phase4', action='store_true',
                        help='Skip UMAP + QC figures')
    parser.add_argument('--n-cells-test', type=int, default=None,
                        help='Subsample N cells for smoke test')
    args = parser.parse_args()

    t_total = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"SPHERICAL K-MEANS (cuML GPU, GH200 ARM64)")
    print(f"  adata  : {args.adata_path}")
    print(f"  output : {args.output_dir}")
    print(f"  k      : {args.n_clusters}")
    print(f"  n_iter : {args.n_iter}  |  n_init: {args.n_init}")
    print(f"{'='*70}\n", flush=True)

    if args.skip_phase0:
        labels_ck = args.output_dir / "kmeans_labels.npy"
        obs_ck    = args.output_dir / "obs_meta.parquet"
        if not labels_ck.exists() or not obs_ck.exists():
            raise FileNotFoundError(
                f"--skip-phase0 requires:\n  {labels_ck}\n  {obs_ck}"
            )
        _log("Loading checkpoint (skip-phase0) ...")
        labels  = np.load(labels_ck)
        obs_df  = pd.read_parquet(obs_ck)
        X       = None  # not needed for phases 2-3; loaded lazily for phase 4
    else:
        X, obs_df = phase0_load(
            adata_path=args.adata_path,
            output_dir=args.output_dir,
            n_cells_test=args.n_cells_test,
        )
        obs_df.to_parquet(args.output_dir / "obs_meta.parquet", index=False)

        labels = phase1_kmeans(
            X=X,
            output_dir=args.output_dir,
            n_clusters=args.n_clusters,
            n_iter=args.n_iter,
            n_init=args.n_init,
        )

    phase2_export_slides(obs_df, labels, args.output_dir)
    props = phase3_proportions(obs_df, labels, args.output_dir)

    if not args.skip_phase4:
        if X is None:
            # reload X after --skip-phase0, keeping the smoke-test subsample if there was one
            idx_ck = args.output_dir / "cell_indices.npy"
            if idx_ck.exists():
                n_reload = int(np.load(idx_ck).shape[0])
                _log(f"Phase 4: reloading {n_reload:,} cells (smoke-test subsample) ...")
                X, _ = phase0_load(args.adata_path, args.output_dir,
                                   n_cells_test=n_reload)
            else:
                _log("Phase 4 requires embeddings — reloading full adata ...")
                X, _ = phase0_load(args.adata_path, args.output_dir)
        phase4_qc(X, obs_df, labels, props, args.output_dir)

    elapsed = (time.perf_counter() - t_total) / 60
    print(f"\n{'='*70}")
    print(f"COMPLETE in {elapsed:.1f} min")
    print(f"Output: {args.output_dir}/")
    for f in sorted(args.output_dir.iterdir()):
        if not f.is_dir():
            print(f"  {f.name}")
    print("=" * 70, flush=True)


if __name__ == '__main__':
    main()
