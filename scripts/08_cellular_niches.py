#!/usr/bin/env python3
"""Discover recurrent cellular neighbourhoods ("niches") from cell phenotype clusters, after Schürch et al. 2020.
Input: per-slide parquets with x, y and global_cluster. Output: per-cell niche assignments, the niche x
cluster composition matrix, a per-niche robustness table, QC plots and the run config."""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)


def _detect_gpu():
    """Return (faiss module or None, whether a GPU is available)."""
    try:
        import faiss
        try:
            ngpu = faiss.get_num_gpus()
        except Exception:
            ngpu = 0
        return faiss, ngpu > 0
    except ImportError:
        return None, False


def knn_indices(coords: np.ndarray, k: int, faiss_mod, use_gpu: bool) -> np.ndarray:
    """Indices of the k nearest neighbours of each point, excluding itself, shape (n, k)."""
    # a fixed neighbour count (as in CODEX), not a fixed radius: the physical extent of a
    # neighbourhood varies with local cell density
    n = len(coords)
    # ask for k+1 because the query point comes back as its own neighbour; keeping it would bias
    # every composition towards the central cell's phenotype
    k_self = min(k + 1, n)
    coords = np.ascontiguousarray(coords.astype(np.float32))

    if faiss_mod is not None:
        index = faiss_mod.IndexFlatL2(2)
        if use_gpu:
            try:
                res = faiss_mod.StandardGpuResources()
                index = faiss_mod.index_cpu_to_gpu(res, 0, index)
            except Exception:
                pass  # fall back to CPU faiss
        index.add(coords)
        _, idx = index.search(coords, k_self)
    else:
        from scipy.spatial import cKDTree
        tree = cKDTree(coords)
        _, idx = tree.query(coords, k=k_self)
        if idx.ndim == 1:  # k_self == 1 edge case
            idx = idx[:, None]

    # drop self by index rather than by column: with tied distances it is not always first
    out = np.empty((n, min(k, k_self - 1)), dtype=np.int64)
    for row in range(n):
        nbrs = idx[row]
        nbrs = nbrs[nbrs != row][:k]
        if len(nbrs) < out.shape[1]:
            # pad by repeating last neighbor (tiny slides); harmless for composition
            pad = np.full(out.shape[1] - len(nbrs), nbrs[-1] if len(nbrs) else row)
            nbrs = np.concatenate([nbrs, pad])
        out[row] = nbrs
    return out


def neighborhood_composition(cluster_ids: np.ndarray, nbr_idx: np.ndarray,
                             n_clusters: int) -> np.ndarray:
    """Fraction of each cluster among each cell's neighbours, shape (n, n_clusters)."""
    n, k = nbr_idx.shape
    nbr_labels = cluster_ids[nbr_idx]  # (n, k)
    comp = np.zeros((n, n_clusters), dtype=np.float32)
    rows = np.repeat(np.arange(n), k)
    cols = nbr_labels.reshape(-1)
    valid = (cols >= 0) & (cols < n_clusters)
    np.add.at(comp, (rows[valid], cols[valid]), 1.0)
    row_sums = comp.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return comp / row_sums


def kmeans_niches(comp_matrix: np.ndarray, n_niches: int,
                  faiss_mod, use_gpu: bool, seed: int = SEED):
    """K-means on the composition vectors; returns (labels, centroids)."""
    x = np.ascontiguousarray(comp_matrix.astype(np.float32))
    d = x.shape[1]

    if faiss_mod is not None:
        km = faiss_mod.Kmeans(d, n_niches, niter=25, seed=seed,
                              gpu=use_gpu, verbose=False, spherical=False)
        km.train(x)
        _, labels = km.index.search(x, 1)
        return labels.reshape(-1).astype(int), km.centroids.reshape(n_niches, d)
    else:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_niches, random_state=seed, n_init=10)
        labels = km.fit_predict(x)
        return labels.astype(int), km.cluster_centers_


def main():
    ap = argparse.ArgumentParser(
        description="Cellular-niche discovery (Module 1: discovery + QC).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--projected-dir", required=True,
                    help="Directory of per-slide projected parquets (x, y, global_cluster)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-clusters", type=int, default=50,
                    help="Number of phenotype clusters (composition-vector dimension)")
    # k=10 is the CODEX window size, not tuned for H&E; conclusions held for k from 4 to 50
    ap.add_argument("--k-neighbors", type=int, default=10,
                    help="Neighborhood window size (Schürch CN default: 10)")
    ap.add_argument("--n-niches", type=int, default=10,
                    help="Number of niches (Schürch identified 9 CNs)")
    ap.add_argument("--x-col", default="x")
    ap.add_argument("--y-col", default="y")
    ap.add_argument("--cluster-col", default="global_cluster")
    ap.add_argument("--slides-subset-n", type=int, default=None,
                    help="Random N-slide subset for a fast test run (default: all slides)")
    ap.add_argument("--min-cells", type=int, default=50,
                    help="Skip slides with fewer than this many cells")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    t0 = time.time()
    faiss_mod, gpu = _detect_gpu()
    backend = "faiss-gpu" if gpu else ("faiss-cpu" if faiss_mod else "scipy/sklearn (CPU)")
    print(f"  Backend: {backend}")

    proj_dir = Path(args.projected_dir)
    out_dir = Path(args.output_dir)
    (out_dir / "niche_assignments").mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(proj_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"  ERROR: no parquets in {proj_dir}")
        sys.exit(1)

    if args.slides_subset_n:
        rng = np.random.default_rng(args.seed)
        pick = rng.choice(len(parquet_files),
                          size=min(args.slides_subset_n, len(parquet_files)),
                          replace=False)
        parquet_files = [parquet_files[i] for i in sorted(pick)]
        print(f"  TEST MODE: {len(parquet_files)} random slides")
    else:
        print(f"  FULL RUN: {len(parquet_files)} slides")

    # compositions are pooled across slides so that niches are defined once for the whole cohort
    all_comp = []
    slide_index = []       # (slide, n_cells), to split the pooled labels back per slide
    per_slide_frames = []

    K = args.n_clusters
    for pf in parquet_files:
        slide = pf.stem
        if slide.endswith("_features"):
            slide = slide[: -len("_features")]
        df = pd.read_parquet(pf)
        # tolerate alternate coordinate column names
        xcol = args.x_col if args.x_col in df.columns else ("centroid_x" if "centroid_x" in df.columns else args.x_col)
        ycol = args.y_col if args.y_col in df.columns else ("centroid_y" if "centroid_y" in df.columns else args.y_col)
        if xcol not in df.columns or ycol not in df.columns or args.cluster_col not in df.columns:
            print(f"    [WARN] {slide}: missing columns, skipping")
            continue
        if len(df) < args.min_cells:
            print(f"    [skip] {slide}: {len(df)} cells < {args.min_cells}")
            continue

        coords = df[[xcol, ycol]].values.astype(np.float32)
        cids = df[args.cluster_col].values.astype(np.int64)

        nbr = knn_indices(coords, args.k_neighbors, faiss_mod, gpu)
        comp = neighborhood_composition(cids, nbr, K)

        all_comp.append(comp)
        slide_index.append((slide, len(df)))
        per_slide_frames.append(df.assign(_slide=slide, _x=df[xcol], _y=df[ycol],
                                          _cluster=cids))
        print(f"    {slide}: {len(df):,} cells")

    if not all_comp:
        print("  ERROR: no slides produced compositions.")
        sys.exit(1)

    comp_pooled = np.vstack(all_comp)
    print(f"\n  Pooled composition matrix: {comp_pooled.shape[0]:,} cells x {K} clusters")

    print(f"  Clustering into {args.n_niches} niches ...")
    niche_labels, centroids = kmeans_niches(comp_pooled, args.n_niches,
                                            faiss_mod, gpu, seed=args.seed)

    cursor = 0
    robustness = {nid: {"n_cells": 0, "slides": set()} for nid in range(args.n_niches)}
    for (slide, n_i), frame in zip(slide_index, per_slide_frames):
        lbl = niche_labels[cursor: cursor + n_i]
        cursor += n_i
        out = pd.DataFrame({
            "x": frame["_x"].values,
            "y": frame["_y"].values,
            "global_cluster": frame["_cluster"].values,
            "niche": lbl,
        })
        out.to_parquet(out_dir / "niche_assignments" / f"{slide}.parquet", index=False)
        for nid in range(args.n_niches):
            c = int((lbl == nid).sum())
            if c:
                robustness[nid]["n_cells"] += c
                robustness[nid]["slides"].add(slide)

    # the centroids are the mean neighbourhood composition of each niche
    comp_df = pd.DataFrame(centroids,
                           index=[f"niche_{i}" for i in range(args.n_niches)],
                           columns=[f"C{j}" for j in range(K)])
    comp_df.to_csv(out_dir / "niche_composition.csv")

    fig, ax = plt.subplots(figsize=(max(10, K * 0.22), max(4, args.n_niches * 0.5)))
    im = ax.imshow(centroids, aspect="auto", cmap="viridis")
    ax.set_xticks(range(K)); ax.set_xticklabels([f"C{j}" for j in range(K)],
                                                rotation=90, fontsize=6)
    ax.set_yticks(range(args.n_niches))
    ax.set_yticklabels([f"niche {i}" for i in range(args.n_niches)], fontsize=8)
    ax.set_xlabel("phenotype cluster"); ax.set_ylabel("niche")
    ax.set_title("Niche composition (mean neighborhood cluster fractions)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="fraction")
    fig.tight_layout()
    fig.savefig(out_dir / "niche_composition_heatmap.png", dpi=200)
    plt.close(fig)

    total_cells = comp_pooled.shape[0]
    rob_rows = []
    for nid in range(args.n_niches):
        rob_rows.append({
            "niche": nid,
            "n_cells": robustness[nid]["n_cells"],
            "n_slides": len(robustness[nid]["slides"]),
            "pct_of_cells": round(100.0 * robustness[nid]["n_cells"] / total_cells, 2),
        })
    rob_df = pd.DataFrame(rob_rows)
    rob_df.to_csv(out_dir / "niche_robustness.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(rob_df["niche"], rob_df["n_slides"], color="#4C72B0")
    ax.set_xlabel("niche"); ax.set_ylabel("n slides present")
    ax.set_title(f"Niche robustness across {len(slide_index)} slides "
                 f"(low n_slides = candidate artefact)")
    ax.axhline(len(slide_index) * 0.5, ls="--", color="red", lw=0.8,
               label="50% of slides")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "niche_robustness.png", dpi=200)
    plt.close(fig)

    config = {
        "n_clusters": K,
        "k_neighbors": args.k_neighbors,
        "n_niches": args.n_niches,
        "n_slides": len(slide_index),
        "n_cells_total": int(total_cells),
        "backend": backend,
        "seed": args.seed,
        "method": "Schurch-style cellular neighborhoods on morphological clusters",
    }
    with open(out_dir / "niche_config.json", "w") as f:
        json.dump(config, f, indent=2)

    dt = time.time() - t0
    print(f"\n  {'='*66}")
    print("  DONE — Module 1 (discovery + QC)")
    print(f"  {'='*66}")
    print(f"  Slides processed : {len(slide_index)}")
    print(f"  Cells total      : {total_cells:,}")
    print(f"  Niches           : {args.n_niches}")
    print(f"  Backend          : {backend}")
    print(f"  Time             : {dt:.1f}s ({dt/60:.1f} min)")
    print(f"\n  QC to inspect:")
    print(f"    niche_composition_heatmap.png  (are niches compositionally distinct?)")
    print(f"    niche_robustness.png           (any niche on <50% of slides = artefact?)")
    print(f"  Output: {out_dir}")


if __name__ == "__main__":
    main()
