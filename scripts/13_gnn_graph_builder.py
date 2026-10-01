#!/usr/bin/env python3
"""Build one spatial cell graph per slide: cells as nodes with a one-hot phenotype, k-NN or radius edges.
Input: per-slide projected parquets (x, y, global_cluster). Output: one torch_geometric .pt per slide for
the step 15 GNN, a manifest, the build config and QC figures."""

import colorsys
import gc
import json
import time
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

try:
    import torch
    from torch_geometric.data import Data
    TORCH_VERSION = torch.__version__
except ImportError as e:
    print(f"Error: {e}")
    print("torch and torch_geometric are required.")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
    print("  pip install torch_geometric")
    sys.exit(1)

try:
    from scipy.spatial import KDTree
except ImportError:
    print("Error: scipy required.  pip install scipy")
    sys.exit(1)


SEED = 42
np.random.seed(SEED)
rng  = np.random.default_rng(SEED)

DPI = 300
plt.rcParams.update({
    'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'savefig.dpi': DPI, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
    'font.family': 'sans-serif', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.linewidth': 0.6, 'lines.linewidth': 0.8,
})


def get_cluster_colors(n_clusters: int) -> list:
    """Generate n visually distinct colors via golden-ratio HSV sampling."""
    golden = 0.618033988749895
    h = 0.1
    colors = []
    for _ in range(n_clusters):
        h = (h + golden) % 1.0
        colors.append(colorsys.hsv_to_rgb(h, 0.72, 0.88))
    return colors


def build_knn_edges(coords: np.ndarray, k: int) -> torch.Tensor:
    """Directed edges from each cell to its k nearest neighbours, self excluded."""
    n       = len(coords)
    k_query = min(k + 1, n)
    tree    = KDTree(coords)
    _, indices = tree.query(coords, k=k_query)

    src_list, dst_list = [], []
    for i, nbrs in enumerate(indices):
        for j in nbrs[1:]:
            src_list.append(i)
            dst_list.append(int(j))

    if not src_list:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([src_list, dst_list], dtype=torch.long)


def build_radius_edges(coords: np.ndarray, radius: float) -> torch.Tensor:
    """Edges between all cells closer than radius, in both directions, self excluded."""
    tree  = KDTree(coords)
    pairs = tree.query_pairs(r=radius, output_type='ndarray')

    if len(pairs) == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    src = np.concatenate([pairs[:, 0], pairs[:, 1]])
    dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
    return torch.tensor([src, dst], dtype=torch.long)


def build_graph(df: pd.DataFrame, n_clusters: int,
                mode: str, k: int, radius: float) -> Data:
    """PyG Data for one slide: one-hot cluster features (x), edge_index and pixel coordinates (pos)."""
    x_col  = 'centroid_x' if 'centroid_x' in df.columns else 'x'
    y_col  = 'centroid_y' if 'centroid_y' in df.columns else 'y'

    coords      = df[[x_col, y_col]].values.astype(np.float32)
    cluster_ids = df['global_cluster'].values.astype(int)
    n_cells     = len(df)

    # nodes carry only the one-hot phenotype, not the embedding: beyond the cluster proportions,
    # the only extra information given to the GNN is the spatial arrangement
    node_feats = np.zeros((n_cells, n_clusters), dtype=np.float32)
    # ids >= n_clusters are dropped; unused ids below it leave an all-zero column
    valid      = (cluster_ids >= 0) & (cluster_ids < n_clusters)
    node_feats[valid, cluster_ids[valid]] = 1.0

    edge_index = build_knn_edges(coords, k) if mode == 'knn' \
                 else build_radius_edges(coords, radius)

    return Data(
        x          = torch.tensor(node_feats, dtype=torch.float),
        edge_index = edge_index,
        pos        = torch.tensor(coords,     dtype=torch.float),
    )


def _find_wsi(wsi_dir: Path, slide_name: str):
    """First WSI file in wsi_dir whose name contains slide_name, or None."""
    for ext in ('*.ndpi', '*.svs', '*.tiff', '*.tif', '*.mrxs'):
        matches = list(wsi_dir.glob(f'*{slide_name}*{ext[1:]}'))
        if matches:
            return matches[0]
    return None


def _read_wsi_roi(wsi_path: Path, ox: float, oy: float,
                  roi_size: int) -> np.ndarray | None:
    """Level-0 RGB region with top-left corner (ox, oy), or None on failure."""
    try:
        import openslide
        slide  = openslide.OpenSlide(str(wsi_path))
        region = slide.read_region(
            (int(ox), int(oy)), level=0, size=(roi_size, roi_size)
        )
        slide.close()
        return np.array(region.convert('RGB'))
    except Exception as e:
        print(f"    WSI read warning ({wsi_path.name}): {e}")
        return None


def _load_sample_graphs(graph_dir: Path, n_slides: int, seed: int = SEED):
    """Load up to n_slides .pt files from graph_dir, chosen randomly."""
    pt_files = sorted(graph_dir.glob("*.pt"))
    rng_local = np.random.default_rng(seed)
    chosen    = rng_local.choice(pt_files, size=min(n_slides, len(pt_files)), replace=False)
    graphs = {}
    for pf in chosen:
        try:
            data = torch.load(pf, map_location='cpu', weights_only=False)
            graphs[pf.stem] = data
        except Exception as e:
            print(f"  QC load warning: {pf.name} — {e}")
    return graphs


def _select_dense_rois(pos: np.ndarray, roi_size: int, n_rois: int) -> list:
    """Top-left corners of the n_rois densest non-overlapping windows on a half-window grid."""
    x_min, y_min = pos[:, 0].min(), pos[:, 1].min()
    x_max, y_max = pos[:, 0].max(), pos[:, 1].max()

    stride = roi_size // 2
    xs = np.arange(x_min, x_max - roi_size + 1, stride)
    ys = np.arange(y_min, y_max - roi_size + 1, stride)

    if len(xs) == 0 or len(ys) == 0:
        return [(float(x_min), float(y_min))]

    candidates = []
    for ox in xs:
        mask_x = (pos[:, 0] >= ox) & (pos[:, 0] < ox + roi_size)
        for oy in ys:
            mask_y = (pos[:, 1] >= oy) & (pos[:, 1] < oy + roi_size)
            n_cells = int((mask_x & mask_y).sum())
            if n_cells >= 5:
                candidates.append((n_cells, float(ox), float(oy)))

    if not candidates:
        return [(float(x_min), float(y_min))]

    candidates.sort(reverse=True)  # densest first

    selected = []
    min_sep  = roi_size * 0.5
    for _, ox, oy in candidates:
        too_close = any(
            abs(ox - sox) < min_sep and abs(oy - soy) < min_sep
            for sox, soy in selected
        )
        if not too_close:
            selected.append((ox, oy))
        if len(selected) >= n_rois:
            break

    return selected or [(candidates[0][1], candidates[0][2])]


def plot_roi_networks(graph_dir: Path, qc_dir: Path, n_clusters: int,
                      n_slides: int = 5, n_rois: int = 3, roi_size: int = 1000,
                      wsi_dir: Path = None):
    """Draw the cell graph in the densest ROIs of a few slides, over H&E when wsi_dir is given."""
    print("  [QC 1/3] ROI network plots (density-selected"
          + (", H&E background)" if wsi_dir else ", no WSI dir provided)"))
    colors = get_cluster_colors(n_clusters)
    graphs = _load_sample_graphs(graph_dir, n_slides)

    for slide_name, data in graphs.items():
        pos    = data.pos.numpy()                # (n_cells, 2), level-0 px
        ei     = data.edge_index.numpy()         # (2, n_edges)
        cl_ids = data.x.numpy().argmax(axis=1)   # (n_cells,)

        wsi_path = None
        if wsi_dir:
            wsi_path = _find_wsi(wsi_dir, slide_name)
            if wsi_path is None:
                print(f"    Warning: no WSI found for {slide_name} in {wsi_dir}")

        roi_origins = _select_dense_rois(pos, roi_size, n_rois)

        for roi_idx, (ox, oy) in enumerate(roi_origins):

            in_roi = (
                (pos[:, 0] >= ox) & (pos[:, 0] < ox + roi_size) &
                (pos[:, 1] >= oy) & (pos[:, 1] < oy + roi_size)
            )
            idx_in = np.where(in_roi)[0]
            if len(idx_in) < 5:
                continue

            mask_e = in_roi[ei[0]] & in_roi[ei[1]]
            ei_roi = ei[:, mask_e]

            he_patch = None
            if wsi_path is not None:
                he_patch = _read_wsi_roi(wsi_path, ox, oy, roi_size)

            fig, ax = plt.subplots(figsize=(5, 5))

            if he_patch is not None:
                # extent is [left, right, bottom, top]; with the inverted y axis below, the
                # patch's first row lands at oy
                ax.imshow(
                    he_patch,
                    extent=[ox, ox + roi_size, oy + roi_size, oy],
                    origin='upper',
                    aspect='equal',
                    zorder=0,
                    interpolation='bilinear',
                )
                edge_color  = 'white'
                edge_alpha  = 0.55
                node_size   = 10
                legend_bg   = 0.55
            else:
                ax.set_facecolor('#f0ede8')
                edge_color  = '#888888'
                edge_alpha  = 0.6
                node_size   = 8
                legend_bg   = 0.75

            for s, d in zip(ei_roi[0], ei_roi[1]):
                ax.plot(
                    [pos[s, 0], pos[d, 0]],
                    [pos[s, 1], pos[d, 1]],
                    color=edge_color, lw=0.25, alpha=edge_alpha, zorder=1,
                )

            node_x      = pos[idx_in, 0]
            node_y      = pos[idx_in, 1]
            node_cl     = cl_ids[idx_in]
            node_colors = [colors[c % n_clusters] for c in node_cl]
            ax.scatter(node_x, node_y, c=node_colors, s=node_size,
                       linewidths=0.2, edgecolors='white',
                       zorder=2, rasterized=True)

            unique_cl = np.unique(node_cl)
            if len(unique_cl) <= 20:
                patches = [mpatches.Patch(color=colors[c % n_clusters],
                                          label=f'C{c}') for c in unique_cl]
                ax.legend(handles=patches, fontsize=5, ncol=3,
                          loc='upper right', framealpha=legend_bg,
                          markerscale=0.8, handlelength=1,
                          facecolor='black' if he_patch is not None else 'white',
                          labelcolor='white' if he_patch is not None else 'black')

            # invert y to match image row order
            ax.set_xlim(ox, ox + roi_size)
            ax.set_ylim(oy + roi_size, oy)
            ax.set_xlabel('X (px)')
            ax.set_ylabel('Y (px)')
            has_he = "H&E" if he_patch is not None else "no WSI"
            ax.set_title(
                f"{slide_name}  [{has_he}]\n"
                f"ROI {roi_idx+1} — {len(idx_in)} cells · {mask_e.sum()} edges"
                f"  @({int(ox)},{int(oy)})+{roi_size}px",
                fontsize=7,
            )
            ax.set_aspect('equal')

            out_path = qc_dir / f"roi_network_{slide_name}_roi{roi_idx+1}.png"
            fig.savefig(out_path, dpi=DPI)
            plt.close(fig)

    print(f"    Saved ROI plots to {qc_dir}")


def plot_degree_distribution(graph_dir: Path, qc_dir: Path, max_slides: int = 50):
    """Histogram of node out-degrees over the first max_slides graphs."""
    print("  [QC 2/3] Degree distribution histogram...")
    pt_files   = sorted(graph_dir.glob("*.pt"))
    sample     = pt_files[:max_slides]
    all_degrees = []

    for pf in tqdm(sample, desc="    Computing degrees", leave=False):
        try:
            data   = torch.load(pf, map_location='cpu', weights_only=False)
            n      = data.x.shape[0]
            if data.edge_index.shape[1] == 0:
                all_degrees.extend([0] * n)
                continue
            src    = data.edge_index[0].numpy()
            counts = np.bincount(src, minlength=n)
            all_degrees.extend(counts.tolist())
        except Exception:
            continue

    if not all_degrees:
        print("    No degree data collected — skipping.")
        return

    degrees = np.array(all_degrees)
    mean_d  = degrees.mean()
    median_d = np.median(degrees)
    pct_isolated = 100 * (degrees == 0).mean()

    fig, ax = plt.subplots(figsize=(6, 4))
    max_deg = min(int(np.percentile(degrees, 99.5)), 50)
    bins    = np.arange(0, max_deg + 2) - 0.5
    ax.hist(degrees[degrees <= max_deg], bins=bins,
            color='#2166AC', alpha=0.8, edgecolor='white', linewidth=0.3)
    ax.axvline(mean_d,   color='#B2182B', lw=1.5, ls='--',
               label=f'Mean = {mean_d:.2f}')
    ax.axvline(median_d, color='#FF7F00', lw=1.5, ls=':',
               label=f'Median = {median_d:.2f}')
    ax.axvspan(5, 8, alpha=0.08, color='green',
               label='Ideal range (5–8)')
    ax.set_xlabel('Node degree (# neighbours)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Node Degree Distribution  '
                 f'({len(sample)} slides, {len(degrees):,} cells)\n'
                 f'Isolated nodes: {pct_isolated:.1f}%')
    ax.legend(fontsize=7)
    ax.set_xlim(-0.5, max_deg + 0.5)
    sns.despine(ax=ax)

    out_path = qc_dir / 'degree_distribution.png'
    fig.savefig(out_path)
    plt.close(fig)
    print(f"    Mean degree: {mean_d:.2f} | Isolated: {pct_isolated:.1f}% | saved {out_path}")


def plot_cooccurrence_matrix(graph_dir: Path, qc_dir: Path,
                             n_clusters: int, max_slides: int = 100):
    """Cluster x cluster adjacency over graph edges, as a row-normalised heatmap and raw counts."""
    print("  [QC 3/3] Co-occurrence matrix...")
    pt_files = sorted(graph_dir.glob("*.pt"))
    sample   = pt_files[:max_slides]
    comat    = np.zeros((n_clusters, n_clusters), dtype=np.float64)

    for pf in tqdm(sample, desc="    Accumulating edges", leave=False):
        try:
            data = torch.load(pf, map_location='cpu', weights_only=False)
            if data.edge_index.shape[1] == 0:
                continue
            cl   = data.x.numpy().argmax(axis=1)  # (n_cells,)
            src  = data.edge_index[0].numpy()
            dst  = data.edge_index[1].numpy()
            ci   = cl[src]
            cj   = cl[dst]
            valid = (ci < n_clusters) & (cj < n_clusters)
            np.add.at(comat, (ci[valid], cj[valid]), 1)
        except Exception:
            continue

    # row-normalised: P(neighbour is cluster j | centre is cluster i)
    row_sums = comat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    comat_norm = comat / row_sums

    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(
        comat_norm,
        ax=ax,
        cmap='YlOrRd',
        square=True,
        linewidths=0,
        cbar_kws={'label': 'P(cluster j adjacent | cluster i)', 'shrink': 0.6},
        xticklabels=[str(i) if i % 5 == 0 else '' for i in range(n_clusters)],
        yticklabels=[str(i) if i % 5 == 0 else '' for i in range(n_clusters)],
    )
    ax.set_xlabel('Cluster j  (neighbour)')
    ax.set_ylabel('Cluster i  (center)')
    ax.set_title(f'Spatial Co-occurrence Matrix  '
                 f'({len(sample)} slides)\n'
                 f'P(cluster j is spatially adjacent to cluster i)')

    out_path = qc_dir / 'cooccurrence_matrix.png'
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)

    pd.DataFrame(
        comat.astype(int),
        index=[f'C{i}' for i in range(n_clusters)],
        columns=[f'C{i}' for i in range(n_clusters)],
    ).to_csv(qc_dir / 'cooccurrence_counts.csv')

    print(f"    Saved {out_path}")


def generate_qc_visuals(graph_dir: Path, n_clusters: int,
                        qc_n_slides: int, qc_n_rois: int, qc_roi_size: int,
                        wsi_dir: Path = None):
    """Run all three QC visualizations into graph_dir/graph_qc_visuals/."""
    qc_dir = graph_dir / 'graph_qc_visuals'
    qc_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating QC visuals → {qc_dir}")

    plot_roi_networks(graph_dir, qc_dir, n_clusters,
                      n_slides=qc_n_slides, n_rois=qc_n_rois, roi_size=qc_roi_size,
                      wsi_dir=wsi_dir)
    plot_degree_distribution(graph_dir, qc_dir, max_slides=100)
    plot_cooccurrence_matrix(graph_dir, qc_dir, n_clusters, max_slides=100)

    print("  QC visuals done.")


def process_slides(projected_dir, output_dir, n_clusters,
                   mode, k, radius, slides_subset, resume):
    projected_dir = Path(projected_dir)
    output_dir    = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir.parent / 'logs').mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(projected_dir.glob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files in {projected_dir}")

    if slides_subset:
        subset_set    = set(slides_subset)
        parquet_files = [
            pf for pf in parquet_files
            if pf.stem.replace('_features', '').replace('_projected', '').strip()
            in subset_set
        ]
        print(f"  Filtered to {len(parquet_files)} specified slides")

    manifest, skipped, failed = [], 0, 0

    for pf in tqdm(parquet_files, desc="Building graphs", unit="slide"):
        slide_name = pf.stem.replace('_features', '').replace('_projected', '').strip()
        out_path   = output_dir / f"{slide_name}.pt"

        if resume and out_path.exists():
            skipped += 1
            manifest.append({'slide_name': slide_name, 'status': 'skipped',
                              'out_path': str(out_path)})
            continue

        try:
            df = pd.read_parquet(pf)

            if 'centroid_x' not in df.columns and 'x' in df.columns:
                df = df.rename(columns={'x': 'centroid_x', 'y': 'centroid_y'})

            if 'global_cluster' not in df.columns:
                print(f"\n  Warning: {slide_name} — missing 'global_cluster', skipping")
                failed += 1
                manifest.append({'slide_name': slide_name,
                                  'status': 'error: no global_cluster', 'out_path': ''})
                continue

            n_cells           = len(df)
            n_unique_clusters = df['global_cluster'].nunique()

            data          = build_graph(df, n_clusters, mode, k, radius)
            n_edges       = data.edge_index.shape[1]
            data.slide_name = slide_name
            data.n_clusters = n_clusters

            torch.save(data, out_path)

            manifest.append({
                'slide_name':         slide_name,
                'n_cells':            n_cells,
                'n_clusters_present': n_unique_clusters,
                'n_edges':            n_edges,
                'avg_degree':         round(n_edges / max(n_cells, 1), 3),
                'status':             'ok',
                'out_path':           str(out_path),
            })

            del df, data
            gc.collect()

        except Exception as e:
            print(f"\n  Error on {slide_name}: {e}")
            failed += 1
            manifest.append({'slide_name': slide_name,
                              'status': f'error: {e}', 'out_path': ''})

    return manifest, skipped, failed


def parse_args():
    p = argparse.ArgumentParser(
        description="Build spatial cell graphs (.pt) from projected slide parquets for GNN training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--projected-dir', required=True,
                   help="Directory containing projected slide parquets")
    p.add_argument('--output-dir', required=True,
                   help="Output directory for .pt graph files and manifest")
    # required: a value above the real number of phenotypes only adds all-zero feature columns
    p.add_argument('--n-clusters', type=int, required=True,
                   help="One-hot node-feature dimension: the number of clusters. "
                        "A value above it only adds permanently-zero columns.")
    p.add_argument('--mode', choices=['knn', 'radius'], default='knn',
                   help="Edge construction method")
    # downstream results were not sensitive to k
    p.add_argument('--k-neighbors', type=int, default=6,
                   help="[knn mode] Nearest neighbours per cell (excluding self)")
    p.add_argument('--radius', type=float, default=200.0,
                   help="[radius mode] Max edge distance in pixels")
    p.add_argument('--slides-subset', type=str, default=None,
                   help="Comma-separated slide names to process (default: all)")
    p.add_argument('--no-resume', action='store_true',
                   help="Reprocess slides even if .pt already exists")
    p.add_argument('--skip-qc', action='store_true',
                   help="Skip QC visualization generation")
    p.add_argument('--qc-n-slides', type=int, default=5,
                   help="Number of slides to sample for ROI network plots")
    p.add_argument('--qc-n-rois', type=int, default=3,
                   help="Number of ROI windows per slide")
    p.add_argument('--qc-roi-size', type=int, default=1000,
                   help="ROI window size in pixels (square)")
    p.add_argument('--wsi-dir', type=str, default=None,
                   help="Directory containing WSI files (.ndpi/.svs) for H&E background overlay")
    return p.parse_args()


def main():
    args = parse_args()
    t0   = time.time()

    print("=" * 70)
    print("GNN GRAPH BUILDER — C3PRO Pipeline (Script 13)")
    print("=" * 70)
    print(f"PyTorch version : {TORCH_VERSION}")
    print(f"Projected dir   : {args.projected_dir}")
    print(f"Output dir      : {args.output_dir}")
    print(f"N clusters      : {args.n_clusters}")
    print(f"Edge mode       : {args.mode}",
          f"(k={args.k_neighbors})" if args.mode == 'knn' else f"(radius={args.radius}px)")
    print(f"Resume          : {not args.no_resume}")
    print(f"QC visuals      : {not args.skip_qc}")
    print()

    slides_subset = ([s.strip() for s in args.slides_subset.split(',')]
                     if args.slides_subset else None)

    manifest, skipped, failed = process_slides(
        projected_dir = args.projected_dir,
        output_dir    = args.output_dir,
        n_clusters    = args.n_clusters,
        mode          = args.mode,
        k             = args.k_neighbors,
        radius        = args.radius,
        slides_subset = slides_subset,
        resume        = not args.no_resume,
    )

    output_dir    = Path(args.output_dir)
    manifest_df   = pd.DataFrame(manifest)
    manifest_path = output_dir / 'graph_manifest.csv'
    manifest_df.to_csv(manifest_path, index=False)

    config = {
        'projected_dir': str(args.projected_dir),
        'output_dir':    str(args.output_dir),
        'n_clusters':    args.n_clusters,
        'mode':          args.mode,
        'k_neighbors':   args.k_neighbors if args.mode == 'knn'    else None,
        'radius':        args.radius       if args.mode == 'radius' else None,
        'torch_version': TORCH_VERSION,
    }
    with open(output_dir / 'graph_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    if not args.skip_qc:
        generate_qc_visuals(
            graph_dir   = output_dir,
            n_clusters  = args.n_clusters,
            qc_n_slides = args.qc_n_slides,
            qc_n_rois   = args.qc_n_rois,
            qc_roi_size = args.qc_roi_size,
            wsi_dir     = Path(args.wsi_dir) if args.wsi_dir else None,
        )

    ok_mask = manifest_df.get('status', pd.Series()) == 'ok'
    n_ok    = int(ok_mask.sum())
    elapsed = time.time() - t0

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Built      : {n_ok}")
    print(f"  Skipped    : {skipped}  (already existed)")
    print(f"  Failed     : {failed}")
    print(f"  Manifest   : {manifest_path}")

    if n_ok > 0:
        ok_rows = manifest_df[ok_mask]
        print(f"\n  Graph statistics (built slides):")
        print(f"    Cells / slide   : {ok_rows['n_cells'].mean():.0f} avg"
              f"  [{ok_rows['n_cells'].min()}–{ok_rows['n_cells'].max()}]")
        print(f"    Edges / slide   : {ok_rows['n_edges'].mean():.0f} avg"
              f"  [{ok_rows['n_edges'].min()}–{ok_rows['n_edges'].max()}]")
        print(f"    Avg node degree : {ok_rows['avg_degree'].mean():.2f}")
        print(f"    Clusters/slide  : {ok_rows['n_clusters_present'].mean():.1f} avg"
              f"  [{ok_rows['n_clusters_present'].min()}–"
              f"{ok_rows['n_clusters_present'].max()}]")

    print(f"\n  Total time : {elapsed:.1f}s  ({elapsed / 60:.1f} min)")
    print("Done.")


if __name__ == '__main__':
    main()
