#!/usr/bin/env python3
"""Render representative regions of each cellular niche for review, with nuclei coloured by cluster.
Input: per-slide niche assignments, the WSIs and the niche composition matrix. Output: per-niche
vignettes and context crops, one montage panel per niche and a vignette manifest."""

import argparse
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import openslide
except ImportError:
    openslide = None

SEED = 42
np.random.seed(SEED)

# two scales: the level-0 crop resolves individual nuclei, the wider context crop on the same
# centre shows the surrounding tissue architecture
TILE = 1024              # vignette size at level 0
CONTEXT_FACTOR = 4
POINT_SIZE = 46
POINT_ALPHA = 1.0
EDGE_K = 6               # same k as the GNN graphs
EDGE_ALPHA = 0.28        # faint enough not to hide the tissue
EDGE_WIDTH = 1.9
EDGE_COLOR = "black"
DRAW_EDGES = True


def _canon_key(name: str) -> str:
    """Normalise a slide or WSI name to a lowercase alphanumeric key (same rule as script 07)."""
    stem = name
    if stem.endswith(".ndpi"):
        stem = stem[:-5]
    if stem.endswith("_features"):
        stem = stem[: -len("_features")]
    return re.sub(r"[^0-9a-z]", "", stem.lower())


def build_wsi_index(wsi_dir: Path):
    """Map normalised key -> WSI path for every .ndpi under wsi_dir."""
    idx = {}
    for p in wsi_dir.rglob("*.ndpi"):
        idx.setdefault(_canon_key(p.name), p)
    return idx


def cluster_palette(n_clusters: int):
    """Deterministic saturated colours per cluster, identical across all panels."""
    vivid = []
    for cmap_name, count in [("tab10", 10), ("Set1", 9), ("Dark2", 8), ("Set2", 8)]:
        cm = plt.cm.get_cmap(cmap_name)
        for i in range(count):
            vivid.append(cm(i / max(1, count - 1)))
    vivid = np.array(vivid)
    reps = int(np.ceil(n_clusters / len(vivid)))
    cols = np.vstack([vivid] * reps)[:n_clusters]
    return cols


def niche_purity(coords: np.ndarray, niche_ids: np.ndarray, target_niche: int,
                 radius: float):
    """For each cell of the target niche, the share of its neighbours within radius in the same niche."""
    # high purity marks a niche core, low purity a border between niches
    from scipy.spatial import cKDTree
    mask = niche_ids == target_niche
    if mask.sum() == 0:
        return np.array([], dtype=int), np.array([])
    tree = cKDTree(coords)
    target_idx = np.where(mask)[0]
    purity = np.empty(len(target_idx), dtype=np.float32)
    for i, ci in enumerate(target_idx):
        neigh = tree.query_ball_point(coords[ci], r=radius)
        if len(neigh) <= 1:
            purity[i] = 0.0
        else:
            same = np.sum(niche_ids[neigh] == target_niche) - 1  # exclude self
            purity[i] = same / (len(neigh) - 1)
    return target_idx, purity


def sample_centers(target_idx, purity, coords, n_want, min_sep):
    """Greedily pick up to n_want of the purest cells as centres, at least min_sep pixels apart."""
    if len(target_idx) == 0:
        return []
    order = np.argsort(-purity)
    chosen = []
    chosen_xy = []
    for o in order:
        ci = target_idx[o]
        xy = coords[ci]
        if all(np.hypot(xy[0] - cx, xy[1] - cy) >= min_sep for cx, cy in chosen_xy):
            chosen.append(int(ci))
            chosen_xy.append((xy[0], xy[1]))
            if len(chosen) >= n_want:
                break
    return chosen


def render_vignette(slide_img, crop_x, crop_y, size, cell_df, palette,
                    niche_label, top3_text, out_path, point_size, alpha,
                    draw_edges=True, edge_alpha=EDGE_ALPHA, edge_width=EDGE_WIDTH,
                    edge_color=EDGE_COLOR):
    """Render one crop with its nuclei coloured by cluster over their k-NN edges."""
    region = slide_img.read_region((crop_x, crop_y), 0, (size, size)).convert("RGB")

    in_crop = cell_df[
        (cell_df["x"] >= crop_x) & (cell_df["x"] < crop_x + size) &
        (cell_df["y"] >= crop_y) & (cell_df["y"] < crop_y + size)
    ]
    lx = in_crop["x"].values - crop_x
    ly = in_crop["y"].values - crop_y
    cc = in_crop["global_cluster"].values

    fig, ax = plt.subplots(figsize=(6, 6.6))
    ax.imshow(region)

    # edges are recomputed on the cells of the crop only
    if draw_edges and len(in_crop) > EDGE_K + 1:
        from scipy.spatial import cKDTree
        pts = np.column_stack([lx, ly])
        tree = cKDTree(pts)
        k = min(EDGE_K + 1, len(pts))
        _, idx = tree.query(pts, k=k)
        seen = set()
        segs_x, segs_y = [], []
        for i in range(len(pts)):
            for j in idx[i][1:]:  # skip self
                a, b = (i, j) if i < j else (j, i)
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                segs_x.extend([pts[a, 0], pts[b, 0], np.nan])
                segs_y.extend([pts[a, 1], pts[b, 1], np.nan])
        ax.plot(segs_x, segs_y, color=edge_color, lw=edge_width,
                alpha=edge_alpha, zorder=2)

    # colour by cluster, not by niche: coloured by niche, a niche core is a uniform fill
    if len(in_crop):
        ax.scatter(lx, ly, s=point_size, c=palette[cc], alpha=alpha,
                   edgecolors="black", linewidths=0.3, zorder=3)
    ax.set_xlim(0, size); ax.set_ylim(size, 0)
    ax.axis("off")
    ax.set_title(f"Niche {niche_label}  |  {top3_text}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def render_context(slide_img, center_x, center_y, size, factor, out_path,
                   niche_label):
    """Wider context crop on the same centre, tissue only."""
    big = size * factor
    cx = max(0, center_x - big // 2)
    cy = max(0, center_y - big // 2)
    region = slide_img.read_region((cx, cy), 0, (big, big)).convert("RGB")
    region = region.resize((size, size))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(region)
    # outline the vignette field
    rect_lo = (size // 2 - size // (2 * factor))
    rect_hi = size // factor
    ax.add_patch(plt.Rectangle((rect_lo, rect_lo), rect_hi, rect_hi,
                               fill=False, edgecolor="red", lw=1.2))
    ax.axis("off")
    ax.set_title(f"Niche {niche_label} — context (x{factor} field)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def make_panel(vignette_paths, out_path, niche_label, ncol=6):
    """Montage the individual vignettes of a niche into one panel image."""
    n = len(vignette_paths)
    if n == 0:
        return
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.2, nrow * 2.4))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, vp in zip(axes, vignette_paths):
        img = plt.imread(vp)
        ax.imshow(img)
        ax.axis("off")
    fig.suptitle(f"Niche {niche_label} — representative regions "
                 f"(nuclei colored by cluster)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=95)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Niche representative-region vignettes for annotation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--niche-dir", required=True,
                    help="niche_assignments/ dir (per-slide parquets)")
    ap.add_argument("--wsi-dir", required=True)
    ap.add_argument("--composition", required=True,
                    help="niche_composition.csv (N_niches x K_clusters)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-clusters", type=int, default=50)
    ap.add_argument("--pool-slides", type=int, default=50,
                    help="Number of high-cellularity slides to sample regions from")
    ap.add_argument("--slides-subset-n", type=int, default=None,
                    help="TEST MODE: use only N slides (overrides pool selection)")
    ap.add_argument("--vignettes-per-niche", type=int, default=30)
    ap.add_argument("--purity-radius", type=float, default=512.0,
                    help="Radius (px) for niche-purity computation")
    ap.add_argument("--min-separation", type=float, default=1536.0,
                    help="Min px between vignette centers (avoid overlap/redundancy)")
    ap.add_argument("--point-size", type=float, default=POINT_SIZE,
                    help="Nucleus marker size on the vignette")
    ap.add_argument("--point-alpha", type=float, default=POINT_ALPHA,
                    help="Nucleus opacity (0-1)")
    ap.add_argument("--edge-alpha", type=float, default=EDGE_ALPHA,
                    help="k-NN edge opacity (0-1); lower = fainter")
    ap.add_argument("--edge-width", type=float, default=EDGE_WIDTH,
                    help="k-NN edge line width")
    ap.add_argument("--edge-color", type=str, default=EDGE_COLOR,
                    help="k-NN edge color (matplotlib color name or hex)")
    ap.add_argument("--no-edges", action="store_true",
                    help="Disable k-NN edge overlay entirely")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    if openslide is None:
        print("  ERROR: openslide not available.")
        sys.exit(1)

    t0 = time.time()
    niche_dir = Path(args.niche_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    comp = pd.read_csv(args.composition, index_col=0)
    n_niches = comp.shape[0]
    top3 = {}
    for ni in range(n_niches):
        row = comp.iloc[ni]
        top = row.sort_values(ascending=False).head(3)
        top3[ni] = ", ".join(f"{c} ({v*100:.0f}%)" for c, v in top.items())

    palette = cluster_palette(args.n_clusters)

    wsi_index = build_wsi_index(Path(args.wsi_dir))
    print(f"  WSI indexed: {len(wsi_index)}")

    parquets = sorted(niche_dir.glob("*.parquet"))
    if not parquets:
        print(f"  ERROR: no niche parquets in {niche_dir}")
        sys.exit(1)

    # only the row count is needed, so read a single column
    print(f"  Ranking {len(parquets)} slides by cellularity ...")
    cellularity = []
    for pf in parquets:
        try:
            n = pd.read_parquet(pf, columns=["niche"]).shape[0]
        except Exception:
            n = 0
        cellularity.append((pf, n))
    cellularity.sort(key=lambda t: -t[1])

    if args.slides_subset_n:
        pool = [pf for pf, _ in cellularity[: args.slides_subset_n]]
        print(f"  TEST MODE: {len(pool)} highest-cellularity slides")
    else:
        pool = [pf for pf, _ in cellularity[: args.pool_slides]]
        print(f"  Pool: {len(pool)} highest-cellularity slides")

    manifest_rows = []
    rng = np.random.default_rng(args.seed)

    for ni in range(n_niches):
        niche_out = out_dir / f"niche_{ni:02d}"
        niche_out.mkdir(exist_ok=True)
        candidates = []  # (purity, parquet, x, y)

        for pf in pool:
            slide = pf.stem
            key = _canon_key(slide)
            if key not in wsi_index:
                continue
            df = pd.read_parquet(pf)
            if not {"x", "y", "global_cluster", "niche"}.issubset(df.columns):
                continue
            coords = df[["x", "y"]].values.astype(np.float64)
            niche_ids = df["niche"].values.astype(int)
            tgt_idx, purity = niche_purity(coords, niche_ids, ni, args.purity_radius)
            if len(tgt_idx) == 0:
                continue
            centers = sample_centers(tgt_idx, purity, coords,
                                     n_want=args.vignettes_per_niche,
                                     min_sep=args.min_separation)
            for ci in centers:
                p_here = purity[np.where(tgt_idx == ci)[0][0]]
                candidates.append((float(p_here), pf, int(coords[ci][0]),
                                   int(coords[ci][1])))

        if not candidates:
            print(f"    niche {ni}: no candidate regions in pool, skipping")
            continue

        candidates.sort(key=lambda t: -t[0])
        # round-robin over slides so that no single slide supplies all of a niche's vignettes
        by_slide = defaultdict(list)
        for c in candidates:
            by_slide[c[1]].append(c)
        picked = []
        slides_cycle = list(by_slide.keys())
        while len(picked) < args.vignettes_per_niche and any(by_slide[s] for s in slides_cycle):
            for s in slides_cycle:
                if by_slide[s]:
                    picked.append(by_slide[s].pop(0))
                    if len(picked) >= args.vignettes_per_niche:
                        break

        vignette_paths = []
        slide_img_cache = {}
        for vi, (p_here, pf, cx, cy) in enumerate(picked):
            slide = pf.stem
            key = _canon_key(slide)
            wsi_path = wsi_index[key]
            if wsi_path not in slide_img_cache:
                try:
                    slide_img_cache[wsi_path] = openslide.OpenSlide(str(wsi_path))
                except Exception as e:
                    print(f"    [WARN] cannot open {wsi_path}: {e}")
                    continue
            slide_img = slide_img_cache[wsi_path]
            df = pd.read_parquet(pf)

            crop_x = max(0, cx - TILE // 2)
            crop_y = max(0, cy - TILE // 2)
            vpath = niche_out / f"niche_{ni:02d}_vignette_{vi:03d}.jpg"
            cpath = niche_out / f"niche_{ni:02d}_context_{vi:03d}.jpg"
            try:
                render_vignette(slide_img, crop_x, crop_y, TILE, df, palette,
                                ni, top3[ni], vpath,
                                point_size=args.point_size, alpha=args.point_alpha,
                                draw_edges=not args.no_edges,
                                edge_alpha=args.edge_alpha, edge_width=args.edge_width,
                                edge_color=args.edge_color)
                render_context(slide_img, cx, cy, TILE, CONTEXT_FACTOR, cpath, ni)
                vignette_paths.append(vpath)
                manifest_rows.append({
                    "niche": ni, "slide": slide, "center_x": cx, "center_y": cy,
                    "purity": round(p_here, 3), "vignette": vpath.name,
                })
            except Exception as e:
                print(f"    [WARN] render failed niche {ni} v{vi}: {e}")

        if vignette_paths:
            make_panel(vignette_paths, out_dir / f"niche_{ni:02d}_panel.jpg", ni)
        print(f"    niche {ni}: {len(vignette_paths)} vignettes "
              f"(top3: {top3[ni]})")

    if manifest_rows:
        pd.DataFrame(manifest_rows).to_csv(out_dir / "vignette_manifest.csv",
                                           index=False)

    dt = time.time() - t0
    print(f"\n  DONE — niche vignettes")
    print(f"  Niches: {n_niches}, vignettes/niche target: {args.vignettes_per_niche}")
    print(f"  Time: {dt:.1f}s ({dt/60:.1f} min)")
    print(f"  Output: {out_dir}")
    print(f"    niche_XX_panel.jpg  = montage per niche (annotation material)")
    print(f"    niche_XX/            = individual vignettes + context crops")


if __name__ == "__main__":
    main()
