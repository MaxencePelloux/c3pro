"""Sample spatially separated cells of each cluster across slides and cut level-0 tiles around them.
Input: per-slide parquets (x, y, global_cluster) and a directory of whole-slide images.
Output: JPEG tiles per cluster, one mosaic per cluster and sampling_summary.json."""

import argparse
import json
import math
import random
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from PIL import Image

try:
    import openslide
except ImportError:
    raise ImportError("openslide-python required: pip install openslide-python")


_WSI_INDEX: dict = {}


def _canon_key(name: str) -> str:
    """PATHID + section + subsection key, insensitive to separators, block number, leading zeros and case."""
    s = re.sub(r'_(features|projected)$', '', name.strip().lower())
    s = s.replace(' ', '').replace('_', '')
    m = re.search(r'(\d{2}(?:hi|bg)\d{6})(.*)$', s)
    if not m:
        return re.sub(r'[^0-9a-z]', '', s)
    pathid, rest = m.group(1), m.group(2)
    secs = re.findall(r'([a-z])(\d+)', rest)
    sec = f'{secs[-1][0]}{int(secs[-1][1])}' if secs else ''
    mm = re.search(r'[a-z]\d+-?0*(\d+)$', rest)
    sub = str(int(mm.group(1))) if mm else ''
    return f'{pathid}{sec}{sub}'


def _find_wsi(wsi_dir: Path, slide_name: str) -> Path | None:
    # the WSI directory is indexed by canonical key once, then cached
    dkey = str(wsi_dir)
    if dkey not in _WSI_INDEX:
        exts = ('.ndpi', '.svs', '.tiff', '.tif', '.mrxs')
        cands = [p for p in wsi_dir.rglob('*') if p.suffix.lower() in exts]
        canon: dict = {}
        for p in cands:
            canon.setdefault(_canon_key(p.stem), p)
        _WSI_INDEX[dkey] = {'canon': canon, 'cands': cands}
    ix = _WSI_INDEX[dkey]
    hit = ix['canon'].get(_canon_key(slide_name))
    if hit is not None:
        return hit
    # fallback for names with unexpected prefixes: alphanumeric substring match
    simple = re.sub(r'[^0-9a-z]', '', slide_name.lower()
                    .replace('_features', '').replace('_projected', ''))
    for p in ix['cands']:
        ks = re.sub(r'[^0-9a-z]', '', p.stem.lower())
        if simple and (simple in ks or ks in simple):
            return p
    return None


def _spatial_subsample(xs: np.ndarray, ys: np.ndarray,
                        n: int, min_radius: float,
                        seed: int = 42) -> np.ndarray:
    """Greedily pick up to n points at least min_radius apart, so tiles show distinct structures."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(xs))

    selected_x = []
    selected_y = []
    selected_idx = []

    radius = float(min_radius)
    attempts = 0

    # the radius is halved up to 3 times, so a tightly grouped cluster is not under-sampled
    while len(selected_idx) < n and attempts <= 3:
        remaining = [i for i in order if i not in set(selected_idx)]
        if not remaining:
            break

        for i in remaining:
            cx, cy = float(xs[i]), float(ys[i])
            if selected_x:
                dx = np.array(selected_x) - cx
                dy = np.array(selected_y) - cy
                if np.any(dx * dx + dy * dy < radius * radius):
                    continue
            selected_x.append(cx)
            selected_y.append(cy)
            selected_idx.append(i)
            if len(selected_idx) >= n:
                break

        if len(selected_idx) < n:
            radius /= 2.0
            attempts += 1

    return np.array(selected_idx, dtype=int)


def _extract_tile(slide: openslide.OpenSlide, x: int, y: int,
                  tile_size: int) -> Image.Image | None:
    """Extract a tile_size × tile_size RGB tile centred on (x, y) at level 0."""
    ox = x - tile_size // 2
    oy = y - tile_size // 2
    try:
        region = slide.read_region((ox, oy), 0, (tile_size, tile_size))
        return region.convert('RGB')
    except Exception:
        return None


def _make_mosaic(tiles: list[Image.Image], tile_size: int,
                 ncols: int = 20) -> Image.Image:
    n = len(tiles)
    nrows = math.ceil(n / ncols)
    mosaic = Image.new('RGB', (ncols * tile_size, nrows * tile_size),
                       color=(240, 240, 240))
    for i, tile in enumerate(tiles):
        row, col = divmod(i, ncols)
        mosaic.paste(tile.resize((tile_size, tile_size)), (col * tile_size, row * tile_size))
    return mosaic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--projected-dir', required=True,
                    help='Directory with per-slide {slide}.parquet files '
                         '(columns: x, y, global_cluster)')
    ap.add_argument('--x-col',      default='x',
                    help='Column name for x coordinate in parquet files (default: x)')
    ap.add_argument('--y-col',      default='y',
                    help='Column name for y coordinate in parquet files (default: y)')
    ap.add_argument('--wsi-dir',       required=True,
                    help='Directory containing WSI files (.ndpi / .svs / ...)')
    ap.add_argument('--output-dir',    required=True)
    ap.add_argument('--n-cells',    type=int,   default=1000,
                    help='Target cells per cluster')
    ap.add_argument('--tile-size',  type=int,   default=256,
                    help='Tile size in level-0 pixels')
    ap.add_argument('--min-radius', type=float, default=512,
                    help='Minimum distance (px level-0) between tiles in same slide')
    ap.add_argument('--mosaic-n',   type=int,   default=200,
                    help='Max tiles shown in mosaic grid per cluster')
    ap.add_argument('--jpeg-quality', type=int, default=95,
                    help='JPEG quality for saved tiles')
    ap.add_argument('--seed',       type=int,   default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    proj_dir = Path(args.projected_dir)
    wsi_dir  = Path(args.wsi_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mosaic_dir = out_dir / 'cluster_mosaic'
    mosaic_dir.mkdir(exist_ok=True)

    print(f"\n  Loading projected slides from {proj_dir} ...")
    parquet_files = sorted(proj_dir.glob('*.parquet'))
    print(f"  Found {len(parquet_files)} slides")

    frames = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)[[args.x_col, args.y_col, 'global_cluster']]
        df = df.rename(columns={args.x_col: 'x', args.y_col: 'y'})
        slide_name = pf.stem.replace(' ', '')
        if slide_name.endswith('_features'):
            slide_name = slide_name[:-len('_features')]
        df['slide'] = slide_name
        frames.append(df)

    cells = pd.concat(frames, ignore_index=True)
    cells['global_cluster'] = cells['global_cluster'].astype(int)

    clusters = sorted(cells['global_cluster'].unique())
    print(f"  {len(cells):,} cells  |  {len(clusters)} clusters")

    summary = []

    for cluster_id in clusters:
        cname = f"cluster_{cluster_id:02d}"
        tile_out = out_dir / cname
        tile_out.mkdir(exist_ok=True)

        c_cells = cells[cells['global_cluster'] == cluster_id]
        slides_in_cluster = c_cells['slide'].unique()
        n_slides = len(slides_in_cluster)
        total_in_cluster = len(c_cells)

        print(f"\n  [{cname}]  {total_in_cluster:,} cells  "
              f"across {n_slides} slides")

        # budget proportional to slide cell count, with a floor of one so the phenotype shows
        # wherever it occurs; tile counts therefore do not reflect real abundance
        slide_counts = c_cells.groupby('slide').size()
        raw_budget   = (slide_counts / slide_counts.sum() * args.n_cells).clip(lower=1)
        # rounded, then adjusted so the total is exactly n_cells
        budget = raw_budget.round().astype(int)
        diff   = args.n_cells - budget.sum()
        if diff != 0:
            order = budget.sort_values(ascending=(diff < 0)).index
            for s in order:
                if diff == 0:
                    break
                budget[s] += int(math.copysign(1, diff))
                diff       -= int(math.copysign(1, diff))
        budget = budget.clip(lower=1)

        sampled_records = []
        for slide_name, n_target in budget.items():
            slide_cells = c_cells[c_cells['slide'] == slide_name]
            xs = slide_cells['x'].values
            ys = slide_cells['y'].values

            n_target = min(n_target, len(xs))
            if n_target == 0:
                continue

            idx = _spatial_subsample(xs, ys, n=n_target,
                                     min_radius=args.min_radius,
                                     seed=args.seed)
            for i in idx:
                sampled_records.append({
                    'slide': slide_name,
                    'x':     int(xs[i]),
                    'y':     int(ys[i]),
                })

        print(f"    Sampled {len(sampled_records)} cells "
              f"(target {args.n_cells}, {n_slides} slides)")

        tiles_saved   = 0
        tiles_failed  = 0

        # grouped by slide so each WSI is opened only once
        by_slide: dict[str, list] = {}
        for rec in sampled_records:
            by_slide.setdefault(rec['slide'], []).append(rec)

        collected_by_slide: dict[str, list] = {}

        for slide_name, recs in by_slide.items():
            wsi_path = _find_wsi(wsi_dir, slide_name)
            if wsi_path is None:
                print(f"    [WARN] WSI not found: {slide_name} — skipping "
                      f"{len(recs)} tiles")
                tiles_failed += len(recs)
                continue

            try:
                slide = openslide.OpenSlide(str(wsi_path))
            except Exception as e:
                print(f"    [WARN] Cannot open {wsi_path.name}: {e}")
                tiles_failed += len(recs)
                continue

            for rec in recs:
                tile = _extract_tile(slide, rec['x'], rec['y'], args.tile_size)
                if tile is None:
                    tiles_failed += 1
                    continue

                fname = f"{slide_name}_{rec['x']}_{rec['y']}.jpg"
                tile.save(tile_out / fname, 'JPEG', quality=args.jpeg_quality,
                          subsampling=0)
                tiles_saved += 1
                collected_by_slide.setdefault(slide_name, []).append(tile)

            slide.close()

        # mosaic slots proportional to each slide's tile count, at least one per slide
        if collected_by_slide:
            slide_counts = {s: len(tiles) for s, tiles in collected_by_slide.items()}
            grand = sum(slide_counts.values())
            raw_budget = {s: max(1, round(c / grand * args.mosaic_n))
                          for s, c in slide_counts.items()}
            diff = args.mosaic_n - sum(raw_budget.values())
            for s in sorted(raw_budget, key=lambda x: raw_budget[x],
                             reverse=(diff < 0)):
                if diff == 0:
                    break
                raw_budget[s] += int(math.copysign(1, diff))
                diff -= int(math.copysign(1, diff))

            mosaic_tiles = []
            rng_local = random.Random(args.seed)
            for slide_name, n_pick in raw_budget.items():
                pool = collected_by_slide[slide_name]
                n_pick = min(n_pick, len(pool))
                mosaic_tiles.extend(rng_local.sample(pool, n_pick))
            rng_local.shuffle(mosaic_tiles)  # interleave slides in the grid

            mosaic = _make_mosaic(mosaic_tiles, args.tile_size, ncols=20)
            mosaic_path = mosaic_dir / f"{cname}_mosaic.jpg"
            mosaic.save(mosaic_path, 'JPEG', quality=90, subsampling=0)
            print(f"    Mosaic: {mosaic_path.name}  "
                  f"({len(mosaic_tiles)} tiles from {len(collected_by_slide)} slides, "
                  f"{mosaic.width}×{mosaic.height}px)")

        summary.append({
            'cluster':       int(cluster_id),
            'total_cells':   int(total_in_cluster),
            'n_slides':      int(n_slides),
            'tiles_sampled': int(len(sampled_records)),
            'tiles_saved':   int(tiles_saved),
            'tiles_failed':  int(tiles_failed),
        })
        print(f"    Saved {tiles_saved} tiles  "
              f"({tiles_failed} failed)")

    summary_path = out_dir / 'sampling_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    total_saved = sum(s['tiles_saved'] for s in summary)
    print(f"\n  Done. {total_saved:,} tiles across {len(clusters)} clusters")
    print(f"  Summary: {summary_path}")


if __name__ == '__main__':
    main()
