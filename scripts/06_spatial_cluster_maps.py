#!/usr/bin/env python3
"""Draw each slide's cells at their coordinates, coloured by cluster with a palette fixed across slides.
Input: per-slide parquets (x, y, global_cluster), optionally a cluster summary CSV to fix the palette.
Output: one PNG map per slide."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import warnings
import gc
import time
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")


def build_global_colormap(n_clusters):
    """Fixed cluster-index to RGBA mapping, so a cluster keeps the same colour on every slide."""
    # qualitative palettes up to 40 clusters; past that nipy_spectral is sampled, so adjacent
    # indices get similar hues (no qualitative palette has that many distinct colours)
    if n_clusters <= 20:
        base_cmap = plt.cm.get_cmap('tab20')
        colors = {i: base_cmap(i % 20) for i in range(n_clusters)}
    elif n_clusters <= 40:
        cmap_a = plt.cm.get_cmap('tab20')
        cmap_b = plt.cm.get_cmap('tab20b')
        colors = {}
        for i in range(n_clusters):
            if i < 20:
                colors[i] = cmap_a(i)
            else:
                colors[i] = cmap_b(i - 20)
    else:
        base_cmap = plt.cm.get_cmap('nipy_spectral')
        colors = {i: base_cmap(i / n_clusters) for i in range(n_clusters)}
    
    return colors


def create_spatial_plot(x_coords, y_coords, labels, slide_name, output_path,
                        global_colors, all_cluster_ids, mode_label, dpi=200):
    """Scatter one slide's cells at their coordinates, coloured by global cluster, and save the figure."""
    fig, ax = plt.subplots(figsize=(16, 12))
    
    unique_labels = np.unique(labels)
    n_clusters_present = len(unique_labels)
    
    for label in sorted(unique_labels):
        mask = labels == label
        count = np.sum(mask)
        color = global_colors.get(label, (0.5, 0.5, 0.5, 1.0))
        # small semi-transparent markers: dense regions read darker instead of saturating
        ax.scatter(
            x_coords[mask], y_coords[mask],
            c=[color], s=1, alpha=0.5, label=f'{label} ({count:,})',
            rasterized=True
        )
    
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'{slide_name}\nSpatial ({mode_label}) - {n_clusters_present} clusters, {len(labels):,} cells')
    
    n_total_clusters = len(all_cluster_ids)
    if n_total_clusters <= 15:
        ncol = 1
    elif n_total_clusters <= 30:
        ncol = 2
    elif n_total_clusters <= 60:
        ncol = 3
    else:
        ncol = 4
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7,
              markerscale=5, frameon=True, ncol=ncol, title='Cluster (count)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Spatial visualization of global cluster projections',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  projected  - every cell of the per-slide parquet, coloured by cluster (default)
  all        - earlier pipeline architecture only: overlays directly-clustered cells with
               perceptron-assigned ones. No current script produces the CSV it needs.

For 'all' mode, provide --sampled-csv.
        """
    )
    parser.add_argument('--projected-dir', required=True,
                        help='Directory containing *_projected.parquet files')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for spatial plots')
    parser.add_argument('--mode', choices=['projected', 'all'], default='projected',
                        help='projected (default) = every cell of the parquet; all = earlier architecture only')
    parser.add_argument('--sampled-csv', default=None,
                        help='Path to the directly-clustered cell CSV of the earlier architecture (required for --mode all)')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Max slides to visualize (for testing)')
    parser.add_argument('--slide', nargs='+', default=None,
                        help='Specific slide name(s) to visualize (e.g., SLIDE_001). '
                             'Matches against *_projected.parquet filenames.')
    parser.add_argument('--cluster-summary', default=None,
                        help='Path to global_cluster_summary.csv for consistent colormap '
                             'without scanning all files. Useful with --slide.')
    parser.add_argument('--dpi', type=int, default=200, help='Plot DPI')
    parser.add_argument('--pattern', default='*_projected.parquet',
                        help='File pattern for projected files')
    
    args = parser.parse_args()
    
    if args.mode == 'all' and args.sampled_csv is None:
        print("Error: --sampled-csv required for --mode all")
        return
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    projected_dir = Path(args.projected_dir)
    projected_files = sorted(projected_dir.glob(args.pattern))
    
    if not projected_files:
        print(f"Error: No files matching '{args.pattern}' in {projected_dir}")
        return
    
    if args.max_files:
        projected_files = projected_files[:args.max_files]
    
    if args.slide:
        slide_set = set(args.slide)
        projected_files = [
            pf for pf in projected_files 
            if pf.stem.replace('_projected', '') in slide_set
        ]
        if not projected_files:
            # fall back to substring matching (partial slide names)
            projected_files_all = sorted(projected_dir.glob(args.pattern))
            projected_files = [
                pf for pf in projected_files_all
                if any(s in pf.stem for s in args.slide)
            ]
        if not projected_files:
            print(f"Error: No projected files found matching slide(s): {args.slide}")
            return
    
    print("=" * 70)
    print("Spatial Visualization - Global Clusters")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Projected slides: {len(projected_files)}")
    if args.slide:
        print(f"Slide filter: {args.slide}")
    print(f"Output: {args.output_dir}")
    
    sampled_by_slide = {}
    
    if args.mode == 'all':
        print(f"\nLoading sampled cells from {args.sampled_csv}...")
        sampled_df = pd.read_csv(args.sampled_csv)
        
        if 'x' not in sampled_df.columns or 'y' not in sampled_df.columns:
            print("  Warning: sampled CSV lacks x/y coordinates.")
            print("  'all' mode requires x/y. Falling back to 'projected' mode.")
            args.mode = 'projected'
        else:
            for slide_name, group in sampled_df.groupby('slide_name'):
                sampled_by_slide[slide_name] = group[['x', 'y', 'global_cluster']].copy()
            
            print(f"  Loaded {len(sampled_df):,} sampled cells from {len(sampled_by_slide)} slides")
            del sampled_df
    
    print("\nScanning cluster IDs for consistent colormap...")
    all_cluster_ids = set()
    
    if args.cluster_summary:
        summary_df = pd.read_csv(args.cluster_summary)
        all_cluster_ids.update(summary_df['global_cluster'].unique())
        print(f"  Loaded {len(all_cluster_ids)} clusters from {args.cluster_summary}")
        del summary_df
    else:
        # scanning every file is slow but keeps the palette consistent across slides
        scan_files = projected_files if args.slide else sorted(projected_dir.glob(args.pattern))
        for pf in scan_files:
            try:
                df = pd.read_parquet(pf, columns=['global_cluster'])
                all_cluster_ids.update(df['global_cluster'].unique())
                del df
            except Exception as e:
                print(f"  Warning: could not read {pf.name}: {e}")
        
        if args.slide:
            print(f"  Note: Colors based on {len(projected_files)} selected slide(s) only.")
            print(f"  Use --cluster-summary for globally consistent colors.")
    
    if args.mode == 'all':
        for group in sampled_by_slide.values():
            all_cluster_ids.update(group['global_cluster'].unique())
    
    all_cluster_ids = sorted(all_cluster_ids)
    n_global_clusters = len(all_cluster_ids)
    print(f"  Found {n_global_clusters} global clusters")
    
    id_to_idx = {cid: idx for idx, cid in enumerate(all_cluster_ids)}
    # one palette over the full cluster set, so a cluster keeps its colour on every map
    base_colors = build_global_colormap(n_global_clusters)
    global_colors = {cid: base_colors[id_to_idx[cid]] for cid in all_cluster_ids}
    
    print(f"\nGenerating spatial plots...")
    total_start = time.time()
    n_success = 0
    n_skipped = 0
    
    for pf in tqdm(projected_files, desc="Visualizing slides"):
        slide_name = pf.stem.replace('_projected', '')
        
        try:
            proj_df = pd.read_parquet(pf)
            
            if 'x' not in proj_df.columns or 'y' not in proj_df.columns:
                print(f"  Skipping {slide_name}: no x/y coordinates")
                n_skipped += 1
                del proj_df
                continue
            
            if args.mode == 'projected':
                x = proj_df['x'].values
                y = proj_df['y'].values
                labels = proj_df['global_cluster'].values
                mode_label = f'projected, {len(labels):,} cells'
                
            else:
                # directly-clustered cells keep their own label, the others the perceptron-assigned one
                
                if slide_name in sampled_by_slide:
                    sampled_slide = sampled_by_slide[slide_name]
                    
                    proj_df = proj_df.reset_index(drop=True)
                    proj_df['_idx'] = proj_df.index
                    
                    # rounded so float coordinates match between the two files
                    proj_df['_xr'] = proj_df['x'].round(6)
                    proj_df['_yr'] = proj_df['y'].round(6)
                    
                    sampled_match = sampled_slide.copy()
                    sampled_match['_xr'] = sampled_match['x'].round(6)
                    sampled_match['_yr'] = sampled_match['y'].round(6)
                    sampled_match = sampled_match.rename(columns={'global_cluster': 'leiden_cluster'})
                    
                    # duplicate coordinates would multiply rows in the merge
                    sampled_match = sampled_match.drop_duplicates(subset=['_xr', '_yr'], keep='first')
                    
                    merged = proj_df.merge(
                        sampled_match[['_xr', '_yr', 'leiden_cluster']],
                        on=['_xr', '_yr'], how='left'
                    )
                    
                    labels = merged['global_cluster'].values.copy()
                    leiden_mask = merged['leiden_cluster'].notna()
                    labels[leiden_mask] = merged.loc[leiden_mask, 'leiden_cluster'].astype(int).values
                    n_leiden = leiden_mask.sum()
                    
                    x = proj_df['x'].values
                    y = proj_df['y'].values
                    
                    mode_label = f'directly clustered:{n_leiden:,} + assigned:{len(labels)-n_leiden:,}'
                    
                    del merged, sampled_match
                else:
                    x = proj_df['x'].values
                    y = proj_df['y'].values
                    labels = proj_df['global_cluster'].values
                    mode_label = f'assigned only (no directly-clustered cells)'
            
            out_file = output_path / f'{slide_name}_spatial_{args.mode}.png'
            create_spatial_plot(
                x, y, labels, slide_name, out_file,
                global_colors, all_cluster_ids, mode_label, dpi=args.dpi
            )
            
            n_success += 1
            del proj_df
            gc.collect()
            
        except Exception as e:
            print(f"  Error on {slide_name}: {e}")
            import traceback
            traceback.print_exc()
            n_skipped += 1
            continue
    
    total_time = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Slides visualized: {n_success}")
    print(f"Slides skipped: {n_skipped}")
    print(f"Time: {total_time/60:.1f} min")
    print(f"Output: {output_path}/")


if __name__ == '__main__':
    main()
