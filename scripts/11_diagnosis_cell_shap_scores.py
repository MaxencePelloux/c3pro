#!/usr/bin/env python3
"""Project the slide-level SHAP attributions of the diagnostic model (step 10) onto cell coordinates.
Input: per-slide projected parquets (step 05) and the saved model. Output: tile_scores.csv (one row
per cell), slide-level SHAP values and a per-slide summary, used by step 12 to draw the maps."""

import numpy as np
import pandas as pd
import json
import time
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

try:
    import shap
    import joblib
except ImportError:
    print("Error: shap and joblib are required")
    print("Install with: pip install shap joblib")
    sys.exit(1)


def load_model(model_path):
    """Load the trained XGBoost model and its feature names."""
    print(f"Loading model: {model_path}")
    model_data = joblib.load(model_path)

    if isinstance(model_data, dict):
        model = model_data['model']
        feature_names = model_data.get('feature_names', None)
        training_info = model_data.get('training_info', {})
        print(f"  Model loaded (dict format)")
        print(f"  Features: {len(feature_names) if feature_names else 'unknown'}")
        print(f"  Train AUC: {training_info.get('train_auc', 'N/A'):.3f}")
        print(f"  Test AUC: {training_info.get('test_auc', 'N/A'):.3f}")
    else:
        model = model_data
        feature_names = model.get_booster().feature_names
        print(f"  Model loaded (direct format)")

    return model, feature_names


def load_faiss_parquets(projected_dir, slides_subset=None):
    """Read the projected parquets; return (slide x cluster proportions, per-slide cell tables)."""
    projected_dir = Path(projected_dir)
    parquet_files = sorted(projected_dir.glob("*.parquet"))
    print(f"\nFound {len(parquet_files)} parquet files")

    if slides_subset:
        print(f"Filtering to {len(slides_subset)} specified slides")

    slide_cluster_counts = {}
    slide_cell_counts = {}
    slide_cell_data = {}
    all_cluster_ids = set()
    total_cells = 0
    slides_loaded = 0

    for pf in tqdm(parquet_files, desc="Reading parquets"):
        slide_name = pf.stem.replace('_features', '').replace('_projected', '').strip()

        if slides_subset and slide_name not in slides_subset:
            continue

        try:
            df = pd.read_parquet(pf)
            # accept either x/y or centroid_x/centroid_y
            if 'centroid_x' not in df.columns and 'x' in df.columns:
                df = df.rename(columns={'x': 'centroid_x', 'y': 'centroid_y'})
            # drop unused columns: every slide's cells stay in memory until the end
            keep = ['global_cluster', 'centroid_x', 'centroid_y']
            if 'tile_id' in df.columns:
                keep.insert(0, 'tile_id')
            if 'vote_confidence' in df.columns:
                keep.append('vote_confidence')
            df = df[keep]
            clusters = df['global_cluster'].values
            unique, counts = np.unique(clusters, return_counts=True)

            slide_cluster_counts[slide_name] = dict(zip(unique.astype(int), counts.astype(int)))
            slide_cell_counts[slide_name] = len(df)
            slide_cell_data[slide_name] = df
            all_cluster_ids.update(unique.astype(int))
            total_cells += len(df)
            slides_loaded += 1
        except Exception as e:
            print(f"  Warning: Failed to read {pf.name}: {e}")

    all_cluster_ids = sorted(all_cluster_ids)
    print(f"\n  {slides_loaded} slides loaded, {total_cells:,} total cells, {len(all_cluster_ids)} clusters")

    print("Building slide-level proportion matrix...")
    proportion_data = []
    slide_names = []

    for slide_name in sorted(slide_cluster_counts.keys()):
        counts = slide_cluster_counts[slide_name]
        total = slide_cell_counts[slide_name]
        row = {}
        for cid in all_cluster_ids:
            row[f"Cluster_{cid}"] = counts.get(cid, 0) / total
        proportion_data.append(row)
        slide_names.append(slide_name)

    slide_proportions = pd.DataFrame(proportion_data, index=slide_names)
    print(f"  Proportions matrix: {slide_proportions.shape}")

    return slide_proportions, slide_cell_data


def compute_shap_values(model, feature_names, slide_proportions):
    """Slide x feature SHAP values for the positive class."""
    print(f"\nComputing SHAP values for {len(slide_proportions)} slides...")

    missing_cols = set(feature_names) - set(slide_proportions.columns)
    extra_cols = set(slide_proportions.columns) - set(feature_names)

    if missing_cols:
        print(f"  Adding {len(missing_cols)} missing columns (zero-filled)")
        for col in missing_cols:
            slide_proportions[col] = 0.0

    if extra_cols:
        print(f"  Dropping {len(extra_cols)} extra columns not in model")

    # trees address features by position, so a column order that differs from training would
    # silently attribute values to the wrong phenotypes
    slide_proportions = slide_proportions[feature_names]

    X = slide_proportions.values.astype(np.float32)

    # recent xgboost saves base_score as a bracketed string, which shap cannot parse
    booster = model.get_booster()
    config = json.loads(booster.save_config())
    bs = config['learner']['learner_model_param']['base_score']
    if isinstance(bs, str) and bs.startswith('['):
        config['learner']['learner_model_param']['base_score'] = bs.strip('[]')
        booster.load_config(json.dumps(config))

    # exact Shapley values from the tree structure, rather than a sampling approximation
    explainer = shap.TreeExplainer(model)
    # shap_values() rather than __call__: for binary xgboost models the Explanation API indexes the
    # class axis inconsistently across shap/xgboost versions; this returns [neg, pos] or one array
    raw = explainer.shap_values(X)
    if isinstance(raw, list):
        raw = raw[1]

    shap_df = pd.DataFrame(
        raw,
        index=slide_proportions.index,
        columns=feature_names
    )

    print(f"  SHAP values shape: {shap_df.shape}")
    print(f"  SHAP range: [{shap_df.values.min():.4f}, {shap_df.values.max():.4f}]")
    print(f"  SHAP mean: {shap_df.values.mean():.4f}")

    return shap_df


def map_shap_to_cells(slide_cell_data, shap_df, feature_names):
    """Give each cell its cluster's SHAP value on that slide (a phenotype-level, not per-cell, value)."""
    print(f"\nMapping SHAP values to {sum(len(d) for d in slide_cell_data.values()):,} cells...")

    shap_lookup = {}
    for slide_name in shap_df.index:
        for feat in feature_names:
            cluster_id = int(feat.split('_')[1])
            shap_lookup[(slide_name, cluster_id)] = shap_df.loc[slide_name, feat]

    all_results = []

    for slide_name in tqdm(sorted(slide_cell_data.keys()), desc="Mapping SHAP to cells"):
        df = slide_cell_data[slide_name]

        if slide_name not in shap_df.index:
            print(f"  Warning: {slide_name} not in SHAP results, skipping")
            continue

        slide_shap = shap_df.loc[slide_name]
        cluster_to_shap = {}
        for feat in feature_names:
            cid = int(feat.split('_')[1])
            cluster_to_shap[cid] = slide_shap[feat]

        shap_values = df['global_cluster'].map(cluster_to_shap).fillna(0.0)

        result = pd.DataFrame({
            'slide_name': slide_name,
            'centroid_x': df['centroid_x'].values,
            'centroid_y': df['centroid_y'].values,
            'global_cluster': df['global_cluster'].values,
            'shap_value': shap_values.values,
        })
        if 'tile_id' in df.columns:
            result.insert(1, 'tile_id', df['tile_id'].values)
        if 'vote_confidence' in df.columns:
            result['vote_confidence'] = df['vote_confidence'].values
        all_results.append(result)

    results_df = pd.concat(all_results, ignore_index=True)
    print(f"  Generated scores for {len(results_df):,} cells across {results_df['slide_name'].nunique()} slides")

    return results_df


def save_results(results_df, shap_df, output_dir):
    """Write per-cell scores, slide-level SHAP values and a per-slide summary."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tile_scores_file = output_path / 'tile_scores.csv'
    print(f"\nSaving tile-level scores to {tile_scores_file}...")
    results_df.to_csv(tile_scores_file, index=False)
    size_mb = tile_scores_file.stat().st_size / 1e6
    print(f"  Size: {size_mb:.0f} MB")

    slide_shap_file = output_path / 'slide_level_shap.csv'
    print(f"Saving slide-level SHAP values to {slide_shap_file}...")
    shap_df.to_csv(slide_shap_file)

    print(f"\n{'='*60}")
    print("SPATIAL SCORES SUMMARY")
    print(f"{'='*60}")
    print(f"  Total cells: {len(results_df):,}")
    print(f"  Slides: {results_df['slide_name'].nunique()}")
    print(f"  Clusters: {results_df['global_cluster'].nunique()}")
    print(f"  SHAP range: [{results_df['shap_value'].min():.4f}, {results_df['shap_value'].max():.4f}]")
    print(f"  SHAP mean: {results_df['shap_value'].mean():.4f}")
    print(f"  SHAP std: {results_df['shap_value'].std():.4f}")

    slide_stats = results_df.groupby('slide_name')['shap_value'].agg(['count', 'mean', 'std', 'min', 'max'])
    slide_stats.to_csv(output_path / 'per_slide_shap_summary.csv')
    print(f"  Per-slide summary saved ({len(slide_stats)} slides)")

    slide_means = results_df.groupby('slide_name')['shap_value'].mean().sort_values()
    print(f"\n  Most anti-cancer slides (mean SHAP):")
    for s, v in slide_means.head(5).items():
        print(f"    {s}: {v:.4f}")
    print(f"\n  Most pro-cancer slides (mean SHAP):")
    for s, v in slide_means.tail(5).items():
        print(f"    {s}: {v:.4f}")


def main():
    parser = argparse.ArgumentParser(description='C3PRO step 11 — project the diagnostic attributions onto cell coordinates')
    parser.add_argument('--projected-dir', required=True,
                        help='Path to the projected_slides directory written by step 05')
    parser.add_argument('--model-path', required=True,
                        help='Path to slide_level_model.joblib')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for tile_scores.csv')
    parser.add_argument('--slides-subset',
                        help='Comma-separated slide names, or path to file with one slide per line')

    args = parser.parse_args()

    start_time = time.time()

    print("=" * 60)
    print("SPATIAL SHAP SCORE GENERATION")
    print("=" * 60)
    print(f"Projected dir: {args.projected_dir}")
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_dir}")

    slides_subset = None
    if args.slides_subset:
        subset_path = Path(args.slides_subset)
        if subset_path.is_file():
            slides_subset = [line.strip() for line in open(subset_path) if line.strip()]
            print(f"Slides subset: {len(slides_subset)} from file {subset_path}")
        else:
            slides_subset = [s.strip() for s in args.slides_subset.split(',')]
            print(f"Slides subset: {len(slides_subset)} specified")

    model, feature_names = load_model(args.model_path)

    slide_proportions, slide_cell_data = load_faiss_parquets(
        args.projected_dir, slides_subset
    )

    shap_df = compute_shap_values(model, feature_names, slide_proportions)

    results_df = map_shap_to_cells(slide_cell_data, shap_df, feature_names)

    save_results(results_df, shap_df, args.output_dir)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} min")
    print("Done!")


if __name__ == '__main__':
    main()
