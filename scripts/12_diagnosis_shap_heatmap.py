#!/usr/bin/env python3
"""Overlay diagnostic SHAP attributions (step 11) on whole-slide images as smoothed, tissue-masked heatmaps.
Inputs: tile_scores.csv, the WSI(s) and optional GeoJSON tissue masks; outputs one PNG per slide.
Attributions are per phenotype, so the map shows where influential phenotypes lie, not which single cell mattered."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
from pathlib import Path
import argparse
import sys
import json
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: opencv-python not found. GeoJSON masking will not be available.")
    print("Install with: pip install opencv-python-headless")

try:
    import openslide
    WSI_AVAILABLE = True
except ImportError:
    WSI_AVAILABLE = False
    print("Error: openslide-python is required for reading whole-slide images")
    print("Install with: pip install openslide-python")
    print("Note: You may also need to install the OpenSlide C library")
    print("  Ubuntu/Debian: sudo apt-get install openslide-tools")
    print("  macOS: brew install openslide")
    print("  Windows: Download from https://openslide.org/download/")
    sys.exit(1)

# WSI thumbnails exceed PIL's decompression-bomb limit
Image.MAX_IMAGE_PIXELS = None

def create_custom_colormaps():
    """Build the custom diverging colormaps used for the SHAP heatmaps."""
    rdylbu_colors = [
        '#A50026',
        '#D73027',
        '#F46D43',
        '#FDAE61',
        '#FEE090',
        '#FFFFBF',
        '#D1E5F0',
        '#92C5DE',
        '#4393C3',
        '#2166AC',
        '#2166AC'
    ]
    
    rdylbu_cmap = ListedColormap(rdylbu_colors, name='RdYlBu_custom')
    
    rdylbu_colors_reversed = rdylbu_colors[::-1]
    rdylbu_cmap_r = ListedColormap(rdylbu_colors_reversed, name='RdYlBu_custom_r')
    
    rdylbu_contrast_colors = [
        '#B2182B',
        '#E08214',
        '#FDB863',
        '#FEE0B6',
        '#F7F7F7',
        '#D1E5F0',
        '#92C5DE',
        '#4393C3',
        '#2166AC'
    ]
    
    rdylbu_contrast_cmap = ListedColormap(rdylbu_contrast_colors, name='RdYlBu_contrast')
    rdylbu_contrast_cmap_r = ListedColormap(rdylbu_contrast_colors[::-1], name='RdYlBu_contrast_r')
    
    return {
        'RdYlBu_custom': rdylbu_cmap,
        'RdYlBu_custom_r': rdylbu_cmap_r,
        'RdYlBu_contrast': rdylbu_contrast_cmap,
        'RdYlBu_contrast_r': rdylbu_contrast_cmap_r
    }

CUSTOM_COLORMAPS = create_custom_colormaps()

def get_colormap(colormap_name):
    """Return one of the custom colormaps, or a matplotlib colormap by name."""
    if colormap_name in CUSTOM_COLORMAPS:
        print(f"Using custom colormap: {colormap_name}")
        return CUSTOM_COLORMAPS[colormap_name]
    
    try:
        cmap = plt.get_cmap(colormap_name)
        print(f"Using standard colormap: {colormap_name}")
        return cmap
    except ValueError:
        print(f"Warning: Colormap '{colormap_name}' not found. Using default 'jet'.")
        return plt.get_cmap('jet')

def find_wsi_file(slide_name, wsi_dir):
    """Find the WSI file for a given slide name, searching recursively in subdirectories."""
    wsi_path = Path(wsi_dir)

    extensions = ['.svs', '.tiff', '.tif', '.ndpi', '.vms', '.vmu', '.scn', '.mrxs', '.czi']

    for ext in extensions:
        candidate = wsi_path / f"{slide_name}{ext}"
        if candidate.exists():
            return candidate

    # file names may carry a prefix or suffix around the slide name
    for ext in extensions:
        candidates = list(wsi_path.glob(f"{slide_name}*{ext}"))
        if candidates:
            return candidates[0]
        candidates = list(wsi_path.glob(f"*{slide_name}*{ext}"))
        if candidates:
            return candidates[0]

    # slides may sit in per-class subdirectories
    for ext in extensions:
        candidates = list(wsi_path.rglob(f"{slide_name}{ext}"))
        if candidates:
            return candidates[0]
        candidates = list(wsi_path.rglob(f"{slide_name}*{ext}"))
        if candidates:
            return candidates[0]
        candidates = list(wsi_path.rglob(f"*{slide_name}*{ext}"))
        if candidates:
            return candidates[0]

    return None

def find_geojson_file(slide_name, geojson_dir):
    """Find the GeoJSON file whose name contains the slide name (file names may carry extra prefixes)."""
    if not geojson_dir:
        return None
        
    geojson_path = Path(geojson_dir)
    
    if not geojson_path.exists():
        print(f"Warning: GeoJSON directory does not exist: {geojson_dir}")
        return None
    
    candidate = geojson_path / f"{slide_name}.geojson"
    if candidate.exists():
        return candidate
    
    for ext in ['.json', '.geojson']:
        candidate = geojson_path / f"{slide_name}{ext}"
        if candidate.exists():
            return candidate
    
    print(f"Searching for slide '{slide_name}' in complex filenames...")
    
    for ext in ['.json', '.geojson']:
        all_files = list(geojson_path.glob(f"*{ext}"))
        
        for file_path in all_files:
            filename = file_path.stem
            
            if slide_name in filename:
                print(f"Found matching GeoJSON: {file_path.name}")
                return file_path
    
    # fallback: match on the first dash-separated part of the slide name
    slide_parts = slide_name.replace('-', ' ').split()
    if len(slide_parts) >= 2:
        main_part = slide_parts[0]
        
        for ext in ['.json', '.geojson']:
            all_files = list(geojson_path.glob(f"*{ext}"))
            
            for file_path in all_files:
                filename = file_path.stem
                
                if main_part in filename and len(main_part) >= 6:  # short fragments would match too many files
                    print(f"Found potential match by main identifier '{main_part}': {file_path.name}")
                    return file_path
    
    print(f"No GeoJSON file found for slide '{slide_name}' in directory '{geojson_dir}'")
    return None

def create_mask_from_geojson(geojson_path, grid_dims, downsampling):
    """Rasterise the GeoJSON polygons into a binary tissue mask at grid resolution."""
    if not CV2_AVAILABLE:
        print("Warning: OpenCV not available, cannot create GeoJSON mask")
        return None
        
    grid_height, grid_width = grid_dims
    mask = np.zeros((grid_height, grid_width), dtype=np.uint8)

    try:
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
            
        features = geojson_data.get('features', [])
        if not features:
            print(f"Warning: No features found in GeoJSON file {geojson_path}")
            return (mask > 0).astype(np.float32)
            
    except FileNotFoundError:
        print(f"Warning: GeoJSON file not found at {geojson_path}")
        return None
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON format in {geojson_path}")
        return None

    polygons_drawn = 0
    for feature in features:
        geom = feature.get('geometry', {})
        if geom.get('type') == 'Polygon':
            coords = geom.get('coordinates', [])
            if coords:
                # outer ring only; holes are ignored
                exterior_ring = coords[0]
                
                # level-0 pixel coordinates -> grid cells
                scaled_coords = []
                for x, y in exterior_ring:
                    scaled_x = int(x / downsampling)
                    scaled_y = int(y / downsampling)
                    scaled_x = max(0, min(scaled_x, grid_width - 1))
                    scaled_y = max(0, min(scaled_y, grid_height - 1))
                    scaled_coords.append([scaled_x, scaled_y])
                
                if len(scaled_coords) >= 3:
                    polygon = np.array(scaled_coords, dtype=np.int32)
                    cv2.fillPoly(mask, [polygon], 255)
                    polygons_drawn += 1

    print(f"Created tissue mask from GeoJSON: {polygons_drawn} polygons drawn")
    
    return (mask > 0).astype(np.float32)

def create_heatmap_and_mask_grids(df_slide, slide_dims, downsampling):
    """Build a max-|SHAP| score grid and a binary tissue-presence grid."""
    grid_width = int(np.ceil(slide_dims[0] / downsampling))
    grid_height = int(np.ceil(slide_dims[1] / downsampling))
    
    score_grid = np.zeros((grid_height, grid_width), dtype=np.float32)
    mask_grid = np.zeros((grid_height, grid_width), dtype=np.float32)

    coords_x = df_slide['centroid_x'].values
    coords_y = df_slide['centroid_y'].values
    scores = df_slide['shap_value'].values
    
    # centroids are level-0 coordinates; a wrong downsampling factor silently shifts the map
    grid_x = np.clip((coords_x / downsampling).astype(int), 0, grid_width - 1)
    grid_y = np.clip((coords_y / downsampling).astype(int), 0, grid_height - 1)

    for i in range(len(scores)):
        gx, gy = grid_x[i], grid_y[i]
        
        score_abs = abs(scores[i])
        score_grid[gy, gx] = max(score_grid[gy, gx], score_abs)
        
        mask_grid[gy, gx] = 1.0
            
    return score_grid, mask_grid

def create_score_and_density_grids(df_slide, slide_dims, downsampling):
    """Build max-|SHAP| score, tile-density and tissue-mask grids."""
    grid_width = int(np.ceil(slide_dims[0] / downsampling))
    grid_height = int(np.ceil(slide_dims[1] / downsampling))
    
    score_grid = np.zeros((grid_height, grid_width), dtype=np.float32)
    # density separates truly discriminant areas from merely crowded ones
    density_grid = np.zeros((grid_height, grid_width), dtype=np.float32)
    mask_grid = np.zeros((grid_height, grid_width), dtype=np.float32)

    coords_x = df_slide['centroid_x'].values
    coords_y = df_slide['centroid_y'].values
    scores = df_slide['shap_value'].values
    
    grid_x = np.clip((coords_x / downsampling).astype(int), 0, grid_width - 1)
    grid_y = np.clip((coords_y / downsampling).astype(int), 0, grid_height - 1)

    for i in range(len(scores)):
        gx, gy = grid_x[i], grid_y[i]
        
        score_abs = abs(scores[i])
        score_grid[gy, gx] = max(score_grid[gy, gx], score_abs)
        density_grid[gy, gx] += 1
        mask_grid[gy, gx] = 1.0
            
    return score_grid, density_grid, mask_grid

def plot_smoothed_heatmap_with_mask(score_grid, mask_grid, slide_thumb, slide_name, output_file, alpha=0.5, sigma=25.0, cmap_name='seismic'):
    """Render and save a smoothed, tissue-masked heatmap of absolute SHAP scores."""
    print(f"Rendering the smoothed, masked heatmap (sigma={sigma})...")

    # smoothing the mask too widens it a little, so the heatmap fades out at the tissue edge
    smoothed_scores = gaussian_filter(score_grid, sigma=sigma)
    smoothed_mask = gaussian_filter(mask_grid, sigma=sigma)

    tissue_mask = smoothed_mask > 1.0

    smoothed_scores[~tissue_mask] = 0

    # exclude zeros so the background does not skew the color scale
    non_zero_scores = smoothed_scores[smoothed_scores > 0]
    if non_zero_scores.size > 0:
        vmin = np.percentile(non_zero_scores, 1)
        vmax = np.percentile(non_zero_scores, 99)
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    else:
        norm = Normalize(vmin=0, vmax=1)

    cmap = get_colormap(cmap_name)
    heatmap_rgba = cmap(norm(smoothed_scores))
    
    # alpha scales with the score, so weak areas stay semi-transparent
    alpha_channel = np.clip(smoothed_scores / (vmax + 1e-6), 0, 1)
    heatmap_rgba[:, :, 3] = alpha_channel * alpha
    heatmap_rgba[~tissue_mask, 3] = 0

    fig, ax = plt.subplots(figsize=(20, 12), dpi=300)
    ax.imshow(slide_thumb, aspect='auto')
    
    heatmap_img = Image.fromarray((heatmap_rgba * 255).astype(np.uint8))
    heatmap_resized = heatmap_img.resize(slide_thumb.size, Image.Resampling.LANCZOS)
    
    ax.imshow(heatmap_resized, aspect='auto')
    
    ax.set_title(f"Heatmap d'Importance SHAP : {slide_name}", fontsize=16)
    ax.axis('off')
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.02)
    cbar.set_label("Importance (Valeur Absolue de SHAP)", rotation=270, labelpad=20, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    print(f"Masked heatmap written to {output_file}")
    plt.close()

def plot_density_normalized_heatmap(score_grid, density_grid, mask_grid, slide_thumb, slide_name, output_file, alpha=0.7, sigma=25.0, cmap_name='seismic'):
    """Render a heatmap whose scores are normalised by local cell density."""
    print(f"Rendering the density-normalised heatmap (sigma={sigma})...")

    smoothed_scores = gaussian_filter(score_grid, sigma=sigma)
    smoothed_density = gaussian_filter(density_grid, sigma=sigma)
    smoothed_mask = gaussian_filter(mask_grid, sigma=sigma)
    
    tissue_mask = smoothed_mask > 0.01

    normalized_heatmap = np.zeros_like(smoothed_scores)
    normalized_heatmap[tissue_mask] = np.divide(
        smoothed_scores[tissue_mask], 
        smoothed_density[tissue_mask] + 1e-6
    )

    non_zero_values = normalized_heatmap[normalized_heatmap > 0]
    if non_zero_values.size > 0:
        vmax = np.percentile(non_zero_values, 99)  # percentile is robust to outliers
        norm = Normalize(vmin=0, vmax=vmax, clip=True)
    else:
        norm = Normalize(vmin=0, vmax=1)

    cmap = get_colormap(cmap_name)
    heatmap_rgba = cmap(norm(normalized_heatmap))
    
    alpha_channel = np.zeros_like(normalized_heatmap)
    alpha_channel[tissue_mask] = alpha
    heatmap_rgba[:, :, 3] = alpha_channel

    fig, ax = plt.subplots(figsize=(20, 12), dpi=300)
    ax.imshow(slide_thumb, aspect='auto')
    
    heatmap_img = Image.fromarray((heatmap_rgba * 255).astype(np.uint8))
    heatmap_resized = heatmap_img.resize(slide_thumb.size, Image.Resampling.LANCZOS)
    
    ax.imshow(heatmap_resized, aspect='auto')
    
    ax.set_title(f"Density-normalised importance heatmap: {slide_name}", fontsize=16)
    ax.axis('off')
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.02)
    cbar.set_label("Importance / density", rotation=270, labelpad=20, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    print(f"Density-normalised heatmap written to {output_file}")
    plt.close()

def create_raw_image_with_geojson(slide_thumb, slide_dims, downsampling=16, geojson_path=None, contour_color=(255, 0, 0), contour_thickness=2):
    """Draw the GeoJSON polygon contours over the raw H&E thumbnail."""
    print(f"Rendering the raw H&E image with its GeoJSON segmentation...")
    
    if slide_thumb.mode != 'RGB':
        slide_thumb = slide_thumb.convert('RGB')
    
    result_image = slide_thumb.copy()
    
    if not geojson_path or not CV2_AVAILABLE:
        print("Warning: GeoJSON path not provided or OpenCV not available. Returning raw image without contours.")
        return result_image
    
    try:
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
            
        features = geojson_data.get('features', [])
        if not features:
            print(f"Warning: No features found in GeoJSON file {geojson_path}")
            return result_image
            
    except FileNotFoundError:
        print(f"Warning: GeoJSON file not found at {geojson_path}")
        return result_image
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON format in {geojson_path}")
        return result_image
    
    img_array = np.array(result_image)
    
    grid_width = int(np.ceil(slide_dims[0] / downsampling))
    grid_height = int(np.ceil(slide_dims[1] / downsampling))
    
    img_resized = cv2.resize(img_array, (grid_width, grid_height))
    
    polygons_drawn = 0
    for feature in features:
        geom = feature.get('geometry', {})
        if geom.get('type') == 'Polygon':
            coords = geom.get('coordinates', [])
            if coords:
                # outer ring only; holes are ignored
                exterior_ring = coords[0]
                
                # level-0 pixel coordinates -> grid cells
                scaled_coords = []
                for x, y in exterior_ring:
                    scaled_x = int(x / downsampling)
                    scaled_y = int(y / downsampling)
                    scaled_x = max(0, min(scaled_x, grid_width - 1))
                    scaled_y = max(0, min(scaled_y, grid_height - 1))
                    scaled_coords.append([scaled_x, scaled_y])
                
                if len(scaled_coords) >= 3:
                    polygon = np.array(scaled_coords, dtype=np.int32)
                    cv2.polylines(img_resized, [polygon], isClosed=True, color=contour_color, thickness=contour_thickness)
                    polygons_drawn += 1
    
    print(f"Drawn {polygons_drawn} polygon contours on raw image")
    
    final_img_array = cv2.resize(img_resized, slide_thumb.size)
    
    result_image = Image.fromarray(final_img_array)
    
    return result_image

def create_tissue_mask_from_thumbnail(slide_thumb, grid_height, grid_width, sat_threshold=15):
    """Boolean tissue mask at grid resolution, from thumbnail saturation (background is near-white)."""
    thumb_array = np.array(slide_thumb.convert('RGB'))
    thumb_resized = np.array(Image.fromarray(thumb_array).resize((grid_width, grid_height), Image.Resampling.LANCZOS))

    hsv = np.zeros_like(thumb_resized, dtype=np.float32)
    r, g, b = thumb_resized[:,:,0].astype(np.float32), thumb_resized[:,:,1].astype(np.float32), thumb_resized[:,:,2].astype(np.float32)

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    diff = cmax - cmin

    saturation = np.zeros_like(cmax)
    nonzero = cmax > 0
    saturation[nonzero] = (diff[nonzero] / cmax[nonzero]) * 255

    tissue_mask = saturation > sat_threshold

    # fill holes without dilating the borders
    from scipy.ndimage import binary_fill_holes
    tissue_mask = binary_fill_holes(tissue_mask)

    print(f"WSI-based tissue mask: {tissue_mask.sum():,} / {tissue_mask.size:,} pixels ({100*tissue_mask.sum()/tissue_mask.size:.1f}% tissue)")
    return tissue_mask

def create_avg_heatmap(df_slide, slide_dims, slide_thumb, downsampling=16, sigma=25.0, percentile_clip=99.0, colormap='RdYlBu_custom', alpha=0.7, geojson_path=None):
    """Mean positive-SHAP heatmap, smoothed and masked to tissue (GeoJSON mask if given, else thumbnail)."""
    print(f"Rendering the MEAN heatmap (sigma={sigma}, percentile={percentile_clip})...")
    grid_width = int(np.ceil(slide_dims[0] / downsampling))
    grid_height = int(np.ceil(slide_dims[1] / downsampling))

    # only evidence for cancer is drawn: negative contributions are clipped to zero
    df_pro_cancer = df_slide.copy()
    df_pro_cancer['shap_value'] = df_pro_cancer['shap_value'].clip(lower=0)

    score_sum_grid = np.zeros((grid_height, grid_width), dtype=np.float64)
    tile_count_grid = np.zeros((grid_height, grid_width), dtype=np.int32)

    coords_x = df_pro_cancer['centroid_x'].values
    coords_y = df_pro_cancer['centroid_y'].values
    scores = df_pro_cancer['shap_value'].values
    
    grid_x = np.clip((coords_x / downsampling).astype(int), 0, grid_width - 1)
    grid_y = np.clip((coords_y / downsampling).astype(int), 0, grid_height - 1)

    for i in range(len(scores)):
        score_sum_grid[grid_y[i], grid_x[i]] += scores[i]
        tile_count_grid[grid_y[i], grid_x[i]] += 1
    
    avg_grid = np.zeros_like(score_sum_grid)
    non_zero_mask = tile_count_grid > 0
    avg_grid[non_zero_mask] = score_sum_grid[non_zero_mask] / tile_count_grid[non_zero_mask]

    # smoothing turns noisy per-cell means into a readable map; sigma=25 is a working value,
    # not a calibrated one (too large erases focal structures, too small keeps the noise)
    smoothed_grid = gaussian_filter(avg_grid, sigma=sigma)

    if geojson_path and CV2_AVAILABLE:
        print(f"Using GeoJSON mask from: {geojson_path}")
        tissue_mask = create_mask_from_geojson(geojson_path, (grid_height, grid_width), downsampling)
        if tissue_mask is not None:
            print("Successfully created precise tissue mask from GeoJSON")
        else:
            print("Failed to create GeoJSON mask, falling back to WSI-based mask")
            tissue_mask = None
    else:
        tissue_mask = None

    if tissue_mask is None:
        tissue_mask = create_tissue_mask_from_thumbnail(slide_thumb, grid_height, grid_width)

    smoothed_grid[tissue_mask == 0] = 0

    non_zero_scores = smoothed_grid[smoothed_grid > 0]
    if non_zero_scores.size == 0: 
        print("No positive score after smoothing; rendering a neutral heatmap")
        vmax = 1.0
    else:
        vmax = np.percentile(non_zero_scores, percentile_clip)
    
    norm = Normalize(vmin=0, vmax=vmax, clip=True)

    cmap = get_colormap(colormap)
    heatmap_rgba = cmap(norm(smoothed_grid))
    
    # alpha over the whole tissue, including zero-score areas
    alpha_channel = tissue_mask.astype(float) * alpha
    heatmap_rgba[:, :, 3] = alpha_channel

    heatmap_img = Image.fromarray((heatmap_rgba * 255).astype(np.uint8))
    heatmap_resized = heatmap_img.resize(slide_thumb.size, Image.Resampling.LANCZOS)
    
    final_image = Image.alpha_composite(slide_thumb.convert('RGBA'), heatmap_resized)
    return final_image.convert('RGB')

def create_heatmap_for_slide(df_scores, slide_name, wsi_path, output_file, method='avg', downsampling=16, alpha=0.5, sigma=25.0, background_quality=4, normalize_by_density=False, percentile_clip=99.0, colormap='RdYlBu_custom', geojson_dir=None, raw_img=False, contour_color=(255, 0, 0), contour_thickness=2):
    """Render one slide's heatmap (or raw image with contours) and save it to output_file."""
    print(f"\nProcessing slide: {slide_name} (method: {method})")
    
    if not raw_img:
        df_slide = df_scores[df_scores['slide_name'] == slide_name].copy()
        if df_slide.empty:
            print(f"Error: no data for '{slide_name}'")
            return False
        print(f"Found {len(df_slide):,} tiles for this slide")
    else:
        print("raw-img mode: rendering the raw H&E image with its GeoJSON segmentation")
    
    geojson_path = None
    if geojson_dir:
        geojson_path = find_geojson_file(slide_name, geojson_dir)
        if geojson_path:
            print(f"GeoJSON found: {geojson_path}")
        else:
            print(f"No GeoJSON found for {slide_name} in {geojson_dir}")
    
    try:
        slide = openslide.OpenSlide(str(wsi_path))
        slide_dims = slide.level_dimensions[0]
        
        thumb_downsampling = background_quality
        thumb_size = (slide_dims[0] // thumb_downsampling, slide_dims[1] // thumb_downsampling)
        slide_thumb = slide.get_thumbnail(thumb_size)
        
        if raw_img:
            final_image = create_raw_image_with_geojson(slide_thumb, slide_dims, 
                                                      downsampling=downsampling, 
                                                      geojson_path=geojson_path,
                                                      contour_color=contour_color,
                                                      contour_thickness=contour_thickness)
            final_image.save(output_file, 'PNG')
            print(f"Raw H&E image with segmentation written to {output_file}")
            
        elif method == 'avg':
            final_image = create_avg_heatmap(df_slide, slide_dims, slide_thumb, 
                                           downsampling=downsampling, 
                                           sigma=sigma, 
                                           percentile_clip=percentile_clip,
                                           colormap=colormap, 
                                           alpha=alpha,
                                           geojson_path=geojson_path)
            final_image.save(output_file, 'PNG')
            print(f"Heatmap (method: {method}) written to {output_file}")
            
        elif method == 'legacy':
            if normalize_by_density:
                score_grid, density_grid, mask_grid = create_score_and_density_grids(df_slide, slide_dims, downsampling)
                plot_density_normalized_heatmap(score_grid, density_grid, mask_grid, slide_thumb, slide_name, output_file, alpha=alpha, sigma=sigma, cmap_name=colormap)
            else:
                score_grid, mask_grid = create_heatmap_and_mask_grids(df_slide, slide_dims, downsampling)
                plot_smoothed_heatmap_with_mask(score_grid, mask_grid, slide_thumb, slide_name, output_file, alpha=alpha, sigma=sigma, cmap_name=colormap)
                
        else:
            print(f"Error: unknown method '{method}'. Choose 'avg' or 'legacy'.")
            slide.close()
            return False
        
        slide.close()
        return True
        
    except Exception as e:
        print(f"Erreur lors du traitement de la lame {slide_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate spatial heatmaps from tile scores")
    parser.add_argument('--scores-csv', 
                       help='Path to the tile_scores.csv file (required for heatmap modes, not needed for --raw-img)')
    parser.add_argument('--slide-name', required=True, 
                       help='Slide name(s) to process (comma-separated for multiple)')
    parser.add_argument('--wsi-path', 
                       help='Path to a single WSI file (use with single slide)')
    parser.add_argument('--wsi-dir', 
                       help='Directory containing WSI files (use with multiple slides)')
    parser.add_argument('--output-image', 
                       help='Path to save a single output heatmap image')
    parser.add_argument('--output-dir', 
                       help='Directory to save multiple output images')
    parser.add_argument('--downsampling', type=int, default=16, 
                       help='Factor to downsample the WSI for the grid (default: 16 for higher resolution)')
    parser.add_argument('--alpha', type=float, default=0.6, 
                       help='Transparency of heatmap overlay (0.0-1.0, default: 0.6)')
    parser.add_argument('--side-by-side', action='store_true', 
                       help='Generate side-by-side comparison instead of high-resolution single heatmap')
    parser.add_argument('--sigma', type=float, default=25.0, 
                       help='Gaussian smoothing factor (default: 25.0, higher = more smoothing)')
    parser.add_argument('--background-quality', type=int, default=4, 
                       help='Background image quality factor (default: 4, lower = higher quality)')
    parser.add_argument('--normalize-by-density', action='store_true', 
                       help='Normalise the heatmap by local cell density')
    parser.add_argument('--method', type=str, default='avg', choices=['avg', 'legacy'],
                       help="Score aggregation: 'avg' (sensitivity, default) or 'legacy' (earlier method)")
    parser.add_argument('--percentile-clip', type=float, default=99.0,
                       help='Percentile at which maximum values are clipped (default: 99.0)')
    parser.add_argument('--cmap', type=str, default='RdYlBu_custom',
                       help='Colormap for the heatmap (default: RdYlBu_custom). '
                            'Standard: seismic, YlOrRd, hot, jet, plasma, viridis. '
                            'Custom: RdYlBu_custom, RdYlBu_custom_r, RdYlBu_contrast, RdYlBu_contrast_r')
    parser.add_argument('--geojson-dir', 
                       help='Directory containing GeoJSON files for tissue masks (optional, enables precise tissue masking)')
    parser.add_argument('--raw-img', action='store_true',
                       help='Generate raw HES image with GeoJSON segmentation overlay (no heatmap)')
    parser.add_argument('--contour-color', type=str, default='255,0,0',
                       help='Color for GeoJSON contours in raw-img mode (R,G,B format, default: 255,0,0 for red)')
    parser.add_argument('--contour-thickness', type=int, default=2,
                       help='Thickness of GeoJSON contours in raw-img mode (default: 2)')
    
    args = parser.parse_args()
    
    try:
        contour_color = tuple(map(int, args.contour_color.split(',')))
        if len(contour_color) != 3 or not all(0 <= c <= 255 for c in contour_color):
            raise ValueError("Invalid color values")
    except (ValueError, AttributeError):
        print("Warning: Invalid contour color format. Using default red (255,0,0)")
        contour_color = (255, 0, 0)
    
    print("Spatial Heatmap Generation")
    print("=" * 50)
    
    slide_names = [s.strip() for s in args.slide_name.split(',')]
    multiple_slides = len(slide_names) > 1
    high_res_mode = not args.side_by_side
    
    print(f"Processing {len(slide_names)} slide(s): {slide_names}")
    print(f"Downsampling factor: {args.downsampling}")
    print(f"Background quality factor: {args.background_quality}")
    print(f"Gaussian smoothing sigma: {args.sigma}")
    print(f"Heatmap transparency: {args.alpha}")
    print(f"Density normalization: {'Enabled' if args.normalize_by_density else 'Disabled'}")
    print(f"Method: {args.method}")
    print(f"Colormap: {args.cmap}")
    print(f"Percentile clip: {args.percentile_clip}")
    print(f"GeoJSON masking: {'Enabled' if args.geojson_dir else 'Disabled'}")
    if args.geojson_dir:
        print(f"GeoJSON directory: {args.geojson_dir}")
    print(f"Raw image mode: {'Enabled' if args.raw_img else 'Disabled'}")
    if args.raw_img:
        print(f"Contour color: {contour_color}")
        print(f"Contour thickness: {args.contour_thickness}")
    print(f"Output mode: {'Raw HES image with GeoJSON' if args.raw_img else ('Side-by-side comparison' if args.side_by_side else ('Density-normalized heatmap' if args.normalize_by_density else f'{args.method.upper()} heatmap'))}")
    
    if not args.raw_img and not args.scores_csv:
        print("Error: --scores-csv is required for heatmap modes (not needed for --raw-img)")
        sys.exit(1)
    
    if multiple_slides:
        if not args.wsi_dir:
            print("Error: --wsi-dir required when processing multiple slides")
            sys.exit(1)
        if not args.output_dir:
            print("Error: --output-dir required when processing multiple slides")
            sys.exit(1)
    else:
        if not args.wsi_path:
            print("Error: --wsi-path required when processing a single slide")
            sys.exit(1)
        if not args.output_image:
            print("Error: --output-image required when processing a single slide")
            sys.exit(1)
    
    if not args.raw_img:
        print(f"\nLoading scores from: {args.scores_csv}")
        try:
            df_scores = pd.read_csv(args.scores_csv)
            print(f"Loaded scores for {len(df_scores):,} tiles from {df_scores['slide_name'].nunique()} slides")
        except Exception as e:
            print(f"Error loading scores CSV: {e}")
            sys.exit(1)
    else:
        print("\nRaw image mode: skipping scores loading")
        df_scores = None
    
    if not args.raw_img:
        available_slides = set(df_scores['slide_name'].unique())
        requested_slides = set(slide_names)
        missing_slides = requested_slides - available_slides
        
        if missing_slides:
            print(f"Warning: The following slides were not found in the scores CSV:")
            for slide in missing_slides:
                print(f"  • {slide}")
            slide_names = [s for s in slide_names if s in available_slides]
            if not slide_names:
                print("Error: No valid slides to process")
                sys.exit(1)
            print(f"Proceeding with {len(slide_names)} available slides")
    
    success_count = 0
    
    if multiple_slides:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for slide_name in tqdm(slide_names, desc="Processing slides"):
            wsi_path = find_wsi_file(slide_name, args.wsi_dir)
            if not wsi_path:
                print(f"Warning: WSI file not found for slide {slide_name}")
                continue
            
            output_file = output_dir / f"{slide_name}_shap_heatmap.png"
            
            if create_heatmap_for_slide(df_scores, slide_name, wsi_path, output_file, method=args.method, downsampling=args.downsampling, alpha=args.alpha, sigma=args.sigma, background_quality=args.background_quality, normalize_by_density=args.normalize_by_density, percentile_clip=args.percentile_clip, colormap=args.cmap, geojson_dir=args.geojson_dir, raw_img=args.raw_img, contour_color=contour_color, contour_thickness=args.contour_thickness):
                success_count += 1
    
    else:
        slide_name = slide_names[0]
        if create_heatmap_for_slide(df_scores, slide_name, args.wsi_path, args.output_image, method=args.method, downsampling=args.downsampling, alpha=args.alpha, sigma=args.sigma, background_quality=args.background_quality, normalize_by_density=args.normalize_by_density, percentile_clip=args.percentile_clip, colormap=args.cmap, geojson_dir=args.geojson_dir, raw_img=args.raw_img, contour_color=contour_color, contour_thickness=args.contour_thickness):
            success_count = 1
    
    print(f"\n" + "=" * 50)
    print(f"Processing complete!")
    print(f"Successfully processed: {success_count}/{len(slide_names)} slides")
    
    if success_count > 0:
        if multiple_slides:
            if args.raw_img:
                print(f"Raw HES images with GeoJSON saved to: {args.output_dir}")
            else:
                print(f"Heatmaps saved to: {args.output_dir}")
        else:
            if args.raw_img:
                print(f"Raw HES image with GeoJSON saved to: {args.output_image}")
            else:
                print(f"Heatmap saved to: {args.output_image}")

if __name__ == "__main__":
    main()
