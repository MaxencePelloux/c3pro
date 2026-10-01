#!/usr/bin/env python3
"""Encode one 224x224 tile per CellViT++ nucleus centroid with UNI2-h (1536-dim embedding).

Input: --wsi_dir (slides) and --cellvit_dir (one cell_detection.json per slide, from step 1).
Output: one <slide>_features.parquet per slide in --output_dir with each cell's embedding and (x, y).
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Iterator
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import warnings
import gc

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import timm
from huggingface_hub import login, hf_hub_download

import openslide

# UNI2-h is a gated model: a Hugging Face token (HF_TOKEN) is needed to download the weights
HUGGING_FACE_TOKEN = os.environ.get('HF_TOKEN', '')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    tile_size: int = 224
    batch_size: int = 256
    chunk_size: int = 5000
    num_io_workers: int = 8
    device: str = 'cuda'
    use_amp: bool = False


class ParallelTileExtractor:
    """Read cell-centred tiles from a WSI with a thread pool."""
    
    def __init__(self, slide_path: str, tile_size: int = 224, num_workers: int = 8):
        self.slide_path = slide_path
        self.tile_size = tile_size
        self.half_tile = tile_size // 2
        self.num_workers = num_workers
        
        self.slide = openslide.OpenSlide(slide_path)
        self.width, self.height = self.slide.dimensions
        self._local_slides = {}
    
    def _get_thread_slide(self) -> openslide.OpenSlide:
        import threading
        tid = threading.current_thread().ident
        if tid not in self._local_slides:
            self._local_slides[tid] = openslide.OpenSlide(self.slide_path)
        return self._local_slides[tid]
    
    def _extract_single(self, cell: Dict) -> Tuple[np.ndarray, int, int, int, int]:
        slide = self._get_thread_slide()
        
        cx, cy = int(cell['centroid'][0]), int(cell['centroid'][1])
        cell_type = cell.get('type', 0)
        cell_idx = cell.get('_idx', 0)
        
        x = cx - self.half_tile
        y = cy - self.half_tile
        
        x = max(0, min(x, self.width - self.tile_size))
        y = max(0, min(y, self.height - self.tile_size))
        
        # at level 0 a 224 px tile covers the nucleus and its surrounding tissue, so every
        # embedding includes peri-cellular context
        try:
            tile = slide.read_region((x, y), 0, (self.tile_size, self.tile_size))
            tile_np = np.array(tile.convert('RGB'))
        except Exception:
            tile_np = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)
        
        return tile_np, cx, cy, cell_type, cell_idx
    
    def extract_batch_parallel(self, cells: List[Dict]) -> List[Tuple]:
        for i, cell in enumerate(cells):
            cell['_idx'] = i
        
        results = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._extract_single, cell) for cell in cells]
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.warning(f"Tile extraction failed: {e}")
        
        return results
    
    def close(self):
        self.slide.close()
        for slide in self._local_slides.values():
            try:
                slide.close()
            except:
                pass
        self._local_slides.clear()


class UNI2Extractor:
    """UNI2-h encoder built from the official MahmoodLab weights and architecture."""
    
    def __init__(self, device: str = 'cuda', use_amp: bool = False):
        self.device = device
        self.use_amp = use_amp
        self.feature_dim = 1536
        
        logger.info("Initializing UNI2-h model...")
        logger.info("Using Official MahmoodLab Method 2: Manual download + load_state_dict")
        
        self._hf_login()
        
        self.model = self._load_uni2h()
        self.model = self.model.to(device)
        self.model.eval()
        
        if device == 'cuda' or device.startswith('cuda:'):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        
        # ImageNet mean/std normalisation, as expected by UNI2-h
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        
        logger.info("Warming up GPU...")
        with torch.no_grad():
            dummy = torch.randn(4, 3, 224, 224, device=device)
            _ = self.model(dummy)
        
        logger.info(f"✓ UNI2-h ready on {device} (dim={self.feature_dim})")
    
    def _hf_login(self):
        try:
            token = HUGGING_FACE_TOKEN or os.environ.get('HF_TOKEN')
            if token:
                login(token=token, add_to_git_credential=False)
                logger.info("✓ HuggingFace login successful")
            else:
                logger.warning("No HuggingFace token provided")
        except Exception as e:
            logger.warning(f"HuggingFace login issue: {e}")
    
    def _load_uni2h(self) -> nn.Module:
        """Download the weights once, then build UNI2-h with the MahmoodLab timm settings."""
        
        local_dir = Path.home() / ".cache" / "uni2-h"
        local_dir.mkdir(parents=True, exist_ok=True)
        weights_path = local_dir / "pytorch_model.bin"
        
        if not weights_path.exists():
            logger.info("Downloading UNI2-h weights...")
            hf_hub_download(
                "MahmoodLab/UNI2-h",
                filename="pytorch_model.bin",
                local_dir=str(local_dir),
                force_download=False
            )
            logger.info(f"✓ Downloaded to {weights_path}")
        else:
            logger.info(f"Using cached weights: {weights_path}")
        
        logger.info("Creating vit_giant_patch14_224 architecture...")
        
        # `vit_giant_patch14_224` is only the timm skeleton; the settings below (depth 24,
        # embed_dim 1536, SwiGLU MLP, 8 register tokens) are what make it UNI2-h.
        timm_kwargs = {
            'model_name': 'vit_giant_patch14_224',
            'img_size': 224,
            'patch_size': 14,
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5,
            'embed_dim': 1536,
            'mlp_ratio': 2.66667 * 2,
            'num_classes': 0,
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked,
            'act_layer': torch.nn.SiLU,
            'reg_tokens': 8,
            'dynamic_img_size': True
        }
        
        model = timm.create_model(pretrained=False, **timm_kwargs)
        
        logger.info("Loading UNI2-h weights...")
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        
        logger.info("✓ Successfully loaded UNI2-h (1536-dim)!")
        return model
    
    def preprocess_batch(self, tiles: np.ndarray) -> torch.Tensor:
        """Convert uint8 (B, H, W, 3) tiles to a normalised float tensor on the device."""
        x = torch.from_numpy(tiles).to(self.device, non_blocking=True)
        x = x.permute(0, 3, 1, 2).float() / 255.0
        x = (x - self.mean) / self.std
        return x
    
    @torch.no_grad()
    def extract_batch(self, tiles: np.ndarray) -> np.ndarray:
        x = self.preprocess_batch(tiles)
        
        if self.use_amp:
            with autocast():
                features = self.model(x)
        else:
            features = self.model(x)
        
        # if the model returns a token sequence (B, N, D), keep the CLS token
        if features.dim() == 3:
            features = features[:, 0]
        
        return features.cpu().numpy()


def load_cells(json_path: str) -> List[Dict]:
    """Load the cell list from a CellViT++ cell_detection.json."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data.get('cells', [])


def filter_valid_cells(cells: List[Dict], width: int, height: int, tile_size: int = 224) -> List[Dict]:
    """Keep cells whose whole tile fits inside the slide (the clamping in _extract_single then never applies)."""
    half = tile_size // 2
    valid = []
    
    for cell in cells:
        cx, cy = cell['centroid']
        cx, cy = int(cx), int(cy)
        
        if (cx - half >= 0 and cx + half <= width and
            cy - half >= 0 and cy + half <= height):
            valid.append(cell)
    
    return valid


def chunks(lst: List, n: int) -> Iterator[List]:
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def process_slide_streaming(
    slide_path: Path,
    json_path: Path,
    output_path: Path,
    extractor: UNI2Extractor,
    config: Config
) -> Optional[Dict]:
    """Encode all in-bounds cells of one slide, write its parquet and return run stats (None if no cells)."""
    slide_name = slide_path.stem
    start_time = time.time()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {slide_name}")
    
    cells = load_cells(str(json_path))
    total_cells = len(cells)
    logger.info(f"Total cells: {total_cells:,}")
    
    if total_cells == 0:
        logger.warning("No cells found, skipping")
        return None
    
    tile_extractor = ParallelTileExtractor(
        str(slide_path),
        tile_size=config.tile_size,
        num_workers=config.num_io_workers
    )
    
    valid_cells = filter_valid_cells(
        cells, 
        tile_extractor.width, 
        tile_extractor.height,
        config.tile_size
    )
    logger.info(f"Valid cells (within bounds): {len(valid_cells):,}")
    
    if len(valid_cells) == 0:
        logger.warning("No valid cells for tiling")
        tile_extractor.close()
        return None
    
    all_features = []
    all_metadata = []
    
    pbar = tqdm(total=len(valid_cells), desc=f"  Extracting", unit="cells")
    
    # chunking bounds memory: only one chunk of tiles is held at a time
    for chunk_cells in chunks(valid_cells, config.chunk_size):
        tile_results = tile_extractor.extract_batch_parallel(chunk_cells)
        
        tiles = np.stack([r[0] for r in tile_results], axis=0)
        coords_x = [r[1] for r in tile_results]
        coords_y = [r[2] for r in tile_results]
        cell_types = [r[3] for r in tile_results]
        
        for batch_start in range(0, len(tiles), config.batch_size):
            batch_end = min(batch_start + config.batch_size, len(tiles))
            
            batch_tiles = tiles[batch_start:batch_end]
            batch_x = coords_x[batch_start:batch_end]
            batch_y = coords_y[batch_start:batch_end]
            batch_types = cell_types[batch_start:batch_end]
            
            features = extractor.extract_batch(batch_tiles)
            all_features.append(features)
            
            for i in range(len(batch_x)):
                all_metadata.append({
                    'x': batch_x[i],
                    'y': batch_y[i],
                    'cell_type': batch_types[i]
                })
            
            pbar.update(len(batch_tiles))
        
        del tiles, tile_results
        gc.collect()
    
    pbar.close()
    tile_extractor.close()
    
    features = np.vstack(all_features)
    
    feature_cols = [f'f{i}' for i in range(features.shape[1])]
    
    df_meta = pd.DataFrame(all_metadata)
    df_meta.insert(0, 'tile_id', [f"{slide_name}_{i:07d}" for i in range(len(df_meta))])
    df_meta.insert(1, 'slide', slide_name)
    
    df_features = pd.DataFrame(features, columns=feature_cols)
    
    df = pd.concat([df_meta, df_features], axis=1)
    
    # parquet, not CSV: a 1536-column CSV runs to ~1 GB per slide and is slow to read back
    df.to_parquet(output_path, index=False)
    
    elapsed = time.time() - start_time
    rate = len(valid_cells) / elapsed if elapsed > 0 else 0
    
    logger.info(f"✓ Saved: {output_path.name}")
    logger.info(f"  Cells: {len(valid_cells):,} | Features: {features.shape[1]}D")
    logger.info(f"  Time: {elapsed:.1f}s ({rate:.0f} cells/sec)")
    
    del features, all_features, df
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        'slide_name': slide_name,
        'total_cells': total_cells,
        'valid_cells': len(valid_cells),
        'feature_dim': extractor.feature_dim,
        'time_seconds': elapsed,
        'cells_per_second': rate
    }


def find_pairs(cellvit_dir: Path, wsi_dir: Path, wsi_ext: str) -> List[Tuple[Path, Path]]:
    """Find matching slide and cell_detection.json pairs."""
    pairs = []
    
    for case_dir in sorted(cellvit_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        
        slide_name = case_dir.name
        json_file = case_dir / 'cell_detection.json'
        slide_file = wsi_dir / f"{slide_name}.{wsi_ext}"
        
        if json_file.exists() and slide_file.exists():
            pairs.append((slide_file, json_file))
    
    return pairs


def load_checkpoint(path: Path) -> Dict:
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return {'completed': [], 'failed': []}


def save_checkpoint(path: Path, data: Dict):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='CellViT Cell-Centered UNI2-h Feature Extraction (Streaming)'
    )
    
    parser.add_argument('--wsi_dir', type=str, required=True)
    parser.add_argument('--cellvit_dir', type=str, required=True)
    parser.add_argument('--wsi_extension', type=str, default='ndpi')
    
    parser.add_argument('--output_dir', type=str, required=True)
    
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--chunk_size', type=int, default=5000)
    parser.add_argument('--io_workers', type=int, default=8)
    parser.add_argument('--gpu', type=int, default=0)
    
    parser.add_argument('--resume', action='store_true')
    
    args = parser.parse_args()
    
    wsi_dir = Path(args.wsi_dir)
    cellvit_dir = Path(args.cellvit_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = Config(
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        num_io_workers=args.io_workers,
        device=f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu',
        use_amp=False
    )
    
    logger.info("="*60)
    logger.info("CellViT Cell-Centered Feature Extraction (UNI2-h)")
    logger.info("="*60)
    logger.info(f"WSI dir: {wsi_dir}")
    logger.info(f"CellViT dir: {cellvit_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Config: batch={config.batch_size}, chunk={config.chunk_size}, "
                f"io_workers={config.num_io_workers}, device={config.device}")
    
    pairs = find_pairs(cellvit_dir, wsi_dir, args.wsi_extension)
    logger.info(f"Found {len(pairs)} slide-JSON pairs")
    
    if len(pairs) == 0:
        logger.error("No valid pairs found!")
        sys.exit(1)
    
    checkpoint_file = output_dir / 'checkpoint.json'
    checkpoint = load_checkpoint(checkpoint_file) if args.resume else {'completed': [], 'failed': []}
    
    if args.resume:
        logger.info(f"Resuming: {len(checkpoint['completed'])} completed, {len(checkpoint['failed'])} failed")
    
    extractor = UNI2Extractor(device=config.device, use_amp=config.use_amp)
    
    results = []
    total_cells = 0
    total_time = 0
    
    for i, (slide_path, json_path) in enumerate(pairs):
        slide_name = slide_path.stem
        
        logger.info(f"\n[{i+1}/{len(pairs)}] {slide_name}")
        
        if slide_name in checkpoint['completed']:
            logger.info("  Skipping (already completed)")
            continue
        
        # with --resume, an existing output over 1 kB counts as done; smaller files are redone
        output_parquet = output_dir / f"{slide_name}_features.parquet"
        if args.resume and output_parquet.exists() and output_parquet.stat().st_size > 1000:
            logger.info("  Skipping (output exists)")
            checkpoint['completed'].append(slide_name)
            continue

        try:
            result = process_slide_streaming(
                slide_path=slide_path,
                json_path=json_path,
                output_path=output_parquet,
                extractor=extractor,
                config=config
            )
            
            if result:
                results.append(result)
                checkpoint['completed'].append(slide_name)
                total_cells += result['valid_cells']
                total_time += result['time_seconds']
            else:
                checkpoint['failed'].append(slide_name)
                
        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            checkpoint['failed'].append(slide_name)
        
        if (i + 1) % 5 == 0:
            save_checkpoint(checkpoint_file, checkpoint)
    
    save_checkpoint(checkpoint_file, checkpoint)
    
    logger.info(f"\n{'='*60}")
    logger.info("EXTRACTION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Slides processed: {len(results)}/{len(pairs)}")
    logger.info(f"Total cells: {total_cells:,}")
    logger.info(f"Total time: {total_time/3600:.2f} hours")
    if total_time > 0:
        logger.info(f"Average rate: {total_cells/total_time:.0f} cells/sec")
    logger.info(f"Failed: {len(checkpoint['failed'])}")
    
    if checkpoint['failed']:
        logger.warning(f"Failed slides: {checkpoint['failed']}")
    
    summary = {
        'total_slides': len(pairs),
        'processed': len(results),
        'failed': len(checkpoint['failed']),
        'total_cells': total_cells,
        'total_time_hours': total_time / 3600,
        'feature_dim': extractor.feature_dim,
        'config': {
            'batch_size': config.batch_size,
            'chunk_size': config.chunk_size,
            'io_workers': config.num_io_workers,
            'tile_size': config.tile_size
        },
        'slides': results
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nOutput: {output_dir}")


if __name__ == '__main__':
    main()
