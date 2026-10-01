#!/usr/bin/env python3
"""Detect cell nuclei in whole-slide images (NDPI/SVS) with CellViT++ (SAM backbone).

Input: a WSI file, or a folder of WSIs with --batch. Output: per-slide CellViT++ results
(GeoJSON nuclei contours and centroids) under --outdir.
"""

import sys
import argparse


# taxonomy defaults to 'binary': nuclei are not typed here, cell identity comes later
# from clustering the cell embeddings
def run_cellvit(wsi_path, outdir='./cellvit_results', model='SAM', 
                taxonomy='binary', overlap=128, batch_size=8,
                use_amp=True, gpu=0):
    """Run CellViT++ on one WSI; `overlap` is only logged, CellViT++ uses its own default."""
    # Force the OpenSlide reader: pathopatch's default backend mishandles the Hamamatsu NDPI format.
    import pathopatch.patch_extraction.dataset as ds
    from pathopatch.wsi_interfaces.openslide_deepzoom import DeepZoomGeneratorOS
    from openslide import OpenSlide

    def _set_hardware_openslide(self):
        self.logger.debug("Using OpenSlide (forced)")
        self.deepzoomgenerator = DeepZoomGeneratorOS
        self.image_loader = OpenSlide
        self.slide_metadata_loader = OpenSlide

    ds.LivePatchWSIDataset._set_hardware = _set_hardware_openslide
    
    # CellViT++ only exposes a CLI entry point, so a programmatic call means simulating a command line.
    sys.argv = [
        'cellvit',
        '--model', model,
        '--nuclei_taxonomy', taxonomy,
        '--outdir', outdir,
        '--geojson',
        '--gpu', str(gpu),
        '--batch_size', str(batch_size),
        '--ray_worker', '1',
    ]

    if use_amp:
        sys.argv.append('--enforce_amp')

    sys.argv.extend([
        'process_wsi',
        '--wsi_path', wsi_path
    ])
    
    print("CellViT++ Optimized Inference")
    print(f"WSI: {wsi_path}")
    print(f"Output: {outdir}")
    print(f"Model: {model}")
    print(f"Taxonomy: {taxonomy}")
    print(f"Overlap: {overlap} (no effect on detection — see --overlap help)")
    print(f"Batch size: {batch_size}")
    print(f"AMP: {use_amp}")
    print(f"GPU: {gpu}")
    
    from cellvit.detect_cells import main
    main()


def run_cellvit_batch(wsi_filelist, outdir='./cellvit_results',
                      model='SAM', taxonomy='binary', batch_size=8,
                      use_amp=True, gpu=0):
    """Run CellViT++ on the slides listed in a CSV (`path` column), loading the model once."""
    # Force the OpenSlide reader: pathopatch's default backend mishandles the Hamamatsu NDPI format.
    import pathopatch.patch_extraction.dataset as ds
    from pathopatch.wsi_interfaces.openslide_deepzoom import DeepZoomGeneratorOS
    from openslide import OpenSlide

    def _set_hardware_openslide(self):
        self.logger.debug("Using OpenSlide (forced)")
        self.deepzoomgenerator = DeepZoomGeneratorOS
        self.image_loader = OpenSlide
        self.slide_metadata_loader = OpenSlide

    ds.LivePatchWSIDataset._set_hardware = _set_hardware_openslide

    # CellViT++ only exposes a CLI entry point, so a programmatic call means simulating a command line.
    sys.argv = [
        'cellvit',
        '--model', model,
        '--nuclei_taxonomy', taxonomy,
        '--outdir', outdir,
        '--geojson',
        '--gpu', str(gpu),
        '--batch_size', str(batch_size),
        '--ray_worker', '1',
    ]

    if use_amp:
        sys.argv.append('--enforce_amp')

    sys.argv.extend([
        'process_dataset',
        '--wsi_filelist', wsi_filelist,
    ])

    print("CellViT++ Batch Processing")
    print(f"Filelist: {wsi_filelist}")
    print(f"Output: {outdir}")

    from cellvit.detect_cells import main
    main()


def main():
    parser = argparse.ArgumentParser(
        description='Run CellViT++ with optimized parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single WSI with default settings
  python 01_nuclei_detection_cellvit.py /path/to/slide.ndpi
  
  # With PanNuke classification
  python 01_nuclei_detection_cellvit.py /path/to/slide.ndpi --taxonomy pannuke
  
  # Batch processing
  python 01_nuclei_detection_cellvit.py /path/to/folder --batch --extension .ndpi
  
  # Larger inference batch
  python 01_nuclei_detection_cellvit.py /path/to/slide.ndpi --batch_size 16

Overlap recommendations:
  64  - Fast, may split nuclei at patch edges
  128 - Balanced (recommended)
  256 - Best quality, slower
        """
    )
    
    parser.add_argument('input', help='Path to WSI file or folder (for batch)')
    parser.add_argument('--outdir', '-o', default='./cellvit_results', 
                        help='Output directory')
    parser.add_argument('--model', choices=['SAM', 'HIPT'], default='SAM',
                        help='Model type (default: SAM)')
    parser.add_argument('--taxonomy', '-t', default='binary',
                        choices=['binary', 'pannuke', 'consep', 'lizard', 'ocelot', 'panoptils'],
                        help='Nuclei taxonomy (default: binary)')
    parser.add_argument('--overlap', type=int, default=128,
                        help='Patch overlap in pixels — accepted for interface compatibility but has no effect on detection; CellViT++ uses its internal default.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable automatic mixed precision')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (default: 0)')
    parser.add_argument('--batch', action='store_true',
                        help='Process all WSIs in folder')
    parser.add_argument('--extension', default='.ndpi',
                        help='WSI file extension for batch mode (default: .ndpi)')
    parser.add_argument('--slide-list', default=None,
                        help='Path to text file with slide names (one per line, no extension). '
                             'Only these slides will be processed in batch mode.')
    parser.add_argument('--chunk', type=int, default=None,
                        help='Chunk index (0-based) for parallel jobs')
    parser.add_argument('--n-chunks', type=int, default=None,
                        help='Total number of chunks for parallel jobs')

    args = parser.parse_args()

    if args.batch:
        import os
        import json
        import tempfile
        from pathlib import Path

        wsi_folder = Path(args.input)
        ext = args.extension if args.extension.startswith('.') else f'.{args.extension}'

        all_slides = sorted(wsi_folder.glob(f'*{ext}'))

        if args.slide_list:
            with open(args.slide_list) as f:
                names = {line.strip() for line in f if line.strip()}
            all_slides = [s for s in all_slides if s.stem.strip() in names or s.name.strip() in names]
            print(f"Filtered to {len(all_slides)} slides from slide list.")

        if args.chunk is not None and args.n_chunks is not None:
            all_slides = [s for i, s in enumerate(all_slides) if i % args.n_chunks == args.chunk]
            print(f"Chunk {args.chunk}/{args.n_chunks}: processing {len(all_slides)} slides.")

        outdir = Path(args.outdir)
        processed_json = outdir / 'processed_files.json'
        processed = set()
        if processed_json.exists():
            with open(processed_json) as f:
                processed = set(json.load(f))

        remaining = [s for s in all_slides if s.name.strip() not in processed and s.stem.strip() not in processed]
        skipped = len(all_slides) - len(remaining)
        if skipped:
            print(f"[SKIP] {skipped} slides already processed.")

        if not remaining:
            print("All slides already processed.")
        else:
            print(f"Processing {len(remaining)} slides using native batch mode (model loaded once).")
            import tempfile, csv, os
            tmpdir = os.environ.get('TMPDIR', tempfile.gettempdir())
            filelist_path = os.path.join(tmpdir, f'chunk_{args.chunk}_filelist.csv')
            with open(filelist_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['path'])
                for s in remaining:
                    writer.writerow([str(s)])
            print(f"[INFO] Filelist written to {filelist_path} ({len(remaining)} slides)")
            run_cellvit_batch(
                wsi_filelist=filelist_path,
                outdir=args.outdir,
                model=args.model,
                taxonomy=args.taxonomy,
                batch_size=args.batch_size,
                use_amp=not args.no_amp,
                gpu=args.gpu
            )

    else:
        run_cellvit(
            wsi_path=args.input,
            outdir=args.outdir,
            model=args.model,
            taxonomy=args.taxonomy,
            overlap=args.overlap,
            batch_size=args.batch_size,
            use_amp=not args.no_amp,
            gpu=args.gpu
        )


if __name__ == '__main__':
    main()
