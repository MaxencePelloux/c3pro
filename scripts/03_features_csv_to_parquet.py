#!/usr/bin/env python3
"""Convert per-slide feature CSVs to parquet; step 2 already writes parquet, so this only serves CSV outputs.

Input: --dir, a directory of *_features.csv files. Output: a .parquet beside each, same basename.
Warning: each CSV is deleted once its parquet is written.
"""
import argparse
from pathlib import Path
from multiprocessing import Pool
import pandas as pd


def convert_one(csv_path):
    csv_path = Path(csv_path)
    parquet_path = csv_path.with_suffix('.parquet')
    if parquet_path.exists():
        return (csv_path.name, "skipped")
    try:
        df = pd.read_csv(csv_path)
        df.to_parquet(parquet_path, index=False)
        # the CSV is removed once the parquet exists: each file is hundreds of MB and keeping
        # both would double the size of the feature directory
        csv_path.unlink()
        return (csv_path.name, "ok")
    except Exception as e:
        return (csv_path.name, f"error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True)
    parser.add_argument('--workers', type=int, default=5)
    args = parser.parse_args()

    d = Path(args.dir)
    csvs = sorted(d.glob('*_features.csv'))
    print(f"Found {len(csvs)} CSVs. Using {args.workers} workers.")

    done = 0
    errors = []
    # processes, not threads: CSV parsing holds the GIL, so threads would not overlap
    with Pool(args.workers) as pool:
        for i, (name, status) in enumerate(pool.imap_unordered(convert_one, csvs), 1):
            if status == "ok":
                done += 1
            elif status.startswith("error"):
                errors.append((name, status))
            if i % 20 == 0:
                print(f"[{i}/{len(csvs)}] done={done} errors={len(errors)}")

    print(f"\nFinished. Converted={done}, errors={len(errors)}, total={len(csvs)}")
    for name, status in errors:
        print(f"  {name}: {status}")


if __name__ == '__main__':
    main()
