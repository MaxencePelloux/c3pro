#!/usr/bin/env python3
"""Build fold_indices.pkl, the 25 outer CV folds (5 repeats x 5 folds) shared by every benchmarked model.

Inputs: --fold-assignments CSV (slide_name, repeat, fold) and --cancer-csv, whose row order defines the indices.
Output: --output-dir/fold_indices.pkl, keyed by (repeat, fold), with train/test slide names and int64 indices.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def build(fold_assignments: Path, cancer_csv: Path, dedup_warn: bool = True) -> dict:
    """Return {(repeat, fold): {'train','test','train_idx','test_idx'}}."""
    fa = pd.read_csv(fold_assignments)
    cd = pd.read_csv(cancer_csv)

    for col in ("slide_name", "repeat", "fold"):
        if col not in fa.columns:
            raise SystemExit(f"{fold_assignments}: missing column '{col}'")
    if "slide_name" not in cd.columns:
        raise SystemExit(f"{cancer_csv}: missing column 'slide_name'")

    # a duplicated slide name maps to its last row, as in the reference pickle (see --compare-to);
    # the earlier row is then never used
    name_to_idx = {name: i for i, name in enumerate(cd["slide_name"])}

    if dedup_warn:
        dupes = cd["slide_name"][cd["slide_name"].duplicated()].tolist()
        if dupes:
            print(f"[warn] {len(dupes)} duplicate slide name(s) in {cancer_csv.name}: {dupes}")
            print("[warn] each resolves to its last row; the shadowed rows are unreachable")

    missing = sorted(set(fa["slide_name"]) - set(name_to_idx))
    if missing:
        raise SystemExit(f"{len(missing)} slide(s) in the fold assignments are absent from "
                         f"{cancer_csv.name}, e.g. {missing[:3]}")

    folds = {}
    for repeat in sorted(fa["repeat"].unique()):
        block = fa[fa["repeat"] == repeat]
        for fold in sorted(block["fold"].unique()):
            test_names = block[block["fold"] == fold]["slide_name"].tolist()
            train_names = block[block["fold"] != fold]["slide_name"].tolist()
            folds[(int(repeat), int(fold))] = {
                "train": train_names,
                "test": test_names,
                "train_idx": np.array([name_to_idx[n] for n in train_names], dtype=np.int64),
                "test_idx": np.array([name_to_idx[n] for n in test_names], dtype=np.int64),
            }
    return folds


def main():
    ap = argparse.ArgumentParser(
        description="C3PRO — rebuild the canonical benchmark fold indices pickle")
    ap.add_argument("--fold-assignments", type=Path, required=True,
                    help="fold_assignments.csv (slide_name, repeat, fold)")
    ap.add_argument("--cancer-csv", type=Path, required=True,
                    help="cancer_dataset.csv — its row order defines the integer indices")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="directory to write fold_indices.pkl into")
    ap.add_argument("--compare-to", type=Path, default=None,
                    help="optional reference pickle; report membership and order agreement")
    ap.add_argument("--no-dedup-warn", action="store_true",
                    help="suppress the duplicate-slide-name warning")
    args = ap.parse_args()

    folds = build(args.fold_assignments, args.cancer_csv, not args.no_dedup_warn)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "fold_indices.pkl"
    with open(out, "wb") as fh:
        pickle.dump(folds, fh)

    sizes = {(len(v["train"]), len(v["test"])) for v in folds.values()}
    print(f"[ok] {len(folds)} folds -> {out}")
    print(f"[ok] sizes (train, test): {sorted(sizes)}")

    if args.compare_to:
        with open(args.compare_to, "rb") as fh:
            ref = pickle.load(fh)
        same_keys = set(ref) == set(folds)
        memb = all(set(ref[k]["train"]) == set(folds[k]["train"])
                   and set(ref[k]["test"]) == set(folds[k]["test"]) for k in ref) if same_keys else False
        order = all(list(ref[k]["train"]) == list(folds[k]["train"])
                    and list(ref[k]["test"]) == list(folds[k]["test"]) for k in ref) if same_keys else False
        print(f"[cmp] keys identical       : {same_keys}")
        print(f"[cmp] membership identical : {memb}")
        print(f"[cmp] order identical      : {order}")


if __name__ == "__main__":
    main()
