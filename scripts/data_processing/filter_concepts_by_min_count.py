#!/usr/bin/env python3
"""
Filter dataset splits by minimum concept frequency in the training split.

This is useful to remove extremely rare concepts which can destabilize training and
make attention diagnostics noisy.

Behavior:
  - Computes concept counts in TRAIN (default column: 'generalized')
  - Keeps only concepts with count >= min-count
  - Filters TRAIN/VAL/TEST to only those concepts
  - Writes filtered CSVs to an output directory (does not overwrite by default)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", type=Path, required=True, help="Directory containing dataset_{train,val,test}.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write filtered CSVs")
    parser.add_argument("--min-count", type=int, default=5, help="Minimum train samples per concept to keep")
    parser.add_argument("--concept-col", type=str, default="generalized", help="Column containing concept labels")
    args = parser.parse_args()

    splits_dir: Path = args.splits_dir
    out_dir: Path = args.output_dir
    min_count: int = int(args.min_count)
    concept_col: str = str(args.concept_col)

    train_path = splits_dir / "dataset_train.csv"
    val_path = splits_dir / "dataset_val.csv"
    test_path = splits_dir / "dataset_test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing {val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing {test_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if concept_col not in df.columns:
            raise ValueError(f"{name} CSV is missing column '{concept_col}'. Did you run map_generalized_concepts.py?")

    counts = train_df[concept_col].value_counts(dropna=True)
    keep = counts[counts >= min_count].index
    keep_set = set(keep.tolist())

    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df2 = df[df[concept_col].isin(keep_set)].copy()
        df2 = df2.reset_index(drop=True)
        after = len(df2)
        print(f"Filtered {before} -> {after} rows")
        return df2

    print(f"Train concepts: {counts.shape[0]}")
    print(f"Keeping concepts with >= {min_count} train samples: {len(keep_set)}")

    train_f = _filter(train_df)
    val_f = _filter(val_df)
    test_f = _filter(test_df)

    # Report how many concepts remain in each split
    for split_name, df in [("train", train_f), ("val", val_f), ("test", test_f)]:
        print(f"{split_name}: unique concepts = {df[concept_col].nunique()}")

    train_out = out_dir / "dataset_train.csv"
    val_out = out_dir / "dataset_val.csv"
    test_out = out_dir / "dataset_test.csv"

    train_f.to_csv(train_out, index=False)
    val_f.to_csv(val_out, index=False)
    test_f.to_csv(test_out, index=False)

    print(f"Wrote filtered splits to: {out_dir}")
    print(f" - {train_out}")
    print(f" - {val_out}")
    print(f" - {test_out}")


if __name__ == "__main__":
    main()












