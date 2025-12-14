import argparse
import sys
from pathlib import Path

import pandas as pd


def load_mapping(mapping_path: Path) -> dict:
    df = pd.read_csv(mapping_path)
    
    # Support both 'generalized' and 'mapped_concept' column names
    if 'original' not in df.columns:
        raise ValueError("Mapping file must contain 'original' column")
    
    if 'mapped_concept' in df.columns:
        target_col = 'mapped_concept'
    elif 'generalized' in df.columns:
        target_col = 'generalized'
    else:
        raise ValueError("Mapping file must contain 'generalized' or 'mapped_concept' column")
    
    # Deduplicate on original; keep first occurrence
    df = df.drop_duplicates(subset=['original'], keep='first')
    return dict(zip(df['original'], df[target_col]))


def add_generalized_column(csv_path: Path, mapping: dict, backup: bool = True) -> None:
    df = pd.read_csv(csv_path)
    if 'meta_name' not in df.columns:
        raise ValueError(f"'meta_name' column not found in {csv_path}")

    if backup:
        bak_path = csv_path.with_suffix(csv_path.suffix + ".bak")
        if not bak_path.exists():
            df.to_csv(bak_path, index=False)
            print(f"Backup created: {bak_path}")

    df['generalized'] = df['meta_name'].map(mapping)
    unmapped = df['generalized'].isna().sum()
    if unmapped:
        print(f"{csv_path.name}: {unmapped} rows had no mapping; dropping them for training")
        df = df[df['generalized'].notna()].copy()

    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} with generalized column (total rows: {len(df):,})")


def main():
    parser = argparse.ArgumentParser(description="Map meta_name to generalized concepts for split CSVs.")
    parser.add_argument(
        "--splits-dir",
        type=Path,
        required=True,
        help="Directory containing dataset_train.csv, dataset_val.csv, dataset_test.csv",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        required=True,
        help="CSV with columns ['original','generalized']",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak backups before writing",
    )
    args = parser.parse_args()

    splits_dir: Path = args.splits_dir
    mapping_path: Path = args.mapping
    backup = not args.no_backup

    if not splits_dir.is_dir():
        print(f"Splits directory not found: {splits_dir}", file=sys.stderr)
        sys.exit(1)
    if not mapping_path.is_file():
        print(f"Mapping file not found: {mapping_path}", file=sys.stderr)
        sys.exit(1)

    mapping = load_mapping(mapping_path)
    print(f"Loaded mapping for {len(mapping):,} concepts from {mapping_path}")

    split_files = [
        splits_dir / "dataset_train.csv",
        splits_dir / "dataset_val.csv",
        splits_dir / "dataset_test.csv",
    ]
    for csv_path in split_files:
        if not csv_path.is_file():
            print(f"Missing split file: {csv_path}", file=sys.stderr)
            continue
        add_generalized_column(csv_path, mapping, backup=backup)


if __name__ == "__main__":
    main()

