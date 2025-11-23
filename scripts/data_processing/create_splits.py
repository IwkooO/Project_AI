#!/usr/bin/env python3
"""
Create persistent Train/Val/Test splits for the dataset.
Standard split: 80% Train, 10% Val, 10% Test.
"""
import pandas as pd
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split

def create_splits(csv_path, output_dir, seed=42):
    print(f"Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Filter valid
    df = df.dropna(subset=['image_path', 'lat', 'lng'])
    print(f"Total valid samples: {len(df)}")
    
    # 1. Split Test (10%)
    train_val_df, test_df = train_test_split(
        df, test_size=0.10, random_state=seed, shuffle=True
    )
    
    # 2. Split Val from Train (10% of total = 11.1% of remaining 90%)
    # 0.10 / 0.90 = 1/9 ~= 0.1111
    train_df, val_df = train_test_split(
        train_val_df, test_size=1/9, random_state=seed, shuffle=True
    )
    
    print("\nSplit Statistics:")
    print(f"Train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val:   {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test:  {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
    
    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    train_path = out_path / "dataset_train.csv"
    val_path = out_path / "dataset_val.csv"
    test_path = out_path / "dataset_test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nSaved splits to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    create_splits(args.csv_path, args.output_dir, args.seed)

if __name__ == "__main__":
    main()

