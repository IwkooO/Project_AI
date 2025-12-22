#!/usr/bin/env python3
"""
Build consolidated report artifacts from all training runs.

Scans results/**/checkpoints/*.pt and results/evals/**/*.json to create:
- Master CSV with all metrics
- Comparison plots
- Summary markdown
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")


def is_vanilla(checkpoint_data: Dict) -> bool:
    """Determine if checkpoint is vanilla (no stage0) or finetuned (uses stage0)."""
    stage0_ckpt = checkpoint_data.get("stage0_checkpoint")
    if stage0_ckpt is not None and stage0_ckpt != "None" and str(stage0_ckpt).lower() != "none":
        return False  # Finetuned (uses stage0)
    return True  # Vanilla (no stage0)


def load_stage1_metrics(checkpoint_path: Path) -> Optional[Dict]:
    """Load metrics from Stage 1 checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    variant = "vanilla" if is_vanilla(ckpt) else "finetuned"
    
    return {
        "stage": 1,
        "checkpoint_path": str(checkpoint_path),
        "variant": variant,
        "ablation_mode": "NA",
        "meta_acc": ckpt.get("meta_acc"),
        "parent_acc": ckpt.get("parent_acc"),
        "meta_acc_top5": None,  # May not be in checkpoint
        "parent_acc_top5": None,
        "epoch": ckpt.get("epoch"),
        "encoder_model": ckpt.get("encoder_model"),
        "num_concepts": ckpt.get("num_concepts"),
        "num_parents": ckpt.get("num_parents"),
        "stage0_checkpoint": ckpt.get("stage0_checkpoint"),
        "splits_json": ckpt.get("splits_json"),
    }


def load_stage2_metrics(checkpoint_path: Path) -> Optional[Dict]:
    """Load metrics from Stage 2 checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    variant = "vanilla" if is_vanilla(ckpt) else "finetuned"
    
    val_metrics = ckpt.get("val_metrics", {})
    test_metrics = ckpt.get("test_metrics", {})
    
    return {
        "stage": 2,
        "checkpoint_path": str(checkpoint_path),
        "variant": variant,
        "ablation_mode": ckpt.get("ablation_mode", "unknown"),
        "val_median_error_km": ckpt.get("val_median_error"),
        "val_cell_acc": val_metrics.get("cell_acc"),
        "val_acc_street": val_metrics.get("acc_street"),
        "val_acc_city": val_metrics.get("acc_city"),
        "val_acc_region": val_metrics.get("acc_region"),
        "val_acc_country": val_metrics.get("acc_country"),
        "test_median_error_km": test_metrics.get("median_error_km") if test_metrics else None,
        "test_cell_acc": test_metrics.get("cell_acc") if test_metrics else None,
        "test_acc_street": test_metrics.get("acc_street") if test_metrics else None,
        "test_acc_city": test_metrics.get("acc_city") if test_metrics else None,
        "test_acc_region": test_metrics.get("acc_region") if test_metrics else None,
        "test_acc_country": test_metrics.get("acc_country") if test_metrics else None,
        "epoch": ckpt.get("epoch"),
        "encoder_model": ckpt.get("encoder_model"),
        "num_cells": ckpt.get("num_cells"),
        "stage0_checkpoint": ckpt.get("stage0_checkpoint"),
        "stage1_checkpoint": ckpt.get("stage1_checkpoint"),
    }


def load_eval_metrics(eval_json_path: Path) -> Optional[Dict]:
    """Load metrics from evaluation JSON file."""
    with open(eval_json_path, 'r') as f:
        data = json.load(f)
    
    metrics = data.get("metrics", {})
    
    # Determine variant from checkpoint if available
    variant = "unknown"
    if "stage2_checkpoint" in data:
        ckpt_path = Path(data["stage2_checkpoint"])
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            variant = "vanilla" if is_vanilla(ckpt) else "finetuned"
    
    return {
        "eval_type": "test_split" if "splits_json" in data else "hf_dataset",
        "checkpoint_path": data.get("stage2_checkpoint") or data.get("stage1_checkpoint"),
        "variant": variant,
        "median_error_km": metrics.get("median_error_km"),
        "mean_error_km": metrics.get("mean_error_km"),
        "cell_acc": metrics.get("cell_acc"),
        "acc_street": metrics.get("acc_street"),
        "acc_city": metrics.get("acc_city"),
        "acc_region": metrics.get("acc_region"),
        "acc_country": metrics.get("acc_country"),
        "meta_acc": metrics.get("meta_acc"),
        "parent_acc": metrics.get("parent_acc"),
        "meta_acc_top5": metrics.get("meta_acc_top5"),
        "parent_acc_top5": metrics.get("parent_acc_top5"),
        "test_samples": data.get("test_samples"),
    }


def scan_results_directory(results_root: Path) -> pd.DataFrame:
    """Scan results directory for all checkpoints and eval files."""
    rows = []
    
    # Scan for Stage 1 checkpoints
    logger.info("Scanning for Stage 1 checkpoints...")
    for ckpt_path in results_root.rglob("**/checkpoints/best_model_stage1.pt"):
        metrics = load_stage1_metrics(ckpt_path)
        if metrics:
            rows.append(metrics)
    
    # Scan for Stage 2 checkpoints
    logger.info("Scanning for Stage 2 checkpoints...")
    for ckpt_path in results_root.rglob("**/checkpoints/best_model_stage2_xattn.pt"):
        metrics = load_stage2_metrics(ckpt_path)
        if metrics:
            rows.append(metrics)
    
    # Scan for evaluation JSONs
    logger.info("Scanning for evaluation results...")
    for eval_json in results_root.rglob("**/evals/**/test_metrics.json"):
        metrics = load_eval_metrics(eval_json)
        if metrics:
            rows.append(metrics)
    
    for eval_json in results_root.rglob("**/evals/**/hf_test_metrics.json"):
        metrics = load_eval_metrics(eval_json)
        if metrics:
            rows.append(metrics)
    
    return pd.DataFrame(rows)


def create_comparison_plots(df: pd.DataFrame, output_dir: Path):
    """Create comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter to stage2 data
    df_stage2 = df[df["stage"] == 2].copy()
    if len(df_stage2) == 0:
        logger.warning("No Stage 2 data found for plotting")
        return
    
    # Plot 1: Median error by variant and ablation mode
    fig, ax = plt.subplots(figsize=(10, 6))
    df_plot = df_stage2[df_stage2["val_median_error_km"].notna()]
    if len(df_plot) > 0:
        sns.barplot(data=df_plot, x="ablation_mode", y="val_median_error_km", hue="variant", ax=ax)
        ax.set_ylabel("Median Error (km)")
        ax.set_xlabel("Ablation Mode")
        ax.set_title("Stage 2 Validation Median Error: Vanilla vs Finetuned")
        plt.tight_layout()
        plt.savefig(output_dir / "stage2_val_median_error.png", dpi=150)
        plt.close()
    
    # Plot 2: Test error comparison (if available)
    df_test = df_stage2[df_stage2["test_median_error_km"].notna()]
    if len(df_test) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_test, x="ablation_mode", y="test_median_error_km", hue="variant", ax=ax)
        ax.set_ylabel("Median Error (km)")
        ax.set_xlabel("Ablation Mode")
        ax.set_title("Stage 2 Test Median Error: Vanilla vs Finetuned")
        plt.tight_layout()
        plt.savefig(output_dir / "stage2_test_median_error.png", dpi=150)
        plt.close()
    
    # Plot 3: Stage 1 accuracy comparison
    df_stage1 = df[df["stage"] == 1].copy()
    if len(df_stage1) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        if df_stage1["meta_acc"].notna().any():
            sns.barplot(data=df_stage1, x="variant", y="meta_acc", ax=axes[0])
            axes[0].set_ylabel("Meta Accuracy")
            axes[0].set_title("Stage 1 Meta Accuracy")
        
        if df_stage1["parent_acc"].notna().any():
            sns.barplot(data=df_stage1, x="variant", y="parent_acc", ax=axes[1])
            axes[1].set_ylabel("Parent Accuracy")
            axes[1].set_title("Stage 1 Parent Accuracy")
        
        plt.tight_layout()
        plt.savefig(output_dir / "stage1_accuracies.png", dpi=150)
        plt.close()
    
    logger.info(f"Saved plots to {output_dir}")


def create_summary_markdown(df: pd.DataFrame, output_dir: Path):
    """Create summary markdown report."""
    output_path = output_dir / "summary.md"
    
    with open(output_path, 'w') as f:
        f.write("# Experiment Results Summary\n\n")
        
        # Stage 1 summary
        df_stage1 = df[df["stage"] == 1].copy()
        if len(df_stage1) > 0:
            f.write("## Stage 1 Results\n\n")
            f.write("| Variant | Meta Acc | Parent Acc |\n")
            f.write("|---------|----------|------------|\n")
            for variant in ["vanilla", "finetuned"]:
                df_var = df_stage1[df_stage1["variant"] == variant]
                if len(df_var) > 0:
                    meta_acc = df_var["meta_acc"].mean()
                    parent_acc = df_var["parent_acc"].mean()
                    f.write(f"| {variant} | {meta_acc:.4f} | {parent_acc:.4f} |\n")
            f.write("\n")
        
        # Stage 2 summary
        df_stage2 = df[df["stage"] == 2].copy()
        if len(df_stage2) > 0:
            f.write("## Stage 2 Results\n\n")
            f.write("### Validation Set\n\n")
            f.write("| Variant | Ablation | Median Error (km) |\n")
            f.write("|---------|----------|-------------------|\n")
            for variant in ["vanilla", "finetuned"]:
                for ablation in ["concept_only", "image_only", "both"]:
                    df_subset = df_stage2[
                        (df_stage2["variant"] == variant) & 
                        (df_stage2["ablation_mode"] == ablation) &
                        (df_stage2["val_median_error_km"].notna())
                    ]
                    if len(df_subset) > 0:
                        median_err = df_subset["val_median_error_km"].mean()
                        f.write(f"| {variant} | {ablation} | {median_err:.2f} |\n")
            
            f.write("\n### Test Set\n\n")
            f.write("| Variant | Ablation | Median Error (km) |\n")
            f.write("|---------|----------|-------------------|\n")
            for variant in ["vanilla", "finetuned"]:
                for ablation in ["concept_only", "image_only", "both"]:
                    df_subset = df_stage2[
                        (df_stage2["variant"] == variant) & 
                        (df_stage2["ablation_mode"] == ablation) &
                        (df_stage2["test_median_error_km"].notna())
                    ]
                    if len(df_subset) > 0:
                        median_err = df_subset["test_median_error_km"].mean()
                        f.write(f"| {variant} | {ablation} | {median_err:.2f} |\n")
    
    logger.info(f"Saved summary to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build consolidated report artifacts")
    parser.add_argument("--results_root", type=str, default="results",
                        help="Root directory containing results")
    parser.add_argument("--output_dir", type=str, default="results/report",
                        help="Output directory for report artifacts")
    
    args = parser.parse_args()
    
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Scan and load all metrics
    logger.info(f"Scanning {results_root} for checkpoints and eval results...")
    df = scan_results_directory(results_root)
    
    if len(df) == 0:
        logger.warning("No metrics found!")
        return
    
    logger.info(f"Found {len(df)} metric entries")
    
    # Save master CSV
    csv_path = output_dir / "master_metrics.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved master CSV to {csv_path}")
    
    # Create plots
    create_comparison_plots(df, output_dir)
    
    # Create summary markdown
    create_summary_markdown(df, output_dir)
    
    logger.info(f"\nReport artifacts saved to {output_dir}")
    logger.info(f"  - master_metrics.csv")
    logger.info(f"  - *.png (plots)")
    logger.info(f"  - summary.md")


if __name__ == "__main__":
    main()


