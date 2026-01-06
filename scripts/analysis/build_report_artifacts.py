#!/usr/bin/env python3
"""
Build consolidated report artifacts from all training runs.

Scans results/**/checkpoints/*.pt and results/evals/**/*.json to create:
- Separate master CSVs for Stage 1 and Stage 2
- Professional matplotlib plots (LaTeX-ready, publishable)
- Interactive plotly plots
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
import matplotlib
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np

# Optional plotly import
PLOTLY_AVAILABLE = False
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    pass

# Set matplotlib to use LaTeX-quality backend and configure for publication
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times', 'Palatino', 'New Century Schoolbook', 'Bookman', 'Computer Modern Roman'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'text.usetex': False,  # Set to True if LaTeX is installed
    'pdf.fonttype': 42,  # TrueType fonts for PDF
    'ps.fonttype': 42,  # TrueType fonts for PS
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
sns.set_palette("colorblind")


def is_vanilla(checkpoint_data: Dict) -> bool:
    """Determine if checkpoint is vanilla (no stage0) or finetuned (uses stage0)."""
    stage0_ckpt = checkpoint_data.get("stage0_checkpoint")
    if stage0_ckpt is not None and stage0_ckpt != "None" and str(stage0_ckpt).lower() != "none":
        return False  # Finetuned (uses stage0)
    return True  # Vanilla (no stage0)


def load_stage1_metrics(checkpoint_path: Path) -> Optional[Dict]:
    """Load metrics from Stage 1 checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
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
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
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
    try:
        with open(eval_json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        # Some older eval outputs may be partially written/corrupted (e.g., concurrent writes).
        # Skip these rather than failing the entire report generation.
        logger.warning(f"Skipping invalid JSON eval file: {eval_json_path} ({e})")
        return None
    
    metrics = data.get("metrics", {})
    
    # Determine variant from checkpoint if available
    variant = "unknown"
    ckpt_path_str = data.get("stage2_checkpoint") or data.get("stage1_checkpoint")
    if ckpt_path_str:
        ckpt_path = Path(ckpt_path_str)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            variant = "vanilla" if is_vanilla(ckpt) else "finetuned"
        else:
            # Fallback: check path for "vanilla" keyword
            if "vanilla" in str(ckpt_path).lower():
                variant = "vanilla"
            elif "geolocal" in str(ckpt_path).lower() or "finetuned" in str(ckpt_path).lower():
                variant = "finetuned"
    
    return {
        "eval_type": "test_split" if "splits_json" in data else "hf_dataset",
        "checkpoint_path": ckpt_path_str,
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


def load_consolidated_csv(csv_path: Path) -> List[Dict]:
    """Load consolidated CSV and convert to metrics dict format."""
    rows = []
    df = pd.read_csv(csv_path)
    
    for _, row in df.iterrows():
        # Determine eval type from CSV path
        csv_name = csv_path.name
        if "stage1" in csv_name:
            eval_type = "test_split"
            stage = 1
        elif "stage2" in csv_name and "hf" in csv_name:
            eval_type = "hf_dataset"
            stage = 2
        elif "stage2" in csv_name:
            eval_type = "test_split"
            stage = 2
        else:
            eval_type = "unknown"
            stage = None
        
        # Convert row to dict
        row_dict = row.to_dict()
        
        # Add metadata
        row_dict["eval_type"] = eval_type
        if stage:
            row_dict["stage"] = stage
        
        rows.append(row_dict)
    
    return rows


def scan_results_directory(results_root: Path) -> pd.DataFrame:
    """Scan results directory for consolidated CSVs only."""
    rows = []
    results_root = Path(results_root)
    
    # Only use consolidated CSVs - these are the source of truth
    logger.info("Loading consolidated CSVs...")
    consolidated_csvs = [
        results_root / "evals" / "stage1_test_consolidated.csv",
        results_root / "evals" / "stage2_test_consolidated.csv",
        results_root / "evals" / "stage2_hf_specific_checkpoints.csv",
    ]
    
    for csv_path in consolidated_csvs:
        if csv_path.exists():
            logger.info(f"Loading consolidated CSV: {csv_path}")
            csv_rows = load_consolidated_csv(csv_path)
            rows.extend(csv_rows)
        else:
            logger.warning(f"Consolidated CSV not found: {csv_path}")
    
    return pd.DataFrame(rows)


def get_zoomed_ylim(values, padding=0.05):
    """Calculate zoomed y-axis range to make small differences more visible."""
    if len(values) == 0:
        return [0, 1]
    min_val, max_val = values.min(), values.max()
    range_val = max_val - min_val
    if range_val == 0:
        return [0, 1]
    # Add padding and ensure we don't go out of [0, 1] bounds too much
    lower = max(0, min_val - range_val * padding)
    upper = min(1, max_val + range_val * padding)
    # If still too tight, use a minimum range of 0.1
    if upper - lower < 0.1:
        mid = (upper + lower) / 2
        lower, upper = mid - 0.05, mid + 0.05
    return [lower, upper]


def create_stage1_plots(df_stage1: pd.DataFrame, output_dir: Path):
    """Create professional Stage 1 plots."""
    if len(df_stage1) == 0:
        logger.warning("No Stage 1 data found for plotting")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter to test split evaluations
    df_test = df_stage1[df_stage1.get("eval_type") == "test_split"].copy()
    if len(df_test) == 0:
        df_test = df_stage1.copy()
    
    # Replace "vanilla" with "default" and capitalize
    df_test = df_test.copy()
    df_test["variant"] = df_test["variant"].replace("vanilla", "default").str.capitalize()
    
    # Matplotlib: Stage 1 Accuracy Comparison - Top-1 and Top-5
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    variant_order = ["Default", "Finetuned"]
    palette_map = {"Default": "#4472C4", "Finetuned": "#ED7D31"}  # Default=blue, Finetuned=orange
    
    # Top-1 Accuracies
    if df_test["meta_acc"].notna().any():
        ax = axes[0, 0]
        sns.barplot(data=df_test, x="variant", y="meta_acc", order=variant_order, ax=ax, 
                   palette=[palette_map[v] for v in variant_order], edgecolor="black", linewidth=2)
        ax.set_ylabel("Accuracy", fontweight='bold', fontsize=12)
        ax.set_xlabel("Model Variant", fontweight='bold', fontsize=12)
        ax.set_title("(a) Child Concept (Top-1)", fontweight='bold', pad=12, fontsize=13)
        y_min, y_max = 0, 1
        ax.set_ylim([y_min, y_max])
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        # Annotate bars above with values
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                y_pos = height + y_range * 0.02
                ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    if df_test["parent_acc"].notna().any():
        ax = axes[0, 1]
        sns.barplot(data=df_test, x="variant", y="parent_acc", order=variant_order, ax=ax,
                   palette=[palette_map[v] for v in variant_order], edgecolor="black", linewidth=2)
        ax.set_ylabel("Accuracy", fontweight='bold', fontsize=12)
        ax.set_xlabel("Model Variant", fontweight='bold', fontsize=12)
        ax.set_title("(b) Parent Concept (Top-1)", fontweight='bold', pad=12, fontsize=13)
        y_min, y_max = get_zoomed_ylim(df_test["parent_acc"])
        ax.set_ylim([y_min, y_max])
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        # Annotate bars above with values
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                y_pos = height + y_range * 0.02
                ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Top-5 Accuracies
    if df_test["meta_acc_top5"].notna().any():
        ax = axes[1, 0]
        sns.barplot(data=df_test, x="variant", y="meta_acc_top5", order=variant_order, ax=ax, 
                   palette=[palette_map[v] for v in variant_order], edgecolor="black", linewidth=2)
        ax.set_ylabel("Accuracy", fontweight='bold', fontsize=12)
        ax.set_xlabel("Model Variant", fontweight='bold', fontsize=12)
        ax.set_title("(c) Child Concept (Top-5)", fontweight='bold', pad=12, fontsize=13)
        y_min, y_max = get_zoomed_ylim(df_test["meta_acc_top5"])
        ax.set_ylim([y_min, y_max])
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        # Annotate bars above with values
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                y_pos = height + y_range * 0.02
                ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    if df_test["parent_acc_top5"].notna().any():
        ax = axes[1, 1]
        sns.barplot(data=df_test, x="variant", y="parent_acc_top5", order=variant_order, ax=ax,
                   palette=[palette_map[v] for v in variant_order], edgecolor="black", linewidth=2)
        ax.set_ylabel("Accuracy", fontweight='bold', fontsize=12)
        ax.set_xlabel("Model Variant", fontweight='bold', fontsize=12)
        ax.set_title("(d) Parent Concept (Top-5)", fontweight='bold', pad=12, fontsize=13)
        y_min, y_max = get_zoomed_ylim(df_test["parent_acc_top5"])
        ax.set_ylim([y_min, y_max])
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        # Annotate bars above with values
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                y_pos = height + y_range * 0.02
                ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle("Stage 1: Concept Classification Performance", fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / "stage1_accuracies.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / "stage1_accuracies.png", format='png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plotly: Interactive Stage 1 plot
    if PLOTLY_AVAILABLE:
        fig_plotly = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Top-1 Accuracy", "Top-5 Accuracy"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Top-1
        for variant in variant_order:
            df_var = df_test[df_test["variant"] == variant]
            meta_mean = df_var["meta_acc"].mean() if df_var["meta_acc"].notna().any() else 0
            parent_mean = df_var["parent_acc"].mean() if df_var["parent_acc"].notna().any() else 0
            
            fig_plotly.add_trace(go.Bar(
                name=variant,
                x=["Child Concept", "Parent Concept"],
                y=[meta_mean, parent_mean],
                marker_color=palette_map.get(variant, "#808080"),
                text=[f"{meta_mean:.3f}", f"{parent_mean:.3f}"],
                textposition='outside',
                showlegend=True,
            ), row=1, col=1)
        
        # Top-5
        for variant in variant_order:
            df_var = df_test[df_test["variant"] == variant]
            meta_top5_mean = df_var["meta_acc_top5"].mean() if df_var["meta_acc_top5"].notna().any() else 0
            parent_top5_mean = df_var["parent_acc_top5"].mean() if df_var["parent_acc_top5"].notna().any() else 0
            
            fig_plotly.add_trace(go.Bar(
                name=variant + " (Top-5)",
                x=["Child Concept", "Parent Concept"],
                y=[meta_top5_mean, parent_top5_mean],
                marker_color=palette_map.get(variant, "#808080"),
                text=[f"{meta_top5_mean:.3f}", f"{parent_top5_mean:.3f}"],
                textposition='outside',
                showlegend=False,
            ), row=1, col=2)
        
        fig_plotly.update_layout(
            title="Stage 1: Concept Classification Performance",
            height=500,
            width=1200,
            barmode='group',
            template='plotly_white',
            font=dict(family="Times New Roman", size=12),
        )
        fig_plotly.update_xaxes(title_text="Concept Type", row=1, col=1)
        fig_plotly.update_yaxes(title_text="Accuracy", range=[0, 1], row=1, col=1)
        fig_plotly.update_xaxes(title_text="Concept Type", row=1, col=2)
        fig_plotly.update_yaxes(title_text="Accuracy", range=[0, 1], row=1, col=2)
        
        fig_plotly.write_html(output_dir / "stage1_accuracies_interactive.html")
    
    logger.info(f"Saved Stage 1 plots to {output_dir}")


def create_finetuned_comparison_charts(df_test_split: pd.DataFrame, output_dir: Path, 
                                      ablation_order: List[str], ablation_labels: Dict[str, str],
                                      variant_order: List[str], palette_map: Dict[str, str]):
    """Create radar and lollipop charts showing finetuned performance improvements."""
    if len(df_test_split) == 0:
        return
    
    # Prepare data for comparison
    metrics_to_compare = {
        "Median Error (km)": "median_error_km",
        "Mean Error (km)": "mean_error_km",
        "Cell Accuracy": "cell_acc",
        "City Accuracy": "acc_city",
        "Region Accuracy": "acc_region",
        "Country Accuracy": "acc_country"
    }
    
    # Calculate improvements (lower is better for errors, higher is better for accuracies)
    comparison_data = []
    for ablation in ablation_order:
        df_ablation = df_test_split[df_test_split["ablation_mode"] == ablation]
        default_row = df_ablation[df_ablation["variant"] == "Default"].iloc[0] if len(df_ablation[df_ablation["variant"] == "Default"]) > 0 else None
        finetuned_row = df_ablation[df_ablation["variant"] == "Finetuned"].iloc[0] if len(df_ablation[df_ablation["variant"] == "Finetuned"]) > 0 else None
        
        if default_row is not None and finetuned_row is not None:
            for metric_name, metric_col in metrics_to_compare.items():
                default_val = default_row.get(metric_col)
                finetuned_val = finetuned_row.get(metric_col)
                
                if pd.notna(default_val) and pd.notna(finetuned_val):
                    if "Error" in metric_name:
                        # For errors, improvement = reduction (negative change is good)
                        improvement = ((default_val - finetuned_val) / default_val) * 100
                    else:
                        # For accuracies, improvement = increase (positive change is good)
                        improvement = ((finetuned_val - default_val) / default_val) * 100 if default_val > 0 else 0
                    
                    comparison_data.append({
                        "ablation": ablation_labels[ablation],
                        "metric": metric_name,
                        "improvement": improvement,
                        "default": default_val,
                        "finetuned": finetuned_val
                    })
    
    if not comparison_data:
        return
    
    df_comp = pd.DataFrame(comparison_data)
    
    # Radar Chart - show all 3 ablation modes
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), subplot_kw=dict(projection='polar'))
    
    # Prepare data for radar chart - one per ablation mode
    angles = np.linspace(0, 2 * np.pi, len(metrics_to_compare), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for idx, ablation in enumerate(ablation_order):  # Show all 3 ablation modes
        ax = axes[idx]
        ablation_label = ablation_labels[ablation]
        
        # Get values for this ablation mode
        df_ablation_comp = df_comp[df_comp["ablation"] == ablation_label]
        
        default_values = []
        finetuned_values = []
        metric_names = []
        
        for metric_name in metrics_to_compare.keys():
            metric_data = df_ablation_comp[df_ablation_comp["metric"] == metric_name]
            if len(metric_data) > 0:
                default_values.append(metric_data.iloc[0]["default"])
                finetuned_values.append(metric_data.iloc[0]["finetuned"])
                metric_names.append(metric_name)
        
        if not default_values:
            continue
        
        # Normalize values to 0-1 scale for radar chart
        def normalize_error(val, max_val, min_val):
            if max_val == min_val:
                return 0.5
            return 1 - (val - min_val) / (max_val - min_val)  # Invert so lower error = higher value
        
        def normalize_accuracy(val):
            return val  # Already 0-1
        
        all_vals = default_values + finetuned_values
        
        normalized_default = []
        normalized_finetuned = []
        
        for i, (metric_name, def_val, fin_val) in enumerate(zip(metric_names, default_values, finetuned_values)):
            if "Error" in metric_name:
                max_val = max(all_vals[i], all_vals[i + len(metric_names)])
                min_val = min(all_vals[i], all_vals[i + len(metric_names)])
                normalized_default.append(normalize_error(def_val, max_val, min_val))
                normalized_finetuned.append(normalize_error(fin_val, max_val, min_val))
            else:
                normalized_default.append(normalize_accuracy(def_val))
                normalized_finetuned.append(normalize_accuracy(fin_val))
        
        normalized_default += normalized_default[:1]  # Complete the circle
        normalized_finetuned += normalized_finetuned[:1]
        
        # Plot
        ax.plot(angles, normalized_default, 'o-', linewidth=2, label='Default', 
               color=palette_map["Default"], markersize=8)
        ax.fill(angles, normalized_default, alpha=0.25, color=palette_map["Default"])
        
        ax.plot(angles, normalized_finetuned, 'o-', linewidth=2, label='Finetuned',
               color=palette_map["Finetuned"], markersize=8)
        ax.fill(angles, normalized_finetuned, alpha=0.25, color=palette_map["Finetuned"])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_title(f"{ablation_label} Mode", fontweight='bold', pad=20, fontsize=12)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    
    plt.suptitle("Radar Chart: Finetuned vs Default Performance", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "stage2_finetuned_radar.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / "stage2_finetuned_radar.png", format='png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Lollipop Chart - showing improvement percentages
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group by metric and calculate average improvement across ablation modes
    metric_improvements = df_comp.groupby("metric")["improvement"].mean().sort_values()
    
    y_pos = np.arange(len(metric_improvements))
    colors = [palette_map["Finetuned"] if val > 0 else palette_map["Default"] for val in metric_improvements.values]
    
    # Create lollipop chart
    for i, (metric, improvement) in enumerate(metric_improvements.items()):
        # Stem
        ax.plot([0, improvement], [i, i], color=colors[i], linewidth=2.5, alpha=0.7)
        # Circle at end
        circle = Circle((improvement, i), radius=0.15, color=colors[i], zorder=5)
        ax.add_patch(circle)
        # Value label
        ax.text(improvement + (2 if improvement > 0 else -2), i, 
               f'{improvement:+.1f}%', va='center', fontsize=10, fontweight='bold',
               color=colors[i])
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metric_improvements.index, fontsize=11, fontweight='bold')
    ax.set_xlabel("Improvement (%)", fontweight='bold', fontsize=12)
    ax.set_title("Finetuned Performance Improvement Over Default", fontweight='bold', fontsize=14, pad=15)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=palette_map["Finetuned"], 
               markersize=10, label='Improvement'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=palette_map["Default"], 
               markersize=10, label='Degradation')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / "stage2_finetuned_lollipop.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / "stage2_finetuned_lollipop.png", format='png', bbox_inches='tight', dpi=300)
    plt.close()
    
    logger.info(f"Saved finetuned comparison charts to {output_dir}")


def create_stage2_plots(df_stage2: pd.DataFrame, output_dir: Path):
    """Create professional Stage 2 plots."""
    if len(df_stage2) == 0:
        logger.warning("No Stage 2 data found for plotting")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Separate test split and HF dataset results
    df_test_split = df_stage2[df_stage2.get("eval_type") == "test_split"].copy()
    df_hf = df_stage2[df_stage2.get("eval_type") == "hf_dataset"].copy()
    
    # Ablation mode order
    ablation_order = ["both", "concept_only", "image_only"]
    ablation_labels = {"both": "Both", "concept_only": "Concept Only", "image_only": "Image Only"}
    
    # Ablation order for HF dataset (includes GeoCLIP)
    ablation_order_hf = ["both", "concept_only", "image_only", "geoclip"]
    ablation_labels_hf = {"both": "Both", "concept_only": "Concept Only", "image_only": "Image Only", "geoclip": "GeoCLIP"}
    
    # Threshold accuracy columns and labels
    threshold_cols = ["acc_city", "acc_region", "acc_country"]
    threshold_labels = ["City (<25km)", "Region (<200km)", "Country (<750km)"]
    
    # Replace "vanilla" with "default" and capitalize variants
    df_test_split = df_test_split.copy()
    df_test_split["variant"] = df_test_split["variant"].replace("vanilla", "default").str.capitalize()
    df_hf = df_hf.copy()
    df_hf["variant"] = df_hf["variant"].replace("vanilla", "default").str.capitalize()
    
    # Map "trained_stage1" to "Default" or "Finetuned" based on stage0_checkpoint
    # "trained_stage1" with stage0_checkpoint -> "Finetuned"
    # "trained_stage1" without stage0_checkpoint -> "Default"
    if "stage0_checkpoint" in df_hf.columns:
        mask_trained = df_hf["variant"].str.lower().str.contains("trained_stage1", na=False)
        mask_has_stage0 = df_hf["stage0_checkpoint"].notna() & (df_hf["stage0_checkpoint"] != "") & (df_hf["stage0_checkpoint"] != "None")
        df_hf.loc[mask_trained & mask_has_stage0, "variant"] = "Finetuned"
        df_hf.loc[mask_trained & ~mask_has_stage0, "variant"] = "Default"
    else:
        # Fallback: map all "trained_stage1" to "Finetuned" if we can't check stage0
        df_hf["variant"] = df_hf["variant"].replace({"Trained_stage1": "Finetuned", "trained_stage1": "Finetuned"})
    
    # Handle GeoCLIP variant mapping (handle different cases)
    df_hf["variant"] = df_hf["variant"].replace({"Geoclip": "GeoCLIP", "geoclip": "GeoCLIP", "GEOCLIP": "GeoCLIP"})
    
    # Convert empty strings to NaN for cell_acc
    df_hf["cell_acc"] = df_hf["cell_acc"].replace("", np.nan)
    df_hf["cell_acc"] = pd.to_numeric(df_hf["cell_acc"], errors='coerce')
    
    variant_order = ["Default", "Finetuned", "GeoCLIP"]
    palette_map = {"Default": "#4472C4", "Finetuned": "#ED7D31", "GeoCLIP": "#70AD47"}  # Default=blue, Finetuned=orange, GeoCLIP=green
    
    # Matplotlib: Stage 2 Test Split Results
    if len(df_test_split) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        
        # Plot 1: Median Error
        df_plot = df_test_split[df_test_split["median_error_km"].notna()].copy()
        if len(df_plot) > 0:
            ax = axes[0, 0]
            sns.barplot(data=df_plot, x="ablation_mode", y="median_error_km", hue="variant",
                       order=ablation_order, hue_order=variant_order, ax=ax, 
                       palette=[palette_map[v] for v in variant_order],
                       edgecolor="black", linewidth=2)
            ax.set_ylabel("Median Error (km)", fontweight='bold', fontsize=12)
            ax.set_xlabel("Ablation Mode", fontweight='bold', fontsize=12)
            ax.set_title("(a) Median Distance Error", fontweight='bold', pad=12, fontsize=13)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, variant_order, title="Variant", title_fontsize=11, fontsize=10)
            ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.2)
            ax.spines['bottom'].set_linewidth(1.2)
            ax.set_xticklabels([ablation_labels.get(x.get_text(), x.get_text()) for x in ax.get_xticklabels()])
            for container in ax.containers:
                for bar in container:
                    height = bar.get_height()
                    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                    y_pos = height + y_range * 0.02
                    ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Plot 2: Cell Accuracy
        if df_test_split["cell_acc"].notna().any():
            ax = axes[0, 1]
            sns.barplot(data=df_test_split, x="ablation_mode", y="cell_acc", hue="variant",
                      order=ablation_order, hue_order=variant_order, ax=ax, 
                      palette=[palette_map[v] for v in variant_order],
                      edgecolor="black", linewidth=2)
            ax.set_ylabel("Cell Accuracy", fontweight='bold', fontsize=12)
            ax.set_xlabel("Ablation Mode", fontweight='bold', fontsize=12)
            ax.set_title("(b) Cell Classification Accuracy", fontweight='bold', pad=12, fontsize=13)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, variant_order, title="Variant", title_fontsize=11, fontsize=10)
            y_min, y_max = get_zoomed_ylim(df_test_split["cell_acc"])
            ax.set_ylim([y_min, y_max])
            ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.2)
            ax.spines['bottom'].set_linewidth(1.2)
            ax.set_xticklabels([ablation_labels.get(x.get_text(), x.get_text()) for x in ax.get_xticklabels()])
            for container in ax.containers:
                for bar in container:
                    height = bar.get_height()
                    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                    y_pos = height + y_range * 0.02
                    ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                           f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Plot 3: Threshold Accuracies (City, Region, Country)
        threshold_data = []
        for _, row in df_test_split.iterrows():
            for col, label in zip(threshold_cols, threshold_labels):
                if pd.notna(row.get(col)):
                    threshold_data.append({
                        "ablation_mode": row["ablation_mode"],
                        "variant": row["variant"],
                        "threshold": label,
                        "accuracy": row[col]
                    })
        
        if threshold_data:
            ax = axes[1, 0]
            df_threshold = pd.DataFrame(threshold_data)
            sns.barplot(data=df_threshold, x="ablation_mode", y="accuracy", hue="threshold",
                       order=ablation_order, ax=ax, palette=["#70AD47", "#FFC000", "#7030A0"],
                       edgecolor="black", linewidth=1.8)
            ax.set_ylabel("Accuracy", fontweight='bold', fontsize=12)
            ax.set_xlabel("Ablation Mode", fontweight='bold', fontsize=12)
            ax.set_title("(c) Threshold Accuracies", fontweight='bold', pad=12, fontsize=13)
            ax.legend(title="Threshold", title_fontsize=11, fontsize=10)
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.2)
            ax.spines['bottom'].set_linewidth(1.2)
            ax.set_xticklabels([ablation_labels.get(x.get_text(), x.get_text()) for x in ax.get_xticklabels()])
            for container in ax.containers:
                for bar in container:
                    height = bar.get_height()
                    y_pos = height + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
                    ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Plot 4: Mean Error comparison
        if df_test_split["mean_error_km"].notna().any():
            ax = axes[1, 1]
            sns.barplot(data=df_test_split, x="ablation_mode", y="mean_error_km", hue="variant",
                       order=ablation_order, hue_order=variant_order, ax=ax, 
                       palette=[palette_map[v] for v in variant_order],
                       edgecolor="black", linewidth=2)
            ax.set_ylabel("Mean Error (km)", fontweight='bold', fontsize=12)
            ax.set_xlabel("Ablation Mode", fontweight='bold', fontsize=12)
            ax.set_title("(d) Mean Distance Error", fontweight='bold', pad=12, fontsize=13)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, variant_order, title="Variant", title_fontsize=11, fontsize=10)
            ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.2)
            ax.spines['bottom'].set_linewidth(1.2)
            ax.set_xticklabels([ablation_labels.get(x.get_text(), x.get_text()) for x in ax.get_xticklabels()])
            for container in ax.containers:
                for bar in container:
                    height = bar.get_height()
                    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                    y_pos = height + y_range * 0.02
                    ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.suptitle("Stage 2: Test Split Evaluation Results", fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_dir / "stage2_test_split_results.pdf", format='pdf', bbox_inches='tight', dpi=300)
        plt.savefig(output_dir / "stage2_test_split_results.png", format='png', bbox_inches='tight', dpi=300)
        plt.close()
    
    # Matplotlib: Stage 2 HF Dataset Results
    if len(df_hf) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        
        # Plot 1: Median Error on HF dataset (exclude GeoCLIP)
        df_plot = df_hf[(df_hf["median_error_km"].notna()) & (df_hf["variant"] != "GeoCLIP")].copy()
        if len(df_plot) > 0:
            ax = axes[0, 0]
            sns.barplot(data=df_plot, x="ablation_mode", y="median_error_km", hue="variant",
                       order=ablation_order, hue_order=["Default", "Finetuned"], ax=ax, 
                       palette=[palette_map[v] for v in ["Default", "Finetuned"]],
                       edgecolor="black", linewidth=2)
            ax.set_ylabel("Median Error (km)", fontweight='bold', fontsize=12)
            ax.set_xlabel("Ablation Mode", fontweight='bold', fontsize=12)
            ax.set_title("(a) Median Distance Error", fontweight='bold', pad=12, fontsize=13)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, ["Default", "Finetuned"], title="Variant", title_fontsize=11, fontsize=10)
            ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.2)
            ax.spines['bottom'].set_linewidth(1.2)
            ax.set_xticklabels([ablation_labels.get(x.get_text(), x.get_text()) for x in ax.get_xticklabels()])
            for container in ax.containers:
                for bar in container:
                    height = bar.get_height()
                    current_ylim = ax.get_ylim()
                    y_range = current_ylim[1] - current_ylim[0]
                    y_pos = height + y_range * 0.02
                    ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Plot 2: Cell Accuracy on HF dataset (exclude GeoCLIP)
        df_cell_acc = df_hf[(df_hf["cell_acc"].notna()) & (df_hf["variant"] != "GeoCLIP")].copy()
        if len(df_cell_acc) > 0:
            ax = axes[0, 1]
            sns.barplot(data=df_cell_acc, x="ablation_mode", y="cell_acc", hue="variant",
                      order=ablation_order, hue_order=["Default", "Finetuned"], ax=ax, 
                      palette=[palette_map[v] for v in ["Default", "Finetuned"]],
                      edgecolor="black", linewidth=2)
            ax.set_ylabel("Cell Accuracy", fontweight='bold', fontsize=12)
            ax.set_xlabel("Ablation Mode", fontweight='bold', fontsize=12)
            ax.set_title("(b) Cell Classification Accuracy", fontweight='bold', pad=12, fontsize=13)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, ["Default", "Finetuned"], title="Variant", title_fontsize=11, fontsize=10)
            y_min, y_max = get_zoomed_ylim(df_cell_acc["cell_acc"])
            ax.set_ylim([y_min, y_max])
            ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.2)
            ax.spines['bottom'].set_linewidth(1.2)
            ax.set_xticklabels([ablation_labels.get(x.get_text(), x.get_text()) for x in ax.get_xticklabels()])
            for container in ax.containers:
                for bar in container:
                    height = bar.get_height()
                    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                    y_pos = height + y_range * 0.02
                    ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                           f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Plot 3: Threshold Accuracies on HF dataset
        threshold_data = []
        for _, row in df_hf.iterrows():
            for col, label in zip(threshold_cols, threshold_labels):
                if pd.notna(row.get(col)):
                    threshold_data.append({
                        "ablation_mode": row["ablation_mode"],
                        "variant": row["variant"],
                        "threshold": label,
                        "accuracy": row[col]
                    })
        
        if threshold_data:
            ax = axes[1, 0]
            df_threshold = pd.DataFrame(threshold_data)
            sns.barplot(data=df_threshold, x="ablation_mode", y="accuracy", hue="threshold",
                       order=ablation_order_hf, ax=ax, palette=["#70AD47", "#FFC000", "#7030A0"],
                       edgecolor="black", linewidth=1.8)
            ax.set_ylabel("Accuracy", fontweight='bold', fontsize=12)
            ax.set_xlabel("Ablation Mode", fontweight='bold', fontsize=12)
            ax.set_title("(c) Threshold Accuracies", fontweight='bold', pad=12, fontsize=13)
            ax.legend(title="Threshold", title_fontsize=11, fontsize=10)
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.2)
            ax.spines['bottom'].set_linewidth(1.2)
            ax.set_xticklabels([ablation_labels_hf.get(x.get_text(), x.get_text()) for x in ax.get_xticklabels()])
            for container in ax.containers:
                for bar in container:
                    height = bar.get_height()
                    y_pos = height + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
                    ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Plot 4: Comparison Test Split vs HF Dataset
        if len(df_test_split) > 0 and len(df_hf) > 0:
            comparison_data = []
            for df_source, source_name in [(df_test_split, "Test Split"), (df_hf, "HF Dataset")]:
                for _, row in df_source.iterrows():
                    if pd.notna(row.get("median_error_km")):
                        comparison_data.append({
                            "ablation_mode": row["ablation_mode"],
                            "variant": row["variant"],
                            "dataset": source_name,
                            "median_error_km": row["median_error_km"]
                        })
            
            # Add GeoCLIP test split value
            geoclip_hf = df_hf[df_hf["variant"] == "GeoCLIP"]
            if len(geoclip_hf) > 0:
                comparison_data.append({
                    "ablation_mode": "geoclip",
                    "variant": "GeoCLIP",
                    "dataset": "Test Split",
                    "median_error_km": 522.3
                })
            
            if comparison_data:
                ax = axes[1, 1]
                df_comp = pd.DataFrame(comparison_data)
                sns.barplot(data=df_comp, x="ablation_mode", y="median_error_km", hue="dataset",
                           order=ablation_order_hf, ax=ax, palette=["#5B9BD5", "#E7E6E6"],
                           edgecolor="black", linewidth=1.8)
                ax.set_ylabel("Median Error (km)", fontweight='bold', fontsize=12)
                ax.set_xlabel("Ablation Mode", fontweight='bold', fontsize=12)
                ax.set_title("(d) Test Split vs HF Dataset", fontweight='bold', pad=12, fontsize=13)
                ax.legend(title="Dataset", title_fontsize=11, fontsize=10)
                ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(1.2)
                ax.spines['bottom'].set_linewidth(1.2)
                ax.set_xticklabels([ablation_labels_hf.get(x.get_text(), x.get_text()) for x in ax.get_xticklabels()])
                for container in ax.containers:
                    for bar in container:
                        height = bar.get_height()
                        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                        y_pos = height + y_range * 0.02
                        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                               f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.suptitle("Stage 2: HF GeoGuessr Dataset Evaluation Results", fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_dir / "stage2_hf_results.pdf", format='pdf', bbox_inches='tight', dpi=300)
        plt.savefig(output_dir / "stage2_hf_results.png", format='png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Create radar and lollipop charts for finetuned performance on HF dataset
        create_finetuned_comparison_charts(df_hf, output_dir, ablation_order, ablation_labels, variant_order, palette_map)
    
    # Plotly: Interactive Stage 2 plots
    if PLOTLY_AVAILABLE and len(df_test_split) > 0:
        fig_plotly = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Median Error", "Cell Accuracy", "Threshold Accuracies", "Variant Comparison"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Median Error
        for variant in variant_order:
            df_var = df_test_split[df_test_split["variant"] == variant]
            if len(df_var) > 0 and df_var["median_error_km"].notna().any():
                fig_plotly.add_trace(
                    go.Bar(name=variant, x=df_var["ablation_mode"], y=df_var["median_error_km"],
                          marker_color=palette_map.get(variant, "#808080")),
                    row=1, col=1
                )
        
        # Cell Accuracy
        for variant in variant_order:
            df_var = df_test_split[df_test_split["variant"] == variant]
            if len(df_var) > 0 and df_var["cell_acc"].notna().any():
                fig_plotly.add_trace(
                    go.Bar(name=variant, x=df_var["ablation_mode"], y=df_var["cell_acc"],
                          marker_color=palette_map.get(variant, "#808080"), showlegend=False),
                    row=1, col=2
                )
        
        fig_plotly.update_layout(
            title="Stage 2: Test Split Evaluation Results (Interactive)",
            height=800,
            width=1200,
            template='plotly_white',
            font=dict(family="Times New Roman", size=12),
        )
        fig_plotly.update_xaxes(title_text="Ablation Mode", row=1, col=1)
        fig_plotly.update_yaxes(title_text="Median Error (km)", row=1, col=1)
        fig_plotly.update_xaxes(title_text="Ablation Mode", row=1, col=2)
        fig_plotly.update_yaxes(title_text="Cell Accuracy", row=1, col=2)
        
        fig_plotly.write_html(output_dir / "stage2_test_split_interactive.html")
    
    logger.info(f"Saved Stage 2 plots to {output_dir}")


def create_summary_markdown(df: pd.DataFrame, output_dir: Path):
    """Create summary markdown report."""
    output_path = output_dir / "summary.md"
    
    # Replace "vanilla" with "default" and capitalize
    df = df.copy()
    df["variant"] = df["variant"].replace("vanilla", "default").str.capitalize()
    
    with open(output_path, 'w') as f:
        f.write("# Experiment Results Summary\n\n")
        f.write("This report summarizes the evaluation results for Stage 1 (concept classification) ")
        f.write("and Stage 2 (geolocation) models.\n\n")
        
        # Stage 1 summary
        df_stage1 = df[df["stage"] == 1].copy()
        if len(df_stage1) > 0:
            f.write("## Stage 1: Concept Classification Results\n\n")
            f.write("### Test Split Evaluation\n\n")
            df_stage1_test = df_stage1[df_stage1.get("eval_type") == "test_split"].copy()
            if len(df_stage1_test) == 0:
                df_stage1_test = df_stage1.copy()
            
            f.write("| Variant | Child Concept (Top-1) | Parent Concept (Top-1) | Child Concept (Top-5) | Parent Concept (Top-5) |\n")
            f.write("|---------|----------------------|----------------------|---------------------|----------------------|\n")
            for variant in ["Default", "Finetuned"]:
                df_var = df_stage1_test[df_stage1_test["variant"] == variant]
                if len(df_var) > 0:
                    meta_acc = df_var["meta_acc"].mean() if df_var["meta_acc"].notna().any() else None
                    parent_acc = df_var["parent_acc"].mean() if df_var["parent_acc"].notna().any() else None
                    meta_top5 = df_var["meta_acc_top5"].mean() if df_var["meta_acc_top5"].notna().any() else None
                    parent_top5 = df_var["parent_acc_top5"].mean() if df_var["parent_acc_top5"].notna().any() else None
                    
                    meta_str = f"{meta_acc:.4f}" if meta_acc is not None else "N/A"
                    parent_str = f"{parent_acc:.4f}" if parent_acc is not None else "N/A"
                    meta_top5_str = f"{meta_top5:.4f}" if meta_top5 is not None else "N/A"
                    parent_top5_str = f"{parent_top5:.4f}" if parent_top5 is not None else "N/A"
                    
                    f.write(f"| {variant} | {meta_str} | {parent_str} | {meta_top5_str} | {parent_top5_str} |\n")
            f.write("\n")
        
        # Stage 2 summary
        df_stage2 = df[df["stage"] == 2].copy()
        if len(df_stage2) > 0:
            f.write("## Stage 2: Geolocation Results\n\n")
            
            # Test Split Results
            df_stage2_test = df_stage2[df_stage2.get("eval_type") == "test_split"].copy()
            if len(df_stage2_test) > 0:
                f.write("### Test Split Evaluation\n\n")
                f.write("| Variant | Ablation | Median Error (km) | Mean Error (km) | Cell Acc | City Acc | Region Acc | Country Acc |\n")
                f.write("|---------|----------|-------------------|----------------|----------|----------|------------|-------------|\n")
                # Replace vanilla with default in the dataframe
                df_stage2_test = df_stage2_test.copy()
                df_stage2_test["variant"] = df_stage2_test["variant"].replace("vanilla", "default").str.capitalize()
                for variant in ["Default", "Finetuned"]:
                    for ablation in ["both", "concept_only", "image_only"]:
                        df_subset = df_stage2_test[
                            (df_stage2_test["variant"] == variant) & 
                            (df_stage2_test["ablation_mode"] == ablation)
                    ]
                    if len(df_subset) > 0:
                            median_err = df_subset["median_error_km"].mean() if df_subset["median_error_km"].notna().any() else None
                            mean_err = df_subset["mean_error_km"].mean() if df_subset["mean_error_km"].notna().any() else None
                            cell_acc = df_subset["cell_acc"].mean() if df_subset["cell_acc"].notna().any() else None
                            city_acc = df_subset["acc_city"].mean() if df_subset["acc_city"].notna().any() else None
                            region_acc = df_subset["acc_region"].mean() if df_subset["acc_region"].notna().any() else None
                            country_acc = df_subset["acc_country"].mean() if df_subset["acc_country"].notna().any() else None
                            
                            median_str = f"{median_err:.2f}" if median_err is not None else "N/A"
                            mean_str = f"{mean_err:.2f}" if mean_err is not None else "N/A"
                            cell_str = f"{cell_acc:.4f}" if cell_acc is not None else "N/A"
                            city_str = f"{city_acc:.4f}" if city_acc is not None else "N/A"
                            region_str = f"{region_acc:.4f}" if region_acc is not None else "N/A"
                            country_str = f"{country_acc:.4f}" if country_acc is not None else "N/A"
                            
                            f.write(f"| {variant} | {ablation} | {median_str} | {mean_str} | {cell_str} | {city_str} | {region_str} | {country_str} |\n")
                f.write("\n")
            
            # HF Dataset Results
            df_stage2_hf = df_stage2[df_stage2.get("eval_type") == "hf_dataset"].copy()
            if len(df_stage2_hf) > 0:
                f.write("### Hugging Face GeoGuessr Dataset Evaluation\n\n")
                f.write("| Variant | Ablation | Median Error (km) | Mean Error (km) | Cell Acc | City Acc | Region Acc | Country Acc |\n")
                f.write("|---------|----------|-------------------|----------------|----------|----------|------------|-------------|\n")
                # Replace vanilla with default in the dataframe
                df_stage2_hf = df_stage2_hf.copy()
                df_stage2_hf["variant"] = df_stage2_hf["variant"].replace("vanilla", "default").str.capitalize()
                for variant in ["Default", "Finetuned"]:
                    for ablation in ["both", "concept_only", "image_only"]:
                        df_subset = df_stage2_hf[
                            (df_stage2_hf["variant"] == variant) & 
                            (df_stage2_hf["ablation_mode"] == ablation)
                    ]
                    if len(df_subset) > 0:
                            median_err = df_subset["median_error_km"].mean() if df_subset["median_error_km"].notna().any() else None
                            mean_err = df_subset["mean_error_km"].mean() if df_subset["mean_error_km"].notna().any() else None
                            cell_acc = df_subset["cell_acc"].mean() if df_subset["cell_acc"].notna().any() else None
                            city_acc = df_subset["acc_city"].mean() if df_subset["acc_city"].notna().any() else None
                            region_acc = df_subset["acc_region"].mean() if df_subset["acc_region"].notna().any() else None
                            country_acc = df_subset["acc_country"].mean() if df_subset["acc_country"].notna().any() else None
                            
                            median_str = f"{median_err:.2f}" if median_err is not None else "N/A"
                            mean_str = f"{mean_err:.2f}" if mean_err is not None else "N/A"
                            cell_str = f"{cell_acc:.4f}" if cell_acc is not None else "N/A"
                            city_str = f"{city_acc:.4f}" if city_acc is not None else "N/A"
                            region_str = f"{region_acc:.4f}" if region_acc is not None else "N/A"
                            country_str = f"{country_acc:.4f}" if country_acc is not None else "N/A"
                            
                            f.write(f"| {variant} | {ablation} | {median_str} | {mean_str} | {cell_str} | {city_str} | {region_str} | {country_str} |\n")
                f.write("\n")
    
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
    
    # Separate Stage 1 and Stage 2 data
    df_stage1 = df[df["stage"] == 1].copy()
    df_stage2 = df[df["stage"] == 2].copy()
    
    # Save separate master CSVs
    if len(df_stage1) > 0:
        csv_path_stage1 = output_dir / "master_metrics_stage1.csv"
        df_stage1.to_csv(csv_path_stage1, index=False)
        logger.info(f"Saved Stage 1 master CSV to {csv_path_stage1}")
    
    if len(df_stage2) > 0:
        csv_path_stage2 = output_dir / "master_metrics_stage2.csv"
        df_stage2.to_csv(csv_path_stage2, index=False)
        logger.info(f"Saved Stage 2 master CSV to {csv_path_stage2}")
    
    # Also save combined CSV for backward compatibility
    csv_path = output_dir / "master_metrics.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved combined master CSV to {csv_path}")
    
    # Create separate plots for Stage 1 and Stage 2
    create_stage1_plots(df_stage1, output_dir)
    create_stage2_plots(df_stage2, output_dir)
    
    # Create summary markdown
    create_summary_markdown(df, output_dir)
    
    logger.info(f"\nReport artifacts saved to {output_dir}")
    logger.info(f"  - master_metrics_stage1.csv")
    logger.info(f"  - master_metrics_stage2.csv")
    logger.info(f"  - master_metrics.csv (combined)")
    logger.info(f"  - stage1_accuracies.pdf/png (publication-ready)")
    logger.info(f"  - stage1_accuracies_interactive.html (plotly)")
    logger.info(f"  - stage2_test_split_results.pdf/png (publication-ready)")
    logger.info(f"  - stage2_hf_results.pdf/png (publication-ready)")
    logger.info(f"  - stage2_test_split_interactive.html (plotly)")
    logger.info(f"  - summary.md")


if __name__ == "__main__":
    main()


