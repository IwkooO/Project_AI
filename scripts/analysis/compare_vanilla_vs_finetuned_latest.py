#!/usr/bin/env python3
"""
Compare vanilla StreetCLIP vs finetuned StreetCLIP performance for Stage 1 and Stage 2.

This script scans the `results/` folder, picks the latest runs, loads metrics from
saved checkpoints, and writes a single CSV dataframe.

Defaults:
- Stage 1 runs: results/stage1-prototype/**
  - best checkpoint: checkpoints/best_model_stage1.pt
- Stage 2 runs: results/stage2_cross_attention_{concept_only,image_only,both}/**
  - best checkpoint: checkpoints/best_model_stage2_xattn.pt

Vanilla vs finetuned labeling:
- If a run dir or embedded `stage1_checkpoint` contains the substring "vanilla",
  it is treated as vanilla.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch


def latest_run_dir(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)


def best_checkpoint_path(run_dir: Path, rel: str) -> Optional[Path]:
    ckpt = run_dir / rel
    if ckpt.exists():
        return ckpt
    return None


def is_vanilla(text: str) -> bool:
    return "vanilla" in (text or "").lower()


def pick_latest_stage1_dirs(stage1_root: Path) -> Tuple[Optional[Path], Optional[Path]]:
    if not stage1_root.exists():
        return None, None

    candidates = [p for p in stage1_root.iterdir() if p.is_dir()]
    vanilla = [p for p in candidates if is_vanilla(str(p))]
    finetuned = [p for p in candidates if not is_vanilla(str(p))]

    vanilla_dir = max(vanilla, key=lambda p: p.stat().st_mtime) if vanilla else None
    finetuned_dir = max(finetuned, key=lambda p: p.stat().st_mtime) if finetuned else None
    return vanilla_dir, finetuned_dir


def load_stage1_metrics(run_dir: Path) -> Optional[Dict]:
    ckpt_path = best_checkpoint_path(run_dir, "checkpoints/best_model_stage1.pt")
    if ckpt_path is None:
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu")
    row = {
        "stage": 1,
        "ablation_mode": "NA",
        "run_dir": str(run_dir),
        "checkpoint_path": str(ckpt_path),
        "variant": "vanilla" if is_vanilla(str(run_dir)) else "finetuned",
        "meta_acc": ckpt.get("meta_acc"),
        "parent_acc": ckpt.get("parent_acc"),
        "epoch": ckpt.get("epoch"),
        "encoder_model": ckpt.get("encoder_model"),
        "num_concepts": ckpt.get("num_concepts"),
        "num_parents": ckpt.get("num_parents"),
    }
    return row


def stage2_mode_root(results_root: Path, ablation_mode: str) -> Path:
    return results_root / f"stage2_cross_attention_{ablation_mode}"


def find_stage2_latest_by_variant(results_root: Path, ablation_mode: str) -> Dict[str, Path]:
    root = stage2_mode_root(results_root, ablation_mode)
    if not root.exists():
        return {}

    run_dirs = [p for p in root.iterdir() if p.is_dir()]
    by_variant: Dict[str, List[Path]] = {"vanilla": [], "finetuned": []}

    for rd in run_dirs:
        ckpt_path = best_checkpoint_path(rd, "checkpoints/best_model_stage2_xattn.pt")
        if ckpt_path is None:
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu")
        stage1_ckpt = str(ckpt.get("stage1_checkpoint", ""))
        variant = "vanilla" if (is_vanilla(stage1_ckpt) or is_vanilla(str(rd))) else "finetuned"
        by_variant[variant].append(rd)

    out: Dict[str, Path] = {}
    for variant, dirs in by_variant.items():
        if dirs:
            out[variant] = max(dirs, key=lambda p: p.stat().st_mtime)
    return out


def flatten_val_metrics(prefix: str, val_metrics: Dict) -> Dict:
    out = {}
    if not isinstance(val_metrics, dict):
        return out
    for k, v in val_metrics.items():
        out[f"{prefix}{k}"] = v
    return out


def load_stage2_metrics(run_dir: Path, ablation_mode: str) -> Optional[Dict]:
    ckpt_path = best_checkpoint_path(run_dir, "checkpoints/best_model_stage2_xattn.pt")
    if ckpt_path is None:
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu")
    stage1_ckpt = str(ckpt.get("stage1_checkpoint", ""))
    variant = "vanilla" if (is_vanilla(stage1_ckpt) or is_vanilla(str(run_dir))) else "finetuned"

    row = {
        "stage": 2,
        "ablation_mode": ablation_mode,
        "run_dir": str(run_dir),
        "checkpoint_path": str(ckpt_path),
        "variant": variant,
        "stage1_checkpoint": stage1_ckpt,
        "val_median_error_km": ckpt.get("val_median_error"),
        "coord_output_dim": ckpt.get("coord_output_dim"),
        "num_cells": ckpt.get("num_cells"),
        "encoder_model": ckpt.get("encoder_model"),
        "epoch": ckpt.get("epoch"),
    }

    row.update(flatten_val_metrics("val_", ckpt.get("val_metrics", {})))
    return row


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=str, default="/scratch-shared/pnair/Project_AI/results")
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--vanilla_stage1_dir", type=str, default=None)
    parser.add_argument("--finetuned_stage1_dir", type=str, default=None)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    stage1_root = results_root / "stage1-prototype"

    if args.output_csv is None:
        output_csv = results_root / "comparisons" / "vanilla_vs_finetuned_latest.csv"
    else:
        output_csv = Path(args.output_csv)
    ensure_parent_dir(output_csv)

    if args.vanilla_stage1_dir and args.finetuned_stage1_dir:
        vanilla_stage1_dir = Path(args.vanilla_stage1_dir)
        finetuned_stage1_dir = Path(args.finetuned_stage1_dir)
    else:
        vanilla_stage1_dir, finetuned_stage1_dir = pick_latest_stage1_dirs(stage1_root)

    rows: List[Dict] = []

    if vanilla_stage1_dir is not None:
        r = load_stage1_metrics(vanilla_stage1_dir)
        if r is not None:
            r["variant"] = "vanilla"
            rows.append(r)

    if finetuned_stage1_dir is not None:
        r = load_stage1_metrics(finetuned_stage1_dir)
        if r is not None:
            r["variant"] = "finetuned"
            rows.append(r)

    for mode in ["concept_only", "image_only", "both"]:
        latest_by_variant = find_stage2_latest_by_variant(results_root, mode)
        for variant, run_dir in latest_by_variant.items():
            r = load_stage2_metrics(run_dir, mode)
            if r is None:
                continue
            r["variant"] = variant
            rows.append(r)

    if not rows:
        raise RuntimeError(f"No comparable runs found under {results_root}")

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["stage", "ablation_mode", "variant"]).reset_index(drop=True)
    df.to_csv(output_csv, index=False)

    print(f"Wrote: {output_csv}")
    summary_cols = ["stage", "ablation_mode", "variant", "meta_acc", "parent_acc", "val_median_error_km"]
    summary_cols = [c for c in summary_cols if c in df.columns]
    print(df[summary_cols].to_string(index=False))


if __name__ == "__main__":
    main()







