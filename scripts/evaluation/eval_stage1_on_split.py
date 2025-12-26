#!/usr/bin/env python3
"""
Evaluate Stage 1 checkpoint on internal test split from splits.json.

Computes:
- Meta concept top-1 and top-5 accuracy
- Parent concept top-1 and top-5 accuracy
- Saves results to JSON and CSV
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


from src.dataset import (
    PanoramaCBMDataset,
    get_transforms_from_processor,
    load_splits_from_json,
)
from src.models.streetclip_encoder import StreetCLIPEncoder, StreetCLIPConfig
from src.models.concept_aware_cbm import Stage1ConceptModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Datetime pattern for run directories
DATETIME_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")

def _is_vanilla_from_stage0(stage0_checkpoint) -> bool:
    if stage0_checkpoint is None:
        return True
    s = str(stage0_checkpoint).strip()
    return s == "" or s.lower() == "none"


def find_latest_datetime_dir(parent_dir: Path, exclude_subdirs: Optional[List[str]] = None) -> Optional[Path]:
    """Find the subdirectory with the latest datetime timestamp."""
    if exclude_subdirs is None:
        exclude_subdirs = []
    
    datetime_dirs = []
    for subdir in parent_dir.iterdir():
        if subdir.is_dir() and subdir.name not in exclude_subdirs:
            if DATETIME_PATTERN.match(subdir.name):
                datetime_dirs.append(subdir)
    
    if not datetime_dirs:
        return None
    
    # Sort by name (datetime format sorts chronologically)
    datetime_dirs.sort(key=lambda x: x.name, reverse=True)
    return datetime_dirs[0]


def find_latest_stage1_checkpoints(results_root: Path) -> List[Path]:
    """Find latest Stage 1 checkpoints for both variants."""
    checkpoints = []
    results_root = Path(results_root)
    
    # Finetuned: results/stage1-prototype/geolocal_StreetCLIP/<latest_datetime>/checkpoints/best_model_stage1.pt
    finetuned_dir = results_root / "stage1-prototype" / "geolocal_StreetCLIP"
    if finetuned_dir.exists():
        latest_dt = find_latest_datetime_dir(finetuned_dir)
        if latest_dt:
            ckpt_path = latest_dt / "checkpoints" / "best_model_stage1.pt"
            if ckpt_path.exists():
                checkpoints.append(ckpt_path)
                logger.info(f"Found latest finetuned Stage 1 checkpoint: {ckpt_path}")
    
    # Vanilla: results/stage1-prototype/vanilla_streetclip_no_stage0/<latest_datetime>/checkpoints/best_model_stage1.pt
    vanilla_dir = results_root / "stage1-prototype" / "vanilla_streetclip_no_stage0"
    if vanilla_dir.exists():
        latest_dt = find_latest_datetime_dir(vanilla_dir)
        if latest_dt:
            ckpt_path = latest_dt / "checkpoints" / "best_model_stage1.pt"
            if ckpt_path.exists():
                checkpoints.append(ckpt_path)
                logger.info(f"Found latest vanilla Stage 1 checkpoint: {ckpt_path}")
    
    return checkpoints


def _default_output_dir_for_stage1_checkpoint(checkpoint_path: Path) -> Path:
    """
    Create a unique output directory for a checkpoint based on the results folder structure:
      results/stage1-prototype/<run_group>/<datetime>/checkpoints/best_model_stage1.pt
    -> results/evals/stage1__<run_group>__<datetime>
    """
    ckpt = checkpoint_path.resolve()
    parts = ckpt.parts
    if "results" in parts:
        rel = Path(*parts[parts.index("results") + 1 :])
    else:
        rel = ckpt

    # Expect: <something>/<run_group>/<datetime>/checkpoints/<file>
    if len(rel.parts) >= 4 and rel.parts[-2] == "checkpoints":
        run_group = rel.parts[-4]
        run_dt = rel.parts[-3]
    else:
        run_group = checkpoint_path.parent.parent.parent.name
        run_dt = checkpoint_path.parent.parent.name

    safe = f"stage1__{run_group}__{run_dt}"
    safe = "".join(ch if (ch.isalnum() or ch in ("_", "-", ".")) else "_" for ch in safe)
    return Path("results") / "evals" / safe


def load_stage1_checkpoint_for_eval(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple:
    """Load Stage 1 checkpoint for evaluation."""
    logger.info(f"Loading Stage 1 checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Initialize encoder
    encoder_model = checkpoint.get("encoder_model", "geolocal/StreetCLIP")
    encoder_config = StreetCLIPConfig(model_name=encoder_model, finetune=False, device=device)
    image_encoder = StreetCLIPEncoder(encoder_config)
    image_encoder.eval()
    for param in image_encoder.parameters():
        param.requires_grad = False
    
    # Create model
    model = Stage1ConceptModel(
        image_encoder=image_encoder,
        T_meta=checkpoint["T_meta_base"].to(device),
        T_parent=checkpoint["T_parent_base"].to(device),
        meta_to_parent_idx=checkpoint["meta_to_parent_idx"].to(device),
        streetclip_dim=768,
        concept_emb_dim=512,
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    concept_info = {
        "concept_names": checkpoint.get("concept_names", []),
        "parent_names": checkpoint.get("parent_names", []),
        "concept_to_idx": checkpoint.get("concept_to_idx", {}),
        "parent_to_idx": checkpoint.get("parent_to_idx", {}),
        "stage0_checkpoint": checkpoint.get("stage0_checkpoint"),
    }
    
    return model, concept_info, checkpoint


@torch.no_grad()
def evaluate_stage1(
    model: Stage1ConceptModel,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict:
    """Evaluate Stage 1 model on test set."""
    model.eval()
    
    total_count = 0
    total_meta_correct = 0
    total_parent_correct = 0
    total_meta_correct_top5 = 0
    total_parent_correct_top5 = 0
    
    for batch in tqdm(test_loader, desc="Evaluating"):
        # Unpack tuple: (images, concept_idx, parent_idx, target_idx, coordinates, metadata)
        images, concept_idx, parent_idx, target_idx, coordinates, metadata = batch
        images = images.to(device)
        if isinstance(concept_idx, torch.Tensor):
            concept_idx = concept_idx.to(device)
        else:
            concept_idx = torch.tensor(concept_idx, dtype=torch.long, device=device)
        if isinstance(parent_idx, torch.Tensor):
            parent_idx = parent_idx.to(device)
        else:
            parent_idx = torch.tensor(parent_idx, dtype=torch.long, device=device)
        
        # Forward pass
        outputs = model(images)
        meta_logits = outputs["meta_logits"]
        parent_logits = outputs["parent_logits"]
        
        # Top-1 accuracy
        pred_meta = meta_logits.argmax(dim=1)
        pred_parent = parent_logits.argmax(dim=1)
        total_meta_correct += (pred_meta == concept_idx).sum().item()
        total_parent_correct += (pred_parent == parent_idx).sum().item()
        
        # Top-5 accuracy
        _, pred_meta_top5 = meta_logits.topk(5, dim=1, largest=True, sorted=True)
        total_meta_correct_top5 += (pred_meta_top5 == concept_idx.view(-1, 1)).sum().item()
        
        _, pred_parent_top5 = parent_logits.topk(5, dim=1, largest=True, sorted=True)
        total_parent_correct_top5 += (pred_parent_top5 == parent_idx.view(-1, 1)).sum().item()
        
        total_count += len(concept_idx)
    
    if total_count == 0:
        return {"error": "No samples processed"}
    
    return {
        "meta_acc": total_meta_correct / total_count,
        "parent_acc": total_parent_correct / total_count,
        "meta_acc_top5": total_meta_correct_top5 / total_count,
        "parent_acc_top5": total_parent_correct_top5 / total_count,
        "num_samples": total_count,
    }


def evaluate_single_stage1_checkpoint(
    checkpoint_path: Path,
    csv_path: str,
    splits_json: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Optional[Dict]:
    """Evaluate a single Stage 1 checkpoint and return results dict."""
    try:
        # Load checkpoint
        model, concept_info, checkpoint = load_stage1_checkpoint_for_eval(
            checkpoint_path,
            device,
        )
        
        # Load splits
        with open(splits_json, 'r') as f:
            splits_data = json.load(f)
        
        test_pano_ids = set(splits_data["test_pano_ids"])
        
        # Load dataset
        full_dataset = PanoramaCBMDataset(
            csv_path=csv_path,
            data_root=data_root,
            transform=get_transforms_from_processor(model.image_encoder.image_processor),
        )
        
        # Filter to test samples
        test_samples = [s for s in full_dataset.samples if s["pano_id"] in test_pano_ids]
        
        if len(test_samples) == 0:
            logger.warning(f"No test samples found for {checkpoint_path}")
            return None
        
        # Create test dataset (subset)
        from src.dataset import SubsetDataset
        test_dataset = SubsetDataset(full_dataset, test_samples)
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        # Evaluate
        metrics = evaluate_stage1(model, test_loader, device)
        
        if "error" in metrics:
            logger.warning(f"Evaluation error for {checkpoint_path}: {metrics['error']}")
            return None
        
        # Determine variant from checkpoint path
        checkpoint_path_str = str(checkpoint_path)
        if "geolocal_StreetCLIP" in checkpoint_path_str:
            variant = "finetuned"
        elif "vanilla_streetclip_no_stage0" in checkpoint_path_str:
            variant = "vanilla"
        else:
            # Fallback to checkpoint contents if path doesn't match expected patterns
            variant = "vanilla" if _is_vanilla_from_stage0(checkpoint.get("stage0_checkpoint")) else "finetuned"
        
        return {
            "checkpoint": str(checkpoint_path),
            "checkpoint_name": checkpoint_path.name,
            "variant": variant,
            "meta_acc": metrics["meta_acc"],
            "parent_acc": metrics["parent_acc"],
            "meta_acc_top5": metrics["meta_acc_top5"],
            "parent_acc_top5": metrics["parent_acc_top5"],
            "num_samples": metrics["num_samples"],
            "stage0_checkpoint": str(checkpoint.get("stage0_checkpoint", "None")),
            "encoder_model": checkpoint.get("encoder_model", "unknown"),
        }
    except Exception as e:
        logger.error(f"Failed to evaluate {checkpoint_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 checkpoint on test split")
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                        help="Path to Stage 1 checkpoint (required if not using --batch_mode)")
    parser.add_argument("--batch_mode", action="store_true",
                        help="Auto-detect and evaluate all latest Stage 1 checkpoints")
    parser.add_argument("--results_root", type=str, default="results",
                        help="Root directory containing results (for batch mode)")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to CSV dataset")
    parser.add_argument("--splits_json", type=str, required=True,
                        help="Path to splits.json file")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results. Default: results/evals/<checkpoint_name>")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Batch mode: auto-detect and evaluate all checkpoints
    if args.batch_mode:
        logger.info("Batch mode: Auto-detecting latest Stage 1 checkpoints...")
        results_root = Path(args.results_root)
        checkpoint_paths = find_latest_stage1_checkpoints(results_root)
        
        if len(checkpoint_paths) == 0:
            logger.error("No Stage 1 checkpoints found!")
            return
        
        logger.info(f"Found {len(checkpoint_paths)} checkpoint(s) to evaluate")
        
        # Evaluate all checkpoints
        all_results = []
        for ckpt_path in checkpoint_paths:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating: {ckpt_path}")
            logger.info(f"{'='*60}")
            
            result = evaluate_single_stage1_checkpoint(
                checkpoint_path=ckpt_path,
                csv_path=args.csv_path,
                splits_json=args.splits_json,
                data_root=args.data_root,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
            )
            
            if result:
                all_results.append(result)
                logger.info(f"  Meta Top-1 Accuracy: {result['meta_acc']:.4f}")
                logger.info(f"  Parent Top-1 Accuracy: {result['parent_acc']:.4f}")
        
        # Save consolidated CSV
        if len(all_results) > 0:
            output_dir = Path(args.results_root) / "evals"
            output_dir.mkdir(parents=True, exist_ok=True)
            consolidated_csv = output_dir / "stage1_test_consolidated.csv"
            
            df = pd.DataFrame(all_results)
            df.to_csv(consolidated_csv, index=False)
            logger.info(f"\nSaved consolidated CSV to {consolidated_csv}")
            logger.info(f"Total evaluations: {len(df)}")
        else:
            logger.warning("No results to save!")
        
        return
    
    # Single checkpoint mode (backward compatible)
    if args.stage1_checkpoint is None:
        parser.error("--stage1_checkpoint is required when not using --batch_mode")
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = _default_output_dir_for_stage1_checkpoint(Path(args.stage1_checkpoint))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    model, concept_info, checkpoint = load_stage1_checkpoint_for_eval(
        Path(args.stage1_checkpoint),
        device,
    )
    
    # Load splits
    logger.info(f"Loading splits from {args.splits_json}")
    with open(args.splits_json, 'r') as f:
        splits_data = json.load(f)
    
    test_pano_ids = set(splits_data["test_pano_ids"])
    logger.info(f"Test split: {len(test_pano_ids)} samples")
    
    # Load dataset
    logger.info(f"Loading dataset from {args.csv_path}")
    full_dataset = PanoramaCBMDataset(
        csv_path=args.csv_path,
        data_root=args.data_root,
        transform=get_transforms_from_processor(model.image_encoder.image_processor),
    )
    
    # Filter to test samples
    test_samples = [s for s in full_dataset.samples if s["pano_id"] in test_pano_ids]
    logger.info(f"Found {len(test_samples)} test samples in dataset")
    
    if len(test_samples) == 0:
        raise RuntimeError("No test samples found!")
    
    # Create test dataset (subset)
    from src.dataset import SubsetDataset
    test_dataset = SubsetDataset(full_dataset, test_samples)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Evaluate
    metrics = evaluate_stage1(model, test_loader, device)
    
    # Save results
    variant = "vanilla" if _is_vanilla_from_stage0(checkpoint.get("stage0_checkpoint")) else "finetuned"
    results_json = {
        "stage1_checkpoint": str(args.stage1_checkpoint),
        "splits_json": str(args.splits_json),
        "variant": variant,
        "stage0_checkpoint": checkpoint.get("stage0_checkpoint"),
        "encoder_model": checkpoint.get("encoder_model"),
        "test_samples": metrics.get("num_samples", 0),
        "metrics": {k: float(v) for k, v in metrics.items() if k != "error"},
    }
    
    json_path = output_dir / "test_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"Saved metrics to {json_path}")
    
    # Save CSV summary
    csv_data = {
        "checkpoint": [Path(args.stage1_checkpoint).name],
        "meta_acc": [metrics["meta_acc"]],
        "parent_acc": [metrics["parent_acc"]],
        "meta_acc_top5": [metrics["meta_acc_top5"]],
        "parent_acc_top5": [metrics["parent_acc_top5"]],
        "num_samples": [metrics["num_samples"]],
        "stage0_checkpoint": [checkpoint.get("stage0_checkpoint", "None")],
    }
    df_results = pd.DataFrame(csv_data)
    csv_path = output_dir / "test_metrics.csv"
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV summary to {csv_path}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Test Evaluation Results")
    logger.info("="*60)
    logger.info(f"Checkpoint: {args.stage1_checkpoint}")
    logger.info(f"Variant: {variant}")
    logger.info(f"Stage0 checkpoint: {checkpoint.get('stage0_checkpoint', 'None')}")
    logger.info(f"Splits: {args.splits_json}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Meta Top-1 Accuracy: {metrics['meta_acc']:.4f}")
    logger.info(f"Meta Top-5 Accuracy: {metrics['meta_acc_top5']:.4f}")
    logger.info(f"Parent Top-1 Accuracy: {metrics['parent_acc']:.4f}")
    logger.info(f"Parent Top-5 Accuracy: {metrics['parent_acc_top5']:.4f}")
    logger.info(f"Test Samples: {metrics['num_samples']}")
    logger.info("="*60)


if __name__ == "__main__":
    main()


