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
from pathlib import Path
from typing import Dict

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


def load_stage1_checkpoint_for_eval(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple:
    """Load Stage 1 checkpoint for evaluation."""
    logger.info(f"Loading Stage 1 checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
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
        images = batch["image"].to(device)
        concept_idx = batch["concept_idx"].to(device)
        parent_idx = batch["parent_idx"].to(device)
        
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 checkpoint on test split")
    parser.add_argument("--stage1_checkpoint", type=str, required=True,
                        help="Path to Stage 1 checkpoint")
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
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ckpt_name = Path(args.stage1_checkpoint).stem
        output_dir = Path("results") / "evals" / ckpt_name
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
    test_dataset = SubsetDataset(full_dataset, indices=[full_dataset.samples.index(s) for s in test_samples])
    
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
    results_json = {
        "stage1_checkpoint": str(args.stage1_checkpoint),
        "splits_json": str(args.splits_json),
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
    logger.info(f"Meta Top-1 Accuracy: {metrics['meta_acc']:.4f}")
    logger.info(f"Meta Top-5 Accuracy: {metrics['meta_acc_top5']:.4f}")
    logger.info(f"Parent Top-1 Accuracy: {metrics['parent_acc']:.4f}")
    logger.info(f"Parent Top-5 Accuracy: {metrics['parent_acc_top5']:.4f}")
    logger.info(f"Test Samples: {metrics['num_samples']}")
    logger.info("="*60)


if __name__ == "__main__":
    main()


