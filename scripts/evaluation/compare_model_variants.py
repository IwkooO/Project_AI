#!/usr/bin/env python3
"""
Compare three model variants on test set:
1. img+conc (both): stage3_joint_both_scratchgeo_dim256_pos_latefusion_2.5ct_0.01gatereg
2. conc only: stage3_joint_concept_only_scratchgeo_dim256_pos_latefusion_2.5ct_0gatereg
3. img only: stage3_joint_image_only_scratchgeo_dim256_pos_latefusion_2.5ct_0gatereg

Analyzes when img+conc is better than img only, and when conc only differs from img only.
Creates comprehensive visualizations for report.

Usage:
    python scripts/evaluation/compare_model_variants.py \
        --checkpoint-both checkpoints/stage3_joint_latefusion/stage3_joint_both_scratchgeo_dim256_pos_latefusion_2.5ct_0.01gatereg \
        --checkpoint-concept checkpoints/stage3_joint_latefusion/stage3_joint_concept_only_scratchgeo_dim256_pos_latefusion_2.5ct_0gatereg \
        --checkpoint-image checkpoints/stage3_joint_latefusion/stage3_joint_image_only_scratchgeo_dim256_pos_latefusion_2.5ct_0gatereg \
        --test-csv data/splits/dataset_test.csv \
        --concept-data-dir data/concept_data_v2 \
        --cached-dir /scratch-shared/igodzwon/Project_AI/data/6921d7831744c5356b098bf7_balanced/cached_streetclip_v2 \
        --output-dir results/model_comparison
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cbm.phase1.model import Phase1CBMTopKMil
from cbm.phase1.data import ConceptDataset
from cbm.phase2.model import ConceptEmbeddingAdapter, Stage2CrossAttentionGeoHead
from cbm.joint.train import eval_epoch, collate_fn_joint, JointDataset, build_concept_vectors
from cbm.phase2.geocells import assign_geocells, compute_offsets
from cbm.phase2.metrics import xyz_to_latlng, haversine_km

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Model names for display
MODEL_NAMES = {
    'both': 'Image + Concepts',
    'concept_only': 'Concepts Only',
    'image_only': 'Image Only'
}

MODEL_COLORS = {
    'both': '#2ecc71',      # Green
    'concept_only': '#e74c3c',  # Red
    'image_only': '#3498db'    # Blue
}


def load_model_from_checkpoint(
    checkpoint_dir: Path,
    num_concepts: int,
    patch_dim: int,
    device: torch.device,
    args
) -> Tuple:
    """
    Load phase1 model, stage2 model, and concept adapter from checkpoint directory.
    
    Returns:
        phase1_model, stage2_model, concept_adapter, centers_xyz, mode, 
        pooled_projection, actual_pooled_dim, checkpoint_pooled_dim
    """
    # Try to load from best_joint.pt first
    joint_ckpt_path = checkpoint_dir / "best_joint.pt"
    use_joint_checkpoint = joint_ckpt_path.exists()
    
    if use_joint_checkpoint:
        phase1_ckpt_path = joint_ckpt_path
        print(f"   Using best_joint.pt for entire architecture")
    else:
        phase1_ckpt_path = checkpoint_dir / "phase1" / "best_phase1.pt"
        if not phase1_ckpt_path.exists():
            raise FileNotFoundError(f"No checkpoint found. Expected either best_joint.pt or phase1/best_phase1.pt in {checkpoint_dir}")
    
    print(f"   Loading Phase1 from: {phase1_ckpt_path}")
    phase1_checkpoint = torch.load(phase1_ckpt_path, map_location=device, weights_only=False)
    
    # Get Phase1 state dict
    if use_joint_checkpoint:
        phase1_state = phase1_checkpoint.get('phase1_model_state_dict', {})
    else:
        phase1_state = phase1_checkpoint.get('model_state_dict', {})
    
    if not phase1_state:
        raise ValueError(f"Could not find Phase1 state dict in checkpoint. Keys: {list(phase1_checkpoint.keys())}")
    
    # Detect Phase1 model configuration from state dict
    has_pos_embed = any('pos_embed' in k for k in phase1_state.keys())
    has_per_concept_tau = any('concept_tau_logit' in k for k in phase1_state.keys())
    
    # Detect concept_dim from query tensor shape
    checkpoint_concept_dim = args.concept_dim
    for k, v in phase1_state.items():
        if k == 'query' or k.endswith('.query'):
            checkpoint_concept_dim = v.shape[1]
            print(f"   Detected concept_dim from Phase1 query: {checkpoint_concept_dim}")
            break
    
    # Detect patch_dim from patch_proj
    checkpoint_patch_dim = patch_dim
    for k, v in phase1_state.items():
        if 'patch_proj' in k and 'weight' in k:
            if len(v.shape) == 2:
                checkpoint_patch_dim = v.shape[1]
                print(f"   Detected patch_dim from Phase1: {checkpoint_patch_dim}")
                break
    
    # Create Phase1 model
    phase1_model = Phase1CBMTopKMil(
        num_concepts=num_concepts,
        patch_dim=checkpoint_patch_dim,
        concept_dim=checkpoint_concept_dim,
        dropout=args.phase1_dropout,
        mil_topk=args.mil_topk,
        mil_tau=args.mil_tau,
        mix_depth=args.mix_depth,
        mix_heads=args.mix_heads,
        mix_mlp_ratio=args.mix_mlp_ratio,
        mix_local_kernel_size=args.mix_local_kernel if args.mix_local_kernel > 0 else None,
        proj_type=args.proj_type,
        use_pos_encoding=has_pos_embed,
        use_per_concept_tau=has_per_concept_tau,
    )
    
    phase1_model.load_state_dict(phase1_state)
    phase1_model = phase1_model.to(device)
    phase1_model.eval()
    print(f"   Phase1 loaded (concept_dim={checkpoint_concept_dim}, pos={has_pos_embed}, adaptive_tau={has_per_concept_tau})")
    
    # Load geocells
    if use_joint_checkpoint and 'geocell_info' in phase1_checkpoint:
        geocells_data = phase1_checkpoint['geocell_info']
        centers_xyz = np.array(geocells_data["centers_xyz"])
        num_cells = len(centers_xyz)
        print(f"   Loaded {num_cells} geocells from joint checkpoint")
    else:
        geocells_path = checkpoint_dir / "phase2" / "geocells.json"
        if not geocells_path.exists():
            raise FileNotFoundError(f"Geocells not found. Expected in joint checkpoint or at {geocells_path}")
        
        with open(geocells_path) as f:
            geocells_data = json.load(f)
        centers_xyz = np.array(geocells_data["centers_xyz"])
        num_cells = len(centers_xyz)
        print(f"   Loaded {num_cells} geocells from phase2/geocells.json")
    
    # Load Stage2
    if use_joint_checkpoint:
        stage2_checkpoint = phase1_checkpoint
        print(f"   Loading Stage2 from best_joint.pt (reusing loaded checkpoint)")
    else:
        stage2_ckpt_path = checkpoint_dir / "phase2" / "best_phase2.pt"
        if not stage2_ckpt_path.exists():
            raise FileNotFoundError(f"No phase2 checkpoint found in {checkpoint_dir}")
        print(f"   Loading Stage2 from: {stage2_ckpt_path}")
        stage2_checkpoint = torch.load(stage2_ckpt_path, map_location=device, weights_only=False)
    
    stage2_state = stage2_checkpoint.get('stage2_model_state_dict', {})
    if not stage2_state:
        raise ValueError(f"Could not find 'stage2_model_state_dict' in checkpoint. Keys: {list(stage2_checkpoint.keys())}")
    
    # Detect mode from Stage2 state dict
    keys = list(stage2_state.keys())
    has_image_adapter = any(k.startswith('image_adapter') for k in keys)
    has_concept_adapter = any(k.startswith('concept_adapter') for k in keys)
    has_fusion_gate = any(k.startswith('fusion_gate') for k in keys)
    has_pooled_proj = any(k.startswith('pooled_proj') for k in keys)
    has_concept_proj = any(k.startswith('concept_proj') for k in keys)
    has_old_concept_proj = any('concept_proj.0.weight' in k for k in keys)
    has_old_pooled_proj = any('pooled_proj.0.weight' in k for k in keys)
    
    if has_fusion_gate or (has_image_adapter and has_concept_adapter):
        mode = "both"
    elif has_concept_adapter or (has_concept_proj and not has_pooled_proj) or has_old_concept_proj:
        mode = "concept_only"
    elif has_image_adapter or has_pooled_proj or has_old_pooled_proj:
        mode = "image_only"
    else:
        mode = "both"
        print(f"   Warning: Could not detect mode from state dict. Defaulting to 'both'.")
    print(f"   Detected mode: {mode}")
    
    # Detect pooled_dim
    checkpoint_pooled_dim = 768  # Default for StreetCLIP
    for k, v in stage2_state.items():
        if k == 'pooled_proj.weight' or k == 'pooled_proj.0.weight':
            checkpoint_pooled_dim = v.shape[1]
            print(f"   Detected pooled_dim from checkpoint: {checkpoint_pooled_dim}")
            break
    
    # Create Stage2 model
    stage2_model = Stage2CrossAttentionGeoHead(
        concept_dim=checkpoint_concept_dim,
        patch_dim=checkpoint_patch_dim,
        num_cells=num_cells,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        mode=mode,
        pooled_dim=checkpoint_pooled_dim,
    )
    
    # Map old Sequential format to new Linear format
    mapped_state = {}
    for k, v in stage2_state.items():
        if 'concept_refinement' in k:
            continue
        if 'fusion_gate.3' in k:
            if 'weight' in k and v.shape[0] != 2:
                print(f"   Skipping {k} (shape {v.shape}, expected [2, {v.shape[1]}])")
                continue
            elif 'bias' in k and v.shape[0] != 2:
                print(f"   Skipping {k} (shape {v.shape}, expected [2])")
                continue
        
        if k == 'concept_proj.0.weight':
            mapped_state['concept_proj.weight'] = v
        elif k == 'concept_proj.0.bias':
            mapped_state['concept_proj.bias'] = v
        elif k == 'pooled_proj.0.weight':
            mapped_state['pooled_proj.weight'] = v
        elif k == 'pooled_proj.0.bias':
            mapped_state['pooled_proj.bias'] = v
        else:
            mapped_state[k] = v
    
    missing_keys, unexpected_keys = stage2_model.load_state_dict(mapped_state, strict=False)
    if missing_keys:
        print(f"   Warning: Missing keys (will use random init): {missing_keys[:5]}..." if len(missing_keys) > 5 else f"   Warning: Missing keys: {missing_keys}")
    if unexpected_keys:
        unexpected_filtered = [k for k in unexpected_keys if 'concept_refinement' not in k]
        if unexpected_filtered:
            print(f"   Warning: Unexpected keys (ignored): {unexpected_filtered[:5]}..." if len(unexpected_filtered) > 5 else f"   Warning: Unexpected keys: {unexpected_filtered}")
    
    stage2_model = stage2_model.to(device)
    stage2_model.eval()
    print(f"   Stage2 loaded (mode={mode})")
    
    # Load concept adapter
    concept_temp = getattr(args, 'concept_temperature', 5.0)
    if 'concept_adapter_state_dict' in stage2_checkpoint:
        adapter_state = stage2_checkpoint['concept_adapter_state_dict']
        concept_vectors = adapter_state['concept_vectors']
        concept_adapter = ConceptEmbeddingAdapter(concept_vectors, temperature=concept_temp)
        concept_adapter.load_state_dict(adapter_state)
        print(f"   Concept adapter loaded (vectors shape: {tuple(concept_vectors.shape)})")
    else:
        print("   Warning: concept_adapter_state_dict not found. Using random vectors.")
        concept_vectors = torch.randn(num_concepts, checkpoint_concept_dim)
        concept_adapter = ConceptEmbeddingAdapter(concept_vectors, temperature=concept_temp)
    
    concept_adapter = concept_adapter.to(device)
    concept_adapter.eval()
    
    # Create pooled_emb projection if needed (not used in this script, but kept for compatibility)
    pooled_projection = None
    actual_pooled_dim = None
    
    return phase1_model, stage2_model, concept_adapter, centers_xyz, mode, pooled_projection, actual_pooled_dim, checkpoint_pooled_dim


@torch.no_grad()
def evaluate_model(
    phase1_model: Phase1CBMTopKMil,
    stage2_model: Stage2CrossAttentionGeoHead,
    concept_adapter: ConceptEmbeddingAdapter,
    test_loader: DataLoader,
    device: torch.device,
    centers_xyz: np.ndarray,
    idx_to_concept: Dict[int, str],
    pooled_projection: Optional[nn.Module] = None,
) -> Dict:
    """Evaluate model and return detailed predictions."""
    
    all_predictions = []
    all_errors = []
    all_concept_preds = []
    all_concept_probs = []
    all_gate_values = []
    all_attention_weights = []
    
    phase1_model.eval()
    stage2_model.eval()
    concept_adapter.eval()
    if pooled_projection is not None:
        pooled_projection.eval()
    
    for batch in tqdm(test_loader, desc="Evaluating"):
        # Unpack batch (collate_fn_joint returns dict with 'patches', not 'patch_tokens')
        patch_tokens = batch['patches'].to(device)
        pooled_emb = batch['pooled_emb'].to(device)
        c_labels = batch['c_labels'].to(device)
        coords = batch['coords'].to(device)
        cell_labels = batch['cell_labels'].to(device)
        
        # Project pooled_emb if needed
        if pooled_projection is not None:
            pooled_emb = pooled_projection(pooled_emb)
        
        # Phase 1: Concept prediction
        c_logits, c_hidden, attn_weights, _ = phase1_model(patch_tokens)
        c_probs = torch.softmax(c_logits, dim=1)
        c_preds = c_logits.argmax(dim=1)
        
        # Phase 2: Geolocation
        concept_emb = concept_adapter(c_logits)
        forward_output = stage2_model(concept_emb, patch_tokens, pooled_emb)
        
        if stage2_model.mode == "both":
            # In "both" mode, forward returns: (cell_logits, offset_pred, gate_info, img_h, concept_h)
            # where gate_info is a dict with 'gate' key
            cell_logits, offset_pred, gate_info, _, _ = forward_output
            if isinstance(gate_info, dict) and 'gate' in gate_info:
                gate = gate_info['gate']
                if gate is not None and isinstance(gate, torch.Tensor) and gate.numel() > 0:
                    # Gate shape is [B, 1] or [B], extract first value per sample
                    if gate.dim() == 2:
                        gate_val = gate[:, 0].cpu().numpy()
                    else:
                        gate_val = gate.cpu().numpy()
                else:
                    gate_val = np.zeros(len(cell_logits))
            else:
                gate_val = np.zeros(len(cell_logits))
        else:
            # In concept_only or image_only mode, forward returns: (cell_logits, offset_pred, gate)
            # where gate is None or a tensor
            cell_logits, offset_pred, gate = forward_output
            if gate is not None and isinstance(gate, torch.Tensor) and gate.numel() > 0:
                if gate.dim() == 2:
                    gate_val = gate[:, 0].cpu().numpy()
                else:
                    gate_val = gate.cpu().numpy()
            else:
                gate_val = np.zeros(len(cell_logits))
        
        # Convert to lat/lng
        pred_cells = cell_logits.argmax(dim=1).cpu().numpy()
        pred_cell_centers = centers_xyz[pred_cells]
        pred_xyz = pred_cell_centers + offset_pred.cpu().numpy()
        pred_lat, pred_lng = xyz_to_latlng(pred_xyz)
        
        # Compute errors
        true_lat = coords[:, 0].cpu().numpy()
        true_lng = coords[:, 1].cpu().numpy()
        errors_km = haversine_km(pred_lat, pred_lng, true_lat, true_lng)
        
        # Store predictions
        batch_size = len(pred_lat)
        for i in range(batch_size):
            all_predictions.append({
                'pred_lat': float(pred_lat[i]),
                'pred_lng': float(pred_lng[i]),
                'true_lat': float(true_lat[i]),
                'true_lng': float(true_lng[i]),
                'error_km': float(errors_km[i]),
                'pred_cell': int(pred_cells[i]),
                'true_cell': int(cell_labels[i].item()),
            })
            all_errors.append(float(errors_km[i]))
            all_concept_preds.append(int(c_preds[i].item()))
            all_concept_probs.append(c_probs[i].cpu().numpy().tolist())
            all_gate_values.append(float(gate_val[i]) if i < len(gate_val) else 0.0)
            # Store attention weights: [K, P] -> numpy
            all_attention_weights.append(attn_weights[i].cpu().numpy())
    
    return {
        'predictions': all_predictions,
        'errors': np.array(all_errors),
        'concept_preds': np.array(all_concept_preds),
        'concept_probs': all_concept_probs,
        'gate_values': np.array(all_gate_values),
        'attention_weights': all_attention_weights,
    }


def compute_metrics(results: Dict) -> Dict:
    """Compute summary metrics from results."""
    errors = results['errors']
    
    return {
        'mean_error': float(np.mean(errors)),
        'median_error': float(np.median(errors)),
        'std_error': float(np.std(errors)),
        'p25_error': float(np.percentile(errors, 25)),
        'p75_error': float(np.percentile(errors, 75)),
        'p90_error': float(np.percentile(errors, 90)),
        'p95_error': float(np.percentile(errors, 95)),
        'acc_1km': float(np.mean(errors <= 1.0)),
        'acc_10km': float(np.mean(errors <= 10.0)),
        'acc_100km': float(np.mean(errors <= 100.0)),
        'acc_1000km': float(np.mean(errors <= 1000.0)),
        'acc_2500km': float(np.mean(errors <= 2500.0)),
    }


def plot_performance_comparison(
    metrics_dict: Dict[str, Dict],
    output_path: Path
):
    """Plot overall performance comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    
    models = list(metrics_dict.keys())
    model_labels = [MODEL_NAMES[m] for m in models]
    colors = [MODEL_COLORS[m] for m in models]
    
    # 1. Mean and Median Error
    ax = axes[0, 0]
    mean_errors = [metrics_dict[m]['mean_error'] for m in models]
    median_errors = [metrics_dict[m]['median_error'] for m in models]
    x = np.arange(len(models))
    width = 0.35
    bars1 = ax.bar(x - width/2, mean_errors, width, label='Mean', color=colors, alpha=0.7)
    bars2 = ax.bar(x + width/2, median_errors, width, label='Median', color=colors, alpha=0.9,
                   hatch='///', edgecolor='white')
    # Add value labels on bars
    for bar, val in zip(bars1, mean_errors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, median_errors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax.set_ylabel('Error (km)', fontsize=11, fontweight='bold')
    ax.set_title('Mean and Median Geolocation Error', fontsize=13, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=10)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(mean_errors) * 1.2)
    
    # 2. Error Distribution (Box Plot)
    ax = axes[0, 1]
    error_data = [metrics_dict[m]['errors'] for m in models]
    bp = ax.boxplot(error_data, labels=model_labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_yscale('log')
    ax.set_ylabel('Error (km, log scale)', fontsize=11, fontweight='bold')
    ax.set_title('Error Distribution', fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', labelsize=10)
    
    # 3. Threshold Accuracies
    ax = axes[0, 2]
    thresholds = ['1km', '10km', '100km', '1000km', '2500km']
    x = np.arange(len(thresholds))
    width = 0.25
    for i, model in enumerate(models):
        accs = [
            metrics_dict[model]['acc_1km'],
            metrics_dict[model]['acc_10km'],
            metrics_dict[model]['acc_100km'],
            metrics_dict[model]['acc_1000km'],
            metrics_dict[model]['acc_2500km'],
        ]
        bars = ax.bar(x + i*width - width, accs, width, label=MODEL_NAMES[model], 
                      color=colors[i], alpha=0.8)
    ax.set_xlabel('Distance Threshold', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('Accuracy at Distance Thresholds', fontsize=13, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(thresholds, fontsize=10)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    
    # 4. Error Percentiles
    ax = axes[1, 0]
    percentiles = ['P25', 'P50', 'P75', 'P90', 'P95']
    x = np.arange(len(percentiles))
    width = 0.25
    for i, model in enumerate(models):
        values = [
            metrics_dict[model]['p25_error'],
            metrics_dict[model]['median_error'],
            metrics_dict[model]['p75_error'],
            metrics_dict[model]['p90_error'],
            metrics_dict[model]['p95_error'],
        ]
        ax.bar(x + i*width - width, values, width, label=MODEL_NAMES[model], 
               color=colors[i], alpha=0.8)
    ax.set_xlabel('Percentile', fontsize=11, fontweight='bold')
    ax.set_ylabel('Error (km)', fontsize=11, fontweight='bold')
    ax.set_title('Error Percentiles', fontsize=13, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(percentiles, fontsize=10)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 5. Error Histogram (log scale)
    ax = axes[1, 1]
    for i, model in enumerate(models):
        errors = metrics_dict[model]['errors']
        ax.hist(errors, bins=50, alpha=0.5, label=MODEL_NAMES[model], 
                color=colors[i], density=True, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Error (km)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax.set_title('Error Distribution (Histogram)', fontsize=13, fontweight='bold', pad=10)
    ax.set_xscale('log')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 6. Improvement over Image Only (with proper clamping)
    ax = axes[1, 2]
    if 'image_only' in metrics_dict and 'both' in metrics_dict:
        img_only_errors = metrics_dict['image_only']['errors']
        both_errors = metrics_dict['both']['errors']
        conc_errors = metrics_dict['concept_only']['errors']
        
        improvement_both = (img_only_errors - both_errors) / np.maximum(img_only_errors, 1e-6) * 100
        improvement_conc = (img_only_errors - conc_errors) / np.maximum(img_only_errors, 1e-6) * 100
        
        # Clamp to reasonable range for visualization
        improvement_both_clamped = np.clip(improvement_both, -200, 100)
        improvement_conc_clamped = np.clip(improvement_conc, -200, 100)
        
        ax.hist(improvement_both_clamped, bins=50, alpha=0.6, label='Image+Concepts vs Image', 
                color=MODEL_COLORS['both'], density=True)
        ax.hist(improvement_conc_clamped, bins=50, alpha=0.6, label='Concepts vs Image', 
                color=MODEL_COLORS['concept_only'], density=True)
        ax.axvline(0, color='black', linestyle='--', linewidth=2, label='No improvement')
        
        # Add mean improvement from mean errors (accurate measure)
        mean_improve_both = (np.mean(img_only_errors) - np.mean(both_errors)) / np.mean(img_only_errors) * 100
        ax.axvline(mean_improve_both, color=MODEL_COLORS['both'], linestyle='-', linewidth=2, alpha=0.8)
        ax.text(mean_improve_both + 5, ax.get_ylim()[1] * 0.9, f'{mean_improve_both:.1f}%', 
                color=MODEL_COLORS['both'], fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Per-Sample Improvement (%)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax.set_title('Improvement over Image Only\n(clamped to [-200%, 100%])', fontsize=13, fontweight='bold', pad=10)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=2.0, h_pad=3.0, w_pad=2.0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


def plot_error_analysis(
    results_dict: Dict[str, Dict],
    idx_to_concept: Dict[int, str],
    output_path: Path
):
    """Plot detailed error analysis comparing models."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    models = list(results_dict.keys())
    colors = [MODEL_COLORS[m] for m in models]
    
    # 1. When is img+conc better than img only?
    if 'both' in results_dict and 'image_only' in results_dict:
        ax = axes[0, 0]
        img_only_errors = results_dict['image_only']['errors']
        both_errors = results_dict['both']['errors']
        
        better_mask = both_errors < img_only_errors
        worse_mask = both_errors > img_only_errors
        same_mask = both_errors == img_only_errors
        
        # Use mean-based improvement (more accurate)
        mean_improvement = (np.mean(img_only_errors) - np.mean(both_errors)) / np.mean(img_only_errors) * 100
        median_improvement = np.median((img_only_errors - both_errors) / np.maximum(img_only_errors, 1e-6) * 100)
        
        ax.scatter(img_only_errors[better_mask], both_errors[better_mask], 
                  alpha=0.4, color='green', s=15, label=f'Better ({np.sum(better_mask)})', rasterized=True)
        ax.scatter(img_only_errors[worse_mask], both_errors[worse_mask], 
                  alpha=0.4, color='red', s=15, label=f'Worse ({np.sum(worse_mask)})', rasterized=True)
        
        # Diagonal line
        max_err = max(img_only_errors.max(), both_errors.max())
        ax.plot([0.1, max_err], [0.1, max_err], 'k--', linewidth=2, label='Equal performance', zorder=10)
        
        ax.set_xlabel('Image Only Error (km)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Image+Concepts Error (km)', fontsize=11, fontweight='bold')
        ax.set_title('Image+Concepts vs Image Only', fontsize=13, fontweight='bold', pad=10)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # Add statistics (positioned to avoid data overlap)
        stats_text = (f'Mean improvement: {mean_improvement:.1f}%\n'
                     f'Median improvement: {median_improvement:.1f}%\n'
                     f'Better: {np.sum(better_mask)} ({np.sum(better_mask)/len(better_mask)*100:.1f}%)')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9, 
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # 2. Concept Only vs Image Only
    if 'concept_only' in results_dict and 'image_only' in results_dict:
        ax = axes[0, 1]
        img_only_errors = results_dict['image_only']['errors']
        conc_errors = results_dict['concept_only']['errors']
        
        better_mask = conc_errors < img_only_errors
        worse_mask = conc_errors > img_only_errors
        
        # Use mean-based improvement (more accurate)
        mean_improvement = (np.mean(img_only_errors) - np.mean(conc_errors)) / np.mean(img_only_errors) * 100
        median_improvement = np.median((img_only_errors - conc_errors) / np.maximum(img_only_errors, 1e-6) * 100)
        
        ax.scatter(img_only_errors[better_mask], conc_errors[better_mask], 
                  alpha=0.4, color='green', s=15, label=f'Better ({np.sum(better_mask)})', rasterized=True)
        ax.scatter(img_only_errors[worse_mask], conc_errors[worse_mask], 
                  alpha=0.4, color='red', s=15, label=f'Worse ({np.sum(worse_mask)})', rasterized=True)
        
        max_err = max(img_only_errors.max(), conc_errors.max())
        ax.plot([0.1, max_err], [0.1, max_err], 'k--', linewidth=2, label='Equal performance', zorder=10)
        
        ax.set_xlabel('Image Only Error (km)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Concepts Only Error (km)', fontsize=11, fontweight='bold')
        ax.set_title('Concepts Only vs Image Only', fontsize=13, fontweight='bold', pad=10)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)
        
        stats_text = (f'Mean improvement: {mean_improvement:.1f}%\n'
                     f'Median improvement: {median_improvement:.1f}%\n'
                     f'Better: {np.sum(better_mask)} ({np.sum(better_mask)/len(better_mask)*100:.1f}%)')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9, 
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # 3. Error by concept prediction (for concepts only)
    if 'concept_only' in results_dict:
        ax = axes[1, 0]
        conc_preds = results_dict['concept_only']['concept_preds']
        conc_errors = results_dict['concept_only']['errors']
        
        # Get top concepts by frequency
        unique_concepts, counts = np.unique(conc_preds, return_counts=True)
        top_concept_indices = unique_concepts[np.argsort(counts)[-8:]]  # Reduced to 8 for better spacing
        
        concept_errors = []
        concept_labels = []
        for concept_idx in top_concept_indices:
            mask = conc_preds == concept_idx
            if np.sum(mask) > 0:
                concept_errors.append(conc_errors[mask])
                label = idx_to_concept.get(int(concept_idx), f'C{concept_idx}')
                # Truncate long labels
                if len(label) > 15:
                    label = label[:12] + '...'
                concept_labels.append(label)
        
        if concept_errors:
            bp = ax.boxplot(concept_errors, labels=concept_labels, patch_artist=True, widths=0.6)
            for patch in bp['boxes']:
                patch.set_facecolor(MODEL_COLORS['concept_only'])
                patch.set_alpha(0.7)
            ax.set_yscale('log')
            ax.set_ylabel('Error (km, log scale)', fontsize=11, fontweight='bold')
            ax.set_title('Error by Top Predicted Concept', fontsize=13, fontweight='bold', pad=10)
            ax.tick_params(axis='x', labelsize=9, rotation=45)
            for label in ax.get_xticklabels():
                label.set_horizontalalignment('right')
            ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Gate value distribution (for both model)
    if 'both' in results_dict:
        ax = axes[1, 1]
        gate_values = results_dict['both']['gate_values']
        
        # Plot gate distribution
        ax.hist(gate_values, bins=50, alpha=0.7, color=MODEL_COLORS['both'], 
                edgecolor='white', linewidth=0.5)
        ax.axvline(np.mean(gate_values), color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {np.mean(gate_values):.3f}')
        ax.axvline(np.median(gate_values), color='blue', linestyle='--', linewidth=2, 
                  label=f'Median: {np.median(gate_values):.3f}')
        ax.set_xlabel('Gate Value (0=Image, 1=Concepts)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Fusion Gate Value Distribution', fontsize=13, fontweight='bold', pad=10)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add statistics (positioned in upper left)
        stats_text = (f'Mean: {np.mean(gate_values):.3f}\n'
                     f'Std: {np.std(gate_values):.3f}\n'
                     f'Range: [{np.min(gate_values):.3f}, {np.max(gate_values):.3f}]')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9, 
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.tight_layout(pad=2.0, h_pad=3.0, w_pad=2.0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


def plot_concept_analysis(
    results_dict: Dict[str, Dict],
    idx_to_concept: Dict[int, str],
    output_path: Path
):
    """Plot concept prediction analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # 1. Concept prediction agreement (as bar chart)
    if 'both' in results_dict and 'concept_only' in results_dict:
        ax = axes[0, 0]
        both_preds = results_dict['both']['concept_preds']
        conc_preds = results_dict['concept_only']['concept_preds']
        
        agreement = (both_preds == conc_preds).astype(float)
        agree_count = np.sum(agreement)
        disagree_count = len(agreement) - agree_count
        
        bars = ax.bar(['Disagree', 'Agree'], [disagree_count, agree_count], 
                     color=['#e74c3c', '#27ae60'], alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 50,
                   f'{int(height):,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax.set_title('Concept Prediction Agreement\n(Image+Concepts vs Concepts Only)', 
                    fontsize=13, fontweight='bold', pad=10)
        ax.tick_params(axis='x', labelsize=11)
        ax.text(0.5, 0.95, f'Agreement Rate: {np.mean(agreement)*100:.1f}%',
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                ha='center', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(agree_count, disagree_count) * 1.15)
    
    # 2. Top concepts by frequency (all models)
    ax = axes[0, 1]
    all_concepts = set()
    for model in results_dict.keys():
        all_concepts.update(results_dict[model]['concept_preds'])
    
    concept_counts = {}
    for model in results_dict.keys():
        preds = results_dict[model]['concept_preds']
        unique, counts = np.unique(preds, return_counts=True)
        for concept_idx, count in zip(unique, counts):
            concept_name = idx_to_concept.get(int(concept_idx), f'C{concept_idx}')
            # Truncate long names
            if len(concept_name) > 12:
                concept_name = concept_name[:9] + '...'
            if concept_name not in concept_counts:
                concept_counts[concept_name] = {}
            concept_counts[concept_name][model] = count
    
    # Get top 8 concepts (reduced for better spacing)
    total_counts = {name: sum(counts.values()) for name, counts in concept_counts.items()}
    top_concepts = sorted(total_counts.items(), key=lambda x: x[1], reverse=True)[:8]
    
    x = np.arange(len(top_concepts))
    width = 0.25
    models_list = list(results_dict.keys())
    for i, model in enumerate(models_list):
        counts = [concept_counts.get(name, {}).get(model, 0) for name, _ in top_concepts]
        ax.bar(x + i*width - width, counts, width, label=MODEL_NAMES[model], 
               color=MODEL_COLORS[model], alpha=0.8)
    
    ax.set_xlabel('Concept', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Top Predicted Concepts', fontsize=13, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in top_concepts], fontsize=9, rotation=45, ha='right')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Error vs concept confidence (for concepts only)
    if 'concept_only' in results_dict:
        ax = axes[1, 0]
        conc_probs = np.array(results_dict['concept_only']['concept_probs'])
        max_probs = np.max(conc_probs, axis=1)
        errors = results_dict['concept_only']['errors']
        
        ax.scatter(max_probs, errors, alpha=0.3, s=15, color=MODEL_COLORS['concept_only'], rasterized=True)
        ax.set_xlabel('Max Concept Probability', fontsize=11, fontweight='bold')
        ax.set_ylabel('Error (km)', fontsize=11, fontweight='bold')
        ax.set_title('Error vs Concept Confidence', fontsize=13, fontweight='bold', pad=10)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Add correlation
        corr = np.corrcoef(max_probs, errors)[0, 1]
        ax.text(0.02, 0.98, f'Correlation: {corr:.3f}',
                transform=ax.transAxes, fontsize=11, fontweight='bold',
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # 4. Concept prediction distribution (horizontal bar, single model for clarity)
    ax = axes[1, 1]
    # Use the 'both' model for cleaner visualization
    if 'both' in results_dict:
        model = 'both'
        preds = results_dict[model]['concept_preds']
        unique, counts = np.unique(preds, return_counts=True)
        # Get top 12
        top_indices = np.argsort(counts)[-12:]
        top_concept_names = []
        for i in top_indices:
            name = idx_to_concept.get(int(unique[i]), f'C{unique[i]}')
            if len(name) > 18:
                name = name[:15] + '...'
            top_concept_names.append(name)
        top_counts = counts[top_indices]
        
        y_pos = np.arange(len(top_concept_names))
        ax.barh(y_pos, top_counts, color=MODEL_COLORS[model], alpha=0.8, edgecolor='white')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_concept_names, fontsize=9)
        
        # Add count labels
        for i, (count, y) in enumerate(zip(top_counts, y_pos)):
            ax.text(count + max(top_counts)*0.01, y, f'{count:,}', 
                   va='center', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title(f'Top Concept Distribution ({MODEL_NAMES.get("both", "Both")})', 
                fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, max(top_counts) * 1.12)
    
    plt.tight_layout(pad=2.0, h_pad=3.0, w_pad=2.0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


def plot_sample_cases(
    results_dict: Dict[str, Dict],
    test_dataset: ConceptDataset,
    output_path: Path,
    num_samples: int = 12
):
    """Plot sample cases where models differ significantly."""
    if 'both' not in results_dict or 'image_only' not in results_dict:
        return
    
    both_errors = results_dict['both']['errors']
    img_only_errors = results_dict['image_only']['errors']
    
    # Find cases where both is much better or much worse
    improvement = (img_only_errors - both_errors) / np.maximum(img_only_errors, 1e-6)
    best_cases = np.argsort(improvement)[-num_samples//2:]  # Best improvements
    worst_cases = np.argsort(improvement)[:num_samples//2]   # Worst (negative improvements)
    
    # Use 4 columns for better layout
    ncols = 4
    nrows = (num_samples + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 5*nrows))
    axes = axes.flatten()
    
    all_samples = list(best_cases) + list(worst_cases)
    
    for idx, sample_idx in enumerate(all_samples[:len(axes)]):
        ax = axes[idx]
        
        # Get predictions
        both_pred = results_dict['both']['predictions'][sample_idx]
        img_pred = results_dict['image_only']['predictions'][sample_idx]
        true_coords = (both_pred['true_lat'], both_pred['true_lng'])
        
        # Plot on map
        ax.scatter([true_coords[1]], [true_coords[0]], c='green', s=250, 
                  marker='*', label='True', zorder=5, edgecolors='black', linewidths=2)
        ax.scatter([both_pred['pred_lng']], [both_pred['pred_lat']], 
                  c=MODEL_COLORS['both'], s=120, marker='o', 
                  label=f'Both ({both_pred["error_km"]:.0f}km)', zorder=4, edgecolors='white', linewidths=1)
        ax.scatter([img_pred['pred_lng']], [img_pred['pred_lat']], 
                  c=MODEL_COLORS['image_only'], s=120, marker='s', 
                  label=f'Image ({img_pred["error_km"]:.0f}km)', zorder=3, edgecolors='white', linewidths=1)
        
        # Draw lines
        ax.plot([true_coords[1], both_pred['pred_lng']], 
               [true_coords[0], both_pred['pred_lat']], 
               color=MODEL_COLORS['both'], linestyle='--', alpha=0.6, linewidth=1.5)
        ax.plot([true_coords[1], img_pred['pred_lng']], 
               [true_coords[0], img_pred['pred_lat']], 
               color=MODEL_COLORS['image_only'], linestyle='--', alpha=0.6, linewidth=1.5)
        
        # Set limits with proper margins
        all_lngs = [true_coords[1], both_pred['pred_lng'], img_pred['pred_lng']]
        all_lats = [true_coords[0], both_pred['pred_lat'], img_pred['pred_lat']]
        lng_range = max(all_lngs) - min(all_lngs) + 1e-6
        lat_range = max(all_lats) - min(all_lats) + 1e-6
        margin = max(lng_range, lat_range) * 0.25 + 1  # Minimum margin of 1 degree
        
        ax.set_xlim(min(all_lngs) - margin, max(all_lngs) + margin)
        ax.set_ylim(min(all_lats) - margin, max(all_lats) + margin)
        
        ax.set_xlabel('Longitude', fontsize=9)
        ax.set_ylabel('Latitude', fontsize=9)
        
        # Color title based on improvement
        improve_val = improvement[sample_idx] * 100
        title_color = 'green' if improve_val > 0 else 'red'
        ax.set_title(f'Sample {sample_idx}\n'
                    f'Improvement: {improve_val:.1f}%', 
                    fontsize=10, fontweight='bold', color=title_color)
        ax.legend(fontsize=7, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=8)
    
    # Hide unused axes
    for idx in range(len(all_samples), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Sample Cases: When Image+Concepts Differs from Image Only\n'
                '(Green titles = fusion helps, Red titles = fusion hurts)', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(pad=2.0, h_pad=2.5, w_pad=2.0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


def plot_why_fusion_helps(
    results_dict: Dict[str, Dict],
    idx_to_concept: Dict[int, str],
    output_path: Path
):
    """
    Deep dive analysis: WHY does image+concept beat image only?
    Shows concepts, gate values, and confidence for cases where fusion helps.
    """
    if 'both' not in results_dict or 'image_only' not in results_dict:
        return
    
    fig = plt.figure(figsize=(22, 20))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.40, wspace=0.35)
    
    img_only_errors = results_dict['image_only']['errors']
    both_errors = results_dict['both']['errors']
    both_concept_preds = results_dict['both']['concept_preds']
    both_concept_probs = np.array(results_dict['both']['concept_probs'])
    both_gate_values = results_dict['both']['gate_values']
    
    # Use safe division to avoid division by zero
    improvement = (img_only_errors - both_errors) / np.maximum(img_only_errors, 1e-6) * 100
    
    # Define improvement thresholds
    significant_improvement = improvement > 20  # >20% improvement
    moderate_improvement = (improvement > 5) & (improvement <= 20)
    no_improvement = (improvement > -5) & (improvement <= 5)
    worse = improvement <= -5
    
    # 1. Gate values vs improvement
    ax1 = fig.add_subplot(gs[0, 0])
    if np.sum(significant_improvement) > 0:
        ax1.scatter(both_gate_values[significant_improvement], improvement[significant_improvement],
                   alpha=0.5, color='green', s=25, label=f'Significant ({np.sum(significant_improvement)})', rasterized=True)
    if np.sum(moderate_improvement) > 0:
        ax1.scatter(both_gate_values[moderate_improvement], improvement[moderate_improvement],
                   alpha=0.5, color='orange', s=25, label=f'Moderate ({np.sum(moderate_improvement)})', rasterized=True)
    if np.sum(no_improvement) > 0:
        ax1.scatter(both_gate_values[no_improvement], improvement[no_improvement],
                   alpha=0.3, color='gray', s=15, label=f'No change ({np.sum(no_improvement)})', rasterized=True)
    if np.sum(worse) > 0:
        ax1.scatter(both_gate_values[worse], improvement[worse],
                   alpha=0.5, color='red', s=25, label=f'Worse ({np.sum(worse)})', rasterized=True)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
    ax1.axvline(0.5, color='black', linestyle=':', linewidth=1.5, alpha=0.6)
    ax1.set_xlabel('Gate Value (0=Image, 1=Concepts)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Improvement (%)', fontsize=10, fontweight='bold')
    ax1.set_title('Gate Value vs Improvement', fontsize=11, fontweight='bold', pad=8)
    ax1.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=9)
    
    # Add statistics
    if np.sum(significant_improvement) > 0:
        mean_gate_help = np.mean(both_gate_values[significant_improvement])
        ax1.text(0.02, 0.98, f'Mean gate (sig. help): {mean_gate_help:.3f}',
                transform=ax1.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # 2. Concept confidence vs improvement
    ax2 = fig.add_subplot(gs[0, 1])
    max_concept_probs = np.max(both_concept_probs, axis=1)
    if np.sum(significant_improvement) > 0:
        ax2.scatter(max_concept_probs[significant_improvement], improvement[significant_improvement],
                   alpha=0.5, color='green', s=25, label='Significant', rasterized=True)
    if np.sum(moderate_improvement) > 0:
        ax2.scatter(max_concept_probs[moderate_improvement], improvement[moderate_improvement],
                   alpha=0.5, color='orange', s=25, label='Moderate', rasterized=True)
    if np.sum(no_improvement) > 0:
        ax2.scatter(max_concept_probs[no_improvement], improvement[no_improvement],
                   alpha=0.3, color='gray', s=15, label='No change', rasterized=True)
    if np.sum(worse) > 0:
        ax2.scatter(max_concept_probs[worse], improvement[worse],
                   alpha=0.5, color='red', s=25, label='Worse', rasterized=True)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
    ax2.set_xlabel('Max Concept Probability', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Improvement (%)', fontsize=10, fontweight='bold')
    ax2.set_title('Concept Confidence vs Improvement', fontsize=11, fontweight='bold', pad=8)
    ax2.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=9)
    
    # 3. Top concepts when fusion helps significantly
    ax3 = fig.add_subplot(gs[0, 2])
    if np.sum(significant_improvement) > 0:
        help_concepts = both_concept_preds[significant_improvement]
        unique_help, counts_help = np.unique(help_concepts, return_counts=True)
        if len(unique_help) > 0:
            top_help_indices = unique_help[np.argsort(counts_help)[-8:]]  # Top 8 for better fit
            
            concept_names_help = []
            for idx in top_help_indices:
                name = idx_to_concept.get(int(idx), f'C{idx}')
                if len(name) > 15:
                    name = name[:12] + '...'
                concept_names_help.append(name)
            counts_help_sorted = sorted(counts_help)[-8:]
            
            y_pos = np.arange(len(concept_names_help))
            bars = ax3.barh(y_pos, counts_help_sorted, color='green', alpha=0.7, edgecolor='white')
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(concept_names_help, fontsize=9)
            ax3.set_xlabel('Frequency', fontsize=10, fontweight='bold')
            ax3.set_title(f'Top Concepts (>20% improvement)\nn={np.sum(significant_improvement)}', 
                         fontsize=11, fontweight='bold', pad=8)
            ax3.grid(True, alpha=0.3, axis='x')
            # Add count labels
            for bar, count in zip(bars, counts_help_sorted):
                ax3.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                        f'{count}', va='center', fontsize=8)
            ax3.tick_params(labelsize=9)
        else:
            ax3.text(0.5, 0.5, 'No concepts found', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=11)
            ax3.set_title('Top Concepts When Fusion Helps', fontsize=11, fontweight='bold', pad=8)
    else:
        ax3.text(0.5, 0.5, 'No significant\nimprovements found', ha='center', va='center',
                transform=ax3.transAxes, fontsize=11)
        ax3.set_title('Top Concepts When Fusion Helps', fontsize=11, fontweight='bold', pad=8)
    
    # 4. Error magnitude where fusion helps most
    ax4 = fig.add_subplot(gs[1, 0])
    bins = np.logspace(0, 4, 21)  # 1km to 10000km in log space
    if np.sum(significant_improvement) > 0:
        ax4.hist(img_only_errors[significant_improvement], bins=bins, alpha=0.7, 
                color='green', label=f'Sig. help ({np.sum(significant_improvement)})', density=True, edgecolor='white')
    if np.sum(moderate_improvement) > 0:
        ax4.hist(img_only_errors[moderate_improvement], bins=bins, alpha=0.7, 
                color='orange', label=f'Mod. help ({np.sum(moderate_improvement)})', density=True, edgecolor='white')
    if np.sum(no_improvement) > 0:
        ax4.hist(img_only_errors[no_improvement], bins=bins, alpha=0.5, 
                color='gray', label=f'No help ({np.sum(no_improvement)})', density=True, edgecolor='white')
    if np.sum(worse) > 0:
        ax4.hist(img_only_errors[worse], bins=bins, alpha=0.7, 
                color='red', label=f'Worse ({np.sum(worse)})', density=True, edgecolor='white')
    ax4.set_xscale('log')
    ax4.set_xlabel('Image Only Error (km)', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Density', fontsize=10, fontweight='bold')
    ax4.set_title('Error Level Where Fusion Helps', fontsize=11, fontweight='bold', pad=8)
    ax4.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(labelsize=9)
    
    # 5. Concept agreement when fusion helps
    if 'concept_only' in results_dict:
        ax5 = fig.add_subplot(gs[1, 1])
        conc_preds = results_dict['concept_only']['concept_preds']
        agreement = (both_concept_preds == conc_preds).astype(float)
        
        # Create grouped bar chart instead of scatter
        categories = ['Significant', 'Moderate', 'No change', 'Worse']
        masks = [significant_improvement, moderate_improvement, no_improvement, worse]
        agree_rates = []
        disagree_rates = []
        for mask in masks:
            if np.sum(mask) > 0:
                agree_rates.append(np.mean(agreement[mask]) * 100)
                disagree_rates.append((1 - np.mean(agreement[mask])) * 100)
            else:
                agree_rates.append(0)
                disagree_rates.append(0)
        
        x = np.arange(len(categories))
        width = 0.35
        ax5.bar(x - width/2, agree_rates, width, label='Agree', color='#27ae60', alpha=0.8)
        ax5.bar(x + width/2, disagree_rates, width, label='Disagree', color='#e74c3c', alpha=0.8)
        ax5.set_xlabel('Improvement Category', fontsize=10, fontweight='bold')
        ax5.set_ylabel('Percentage (%)', fontsize=10, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(categories, fontsize=9)
        ax5.set_title('Concept Agreement by Category', fontsize=11, fontweight='bold', pad=8)
        ax5.legend(fontsize=8, loc='upper right', framealpha=0.9)
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.tick_params(labelsize=9)
        ax5.set_ylim(0, 105)
    else:
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.text(0.5, 0.5, 'Concept-only model\nnot available', ha='center', va='center',
                transform=ax5.transAxes, fontsize=11)
        ax5.set_title('Concept Agreement vs Improvement', fontsize=11, fontweight='bold', pad=8)
    
    # 6. Gate value distribution by improvement category
    ax6 = fig.add_subplot(gs[1, 2])
    # Use boxplot for cleaner visualization
    gate_data = []
    gate_labels = []
    gate_colors = []
    if np.sum(significant_improvement) > 0:
        gate_data.append(both_gate_values[significant_improvement])
        gate_labels.append(f'Sig.\n(n={np.sum(significant_improvement)})')
        gate_colors.append('green')
    if np.sum(moderate_improvement) > 0:
        gate_data.append(both_gate_values[moderate_improvement])
        gate_labels.append(f'Mod.\n(n={np.sum(moderate_improvement)})')
        gate_colors.append('orange')
    if np.sum(no_improvement) > 0:
        gate_data.append(both_gate_values[no_improvement])
        gate_labels.append(f'None\n(n={np.sum(no_improvement)})')
        gate_colors.append('gray')
    if np.sum(worse) > 0:
        gate_data.append(both_gate_values[worse])
        gate_labels.append(f'Worse\n(n={np.sum(worse)})')
        gate_colors.append('red')
    
    if gate_data:
        bp = ax6.boxplot(gate_data, labels=gate_labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], gate_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax6.axhline(0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.6, label='Gate=0.5')
    ax6.set_xlabel('Improvement Category', fontsize=10, fontweight='bold')
    ax6.set_ylabel('Gate Value', fontsize=10, fontweight='bold')
    ax6.set_title('Gate Distribution by Category', fontsize=11, fontweight='bold', pad=8)
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.tick_params(labelsize=9)
    
    # 7. Gate vs Concept Confidence (2D scatter with colorbar)
    ax7 = fig.add_subplot(gs[2, 1])
    # Show all points with improvement color
    scatter = ax7.scatter(both_gate_values, max_concept_probs, 
                         c=np.clip(improvement, -100, 100), cmap='RdYlGn', 
                         s=15, alpha=0.5, vmin=-100, vmax=100, rasterized=True)
    cbar = plt.colorbar(scatter, ax=ax7, shrink=0.8)
    cbar.set_label('Improvement (%)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    ax7.axvline(0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
    ax7.set_xlabel('Gate Value', fontsize=10, fontweight='bold')
    ax7.set_ylabel('Max Concept Probability', fontsize=10, fontweight='bold')
    ax7.set_title('Gate × Concept Confidence', fontsize=11, fontweight='bold', pad=8)
    ax7.grid(True, alpha=0.3)
    ax7.tick_params(labelsize=9)
    
    # 8. Summary statistics
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    # Calculate summary statistics
    total_samples = len(improvement)
    sig_count = np.sum(significant_improvement)
    mod_count = np.sum(moderate_improvement)
    no_count = np.sum(no_improvement)
    worse_count = np.sum(worse)
    
    # Calculate overall improvement from mean errors (accurate)
    overall_mean_improvement = (np.mean(img_only_errors) - np.mean(both_errors)) / np.mean(img_only_errors) * 100
    
    summary_text = f"""SUMMARY STATISTICS
{'─'*40}

Total samples: {total_samples:,}

Improvement Categories:
  • Significant (>20%): {sig_count:,} ({sig_count/total_samples*100:.1f}%)
  • Moderate (5-20%):   {mod_count:,} ({mod_count/total_samples*100:.1f}%)
  • No change (±5%):    {no_count:,} ({no_count/total_samples*100:.1f}%)
  • Worse (<-5%):       {worse_count:,} ({worse_count/total_samples*100:.1f}%)

Overall Performance:
  • Mean Error (Image+Concepts): {np.mean(both_errors):.1f} km
  • Mean Error (Image Only):     {np.mean(img_only_errors):.1f} km
  • Mean Improvement:            {overall_mean_improvement:.1f}%
  • Median Improvement:          {np.median(improvement):.1f}%
  • Fusion helps on:             {np.sum(improvement > 0):,} samples
                                ({np.sum(improvement > 0)/total_samples*100:.1f}%)"""

    if sig_count > 0:
        mean_gate_sig = np.mean(both_gate_values[significant_improvement])
        mean_conf_sig = np.mean(max_concept_probs[significant_improvement])
        summary_text += f"""

When Fusion Helps Significantly:
  • Mean gate value: {mean_gate_sig:.3f}
  • Mean confidence: {mean_conf_sig:.3f}"""
    
    ax8.text(0.02, 0.98, summary_text, transform=ax8.transAxes, fontsize=9,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', alpha=0.95, edgecolor='gray'))
    
    # 9. Most Useful Concepts (full width, new row)
    ax9 = fig.add_subplot(gs[3, :])
    # Find concepts with highest median improvement (most useful)
    unique_concepts, counts = np.unique(both_concept_preds, return_counts=True)
    
    # Only consider concepts with at least 10 samples for statistical reliability
    min_samples = 10
    concept_median_improvements = []
    for concept_idx in unique_concepts:
        mask = both_concept_preds == concept_idx
        if np.sum(mask) >= min_samples:
            median_imp = np.median(improvement[mask])
            mean_imp = np.mean(improvement[mask])
            concept_median_improvements.append((concept_idx, median_imp, mean_imp, np.sum(mask)))
    
    # Sort by median improvement (most useful first)
    concept_median_improvements.sort(key=lambda x: x[1], reverse=True)
    top_useful_concepts = concept_median_improvements[:20]  # Top 20 most useful
    
    concept_improvements = []
    concept_labels = []
    for concept_idx, median_imp, mean_imp, count in top_useful_concepts:
        mask = both_concept_preds == concept_idx
        concept_improvements.append(improvement[mask])
        name = idx_to_concept.get(int(concept_idx), f'C{concept_idx}')
        # Don't truncate since we have full width and rotated labels
        concept_labels.append(f'{name}\n(n={count})')
    
    if concept_improvements:
        bp = ax9.boxplot(concept_improvements, labels=concept_labels, patch_artist=True, 
                        widths=0.7, showfliers=False)  # Remove outliers
        for patch in bp['boxes']:
            patch.set_facecolor(MODEL_COLORS['both'])
            patch.set_alpha(0.7)
        ax9.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.6)
        ax9.set_ylabel('Improvement (%)', fontsize=11, fontweight='bold')
        ax9.set_title('Most Useful Concepts (by median improvement, outliers hidden)', 
                     fontsize=12, fontweight='bold', pad=10)
        ax9.tick_params(axis='x', labelsize=9, rotation=90)
        for label in ax9.get_xticklabels():
            label.set_horizontalalignment('center')
            label.set_verticalalignment('top')
        ax9.grid(True, alpha=0.3, axis='y')
        ax9.tick_params(axis='y', labelsize=10)
    
    plt.suptitle('Why Does Image+Concepts Beat Image Only?', 
                fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout(pad=2.0, h_pad=2.5, w_pad=2.5, rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


def save_individual_image_analyses(
    results_dict: Dict[str, Dict],
    test_dataset: ConceptDataset,
    idx_to_concept: Dict[int, str],
    output_dir: Path,
    num_samples: int = 20,
    selection_criteria: str = "significant_improvement"
):
    """
    Save individual images with concept activations and location predictions.
    
    Args:
        results_dict: Results from evaluate_model
        test_dataset: The test dataset (for accessing images)
        idx_to_concept: Concept index to name mapping
        output_dir: Directory to save images
        num_samples: Number of samples to save
        selection_criteria: How to select samples:
            - "significant_improvement": Where fusion helps >20%
            - "best_improvement": Top N improvements
            - "worst_cases": Where fusion hurts most
            - "random": Random samples
    """
    if 'both' not in results_dict or 'image_only' not in results_dict:
        print("Warning: Both models required for image analysis. Skipping.")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    both_errors = results_dict['both']['errors']
    img_only_errors = results_dict['image_only']['errors']
    improvement = (img_only_errors - both_errors) / np.maximum(img_only_errors, 1e-6) * 100
    
    # Select samples based on criteria
    if selection_criteria == "significant_improvement":
        mask = improvement > 20
        selected_indices = np.where(mask)[0]
        if len(selected_indices) > num_samples:
            selected_indices = selected_indices[np.argsort(improvement[selected_indices])[-num_samples:]]
    elif selection_criteria == "best_improvement":
        selected_indices = np.argsort(improvement)[-num_samples:]
    elif selection_criteria == "worst_cases":
        selected_indices = np.argsort(improvement)[:num_samples]
    elif selection_criteria == "random":
        selected_indices = np.random.choice(len(improvement), min(num_samples, len(improvement)), replace=False)
    else:
        selected_indices = np.argsort(improvement)[-num_samples:]
    
    print(f"\nSaving {len(selected_indices)} individual image analyses...")
    
    # Try to import cartopy for maps
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        HAS_CARTOPY = True
    except ImportError:
        HAS_CARTOPY = False
    
    for idx in tqdm(selected_indices, desc="Processing images"):
        try:
            # Get predictions
            both_pred = results_dict['both']['predictions'][idx]
            img_pred = results_dict['image_only']['predictions'][idx]
            
            # Get concept information
            both_concept_idx = results_dict['both']['concept_preds'][idx]
            both_concept_probs = np.array(results_dict['both']['concept_probs'][idx])
            both_attn = results_dict['both']['attention_weights'][idx]  # [K, P]
            both_gate = results_dict['both']['gate_values'][idx]
            
            # Get top concepts
            top5_idx = np.argsort(both_concept_probs)[-5:][::-1]
            top5_probs = both_concept_probs[top5_idx]
            top5_names = []
            for i in top5_idx:
                name = idx_to_concept.get(int(i), f'Concept {i}')
                if len(name) > 25:
                    name = name[:22] + '...'
                top5_names.append(name)
            
            # Try to load image
            img = None
            img_path = None
            try:
                if hasattr(test_dataset, 'df') and 'image_path' in test_dataset.df.columns:
                    img_path = test_dataset.df.iloc[idx]['image_path']
                    if isinstance(img_path, str) and Path(img_path).exists():
                        img = plt.imread(img_path)
                elif hasattr(test_dataset, 'image_dir') and test_dataset.image_dir is not None:
                    pano_id = test_dataset._pano_ids[idx]
                    img_path = test_dataset.image_dir / f"image_{pano_id}.jpg"
                    if img_path.exists():
                        img = plt.imread(img_path)
            except Exception as e:
                pass  # Image not available, will show placeholder
            
            # Prepare attention map for overlay
            top_concept_attn = both_attn[top5_idx[0]]  # Attention for top concept
            num_patches = len(top_concept_attn)
            side = int(np.sqrt(num_patches))
            if side * side == num_patches:
                attn_map = top_concept_attn.reshape(side, side)
            else:
                attn_map = top_concept_attn[:side*side].reshape(side, side)
            
            # Create visualization with new layout
            fig = plt.figure(figsize=(18, 16))
            gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.30, wspace=0.25,
                                  height_ratios=[1, 0.8, 1.2])
            
            # Row 1: Original image and Attention overlay
            # 1. Original image
            ax_img = fig.add_subplot(gs[0, 0])
            if img is not None:
                ax_img.imshow(img)
                ax_img.set_title('Original Image', fontsize=12, fontweight='bold', pad=8)
            else:
                ax_img.text(0.5, 0.5, f'Image not available\nSample {idx}', 
                           ha='center', va='center', transform=ax_img.transAxes, fontsize=11)
                ax_img.set_facecolor('#f0f0f0')
                ax_img.set_title('Original Image', fontsize=12, fontweight='bold', pad=8)
            ax_img.axis('off')
            
            # 2. Attention overlay on image
            ax_overlay = fig.add_subplot(gs[0, 1])
            if img is not None:
                # Upsample attention to image size
                attn_tensor = torch.tensor(attn_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                img_h, img_w = img.shape[:2]
                attn_upsampled = F.interpolate(attn_tensor, size=(img_h, img_w), mode='bilinear', align_corners=False)
                attn_upsampled = attn_upsampled.squeeze().numpy()
                
                ax_overlay.imshow(img)
                im = ax_overlay.imshow(attn_upsampled, cmap='hot', alpha=0.5, interpolation='bilinear')
                # Truncate concept name for title
                top_name_short = top5_names[0] if len(top5_names[0]) <= 20 else top5_names[0][:17] + '...'
                ax_overlay.set_title(f'Attention: {top_name_short}', fontsize=12, fontweight='bold', pad=8)
            else:
                ax_overlay.text(0.5, 0.5, 'Image not available', ha='center', va='center',
                               transform=ax_overlay.transAxes, fontsize=11)
                ax_overlay.set_facecolor('#f0f0f0')
                ax_overlay.set_title('Attention Overlay', fontsize=12, fontweight='bold', pad=8)
            ax_overlay.axis('off')
            
            # Row 2: Top 5 concepts and Sample analysis
            # 3. Top concepts bar chart
            ax_concepts = fig.add_subplot(gs[1, 0])
            y_pos = np.arange(len(top5_names))
            colors_bar = ['#27ae60' if i == both_concept_idx else '#3498db' for i in top5_idx]
            bars = ax_concepts.barh(y_pos, top5_probs, color=colors_bar, alpha=0.8, edgecolor='white')
            ax_concepts.set_yticks(y_pos)
            ax_concepts.set_yticklabels(top5_names, fontsize=9)
            ax_concepts.set_xlabel('Probability', fontsize=10, fontweight='bold')
            ax_concepts.set_title('Top 5 Predicted Concepts', fontsize=12, fontweight='bold', pad=8)
            ax_concepts.grid(True, alpha=0.3, axis='x')
            ax_concepts.set_xlim(0, max(top5_probs) * 1.15)
            # Add probability labels
            for bar, prob in zip(bars, top5_probs):
                ax_concepts.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                               f'{prob:.3f}', va='center', fontsize=8, fontweight='bold')
            
            # 4. Statistics and info
            ax_info = fig.add_subplot(gs[1, 1])
            ax_info.axis('off')
            
            # Get top concept name (truncated)
            top_concept_name = idx_to_concept.get(int(both_concept_idx), f'Concept {both_concept_idx}')
            if len(top_concept_name) > 30:
                top_concept_name = top_concept_name[:27] + '...'
            
            info_text = f"""SAMPLE {idx} ANALYSIS
{'─'*45}

Location Predictions:
  True:       ({both_pred['true_lat']:.4f}, {both_pred['true_lng']:.4f})
  Both:       ({both_pred['pred_lat']:.4f}, {both_pred['pred_lng']:.4f})
              Error: {both_pred['error_km']:.1f} km
  Image Only: ({img_pred['pred_lat']:.4f}, {img_pred['pred_lng']:.4f})
              Error: {img_pred['error_km']:.1f} km

Improvement: {improvement[idx]:.1f}%

Concept Predictions:
  Top Concept: {top_concept_name}
  Confidence:  {both_concept_probs[both_concept_idx]:.3f}"""
            
            if both_gate is not None and not np.isnan(both_gate):
                info_text += f"\n  Gate Value: {both_gate:.3f}"
            
            ax_info.text(0.02, 0.98, info_text, transform=ax_info.transAxes,
                        fontsize=9, verticalalignment='top', family='monospace',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', alpha=0.95, edgecolor='gray'))
            
            # Row 3: Map with predictions (full width)
            ax_map = fig.add_subplot(gs[2, :], projection=ccrs.PlateCarree() if HAS_CARTOPY else None)
            
            if HAS_CARTOPY:
                ax_map.set_global()
                ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5)
                ax_map.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle='--')
                ax_map.add_feature(cfeature.LAND, alpha=0.3, facecolor='#e8e8e8')
                ax_map.add_feature(cfeature.OCEAN, alpha=0.3, facecolor='#d4e6f1')
                ax_map.gridlines(draw_labels=True, alpha=0.4, linewidth=0.5)
                
                # Plot true location
                ax_map.scatter(both_pred['true_lng'], both_pred['true_lat'],
                             c='green', s=350, marker='*', label='True Location',
                             transform=ccrs.PlateCarree(), zorder=5, 
                             edgecolors='black', linewidths=2)
                
                # Plot predictions
                ax_map.scatter(both_pred['pred_lng'], both_pred['pred_lat'],
                             c='#3498db', s=220, marker='o', 
                             label=f"Both ({both_pred['error_km']:.0f}km)",
                             transform=ccrs.PlateCarree(), zorder=4,
                             edgecolors='black', linewidths=1.5)
                ax_map.scatter(img_pred['pred_lng'], img_pred['pred_lat'],
                             c='#e74c3c', s=220, marker='s',
                             label=f"Image Only ({img_pred['error_km']:.0f}km)",
                             transform=ccrs.PlateCarree(), zorder=4,
                             edgecolors='black', linewidths=1.5)
            else:
                ax_map.scatter(both_pred['true_lng'], both_pred['true_lat'],
                             c='green', s=350, marker='*', label='True Location',
                             zorder=5, edgecolors='black', linewidths=2)
                ax_map.scatter(both_pred['pred_lng'], both_pred['pred_lat'],
                             c='#3498db', s=220, marker='o', 
                             label=f"Both ({both_pred['error_km']:.0f}km)",
                             zorder=4, edgecolors='black', linewidths=1.5)
                ax_map.scatter(img_pred['pred_lng'], img_pred['pred_lat'],
                             c='#e74c3c', s=220, marker='s',
                             label=f"Image Only ({img_pred['error_km']:.0f}km)",
                             zorder=4, edgecolors='black', linewidths=1.5)
                ax_map.set_xlim(-180, 180)
                ax_map.set_ylim(-90, 90)
                ax_map.set_xlabel('Longitude', fontsize=10)
                ax_map.set_ylabel('Latitude', fontsize=10)
                ax_map.grid(True, alpha=0.3)
            
            ax_map.set_title('Location Predictions', fontsize=12, fontweight='bold', pad=10)
            ax_map.legend(loc='upper right', fontsize=10, framealpha=0.95)
            
            # Save
            safe_idx = str(idx).zfill(6)
            output_path = output_dir / f"sample_{safe_idx}_improvement_{improvement[idx]:.1f}percent.png"
            
            # Color suptitle based on improvement
            improve_color = '#27ae60' if improvement[idx] > 0 else '#e74c3c'
            plt.suptitle(f'Sample {idx}: Image+Concepts Analysis (Improvement: {improvement[idx]:.1f}%)', 
                        fontsize=14, fontweight='bold', y=0.995, color=improve_color)
            plt.tight_layout(pad=1.5, rect=[0, 0, 1, 0.98])
            plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            
        except Exception as e:
            print(f"  Warning: Failed to process sample {idx}: {e}")
            continue
    
    print(f"  ✅ Saved {len(selected_indices)} image analyses to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compare model variants on test set")
    
    # Checkpoint paths
    parser.add_argument("--checkpoint-both", type=str, required=True,
                       help="Checkpoint directory for img+conc model")
    parser.add_argument("--checkpoint-concept", type=str, required=True,
                       help="Checkpoint directory for concept only model")
    parser.add_argument("--checkpoint-image", type=str, required=True,
                       help="Checkpoint directory for image only model")
    
    # Data paths
    parser.add_argument("--test-csv", type=str, required=True, help="Test CSV path")
    parser.add_argument("--concept-data-dir", type=str, required=True,
                       help="Concept data directory")
    parser.add_argument("--cached-dir", type=str, required=True,
                       help="Cached embeddings directory")
    
    # Output
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for results")
    
    # Model config
    parser.add_argument("--concept-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--phase1-dropout", type=float, default=0.45)
    parser.add_argument("--mil-topk", type=int, default=6)
    parser.add_argument("--mil-tau", type=float, default=0.25)
    parser.add_argument("--mix-depth", type=int, default=1)
    parser.add_argument("--mix-heads", type=int, default=4)
    parser.add_argument("--mix-mlp-ratio", type=float, default=2.0)
    parser.add_argument("--mix-local-kernel", type=int, default=0)
    parser.add_argument("--proj-type", type=str, default="simple")
    parser.add_argument("--concept-temperature", type=float, default=2.5)
    
    # Options
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Image analysis options
    parser.add_argument("--save-individual-images", action="store_true",
                       help="Save individual images with concept activations")
    parser.add_argument("--num-image-samples", type=int, default=20,
                       help="Number of individual images to save")
    parser.add_argument("--image-selection", type=str, default="significant_improvement",
                       choices=["significant_improvement", "best_improvement", "worst_cases", "random"],
                       help="How to select images for individual analysis")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("MODEL VARIANT COMPARISON")
    print("="*80)
    
    # Load concept vocabulary
    concept_vocab_path = Path(args.concept_data_dir) / "concept_vocab.json"
    with open(concept_vocab_path) as f:
        concept_vocab = json.load(f)
    idx_to_concept = {int(k): v for k, v in concept_vocab["idx_to_concept"].items()}
    num_concepts = concept_vocab["num_concepts"]
    
    print(f"\n📂 Loaded {num_concepts} concepts")
    
    # Load test dataset
    print(f"\n📊 Loading test dataset...")
    s2_vocab_path = Path(args.concept_data_dir) / "s2_cells.json"
    base_test_ds = ConceptDataset(
        str(args.test_csv),
        args.cached_dir,
        str(concept_vocab_path),
        str(s2_vocab_path),
        split="test",
        load_pooled_embeddings=True,
    )
    print(f"   Test samples: {len(base_test_ds)}")
    
    # Detect patch dimension
    patch_dim = base_test_ds.patch_tokens.shape[2]
    print(f"   Patch dimension: {patch_dim}")
    
    # Load and evaluate each model
    checkpoint_dirs = {
        'both': Path(args.checkpoint_both),
        'concept_only': Path(args.checkpoint_concept),
        'image_only': Path(args.checkpoint_image),
    }
    
    results_dict = {}
    metrics_dict = {}
    
    for model_name, checkpoint_dir in checkpoint_dirs.items():
        print(f"\n{'='*80}")
        print(f"Evaluating {MODEL_NAMES[model_name]} model")
        print(f"{'='*80}")
        print(f"Checkpoint: {checkpoint_dir}")
        
        # Load model
        phase1_model, stage2_model, concept_adapter, centers_xyz, mode, \
        pooled_projection, actual_pooled_dim, checkpoint_pooled_dim = load_model_from_checkpoint(
            checkpoint_dir, num_concepts, patch_dim, device, args
        )
        
        # Assign geocells to test set
        test_coords = base_test_ds._coords
        test_cell_labels = assign_geocells(test_coords, centers_xyz=centers_xyz)
        test_offsets = compute_offsets(test_coords, test_cell_labels, centers_xyz)
        
        test_ds = JointDataset(base_test_ds, test_cell_labels, test_offsets)
        
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn_joint(b, test_ds.pooled_embeddings),
            num_workers=args.num_workers,
        )
        
        # Evaluate
        results = evaluate_model(
            phase1_model, stage2_model, concept_adapter,
            test_loader, device, centers_xyz, idx_to_concept,
            pooled_projection=pooled_projection
        )
        
        # Compute metrics
        metrics = compute_metrics(results)
        metrics['errors'] = results['errors']  # Keep errors for plotting
        
        results_dict[model_name] = results
        metrics_dict[model_name] = metrics
        
        print(f"\n✅ {MODEL_NAMES[model_name]} Results:")
        print(f"   Mean Error: {metrics['mean_error']:.2f} km")
        print(f"   Median Error: {metrics['median_error']:.2f} km")
        print(f"   Acc@100km: {metrics['acc_100km']:.4f}")
        print(f"   Acc@1000km: {metrics['acc_1000km']:.4f}")
    
    # Create visualizations
    print(f"\n{'='*80}")
    print("Creating visualizations...")
    print(f"{'='*80}")
    
    # 1. Performance comparison
    print("  1. Performance comparison...")
    plot_performance_comparison(metrics_dict, output_dir / "performance_comparison.png")
    
    # 2. Error analysis
    print("  2. Error analysis...")
    plot_error_analysis(results_dict, idx_to_concept, output_dir / "error_analysis.png")
    
    # 3. Concept analysis
    print("  3. Concept analysis...")
    plot_concept_analysis(results_dict, idx_to_concept, output_dir / "concept_analysis.png")
    
    # 4. Sample cases
    print("  4. Sample cases...")
    plot_sample_cases(results_dict, base_test_ds, output_dir / "sample_cases.png")
    
    # 5. Why fusion helps
    print("  5. Why fusion helps analysis...")
    plot_why_fusion_helps(results_dict, idx_to_concept, output_dir / "why_fusion_helps.png")
    
    # 6. Individual image analyses
    if args.save_individual_images:
        print("  6. Saving individual image analyses...")
        images_output_dir = output_dir / "individual_images"
        save_individual_image_analyses(
            results_dict, base_test_ds, idx_to_concept, images_output_dir,
            num_samples=args.num_image_samples,
            selection_criteria=args.image_selection
        )
    
    # Save metrics to JSON
    print("  5. Saving metrics...")
    json_metrics = {}
    for model_name, metrics in metrics_dict.items():
        json_metrics[model_name] = {
            k: v for k, v in metrics.items() if k != 'errors'
        }
    
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(json_metrics, f, indent=2)
    
    # Save full results_dict for geographic analysis
    print("  6. Saving full results for geographic analysis...")
    # Convert numpy arrays to lists for JSON serialization
    json_results_dict = {}
    for model_name, results in results_dict.items():
        json_results_dict[model_name] = {
            'predictions': results['predictions'],
            'errors': results['errors'].tolist() if isinstance(results['errors'], np.ndarray) else results['errors'],
            'concept_preds': results['concept_preds'].tolist() if isinstance(results['concept_preds'], np.ndarray) else results['concept_preds'],
            'concept_probs': results.get('concept_probs', []),
            'gate_values': results.get('gate_values', []).tolist() if isinstance(results.get('gate_values', []), np.ndarray) else results.get('gate_values', []),
        }
    
    with open(output_dir / "results_dict.json", 'w') as f:
        json.dump(json_results_dict, f, indent=2)
    
    # Create summary report
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    print("\n📊 Overall Performance:")
    for model_name in ['both', 'concept_only', 'image_only']:
        if model_name in metrics_dict:
            m = metrics_dict[model_name]
            print(f"\n{MODEL_NAMES[model_name]}:")
            print(f"  Mean Error: {m['mean_error']:.2f} km")
            print(f"  Median Error: {m['median_error']:.2f} km")
            print(f"  Acc@100km: {m['acc_100km']:.4f} ({m['acc_100km']*100:.2f}%)")
            print(f"  Acc@1000km: {m['acc_1000km']:.4f} ({m['acc_1000km']*100:.2f}%)")
    
    if 'both' in metrics_dict and 'image_only' in metrics_dict:
        print("\n🔍 Image+Concepts vs Image Only:")
        both_mean = metrics_dict['both']['mean_error']
        img_mean = metrics_dict['image_only']['mean_error']
        improvement = (img_mean - both_mean) / img_mean * 100
        print(f"  Mean Error Improvement: {improvement:.2f}%")
        
        both_errors = metrics_dict['both']['errors']
        img_errors = metrics_dict['image_only']['errors']
        better_count = np.sum(both_errors < img_errors)
        print(f"  Better on {better_count}/{len(both_errors)} samples ({better_count/len(both_errors)*100:.1f}%)")
    
    if 'concept_only' in metrics_dict and 'image_only' in metrics_dict:
        print("\n🔍 Concepts Only vs Image Only:")
        conc_mean = metrics_dict['concept_only']['mean_error']
        img_mean = metrics_dict['image_only']['mean_error']
        improvement = (img_mean - conc_mean) / img_mean * 100
        print(f"  Mean Error Improvement: {improvement:.2f}%")
        
        conc_errors = metrics_dict['concept_only']['errors']
        img_errors = metrics_dict['image_only']['errors']
        better_count = np.sum(conc_errors < img_errors)
        print(f"  Better on {better_count}/{len(conc_errors)} samples ({better_count/len(conc_errors)*100:.1f}%)")
    
    print(f"\n✅ Results saved to: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - performance_comparison.png")
    print(f"  - error_analysis.png")
    print(f"  - concept_analysis.png")
    print(f"  - sample_cases.png")
    print(f"  - why_fusion_helps.png")
    print(f"  - metrics.json")
    if args.save_individual_images:
        print(f"  - individual_images/ (directory with {args.num_image_samples} image analyses)")


if __name__ == "__main__":
    main()

