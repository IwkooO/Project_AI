#!/usr/bin/env python3
"""
Visualize top images per concept where concept activation is highest.
Shows attention overlays to verify if attention matches the concept.

Usage:
    python scripts/evaluation/visualize_top_images_per_concept.py \
        --checkpoint-both checkpoints/stage3_joint_latefusion/stage3_joint_both_scratchgeo_dim256_pos_latefusion_2.5ct_0.01gatereg \
        --test-csv data/splits/dataset_test.csv \
        --concept-data-dir data/concept_data_v2 \
        --cached-dir /scratch-shared/igodzwon/Project_AI/data/6921d7831744c5356b098bf7_balanced/cached_streetclip_v2 \
        --output-dir results/top_images_per_concept \
        --num-concepts 20 \
        --num-images-per-concept 10
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


def load_model_from_checkpoint(
    checkpoint_dir: Path,
    num_concepts: int,
    patch_dim: int,
    device: torch.device,
    args
) -> Tuple:
    """
    Load phase1 model, stage2 model, and concept adapter from checkpoint directory.
    (Matches logic in compare_model_variants.py for consistency)
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
        # Fallback for older checkpoints
        phase1_state = phase1_checkpoint.get('state_dict', phase1_checkpoint)
    
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
    
    phase1_model.load_state_dict(phase1_state, strict=False)
    phase1_model = phase1_model.to(device)
    phase1_model.eval()
    print(f"   Phase1 loaded (concept_dim={checkpoint_concept_dim}, pos={has_pos_embed}, adaptive_tau={has_per_concept_tau})")
    
    # Load geocells (if needed, though not strictly for concept visualization)
    centers_xyz = None
    if use_joint_checkpoint and 'geocell_info' in phase1_checkpoint:
        geocells_data = phase1_checkpoint['geocell_info']
        centers_xyz = np.array(geocells_data["centers_xyz"])
    
    # Load Stage2 and concept adapter if available
    stage2_model = None
    concept_adapter = None
    mode = "phase1_only"
    pooled_projection = None
    actual_pooled_dim = None
    checkpoint_pooled_dim = None
    
    if use_joint_checkpoint:
        stage2_state = phase1_checkpoint.get('stage2_state_dict', {})
        adapter_state = phase1_checkpoint.get('concept_adapter_state_dict', {})
        
        if stage2_state and adapter_state:
            # Initialize Stage2 model
            stage2_model = Stage2CrossAttentionGeoHead(
                patch_dim=checkpoint_patch_dim,
                hidden_dim=args.hidden_dim,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                dropout=args.dropout,
                mix_depth=args.mix_depth,
                mix_heads=args.mix_heads,
                mix_mlp_ratio=args.mix_mlp_ratio,
                mix_local_kernel=args.mix_local_kernel,
                proj_type=args.proj_type,
            ).to(device)
            
            stage2_model.load_state_dict(stage2_state, strict=False)
            stage2_model.eval()
            
            # Initialize concept adapter
            concept_adapter = ConceptEmbeddingAdapter(
                num_concepts=num_concepts,
                concept_dim=checkpoint_concept_dim,
                hidden_dim=args.hidden_dim,
                temperature=args.concept_temperature,
            ).to(device)
            
            concept_adapter.load_state_dict(adapter_state, strict=False)
            concept_adapter.eval()
            
            mode = "joint"
            if centers_xyz is not None:
                centers_xyz = torch.tensor(centers_xyz, dtype=torch.float32).to(device)
    
    return phase1_model, stage2_model, concept_adapter, centers_xyz, mode, \
           pooled_projection, actual_pooled_dim, checkpoint_pooled_dim


def collate_fn_simple(batch: List):
    """
    Custom collate function for ConceptDataset.
    Converts tuple output to dict format for evaluation.
    """
    # ConceptDataset returns: (vision_input, c_label, coords, cell_label, country_label, cache_idx, pooled_emb)
    patches = torch.stack([b[0] for b in batch], dim=0)
    concept_labels = torch.stack([b[1] for b in batch], dim=0)
    coords = torch.stack([b[2] for b in batch], dim=0)
    
    # Handle pooled embeddings (may be None)
    pooled_embs = [b[6] for b in batch]
    if pooled_embs[0] is not None:
        pooled_emb = torch.stack(pooled_embs, dim=0)
    else:
        pooled_emb = None
    
    return {
        'patches': patches,
        'concept_labels': concept_labels,
        'coords': coords,
        'pooled_emb': pooled_emb,
    }


def evaluate_model_for_concepts(
    phase1_model: nn.Module,
    stage2_model: Optional[nn.Module],
    concept_adapter: Optional[nn.Module],
    test_loader: DataLoader,
    device: torch.device,
    idx_to_concept: Dict[int, str],
) -> Dict:
    """
    Evaluate model and return concept probabilities and attention weights for all samples.
    
    Returns:
        Dictionary with:
        - concept_probs: [N, K] array of concept probabilities
        - attention_weights: [N, K, P] array of attention weights per concept
        - sample_indices: [N] array of sample indices
        - ground_truth_labels: [N] array of ground truth concept indices
    """
    phase1_model.eval()
    if stage2_model is not None:
        stage2_model.eval()
    if concept_adapter is not None:
        concept_adapter.eval()
    
    all_concept_probs = []
    all_attention_weights = []
    all_sample_indices = []
    all_ground_truth = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            patches = batch['patches'].to(device)  # [B, P, D]
            concept_labels = batch['concept_labels'].to(device)  # [B]
            pooled_emb = batch.get('pooled_emb', None)
            if pooled_emb is not None:
                pooled_emb = pooled_emb.to(device)
            
            batch_size = patches.shape[0]
            start_idx = batch_idx * test_loader.batch_size
            
            # Forward through Phase1
            concept_logits, hidden, attn_weights, _ = phase1_model(patches)  # [B, K], [B, D], [B, K, P], _
            concept_probs = torch.softmax(concept_logits, dim=-1)  # [B, K]
            
            # Store results
            all_concept_probs.append(concept_probs.cpu().numpy())
            all_attention_weights.append(attn_weights.cpu().numpy())
            all_ground_truth.append(concept_labels.cpu().numpy())
            all_sample_indices.extend(range(start_idx, start_idx + batch_size))
    
    return {
        'concept_probs': np.concatenate(all_concept_probs, axis=0),  # [N, K]
        'attention_weights': np.concatenate(all_attention_weights, axis=0),  # [N, K, P]
        'sample_indices': np.array(all_sample_indices),
        'ground_truth_labels': np.concatenate(all_ground_truth, axis=0),  # [N]
    }


def find_top_images_per_concept(
    concept_probs: np.ndarray,  # [N, K]
    sample_indices: np.ndarray,  # [N]
    ground_truth_labels: np.ndarray,  # [N]
    num_concepts: int,
    num_images_per_concept: int,
    min_samples: int = 10,
    only_correct: bool = True,
) -> Dict[int, List[int]]:
    """
    Find top images for each concept based on concept probability.
    If only_correct=True, only includes images where the concept is correctly predicted.
    
    Returns:
        Dictionary mapping concept_idx -> list of sample indices (sorted by probability, highest first)
    """
    num_samples, num_concepts_total = concept_probs.shape
    
    # For each concept, find images with highest probability
    top_images_per_concept = {}
    
    for concept_idx in range(num_concepts_total):
        # Get probabilities for this concept across all samples
        concept_scores = concept_probs[:, concept_idx]  # [N]
        
        if only_correct:
            # Filter to only samples where ground truth matches this concept
            correct_mask = (ground_truth_labels == concept_idx)  # [N]
            correct_indices = np.where(correct_mask)[0]
            
            if len(correct_indices) == 0:
                # No correct samples for this concept
                continue
            
            # Get scores for correct predictions only
            correct_scores = concept_scores[correct_indices]  # [num_correct]
            
            # Find top samples among correct predictions (show all available, up to num_images_per_concept)
            top_k = min(num_images_per_concept, len(correct_indices))
            top_local_indices = np.argsort(correct_scores)[-top_k:][::-1]
            top_global_indices = correct_indices[top_local_indices]
            top_scores = correct_scores[top_local_indices]
        else:
            # Original behavior: find top samples regardless of correctness
            top_global_indices = np.argsort(concept_scores)[-num_images_per_concept:][::-1]
            top_scores = concept_scores[top_global_indices]
        
        # Only include if we have enough samples with non-zero probability
            if np.sum(concept_scores > 0.01) < min_samples:
                continue
        
            # Map back to actual sample indices
        actual_indices = sample_indices[top_global_indices].tolist()
        top_images_per_concept[concept_idx] = {
            'indices': actual_indices,
            'scores': top_scores.tolist(),
        }
    
    return top_images_per_concept


def load_image_from_dataset(dataset: ConceptDataset, idx: int, image_dir: Optional[Path] = None) -> Optional[np.ndarray]:
    """Load image from dataset by index, using provided image_dir."""
    try:
        # First try explicit image_dir if provided
        if image_dir is not None and hasattr(dataset, '_pano_ids'):
            pano_id = dataset._pano_ids[idx]
            img_path = image_dir / f"image_{pano_id}.jpg"
            if img_path.exists():
                return plt.imread(img_path)
        
        # Fallback: try dataset's own image_dir
        if hasattr(dataset, 'image_dir') and dataset.image_dir is not None:
            pano_id = dataset._pano_ids[idx]
            img_path = dataset.image_dir / f"image_{pano_id}.jpg"
            if img_path.exists():
                return plt.imread(img_path)
        
        # Fallback: try df image_path column
        if hasattr(dataset, 'df') and 'image_path' in dataset.df.columns:
            img_path = dataset.df.iloc[idx]['image_path']
            if isinstance(img_path, str) and Path(img_path).exists():
                return plt.imread(img_path)
    except Exception as e:
        pass
    return None


def visualize_concept_images(
    concept_idx: int,
    concept_name: str,
    top_images_data: Dict,
    dataset: ConceptDataset,
    attention_weights: np.ndarray,  # [N, K, P]
    sample_indices_map: Dict[int, int],  # Maps dataset index -> results index
    output_path: Path,
    num_images: int = 10,
    image_dir: Optional[Path] = None,
):
    """
    Create visualization for a single concept showing top images with attention overlays.
    """
    # Use all available images (may be less than num_images if fewer correct predictions exist)
    indices = top_images_data['indices']
    scores = top_images_data['scores']
    
    # Calculate grid dimensions based on actual number of images
    n_images = len(indices)
    if n_images == 0:
        print(f"   Warning: No images found for concept {concept_name}, skipping visualization")
        return
    
    n_cols = 5
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(20, 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.2)
    
    for i, (dataset_idx, score) in enumerate(zip(indices, scores)):
        row = i // n_cols
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        # Get original meta_name (unmapped concept) from dataset
        # NOTE: dataset_idx is an index into the FILTERED dataset, but dataset.df
        # contains ALL rows (including those skipped during init). We must use
        # pano_id to find the correct row in the original dataframe.
        original_meta = None
        if hasattr(dataset, 'df') and 'meta_name' in dataset.df.columns:
            if hasattr(dataset, '_pano_ids') and dataset_idx < len(dataset._pano_ids):
                pano_id = dataset._pano_ids[dataset_idx]
                matching_rows = dataset.df[dataset.df['pano_id'] == pano_id]
                if len(matching_rows) > 0:
                    original_meta = matching_rows.iloc[0]['meta_name']
                    if pd.isna(original_meta) or original_meta == '':
                        original_meta = None
        
        # Load image
        img = load_image_from_dataset(dataset, dataset_idx, image_dir=image_dir)
        
        if img is not None:
            # Get attention weights for this concept
            results_idx = sample_indices_map.get(dataset_idx, -1)
            if results_idx >= 0 and results_idx < attention_weights.shape[0]:
                attn = attention_weights[results_idx, concept_idx]  # [P]
                
                # Reshape attention to spatial map
                num_patches = len(attn)
                side = int(np.sqrt(num_patches))
                if side * side == num_patches:
                    attn_map = attn.reshape(side, side)
                else:
                    attn_map = attn[:side*side].reshape(side, side)
                
                # Upsample attention to image size
                img_h, img_w = img.shape[:2]
                attn_tensor = torch.tensor(attn_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                attn_upsampled = F.interpolate(
                    attn_tensor, 
                    size=(img_h, img_w), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze().numpy()
                
                # Display image with attention overlay
                ax.imshow(img)
                ax.imshow(attn_upsampled, cmap='hot', alpha=0.5, interpolation='bilinear')
            else:
                # Just show image without attention
                ax.imshow(img)
            
            # Create title with score and original meta
            title_parts = [f'Score: {score:.3f}']
            if original_meta:
                # Truncate long meta names
                meta_display = original_meta[:30] + '...' if len(original_meta) > 30 else original_meta
                title_parts.append(f'Original: {meta_display}')
            ax.set_title('\n'.join(title_parts), fontsize=9, fontweight='bold')
        else:
            ax.text(0.5, 0.5, f'Image {dataset_idx}\nnot available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_facecolor('#f0f0f0')
            title_parts = [f'Score: {score:.3f}']
            if original_meta:
                meta_display = original_meta[:30] + '...' if len(original_meta) > 30 else original_meta
                title_parts.append(f'Original: {meta_display}')
            ax.set_title('\n'.join(title_parts), fontsize=9, fontweight='bold')
        
        ax.axis('off')
    
    # Hide unused subplot slots if we have fewer images than grid size
    total_slots = n_rows * n_cols
    for i in range(n_images, total_slots):
        row = i // n_cols
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col])
        ax.axis('off')
    
    # Add overall title (show actual number of images found)
    plt.suptitle(f'Top {n_images} Correct Predictions for Concept: {concept_name}\n(Concept Index: {concept_idx})', 
                fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


def visualize_activation_distributions(
    concept_probs: np.ndarray,  # [N, K]
    concept_indices: List[int],
    idx_to_concept: Dict[int, str],
    ground_truth_labels: np.ndarray,  # [N]
    output_path: Path,
    train_counts: Optional[Dict[int, int]] = None,  # Mapping from concept_idx to count in training data
):
    """
    Visualize the distribution of activation scores (probabilities) for specified concepts using violin plots.
    Only includes samples where the concept prediction is correct (ground truth matches concept).
    
    Args:
        concept_probs: Concept probabilities for all samples [N, K]
        concept_indices: List of concept indices to visualize
        idx_to_concept: Mapping from concept index to name
        ground_truth_labels: Ground truth concept labels for all samples [N]
        output_path: Path to save the visualization
    """
    n_concepts = len(concept_indices)
    if n_concepts == 0:
        return
    
    # Prepare data for violin plots (only correct predictions)
    data = []
    labels = []
    
    for concept_idx in concept_indices:
        concept_name = idx_to_concept.get(concept_idx, f'Concept {concept_idx}')
        
        # Filter to only samples where ground truth matches this concept
        correct_mask = (ground_truth_labels == concept_idx)
        activations = concept_probs[correct_mask, concept_idx]  # [num_correct]
        
        if len(activations) == 0:
            print(f"   Warning: No correct predictions found for {concept_name}, skipping")
            continue
        
        data.append(activations)
        
        # Create label with training count - use full name but split intelligently
        formatted_name = concept_name.replace('_', ' ').title()
        
        # Add training count if available
        if train_counts and concept_idx in train_counts:
            count = train_counts[concept_idx]
            labels.append(f'{formatted_name}\n(idx {concept_idx}, n={count})')
        else:
            labels.append(f'{formatted_name}\n(idx {concept_idx})')
    
    if len(data) == 0:
        print("   Error: No data to plot (no correct predictions for any concept)")
        return
    
    # Create figure with violin plots - increase width for longer labels and height for less cramping
    fig, ax = plt.subplots(1, 1, figsize=(max(16, len(data) * 1.5), 12))
    
    # Create violin plots
    parts = ax.violinplot(
        data, 
        positions=range(len(data)),
        showmeans=True,
        showmedians=True,
        widths=0.7
    )
    
    # Customize violin plot colors
    for pc in parts['bodies']:
        pc.set_facecolor('steelblue')
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
    
    # Customize median and mean lines
    parts['cmedians'].set_color('orange')
    parts['cmedians'].set_linewidth(2)
    parts['cmeans'].set_color('red')
    parts['cmeans'].set_linewidth(2)
    
    # Set labels with better spacing
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(labels, rotation=90, ha='center', va='top', fontsize=8)
    ax.set_ylabel('Activation Score (Probability)', fontsize=10, fontweight='bold')
    ax.set_title('Distribution of Activation Scores for Correct Predictions',
                 fontsize=12, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(bottom=0)
    
    # Add legend for mean and median lines
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='Mean'),
        Line2D([0], [0], color='orange', linewidth=2, label='Median')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Calculate bottom margin based on longest label - need more space for rotated labels
    # Count total characters including newlines
    max_label_length = max(len(label.replace('\n', ' ')) for label in labels)
    # For 90-degree rotated labels, need significant vertical space
    # Each character when rotated 90 degrees takes more vertical space
    # Use a fixed larger margin to ensure labels don't overlap
    bottom_margin = 0.4  # Fixed 40% bottom margin for rotated labels
    
    plt.tight_layout(rect=[0, bottom_margin, 1, 0.98])  # Leave 40% at bottom for labels
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize top images per concept with attention")
    
    # Checkpoint path
    parser.add_argument("--checkpoint-both", type=str, required=True,
                       help="Checkpoint directory for img+conc model")
    
    # Data paths
    parser.add_argument("--test-csv", type=str, required=True, help="Test CSV path")
    parser.add_argument("--concept-data-dir", type=str, required=True,
                       help="Concept data directory")
    parser.add_argument("--cached-dir", type=str, required=True,
                       help="Cached embeddings directory")
    parser.add_argument("--image-dir", type=str, default=None,
                       help="Image directory (if images are available)")
    
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
    
    # Visualization options
    parser.add_argument("--num-concepts", type=int, default=20,
                       help="Number of top concepts to visualize")
    parser.add_argument("--num-images-per-concept", type=int, default=10,
                       help="Number of top images per concept to show")
    parser.add_argument("--min-samples", type=int, default=10,
                       help="Minimum number of samples required for a concept to be included")
    parser.add_argument("--concept-names", type=str, nargs="+", default=None,
                       help="Specific concept names to visualize (e.g., 'license_plate' 'chevron'). If provided, --num-concepts is ignored.")
    parser.add_argument("--distribution-only", action="store_true",
                       help="Only create activation distribution plot, skip individual concept visualizations")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("TOP IMAGES PER CONCEPT VISUALIZATION")
    print("="*80)
    
    # Load concept vocabulary
    concept_vocab_path = Path(args.concept_data_dir) / "concept_vocab.json"
    with open(concept_vocab_path) as f:
        concept_vocab = json.load(f)
    idx_to_concept = {int(k): v for k, v in concept_vocab["idx_to_concept"].items()}
    num_concepts = concept_vocab["num_concepts"]
    
    print(f"\n📂 Loaded {num_concepts} concepts")
    
    # Load test dataset (using cached embeddings only, NOT raw images)
    print(f"\n📊 Loading test dataset...")
    s2_vocab_path = Path(args.concept_data_dir) / "s2_cells.json"
    test_ds = ConceptDataset(
        str(args.test_csv),
        args.cached_dir,
        str(concept_vocab_path),
        str(s2_vocab_path),
        split="test",
        load_pooled_embeddings=True,
        # Don't pass image_dir here - we use cached embeddings for inference
        # Images are loaded separately during visualization
    )
    print(f"   Test samples: {len(test_ds)}")
    
    # Store image directory for visualization (separate from dataset)
    image_dir = Path(args.image_dir) if args.image_dir else None
    if image_dir and image_dir.exists():
        print(f"   Image directory: {image_dir}")
    else:
        print(f"   Warning: Image directory not found or not provided, attention overlays will not be shown")
    
    # Detect patch dimension
    patch_dim = test_ds.patch_tokens.shape[2]
    print(f"   Patch dimension: {patch_dim}")
    
    # Load model
    print(f"\n{'='*80}")
    print(f"Loading model")
    print(f"{'='*80}")
    checkpoint_dir = Path(args.checkpoint_both)
    print(f"Checkpoint: {checkpoint_dir}")
    
    phase1_model, stage2_model, concept_adapter, centers_xyz, mode, \
    pooled_projection, actual_pooled_dim, checkpoint_pooled_dim = load_model_from_checkpoint(
        checkpoint_dir, num_concepts, patch_dim, device, args
    )
    
    # Create data loader with custom collate function
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_simple,
    )
    
    # Evaluate model to get concept probabilities and attention
    print(f"\n{'='*80}")
    print("Evaluating model...")
    print(f"{'='*80}")
    results = evaluate_model_for_concepts(
        phase1_model, stage2_model, concept_adapter,
        test_loader, device, idx_to_concept
    )
    
    concept_probs = results['concept_probs']  # [N, K]
    attention_weights = results['attention_weights']  # [N, K, P]
    sample_indices = results['sample_indices']  # [N]
    ground_truth_labels = results['ground_truth_labels']  # [N]
    
    print(f"   Evaluated {len(sample_indices)} samples")
    print(f"   Concept probabilities shape: {concept_probs.shape}")
    print(f"   Attention weights shape: {attention_weights.shape}")
    print(f"   Ground truth labels shape: {ground_truth_labels.shape}")
    
    # Create mapping from dataset index to results index
    sample_indices_map = {idx: i for i, idx in enumerate(sample_indices)}
    
    # Find top images per concept (only correct predictions)
    print(f"\n{'='*80}")
    print("Finding top images per concept (correct predictions only)...")
    print(f"{'='*80}")
    top_images_per_concept = find_top_images_per_concept(
        concept_probs, sample_indices, ground_truth_labels, num_concepts, 
        args.num_images_per_concept, args.min_samples, only_correct=True
    )
    
    print(f"   Found top images for {len(top_images_per_concept)} concepts")
    
    # Log correct prediction counts for each concept
    for concept_idx, data in top_images_per_concept.items():
        concept_name = idx_to_concept.get(concept_idx, f'Concept {concept_idx}')
        num_correct = len(data['indices'])
        requested = args.num_images_per_concept
        if num_correct < requested:
            print(f"     {concept_name} (idx {concept_idx}): {num_correct} correct predictions (showing all {num_correct}, requested {requested})")
        else:
            print(f"     {concept_name} (idx {concept_idx}): {num_correct} correct predictions (showing top {requested})")
    
    # Select concepts to visualize
    if args.concept_names:
        # Filter by specific concept names
        concept_to_idx = {v: k for k, v in idx_to_concept.items()}
        top_concept_indices = []
        for concept_name in args.concept_names:
            if concept_name in concept_to_idx:
                concept_idx = concept_to_idx[concept_name]
                if concept_idx in top_images_per_concept:
                    top_concept_indices.append(concept_idx)
                    print(f"   Found concept: {concept_name} (idx: {concept_idx})")
                else:
                    print(f"   Warning: Concept '{concept_name}' not found in top images (may have < {args.min_samples} samples)")
            else:
                print(f"   Warning: Concept '{concept_name}' not found in vocabulary")
        
        if not top_concept_indices:
            print(f"   Error: No valid concepts found. Available concepts include:")
            for name in sorted(concept_to_idx.keys())[:20]:
                print(f"     - {name}")
            return
        
        print(f"\n{'='*80}")
        print(f"Visualizing {len(top_concept_indices)} specified concepts: {args.concept_names}")
        print(f"{'='*80}")
    else:
        # Select top N concepts by frequency or average score
        concept_avg_scores = {}
        for concept_idx, data in top_images_per_concept.items():
            concept_avg_scores[concept_idx] = np.mean(data['scores'])
        
        # Sort by average score
        sorted_concepts = sorted(concept_avg_scores.items(), key=lambda x: x[1], reverse=True)
        top_concept_indices = [idx for idx, _ in sorted_concepts[:args.num_concepts]]
        
        print(f"\n{'='*80}")
        print(f"Visualizing top {args.num_concepts} concepts...")
        print(f"{'='*80}")
    
    # Create individual concept visualizations (unless distribution-only mode)
    if not args.distribution_only:
        for concept_idx in tqdm(top_concept_indices, desc="Creating visualizations"):
            concept_name = idx_to_concept.get(concept_idx, f'Concept {concept_idx}')
            
            # Sanitize filename
            safe_name = concept_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
            safe_name = ''.join(c for c in safe_name if c.isalnum() or c in ('_', '-'))[:50]
            
            output_path = output_dir / f"concept_{concept_idx:04d}_{safe_name}.png"
            
            visualize_concept_images(
                concept_idx, concept_name,
                top_images_per_concept[concept_idx],
                test_ds,
                attention_weights,
                sample_indices_map,
                output_path,
                num_images=args.num_images_per_concept,
                image_dir=image_dir,
            )
    
    # Create activation distribution plot (if concept_names provided or distribution-only mode)
    if (args.concept_names or args.distribution_only) and len(top_concept_indices) > 0:
        print(f"\n{'='*80}")
        print("Creating activation distribution visualization (correct predictions only)...")
        print(f"{'='*80}")
        
        # Load training counts for each concept
        train_counts = {}
        train_csv_path = Path(args.test_csv).parent / "dataset_train.csv"
        if train_csv_path.exists():
            print(f"   Loading training counts from: {train_csv_path}")
            train_df = pd.read_csv(train_csv_path)
            if 'generalized' in train_df.columns:
                concept_counts = train_df['generalized'].value_counts()
                concept_to_idx = {v: k for k, v in idx_to_concept.items()}
                for concept_name, count in concept_counts.items():
                    if concept_name in concept_to_idx:
                        concept_idx = concept_to_idx[concept_name]
                        train_counts[concept_idx] = int(count)
                print(f"   Loaded training counts for {len(train_counts)} concepts")
            else:
                print(f"   Warning: 'generalized' column not found in training CSV, skipping counts")
        else:
            print(f"   Warning: Training CSV not found at {train_csv_path}, skipping counts")
        
        distribution_path = output_dir / "activation_distributions.png"
        visualize_activation_distributions(
            concept_probs,
            top_concept_indices,
            idx_to_concept,
            ground_truth_labels,
            distribution_path,
            train_counts=train_counts,
        )
        print(f"   ✅ Activation distributions saved to: {distribution_path}")
    
    print(f"\n✅ Visualizations saved to: {output_dir}")
    if not args.distribution_only:
        print(f"   Generated {len(top_concept_indices)} concept visualizations")
    else:
        print(f"   Generated activation distribution plot for {len(top_concept_indices)} concepts")


if __name__ == "__main__":
    main()

