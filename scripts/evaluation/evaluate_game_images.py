#!/usr/bin/env python3
"""
Evaluate both model variants on game images.

Takes a game directory containing images 1.png to 5.png, predicts their location
and country, and visualizes:
- Predictions on map using cartopy
- Attention maps
- Top activated concepts
- Gate values

Usage:
    python scripts/evaluation/evaluate_game_images.py \
        --game-dir data/Game1 \
        --checkpoint-both checkpoints/stage3_joint_latefusion/stage3_joint_both_scratchgeo_dim256_pos_latefusion_2.5ct_0.01gatereg \
        --checkpoint-concept checkpoints/stage3_joint_latefusion/stage3_joint_concept_only_scratchgeo_dim256_pos_latefusion_2.5ct_0gatereg \
        --concept-data-dir data/concept_data_v2 \
        --cached-dir /scratch-shared/igodzwon/Project_AI/data/6921d7831744c5356b098bf7_balanced/cached_streetclip_v2 \
        --output-dir results/game_evaluation
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Try to import cartopy
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Warning: cartopy not available. Install with: pip install cartopy")

# Try to import geopy for reverse geocoding
try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    HAS_GEOPY = True
except ImportError:
    HAS_GEOPY = False
    print("Warning: geopy not available. Install with: pip install geopy")

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cbm.phase1.model import Phase1CBMTopKMil
from cbm.phase2.model import ConceptEmbeddingAdapter, Stage2CrossAttentionGeoHead
from cbm.phase2.geocells import assign_geocells, compute_offsets, latlng_to_xyz
from cbm.phase2.metrics import xyz_to_latlng, haversine_km
from transformers import CLIPModel, CLIPProcessor

# CLIP normalization constants
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def load_streetclip_model(model_name: str = "geolocal/StreetCLIP", device: torch.device = None):
    """Load StreetCLIP model for extracting embeddings."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading StreetCLIP model: {model_name}")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    
    # Get expected number of patches
    if hasattr(model.config, 'vision_config'):
        image_size = getattr(model.config.vision_config, 'image_size', 336)
        patch_size = getattr(model.config.vision_config, 'patch_size', 14)
    else:
        image_size = getattr(model.config, 'image_size', 336)
        patch_size = getattr(model.config, 'patch_size', 14)
    expected_num_patches = (image_size // patch_size) ** 2
    
    return model, processor, expected_num_patches, image_size


@torch.no_grad()
def extract_embeddings_from_image(
    model: CLIPModel,
    processor: CLIPProcessor,
    image_path: Path,
    device: torch.device,
    expected_num_patches: int,
):
    """Extract patch tokens and pooled embeddings from a single image."""
    # Load and process image
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
    
    # Get pooled embedding
    pooled_emb = model.get_image_features(pixel_values=pixel_values)
    pooled_emb = torch.nn.functional.normalize(pooled_emb, p=2, dim=-1)
    
    # Get patch tokens
    outputs = model.vision_model(pixel_values=pixel_values)
    hidden_states = outputs.last_hidden_state  # [B, seq_len, D]
    
    seq_len = int(hidden_states.shape[1])
    if seq_len == expected_num_patches:
        patch_tokens = hidden_states
    elif seq_len > expected_num_patches:
        num_extra = seq_len - expected_num_patches
        patch_tokens = hidden_states[:, num_extra:, :]
    else:
        # Fallback
        side = int(np.sqrt(seq_len))
        if side * side == seq_len:
            patch_tokens = hidden_states
        else:
            patch_tokens = hidden_states[:, 1:, :]
    
    return patch_tokens.cpu(), pooled_emb.cpu()


def reverse_geocode(lat: float, lng: float, geocoder: Optional = None) -> Optional[str]:
    """Reverse geocode coordinates to get country name."""
    if not HAS_GEOPY:
        return None
    
    if geocoder is None:
        try:
            geocoder = Nominatim(user_agent="geolocation_evaluation", timeout=10)
        except Exception:
            return None
    
    try:
        location = geocoder.reverse((lat, lng), exactly_one=True, timeout=10)
        if location and location.raw and 'address' in location.raw:
            address = location.raw['address']
            # Try to get country
            country = address.get('country', None)
            if country:
                return country
            # Fallback to country_code
            country_code = address.get('country_code', None)
            if country_code:
                return country_code.upper()
    except (GeocoderTimedOut, GeocoderServiceError, Exception) as e:
        print(f"  Warning: Reverse geocoding failed for ({lat:.4f}, {lng:.4f}): {e}")
    
    return None


def load_model_from_checkpoint(
    checkpoint_dir: Path,
    num_concepts: int,
    patch_dim: int,
    device: torch.device,
    args
) -> Tuple:
    """Load phase1 model, stage2 model, and concept adapter from checkpoint directory."""
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
    
    return phase1_model, stage2_model, concept_adapter, centers_xyz, mode


@torch.no_grad()
def predict_single_image(
    phase1_model: Phase1CBMTopKMil,
    stage2_model: Stage2CrossAttentionGeoHead,
    concept_adapter: ConceptEmbeddingAdapter,
    patch_tokens: torch.Tensor,
    pooled_emb: torch.Tensor,
    centers_xyz: np.ndarray,
    device: torch.device,
) -> Dict:
    """Predict location and extract attention/concepts/gate for a single image."""
    # Move to device
    patch_tokens = patch_tokens.to(device)
    pooled_emb = pooled_emb.to(device)
    
    # Phase 1: Concept prediction
    c_logits, c_hidden, attn_weights, _ = phase1_model(patch_tokens)
    c_probs = torch.softmax(c_logits, dim=1)
    c_preds = c_logits.argmax(dim=1)
    
    # Get top concepts
    top5_probs, top5_idx = torch.topk(c_probs[0], min(5, c_probs.shape[1]))
    
    # Phase 2: Geolocation
    concept_emb = concept_adapter(c_logits)
    forward_output = stage2_model(concept_emb, patch_tokens, pooled_emb)
    
    # Extract gate value
    gate_val = None
    if stage2_model.mode == "both":
        cell_logits, offset_pred, gate_info, _, _ = forward_output
        if isinstance(gate_info, dict) and 'gate' in gate_info:
            gate = gate_info['gate']
            if gate is not None and isinstance(gate, torch.Tensor) and gate.numel() > 0:
                if gate.dim() == 2:
                    gate_val = gate[0, 0].item()
                else:
                    gate_val = gate[0].item()
    else:
        cell_logits, offset_pred, gate = forward_output
        if gate is not None and isinstance(gate, torch.Tensor) and gate.numel() > 0:
            if gate.dim() == 2:
                gate_val = gate[0, 0].item()
            else:
                gate_val = gate[0].item()
    
    # Convert to lat/lng
    pred_cell = cell_logits.argmax(dim=1)[0].item()
    pred_cell_center = centers_xyz[pred_cell]
    pred_xyz = pred_cell_center + offset_pred[0].cpu().numpy()
    pred_lat, pred_lng = xyz_to_latlng(pred_xyz.reshape(1, -1))
    pred_lat, pred_lng = pred_lat[0], pred_lng[0]
    
    return {
        'pred_lat': pred_lat,
        'pred_lng': pred_lng,
        'pred_cell': pred_cell,
        'attention_weights': attn_weights[0].cpu().numpy(),  # [K, P]
        'top5_concept_indices': top5_idx.cpu().tolist(),
        'top5_concept_probs': top5_probs.cpu().tolist(),
        'predicted_concept_idx': c_preds[0].item(),
        'concept_probs': c_probs[0].cpu().numpy(),
        'gate_value': gate_val,
    }


def visualize_image_results(
    image_path: Path,
    results_both: Dict,
    results_concept: Dict,
    idx_to_concept: Dict[int, str],
    output_path: Path,
    country_both: Optional[str] = None,
    country_concept: Optional[str] = None,
):
    """Create comprehensive visualization for a single image."""
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Load image
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Warning: Could not load image {image_path}: {e}")
        img = None
    
    # 1. Original image
    ax_img = fig.add_subplot(gs[0, 0])
    if img:
        ax_img.imshow(img)
    ax_img.axis('off')
    ax_img.set_title(f'Image: {image_path.name}', fontsize=12, fontweight='bold')
    
    # 2. Map with predictions (using cartopy if available)
    ax_map = fig.add_subplot(gs[0, 1:3], projection=ccrs.PlateCarree() if HAS_CARTOPY else None)
    
    if HAS_CARTOPY:
        ax_map.set_global()
        ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax_map.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle='--')
        ax_map.add_feature(cfeature.LAND, alpha=0.3)
        ax_map.add_feature(cfeature.OCEAN, alpha=0.3)
        ax_map.gridlines(draw_labels=True, alpha=0.5)
        
        # Plot predictions
        ax_map.scatter(
            results_both['pred_lng'], results_both['pred_lat'],
            c='green', s=200, marker='*', label=f"Both: ({results_both['pred_lat']:.4f}, {results_both['pred_lng']:.4f})",
            transform=ccrs.PlateCarree(), zorder=5, edgecolors='black', linewidths=2
        )
        ax_map.scatter(
            results_concept['pred_lng'], results_concept['pred_lat'],
            c='red', s=200, marker='s', label=f"Concept: ({results_concept['pred_lat']:.4f}, {results_concept['pred_lng']:.4f})",
            transform=ccrs.PlateCarree(), zorder=5, edgecolors='black', linewidths=2
        )
        
        # Add country labels
        if country_both:
            ax_map.text(results_both['pred_lng'], results_both['pred_lat'] + 2, 
                       f"Country: {country_both}", transform=ccrs.PlateCarree(),
                       fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        if country_concept:
            ax_map.text(results_concept['pred_lng'], results_concept['pred_lat'] - 2,
                       f"Country: {country_concept}", transform=ccrs.PlateCarree(),
                       fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        # Fallback: simple scatter plot
        ax_map.scatter(
            results_both['pred_lng'], results_both['pred_lat'],
            c='green', s=200, marker='*', label=f"Both: ({results_both['pred_lat']:.4f}, {results_both['pred_lng']:.4f})",
            zorder=5, edgecolors='black', linewidths=2
        )
        ax_map.scatter(
            results_concept['pred_lng'], results_concept['pred_lat'],
            c='red', s=200, marker='s', label=f"Concept: ({results_concept['pred_lat']:.4f}, {results_concept['pred_lng']:.4f})",
            zorder=5, edgecolors='black', linewidths=2
        )
        ax_map.set_xlim(-180, 180)
        ax_map.set_ylim(-90, 90)
        ax_map.set_xlabel('Longitude')
        ax_map.set_ylabel('Latitude')
        ax_map.grid(True, alpha=0.3)
        
        if country_both:
            ax_map.text(results_both['pred_lng'], results_both['pred_lat'] + 2,
                       f"Country: {country_both}", fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        if country_concept:
            ax_map.text(results_concept['pred_lng'], results_concept['pred_lat'] - 2,
                       f"Country: {country_concept}", fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax_map.set_title('Location Predictions', fontsize=12, fontweight='bold')
    ax_map.legend(loc='upper right', fontsize=9)
    
    # 3. Gate value (only for both model)
    ax_gate = fig.add_subplot(gs[0, 3])
    if results_both['gate_value'] is not None:
        gate_val = results_both['gate_value']
        ax_gate.barh([0], [gate_val], color='green' if gate_val > 0.5 else 'orange', alpha=0.7)
        ax_gate.set_xlim(0, 1)
        ax_gate.set_yticks([0])
        ax_gate.set_yticklabels(['Gate'])
        ax_gate.set_xlabel('Value', fontsize=10)
        ax_gate.set_title(f'Fusion Gate: {gate_val:.3f}', fontsize=11, fontweight='bold')
        ax_gate.axvline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Midpoint')
        ax_gate.legend(fontsize=8)
    else:
        ax_gate.text(0.5, 0.5, 'N/A\n(not both mode)', ha='center', va='center', transform=ax_gate.transAxes)
        ax_gate.set_title('Fusion Gate', fontsize=11, fontweight='bold')
    ax_gate.grid(True, alpha=0.3, axis='x')
    
    # 4. Top concepts - Both model
    ax_concepts_both = fig.add_subplot(gs[1, 0:2])
    top5_idx = results_both['top5_concept_indices']
    top5_probs = results_both['top5_concept_probs']
    top5_names = [idx_to_concept.get(idx, f'Concept {idx}') for idx in top5_idx]
    
    y_pos = np.arange(len(top5_names))
    ax_concepts_both.barh(y_pos, top5_probs, color='steelblue', alpha=0.7)
    ax_concepts_both.set_yticks(y_pos)
    ax_concepts_both.set_yticklabels(top5_names, fontsize=9)
    ax_concepts_both.set_xlabel('Probability', fontsize=10)
    ax_concepts_both.set_title('Top 5 Concepts (Both Model)', fontsize=11, fontweight='bold')
    ax_concepts_both.grid(True, alpha=0.3, axis='x')
    
    # 5. Top concepts - Concept model
    ax_concepts_conc = fig.add_subplot(gs[1, 2:4])
    top5_idx = results_concept['top5_concept_indices']
    top5_probs = results_concept['top5_concept_probs']
    top5_names = [idx_to_concept.get(idx, f'Concept {idx}') for idx in top5_idx]
    
    y_pos = np.arange(len(top5_names))
    ax_concepts_conc.barh(y_pos, top5_probs, color='coral', alpha=0.7)
    ax_concepts_conc.set_yticks(y_pos)
    ax_concepts_conc.set_yticklabels(top5_names, fontsize=9)
    ax_concepts_conc.set_xlabel('Probability', fontsize=10)
    ax_concepts_conc.set_title('Top 5 Concepts (Concept Model)', fontsize=11, fontweight='bold')
    ax_concepts_conc.grid(True, alpha=0.3, axis='x')
    
    # 6-7. Attention maps for top concepts (Both model)
    attn_weights = results_both['attention_weights']  # [K, P]
    top_concept_idx = results_both['top5_concept_indices'][0]
    top_concept_name = idx_to_concept.get(top_concept_idx, f'Concept {top_concept_idx}')
    
    # Reshape attention to spatial grid
    num_patches = attn_weights.shape[1]
    side = int(np.sqrt(num_patches))
    if side * side == num_patches:
        attn_map = attn_weights[top_concept_idx].reshape(side, side)
    else:
        # Fallback: pad or crop
        attn_map = attn_weights[top_concept_idx][:side*side].reshape(side, side)
    
    ax_attn_both = fig.add_subplot(gs[2, 0])
    im = ax_attn_both.imshow(attn_map, cmap='hot', interpolation='bilinear')
    ax_attn_both.set_title(f'Attention: {top_concept_name}\n(Both Model)', fontsize=10, fontweight='bold')
    ax_attn_both.axis('off')
    plt.colorbar(im, ax=ax_attn_both, fraction=0.046, pad=0.04)
    
    # 8. Attention overlay on image (Both model)
    ax_overlay_both = fig.add_subplot(gs[2, 1])
    if img:
        # Upsample attention to image size
        attn_tensor = torch.tensor(attn_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        img_size = img.size[1]  # height
        attn_upsampled = F.interpolate(attn_tensor, size=(img_size, img_size), mode='bilinear', align_corners=False)
        attn_upsampled = attn_upsampled.squeeze().numpy()
        
        ax_overlay_both.imshow(img)
        ax_overlay_both.imshow(attn_upsampled, cmap='hot', alpha=0.5, interpolation='bilinear')
    ax_overlay_both.set_title(f'Attention Overlay\n(Both Model)', fontsize=10, fontweight='bold')
    ax_overlay_both.axis('off')
    
    # 9-10. Attention maps for top concepts (Concept model)
    attn_weights = results_concept['attention_weights']  # [K, P]
    top_concept_idx = results_concept['top5_concept_indices'][0]
    top_concept_name = idx_to_concept.get(top_concept_idx, f'Concept {top_concept_idx}')
    
    # Reshape attention to spatial grid
    num_patches = attn_weights.shape[1]
    side = int(np.sqrt(num_patches))
    if side * side == num_patches:
        attn_map = attn_weights[top_concept_idx].reshape(side, side)
    else:
        attn_map = attn_weights[top_concept_idx][:side*side].reshape(side, side)
    
    ax_attn_conc = fig.add_subplot(gs[2, 2])
    im = ax_attn_conc.imshow(attn_map, cmap='hot', interpolation='bilinear')
    ax_attn_conc.set_title(f'Attention: {top_concept_name}\n(Concept Model)', fontsize=10, fontweight='bold')
    ax_attn_conc.axis('off')
    plt.colorbar(im, ax=ax_attn_conc, fraction=0.046, pad=0.04)
    
    # 11. Attention overlay on image (Concept model)
    ax_overlay_conc = fig.add_subplot(gs[2, 3])
    if img:
        # Upsample attention to image size
        attn_tensor = torch.tensor(attn_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        img_size = img.size[1]  # height
        attn_upsampled = F.interpolate(attn_tensor, size=(img_size, img_size), mode='bilinear', align_corners=False)
        attn_upsampled = attn_upsampled.squeeze().numpy()
        
        ax_overlay_conc.imshow(img)
        ax_overlay_conc.imshow(attn_upsampled, cmap='hot', alpha=0.5, interpolation='bilinear')
    ax_overlay_conc.set_title(f'Attention Overlay\n(Concept Model)', fontsize=10, fontweight='bold')
    ax_overlay_conc.axis('off')
    
    plt.suptitle(f'Game Image Evaluation: {image_path.name}', fontsize=14, fontweight='bold', y=0.995)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on game images")
    
    # Input paths
    parser.add_argument("--game-dir", type=str, required=True, help="Game directory with images 1.png to 5.png")
    parser.add_argument("--checkpoint-both", type=str, required=True, help="Checkpoint directory for img+conc model")
    parser.add_argument("--checkpoint-concept", type=str, required=True, help="Checkpoint directory for concept only model")
    
    # Data paths
    parser.add_argument("--concept-data-dir", type=str, required=True, help="Concept data directory")
    parser.add_argument("--cached-dir", type=str, required=True, help="Cached embeddings directory")
    
    # Output
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for results")
    
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
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--streetclip-model", type=str, default="geolocal/StreetCLIP", help="StreetCLIP model name")
    parser.add_argument("--skip-reverse-geocode", action="store_true", help="Skip reverse geocoding (faster)")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    game_dir = Path(args.game_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("GAME IMAGE EVALUATION")
    print("="*80)
    
    # Load concept vocabulary
    concept_vocab_path = Path(args.concept_data_dir) / "concept_vocab.json"
    with open(concept_vocab_path) as f:
        concept_vocab = json.load(f)
    idx_to_concept = {int(k): v for k, v in concept_vocab["idx_to_concept"].items()}
    num_concepts = concept_vocab["num_concepts"]
    
    print(f"\n📂 Loaded {num_concepts} concepts")
    
    # Load StreetCLIP model
    print(f"\n📸 Loading StreetCLIP model...")
    streetclip_model, processor, expected_num_patches, image_size = load_streetclip_model(
        args.streetclip_model, device
    )
    
    # Load models
    checkpoint_dirs = {
        'both': Path(args.checkpoint_both),
        'concept': Path(args.checkpoint_concept),
    }
    
    models = {}
    for model_name, checkpoint_dir in checkpoint_dirs.items():
        print(f"\n{'='*80}")
        print(f"Loading {model_name} model")
        print(f"{'='*80}")
        
        # Detect patch_dim from cached embeddings (sample one)
        cached_path = Path(args.cached_dir) / "patch_tokens.pt"
        if cached_path.exists():
            sample_patches = torch.load(cached_path, map_location='cpu')
            if isinstance(sample_patches, dict):
                # Get first tensor value
                patch_dim = next(iter(sample_patches.values())).shape[-1]
            else:
                patch_dim = sample_patches.shape[-1] if sample_patches.dim() > 1 else 768
        else:
            patch_dim = 768  # Default for StreetCLIP
        
        phase1_model, stage2_model, concept_adapter, centers_xyz, mode = load_model_from_checkpoint(
            checkpoint_dir, num_concepts, patch_dim, device, args
        )
        
        models[model_name] = {
            'phase1': phase1_model,
            'stage2': stage2_model,
            'adapter': concept_adapter,
            'centers_xyz': centers_xyz,
            'mode': mode,
        }
    
    # Initialize geocoder
    geocoder = None
    if HAS_GEOPY and not args.skip_reverse_geocode:
        try:
            geocoder = Nominatim(user_agent="geolocation_evaluation", timeout=10)
            print("\n🌍 Geocoder initialized")
        except Exception as e:
            print(f"\n⚠️  Warning: Could not initialize geocoder: {e}")
    
    # Process images
    print(f"\n{'='*80}")
    print("Processing game images...")
    print(f"{'='*80}")
    
    image_files = sorted([f for f in game_dir.glob("*.png") if f.stem.isdigit()])
    if not image_files:
        raise ValueError(f"No numbered PNG files (1.png, 2.png, etc.) found in {game_dir}")
    
    print(f"Found {len(image_files)} images: {[f.name for f in image_files]}")
    
    all_results = []
    
    for img_path in tqdm(image_files, desc="Processing images"):
        print(f"\n📷 Processing {img_path.name}...")
        
        # Extract embeddings
        patch_tokens, pooled_emb = extract_embeddings_from_image(
            streetclip_model, processor, img_path, device, expected_num_patches
        )
        
        # Predict with both models
        results_both = predict_single_image(
            models['both']['phase1'],
            models['both']['stage2'],
            models['both']['adapter'],
            patch_tokens,
            pooled_emb,
            models['both']['centers_xyz'],
            device,
        )
        
        results_concept = predict_single_image(
            models['concept']['phase1'],
            models['concept']['stage2'],
            models['concept']['adapter'],
            patch_tokens,
            pooled_emb,
            models['concept']['centers_xyz'],
            device,
        )
        
        # Reverse geocode
        country_both = None
        country_concept = None
        if not args.skip_reverse_geocode:
            print(f"  Reverse geocoding...")
            country_both = reverse_geocode(results_both['pred_lat'], results_both['pred_lng'], geocoder)
            country_concept = reverse_geocode(results_concept['pred_lat'], results_concept['pred_lng'], geocoder)
            if country_both:
                print(f"  Both model country: {country_both}")
            if country_concept:
                print(f"  Concept model country: {country_concept}")
        
        # Save visualization
        output_path = output_dir / f"{img_path.stem}_evaluation.png"
        visualize_image_results(
            img_path, results_both, results_concept, idx_to_concept,
            output_path, country_both, country_concept
        )
        print(f"  ✅ Saved visualization to {output_path}")
        
        # Store results
        all_results.append({
            'image': img_path.name,
            'both': {
                'lat': results_both['pred_lat'],
                'lng': results_both['pred_lng'],
                'country': country_both,
                'top_concept': idx_to_concept.get(results_both['top5_concept_indices'][0], 'Unknown'),
                'gate': results_both['gate_value'],
            },
            'concept': {
                'lat': results_concept['pred_lat'],
                'lng': results_concept['pred_lng'],
                'country': country_concept,
                'top_concept': idx_to_concept.get(results_concept['top5_concept_indices'][0], 'Unknown'),
            },
        })
    
    # Create summary map
    print(f"\n{'='*80}")
    print("Creating summary map...")
    print(f"{'='*80}")
    
    fig = plt.figure(figsize=(16, 8))
    if HAS_CARTOPY:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle='--')
        ax.add_feature(cfeature.LAND, alpha=0.3)
        ax.add_feature(cfeature.OCEAN, alpha=0.3)
        ax.gridlines(draw_labels=True, alpha=0.5)
    else:
        ax = plt.axes()
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, alpha=0.3)
    
    # Plot all predictions
    for i, result in enumerate(all_results):
        img_name = result['image']
        both_lat, both_lng = result['both']['lat'], result['both']['lng']
        conc_lat, conc_lng = result['concept']['lat'], result['concept']['lng']
        
        if HAS_CARTOPY:
            ax.scatter(both_lng, both_lat, c='green', s=300, marker='*', 
                      label='Both' if i == 0 else '', transform=ccrs.PlateCarree(),
                      zorder=5, edgecolors='black', linewidths=2)
            ax.scatter(conc_lng, conc_lat, c='red', s=300, marker='s',
                      label='Concept' if i == 0 else '', transform=ccrs.PlateCarree(),
                      zorder=5, edgecolors='black', linewidths=2)
            ax.text(both_lng, both_lat + 1, img_name, transform=ccrs.PlateCarree(),
                   fontsize=8, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        else:
            ax.scatter(both_lng, both_lat, c='green', s=300, marker='*',
                      label='Both' if i == 0 else '', zorder=5, edgecolors='black', linewidths=2)
            ax.scatter(conc_lng, conc_lat, c='red', s=300, marker='s',
                      label='Concept' if i == 0 else '', zorder=5, edgecolors='black', linewidths=2)
            ax.text(both_lng, both_lat + 1, img_name, fontsize=8, ha='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_title('All Predictions Summary', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    plt.savefig(output_dir / "summary_map.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved summary map to {output_dir / 'summary_map.png'}")
    
    # Save results JSON
    results_json = {
        'game_dir': str(game_dir),
        'results': all_results,
    }
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"  ✅ Saved results to {output_dir / 'results.json'}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for result in all_results:
        print(f"\n{result['image']}:")
        print(f"  Both model: ({result['both']['lat']:.4f}, {result['both']['lng']:.4f}) - {result['both']['country'] or 'Unknown'}")
        print(f"    Top concept: {result['both']['top_concept']}")
        if result['both']['gate'] is not None:
            print(f"    Gate value: {result['both']['gate']:.3f}")
        print(f"  Concept model: ({result['concept']['lat']:.4f}, {result['concept']['lng']:.4f}) - {result['concept']['country'] or 'Unknown'}")
        print(f"    Top concept: {result['concept']['top_concept']}")
    
    print(f"\n✅ All results saved to: {output_dir}")


if __name__ == "__main__":
    main()

