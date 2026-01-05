#!/usr/bin/env python3
"""
Evaluate CBM model on game images (1.png, 2.png, etc.) from a directory.

For each image, shows:
- Geolocation prediction
- Top-5 concept predictions with probabilities
- Attention visualizations for top concepts
- All predictions visualized on a world map

Usage:
    python scripts/evaluation/evaluate_game_images.py \
        --checkpoint checkpoints/stage3_joint/stage3_joint_both_scratchgeo_dim256_pos_both_gate_ct2.5_gemini \
        --image-dir data/Game1 \
        --concept-data-dir data/concept_data_v2 \
        --output-dir results/game1_evaluation
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns

# Global variable for world map dataset
WORLD_DATA = None

def get_country_from_coords(lat: float, lng: float) -> str:
    """Map latitude and longitude to a country name using reverse_geocoder or geopandas."""
    global WORLD_DATA
    
    # Try reverse_geocoder first (simpler and more reliable)
    try:
        import reverse_geocoder as rg
        result = rg.search((lat, lng))
        if result and len(result) > 0:
            country = result[0].get('cc', '')
            # Convert country code to name if needed, or return code
            # For now, return the admin1 (state/province) or country code
            admin1 = result[0].get('admin1', '')
            if admin1:
                return f"{admin1}, {country}"
            return country
    except ImportError:
        pass
    except Exception as e:
        pass
    
    # Fallback to geopandas
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        
        if WORLD_DATA is None:
            # Try different ways to load the world dataset
            try:
                # Method 1: Use geopandas built-in dataset
                world_path = gpd.datasets.get_path('naturalearth_lowres')
                WORLD_DATA = gpd.read_file(world_path)
            except (AttributeError, FileNotFoundError):
                # Method 2: Try direct download
                try:
                    import urllib.request
                    import tempfile
                    url = "https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/world.geojson"
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.geojson') as tmp:
                        urllib.request.urlretrieve(url, tmp.name)
                        WORLD_DATA = gpd.read_file(tmp.name)
                except:
                    # Method 3: Try naturalearth_lowres from cartopy
                    try:
                        import cartopy.io.shapereader as shpreader
                        shpfilename = shpreader.natural_earth(resolution='110m', category='cultural', name='admin_0_countries')
                        WORLD_DATA = gpd.read_file(shpfilename)
                    except:
                        return "Unknown (Dataset unavailable)"
        
        # Create a point and check for intersection
        point = Point(lng, lat)  # Note: Point takes (x, y) = (lng, lat)
        # Search for country
        match = WORLD_DATA[WORLD_DATA.geometry.contains(point)]
        if not match.empty:
            # Try different possible name columns
            for col in ['NAME', 'name', 'NAME_EN', 'name_en', 'ADMIN', 'admin']:
                if col in match.columns:
                    return match.iloc[0][col]
            # If no name column found, return first non-geometry column
            return str(match.iloc[0][match.columns[0]])
        else:
            return "Unknown (Ocean/No match)"
            
    except ImportError:
        return "Unknown (geopandas missing)"
    except Exception as e:
        return f"Unknown ({str(e)[:30]})"


# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cbm.phase1.model import Phase1CBMTopKMil
from cbm.phase2.model import ConceptEmbeddingAdapter, Stage2CrossAttentionGeoHead
from cbm.phase2.metrics import xyz_to_latlng
from cbm.phase2.geocells import assign_geocells, compute_offsets

# Import StreetCLIP for embedding extraction
from transformers import CLIPModel, CLIPProcessor

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# CLIP normalization constants
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def load_streetclip_model(device: torch.device, model_name: str = "geolocal/StreetCLIP"):
    """Load StreetCLIP model for extracting embeddings."""
    print(f"Loading StreetCLIP model: {model_name}")
    clip_model = CLIPModel.from_pretrained(model_name).to(device)
    clip_model.eval()
    vision_model = clip_model.vision_model
    processor = CLIPProcessor.from_pretrained(model_name)
    
    return clip_model, vision_model, processor


@torch.no_grad()
def extract_embeddings(
    vision_model,
    processor,
    image_path: Path,
    device: torch.device,
    image_size: int = 336,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract pooled and patch embeddings from an image.
    
    Returns:
        pooled_emb: [1, D] pooled embedding (CLS token)
        patch_tokens: [1, P, D] patch token embeddings
    """
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size))
    
    # Process with CLIP processor
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    
    # Forward through vision model
    vision_outputs = vision_model(pixel_values=pixel_values)
    
    # Get pooled embedding (CLS token)
    pooled_emb = vision_outputs.pooler_output  # [1, hidden_size]
    if pooled_emb is None:
        # Fallback: use mean pooling of last hidden state
        pooled_emb = vision_outputs.last_hidden_state.mean(dim=1)
    
    # Get patch tokens (exclude CLS token)
    patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]  # [1, P, hidden_size]
    
    # Normalize
    pooled_emb = F.normalize(pooled_emb, p=2, dim=-1)
    
    return pooled_emb, patch_tokens


def load_model(checkpoint_dir: Path, num_concepts: int, patch_dim: int, device: torch.device, args) -> Tuple:
    """
    Load phase1 model, stage2 model, and concept adapter from checkpoint directory.
    
    Handles checkpoint formats from:
    - cbm.joint.train: saves phase1/best_phase1.pt, phase2/best_phase2.pt, best_joint.pt
    - cbm.phase2.train: saves phase2/best_phase2.pt
    
    Checkpoint structures:
    - Phase1: {'model_state_dict': ...}
    - Phase2: {'stage2_model_state_dict': ..., 'concept_adapter_state_dict': ..., 'geocell_info': ...}
    - Joint: {'phase1_model_state_dict': ..., 'stage2_model_state_dict': ..., 'concept_adapter_state_dict': ...}
    """
    
    # =========================================================================
    # 1. LOAD PHASE 1 MODEL
    # =========================================================================
    # Prefer best_joint.pt for the whole architecture
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
    
    # Get the correct state dict key based on checkpoint type
    if use_joint_checkpoint:
        phase1_state = phase1_checkpoint.get('phase1_model_state_dict', {})
    else:
        phase1_state = phase1_checkpoint.get('model_state_dict', {})
    
    if not phase1_state:
        raise ValueError(f"Could not find Phase1 state dict in checkpoint. Keys: {list(phase1_checkpoint.keys())}")
    
    # Detect Phase1 model configuration from state dict
    has_pos_embed = any('pos_embed' in k for k in phase1_state.keys())
    has_per_concept_tau = any('concept_tau_logit' in k for k in phase1_state.keys())
    
    # Detect concept_dim from query tensor shape: query is [num_concepts, concept_dim]
    checkpoint_concept_dim = args.concept_dim  # Fallback
    for k, v in phase1_state.items():
        if k == 'query' or k.endswith('.query'):
            checkpoint_concept_dim = v.shape[1]
            print(f"   Detected concept_dim from Phase1 query: {checkpoint_concept_dim}")
            break
    
    # Detect patch_dim from patch_proj 
    checkpoint_patch_dim = patch_dim  # Fallback
    for k, v in phase1_state.items():
        if 'patch_proj' in k and 'weight' in k:
            if len(v.shape) == 2:
                checkpoint_patch_dim = v.shape[1]
                print(f"   Detected patch_dim from Phase1: {checkpoint_patch_dim}")
                break
    
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
    
    # Load Phase1 weights
    phase1_model.load_state_dict(phase1_state)
    phase1_model = phase1_model.to(device)
    phase1_model.eval()
    print(f"   Phase1 loaded (concept_dim={checkpoint_concept_dim}, pos={has_pos_embed}, adaptive_tau={has_per_concept_tau})")
    
    # =========================================================================
    # 2. LOAD GEOCELLS
    # =========================================================================
    # Try to get geocells from joint checkpoint first, then fall back to phase2/geocells.json
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
    
    # =========================================================================
    # 3. LOAD PHASE 2 / STAGE 2 MODEL
    # =========================================================================
    if use_joint_checkpoint:
        # Reuse the same checkpoint object we already loaded
        stage2_checkpoint = phase1_checkpoint
        print(f"   Loading Stage2 from best_joint.pt (reusing loaded checkpoint)")
    else:
        stage2_ckpt_path = checkpoint_dir / "phase2" / "best_phase2.pt"
        if not stage2_ckpt_path.exists():
            raise FileNotFoundError(f"No phase2 checkpoint found in {checkpoint_dir}")
        print(f"   Loading Stage2 from: {stage2_ckpt_path}")
        stage2_checkpoint = torch.load(stage2_ckpt_path, map_location=device, weights_only=False)
    
    # Get Stage2 state dict (key is 'stage2_model_state_dict' in both phase2 and joint checkpoints)
    stage2_state = stage2_checkpoint.get('stage2_model_state_dict', {})
    if not stage2_state:
        raise ValueError(f"Could not find 'stage2_model_state_dict' in checkpoint. Keys: {list(stage2_checkpoint.keys())}")
    
    # Detect mode from Stage2 state dict (matches cbm/joint/train.py logic)
    keys = list(stage2_state.keys())
    has_image_adapter = any(k.startswith('image_adapter') for k in keys)
    has_concept_adapter = any(k.startswith('concept_adapter') for k in keys)
    has_fusion_gate = any(k.startswith('fusion_gate') for k in keys)
    has_pooled_proj = any(k.startswith('pooled_proj') for k in keys)
    has_concept_proj = any(k.startswith('concept_proj') for k in keys)
    
    # Check for old architecture (Sequential layers: concept_proj.0.weight, pooled_proj.0.weight)
    has_old_concept_proj = any('concept_proj.0.weight' in k for k in keys)
    has_old_pooled_proj = any('pooled_proj.0.weight' in k for k in keys)
    
    # Mode detection: fusion_gate is the strongest indicator of "both" mode
    if has_fusion_gate or (has_image_adapter and has_concept_adapter):
        mode = "both"
    elif has_concept_adapter or (has_concept_proj and not has_pooled_proj) or has_old_concept_proj:
        mode = "concept_only"
    elif has_image_adapter or has_pooled_proj or has_old_pooled_proj:
        mode = "image_only"
    else:
        mode = "both"  # Default fallback
        print(f"   Warning: Could not detect mode from state dict. Defaulting to 'both'.")
    print(f"   Detected mode: {mode}")
    
    # Detect pooled_dim from pooled_proj weight shape (handle both old and new formats)
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
    
    # Load Stage2 weights with key mapping for old architecture
    # Map old Sequential format (concept_proj.0.weight) to new Linear format (concept_proj.weight)
    mapped_state = {}
    for k, v in stage2_state.items():
        # Skip concept_refinement keys (old architecture, not in current model)
        if 'concept_refinement' in k:
            continue
        
        # Skip fusion_gate layers with size mismatches (old architecture had different structure)
        # Old: fusion_gate.3 was Linear(hidden_dim, hidden_dim)
        # New: fusion_gate.3 is Linear(hidden_dim, 2)
        if 'fusion_gate.3' in k:
            # Check if shape matches expected (for fusion_gate.3, should be [2, hidden_dim] for weight, [2] for bias)
            if 'weight' in k and v.shape[0] != 2:
                print(f"   Skipping {k} (shape {v.shape}, expected [2, {v.shape[1]}])")
                continue
            elif 'bias' in k and v.shape[0] != 2:
                print(f"   Skipping {k} (shape {v.shape}, expected [2])")
                continue
        
        # Map old Sequential format to new Linear format
        if k == 'concept_proj.0.weight':
            mapped_state['concept_proj.weight'] = v
        elif k == 'concept_proj.0.bias':
            mapped_state['concept_proj.bias'] = v
        elif k == 'pooled_proj.0.weight':
            mapped_state['pooled_proj.weight'] = v
        elif k == 'pooled_proj.0.bias':
            mapped_state['pooled_proj.bias'] = v
        else:
            # Keep other keys as-is
            mapped_state[k] = v
    
    # Load with strict=False to handle any remaining mismatches
    missing_keys, unexpected_keys = stage2_model.load_state_dict(mapped_state, strict=False)
    if missing_keys:
        print(f"   Warning: Missing keys (will use random init): {missing_keys[:5]}..." if len(missing_keys) > 5 else f"   Warning: Missing keys: {missing_keys}")
    if unexpected_keys:
        # Filter out concept_refinement keys (expected in old checkpoints)
        unexpected_filtered = [k for k in unexpected_keys if 'concept_refinement' not in k]
        if unexpected_filtered:
            print(f"   Warning: Unexpected keys (ignored): {unexpected_filtered[:5]}..." if len(unexpected_filtered) > 5 else f"   Warning: Unexpected keys: {unexpected_filtered}")
    
    stage2_model = stage2_model.to(device)
    stage2_model.eval()
    print(f"   Stage2 loaded (mode={mode})")
    
    # =========================================================================
    # 4. LOAD CONCEPT ADAPTER
    # =========================================================================
    concept_temp = getattr(args, 'concept_temperature', 5.0)
    if 'concept_adapter_state_dict' in stage2_checkpoint:
        adapter_state = stage2_checkpoint['concept_adapter_state_dict']
        concept_vectors = adapter_state['concept_vectors']  # [K, D]
        concept_adapter = ConceptEmbeddingAdapter(concept_vectors, temperature=concept_temp)
        concept_adapter.load_state_dict(adapter_state)
        print(f"   Concept adapter loaded (vectors shape: {tuple(concept_vectors.shape)})")
    else:
        print("   Warning: concept_adapter_state_dict not found. Using random vectors.")
        concept_vectors = torch.randn(num_concepts, checkpoint_concept_dim)
        concept_adapter = ConceptEmbeddingAdapter(concept_vectors, temperature=concept_temp)
    
    concept_adapter = concept_adapter.to(device)
    concept_adapter.eval()
    
    # Create pooled_emb projection if needed (maps actual StreetCLIP output to expected pooled_dim)
    pooled_projection = None
    actual_pooled_dim = None  # Will be detected from actual StreetCLIP output
    
    return phase1_model, stage2_model, concept_adapter, centers_xyz, mode, pooled_projection, actual_pooled_dim, checkpoint_pooled_dim


@torch.no_grad()
def predict_image(
    phase1_model: Phase1CBMTopKMil,
    stage2_model: Stage2CrossAttentionGeoHead,
    concept_adapter: ConceptEmbeddingAdapter,
    patch_tokens: torch.Tensor,
    pooled_emb: torch.Tensor,
    device: torch.device,
    centers_xyz: np.ndarray,
    idx_to_concept: Dict[int, str],
) -> Dict:
    """Run inference on a single image and return predictions."""
    
    # Phase 1: Concept prediction
    c_logits, c_hidden, attn_weights, _ = phase1_model(patch_tokens)
    c_probs = torch.softmax(c_logits, dim=1)
    
    # Get top-5 concepts
    top5_probs, top5_idx = torch.topk(c_probs, min(5, c_probs.size(1)), dim=1)
    top5_concepts = [idx_to_concept[int(idx)] for idx in top5_idx[0]]
    
    # Phase 2: Geolocation
    concept_emb = concept_adapter(c_logits)
    forward_output = stage2_model(concept_emb, patch_tokens, pooled_emb)
    
    if stage2_model.mode == "both":
        cell_logits, offset_pred, gate, _, _ = forward_output
        # Gate is [B, hidden_dim] where all values in each row are the same (concept weight)
        # Extract using gate[:, 0] to get first column (matches analyze_model_errors.py)
        if gate is not None and gate.numel() > 0:
            # gate[:, 0] gets first column, then [0] gets first sample's value
            gate_val = float(gate[:, 0][0].item())
            # Debug: verify gate shape and value
            if gate_val == 0.0:
                print(f"   Warning: Gate value is 0.0 (gate shape: {gate.shape}, gate min: {gate.min().item():.6f}, gate max: {gate.max().item():.6f})")
        else:
            gate_val = 0.0
            print(f"   Warning: Gate is None or empty")
    else:
        cell_logits, offset_pred, _ = forward_output
        gate_val = 0.0
    
    # Convert to lat/lng
    pred_cell = int(cell_logits.argmax(dim=1)[0].item())
    pred_cell_center = centers_xyz[pred_cell]
    pred_xyz = pred_cell_center + offset_pred[0].cpu().numpy()
    pred_lat, pred_lng = xyz_to_latlng(pred_xyz.reshape(1, -1))
    
    # Get country mapping
    country = get_country_from_coords(float(pred_lat[0]), float(pred_lng[0]))
    
    return {
        "pred_lat": float(pred_lat[0]),
        "pred_lng": float(pred_lng[0]),
        "pred_country": country,
        "top5_concepts": top5_concepts,
        "top5_probs": top5_probs[0].cpu().tolist(),
        "top1_concept": top5_concepts[0],
        "top1_prob": float(top5_probs[0, 0].item()),
        "attention_weights": attn_weights[0].cpu().numpy() if attn_weights is not None else None,
        "gate_value": gate_val,
    }


def visualize_image_predictions(
    image_path: Path,
    predictions: Dict,
    idx_to_concept: Dict[int, str],
    output_path: Path,
    image_size: int = 336,
):
    """Create comprehensive visualization for a single image."""
    
    # Load image
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size))
    img_array = np.array(img)
    
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 4, figure=fig, width_ratios=[1.5, 1, 1, 1], height_ratios=[1, 1])
    
    # 1. Original image with prediction info
    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(img_array)
    ax_img.axis('off')
    
    # Add prediction info box
    info_text = (
        f"Geolocation Prediction\n"
        f"Lat: {predictions['pred_lat']:.4f} deg\n"
        f"Lng: {predictions['pred_lng']:.4f} deg\n"
        f"Country: {predictions['pred_country']}\n"
        f"\nTop Concept: {predictions['top1_concept']}\n"
        f"Confidence: {predictions['top1_prob']*100:.1f}%\n"
    )
    if predictions['gate_value'] > 0:
        info_text += f"\nGate: {predictions['gate_value']:.3f}\n(0=Image, 1=Concepts)"
    
    ax_img.text(0.02, 0.98, info_text, transform=ax_img.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # 2-4. Attention overlays for top-3 concepts
    attn_weights = predictions['attention_weights']
    if attn_weights is not None:
        K, P = attn_weights.shape
        side = int(math.isqrt(P))
        
        for i, (concept, prob) in enumerate(zip(predictions['top5_concepts'][:3], predictions['top5_probs'][:3])):
            ax = fig.add_subplot(gs[0, i+1])
            
            # Find concept index
            concept_idx = None
            for idx, name in idx_to_concept.items():
                if name == concept:
                    concept_idx = idx
                    break
            
            if concept_idx is not None and concept_idx < K:
                attn = attn_weights[concept_idx]
                attn_grid = attn.reshape(side, side)
                
                # Upsample attention
                attn_tensor = torch.tensor(attn_grid).unsqueeze(0).unsqueeze(0).float()
                attn_up = F.interpolate(
                    attn_tensor,
                    size=(image_size, image_size),
                    mode='bilinear',
                    align_corners=False
                ).squeeze().numpy()
                
                # Normalize
                attn_up = (attn_up - attn_up.min()) / (attn_up.max() - attn_up.min() + 1e-8)
                
                ax.imshow(img_array)
                ax.imshow(attn_up, cmap='magma', alpha=0.5)
                ax.set_title(f"{concept}\np={prob:.3f}", fontsize=10, fontweight='bold')
            else:
                ax.imshow(img_array)
                ax.set_title(f"{concept}\n(No attention)", fontsize=10)
            ax.axis('off')
    
    # 5. Top-5 concept probabilities bar chart
    ax_probs = fig.add_subplot(gs[1, 1])
    top5_concepts = predictions['top5_concepts']
    top5_probs = predictions['top5_probs']
    y_pos = np.arange(len(top5_concepts))
    colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(top5_concepts))]
    ax_probs.barh(y_pos, top5_probs, color=colors)
    ax_probs.set_yticks(y_pos)
    ax_probs.set_yticklabels(top5_concepts, fontsize=9)
    ax_probs.invert_yaxis()
    ax_probs.set_xlim(0, 1)
    ax_probs.set_xlabel("Probability", fontsize=10)
    ax_probs.set_title("Top-5 Concept Predictions", fontsize=11, fontweight='bold')
    for i, prob in enumerate(top5_probs):
        ax_probs.text(prob + 0.01, i, f'{prob:.3f}', va='center', fontsize=9)
    
    # 6. Gate visualization (if applicable)
    ax_gate = fig.add_subplot(gs[1, 2])
    gate_val = predictions['gate_value']
    ax_gate.barh(['Gate'], [gate_val], color='#3498db')
    ax_gate.barh(['Gate'], [1-gate_val], left=[gate_val], color='#95a5a6', alpha=0.3)
    ax_gate.set_xlim(0, 1)
    ax_gate.set_xlabel("Gate Value", fontsize=10)
    ax_gate.set_title(f"Fusion Gate\n{gate_val:.3f}", fontsize=11, fontweight='bold')
    ax_gate.text(gate_val/2, 0, f'{gate_val:.2%}', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 7. Geolocation info
    ax_geo = fig.add_subplot(gs[1, 3])
    ax_geo.axis('off')
    geo_text = (
        f"Geolocation\n\n"
        f"Latitude: {predictions['pred_lat']:.6f} deg\n"
        f"Longitude: {predictions['pred_lng']:.6f} deg\n"
        f"Country: {predictions['pred_country']}\n\n"
        f"Map coordinates:\n"
        f"({predictions['pred_lat']:.4f}, {predictions['pred_lng']:.4f})"
    )
    ax_geo.text(0.5, 0.5, geo_text, transform=ax_geo.transAxes, fontsize=11,
                ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.suptitle(f"Image: {image_path.name}", fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_map(
    predictions_list: List[Dict],
    image_names: List[str],
    output_path: Path,
):
    """Visualize all predictions on a physical world map using Cartopy."""
    
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        USE_CARTOPY = True
    except ImportError:
        USE_CARTOPY = False
        print("⚠️ Warning: Cartopy not installed. Using simple coordinate plot fallback.")
        print("   Install with: pip install cartopy")

    # Extract coordinates and countries
    lats = [p['pred_lat'] for p in predictions_list]
    lngs = [p['pred_lng'] for p in predictions_list]
    countries = [p.get('pred_country', 'Unknown') for p in predictions_list]
    
    if USE_CARTOPY:
        # Create figure with proper cartopy projection
        fig = plt.figure(figsize=(20, 12))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Add physical map features with higher resolution
        ax.add_feature(cfeature.LAND, facecolor='#f0f0e8', edgecolor='none', zorder=1)
        ax.add_feature(cfeature.OCEAN, facecolor='#c8e4f0', edgecolor='none', zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#666666', zorder=2)
        ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5, edgecolor='#888888', alpha=0.6, zorder=2)
        ax.add_feature(cfeature.LAKES, facecolor='#c8e4f0', edgecolor='#666666', linewidth=0.5, alpha=0.7, zorder=2)
        ax.add_feature(cfeature.RIVERS, edgecolor='#c8e4f0', linewidth=0.5, alpha=0.5, zorder=2)
        
        # Set global extent
        ax.set_global()
        
        # Add gridlines with labels
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.5,
            color='gray',
            alpha=0.5,
            linestyle='--',
            zorder=3
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}
        
        # Plot predictions with proper cartopy transform
        colors = plt.cm.Set3(np.linspace(0, 1, len(predictions_list)))
        
        for i, (lat, lng, name, country) in enumerate(zip(lats, lngs, image_names, countries)):
            label = name.replace('.png', '')
            
            # Plot marker
            ax.scatter(
                lng, lat,
                c=[colors[i]],
                s=300,
                alpha=0.9,
                edgecolors='black',
                linewidths=2.5,
                transform=ccrs.PlateCarree(),
                zorder=10,
                marker='o'
            )
            
            # Add annotation with country info
            annotation_text = f"{label}\n({country})" if country and country != "Unknown" else label
            ax.text(
                lng + 3, lat + 3,
                annotation_text,
                fontsize=11,
                fontweight='bold',
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    facecolor='white',
                    alpha=0.9,
                    edgecolor=colors[i],
                    linewidth=2
                ),
                transform=ccrs.PlateCarree(),
                zorder=11,
                verticalalignment='bottom'
            )
        
        ax.set_title("Geolocation Predictions Map", fontsize=20, fontweight='bold', pad=20)
        
    else:
        # Fallback: simple plot
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude", fontsize=12)
        ax.set_ylabel("Latitude", fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(predictions_list)))
        
        for i, (lat, lng, name, country) in enumerate(zip(lats, lngs, image_names, countries)):
            label = name.replace('.png', '')
            ax.scatter(lng, lat, c=[colors[i]], s=200, alpha=0.7, edgecolors='black', linewidths=2)
            annotation_text = f"{label}\n({country})" if country and country != "Unknown" else label
            ax.annotate(
                annotation_text,
                (lng, lat),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=11,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
            )
        
        ax.set_title("Geolocation Predictions Map", fontsize=18, fontweight='bold', pad=20)
    
    # Add info box with country details
    info_lines = [f"Total Predictions: {len(predictions_list)}"]
    for i, (name, country) in enumerate(zip(image_names, countries)):
        info_lines.append(f"{name.replace('.png', '')}: {country}")
    
    info_text = "\n".join(info_lines[:6])  # Limit to first 6 for readability
    if len(info_lines) > 6:
        info_text += f"\n... and {len(info_lines) - 6} more"
    
    ax.text(
        0.02, 0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'),
        family='monospace'
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Map saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CBM model on game images")
    
    # Required paths
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory with images (1.png, 2.png, ...)")
    parser.add_argument("--concept-data-dir", type=str, required=True, help="Concept data directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    
    # Model config
    parser.add_argument("--concept-dim", type=int, default=256, help="Concept dimension")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--phase1-dropout", type=float, default=0.3, help="Phase1 dropout")
    parser.add_argument("--mil-topk", type=int, default=6, help="MIL top-K")
    parser.add_argument("--mil-tau", type=float, default=0.25, help="MIL tau")
    parser.add_argument("--mix-depth", type=int, default=1, help="Mix depth")
    parser.add_argument("--mix-heads", type=int, default=4, help="Mix heads")
    parser.add_argument("--mix-mlp-ratio", type=float, default=2.0, help="Mix MLP ratio")
    parser.add_argument("--mix-local-kernel", type=int, default=5, help="Mix local kernel size")
    parser.add_argument("--proj-type", type=str, default="simple", help="Projection type")
    parser.add_argument("--concept-temperature", type=float, default=5.0, help="Concept adapter temperature")
    
    # Options
    parser.add_argument("--image-size", type=int, default=336, help="Image size for processing")
    parser.add_argument("--vision-model", type=str, default="geolocal/StreetCLIP", help="Vision model name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_dir = Path(args.image_dir)
    
    print("="*70)
    print("🎮 GAME IMAGE EVALUATION")
    print("="*70)
    
    # Load concept vocabulary
    concept_vocab_path = Path(args.concept_data_dir) / "concept_vocab.json"
    with open(concept_vocab_path) as f:
        concept_vocab = json.load(f)
    idx_to_concept = {int(k): v for k, v in concept_vocab["idx_to_concept"].items()}
    num_concepts = concept_vocab["num_concepts"]
    
    print(f"\n📂 Loaded {num_concepts} concepts")
    
    # Load StreetCLIP model
    print("\n🤖 Loading StreetCLIP model...")
    vision_model, processor = load_streetclip_model(device, args.vision_model)
    patch_dim = vision_model.config.hidden_size
    print(f"   Patch dimension: {patch_dim}")
    
    # Load CBM models
    print("\n🤖 Loading CBM models...")
    checkpoint_dir = Path(args.checkpoint)
    phase1_model, stage2_model, concept_adapter, centers_xyz, mode, pooled_projection, actual_pooled_dim, checkpoint_pooled_dim = load_model(
        checkpoint_dir, num_concepts, patch_dim, device, args
    )
    print(f"   Stage2 mode: {mode}")
    
    # Detect actual pooled_dim from StreetCLIP by extracting from a dummy image
    if actual_pooled_dim is None:
        print("   Detecting actual pooled_dim from StreetCLIP...")
        dummy_image = Image.new('RGB', (args.image_size, args.image_size), color='black')
        inputs = processor(images=dummy_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            vision_outputs = vision_model(pixel_values=pixel_values)
            dummy_pooled = vision_outputs.pooler_output
            if dummy_pooled is None:
                dummy_pooled = vision_outputs.last_hidden_state.mean(dim=1)
            actual_pooled_dim = dummy_pooled.shape[-1]
        print(f"   Actual StreetCLIP pooled_dim: {actual_pooled_dim}")
        
        # Create projection if dimensions don't match
        if actual_pooled_dim != checkpoint_pooled_dim:
            print(f"   Creating projection layer: {actual_pooled_dim} -> {checkpoint_pooled_dim}")
            pooled_projection = nn.Linear(actual_pooled_dim, checkpoint_pooled_dim).to(device)
            pooled_projection.eval()
        else:
            pooled_projection = None
    
    # Find images (1.png, 2.png, ...)
    image_files = sorted([f for f in image_dir.glob("*.png") if f.stem.isdigit()],
                        key=lambda x: int(x.stem))
    
    if len(image_files) == 0:
        print(f"❌ ERROR: No numbered PNG files found in {image_dir}")
        print("   Expected files like: 1.png, 2.png, 3.png, ...")
        return
    
    print(f"\n📸 Found {len(image_files)} images: {[f.name for f in image_files]}")
    
    # Process each image
    all_predictions = []
    image_names = []
    
    for image_path in tqdm(image_files, desc="Processing images"):
        # Extract embeddings
        pooled_emb, patch_tokens = extract_embeddings(
            vision_model, processor, image_path, device, args.image_size
        )
        
        # Project pooled_emb if needed
        if pooled_projection is not None:
            pooled_emb = pooled_projection(pooled_emb)
        
        # Run prediction
        predictions = predict_image(
            phase1_model, stage2_model, concept_adapter,
            patch_tokens, pooled_emb, device, centers_xyz, idx_to_concept
        )
        
        predictions['image_name'] = image_path.name
        all_predictions.append(predictions)
        image_names.append(image_path.name)
        
        # Create individual visualization
        viz_path = output_dir / f"{image_path.stem}_prediction.png"
        visualize_image_predictions(image_path, predictions, idx_to_concept, viz_path, args.image_size)
    
    # Create map visualization
    print("\n🗺️ Creating map visualization...")
    map_path = output_dir / "predictions_map.png"
    visualize_map(all_predictions, image_names, map_path)
    
    # Save results JSON (convert numpy arrays to lists for JSON serialization)
    results_path = output_dir / "predictions.json"
    
    # Convert numpy arrays to lists for JSON serialization
    json_predictions = []
    for pred in all_predictions:
        json_pred = pred.copy()
        if 'attention_weights' in json_pred and json_pred['attention_weights'] is not None:
            # Convert numpy array to nested list
            json_pred['attention_weights'] = json_pred['attention_weights'].tolist()
        # Convert any other numpy arrays
        for key, value in json_pred.items():
            if isinstance(value, np.ndarray):
                json_pred[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                json_pred[key] = float(value)
        json_predictions.append(json_pred)
    
    with open(results_path, 'w') as f:
        json.dump({
            'predictions': json_predictions,
            'image_names': image_names,
        }, f, indent=2)
    
    # Create summary
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nGenerated files:")
    for img_name in image_names:
        print(f"  - {Path(img_name).stem}_prediction.png")
    print(f"  - predictions_map.png")
    print(f"  - predictions.json")
    
    print(f"\n📊 Summary:")
    for i, (pred, name) in enumerate(zip(all_predictions, image_names), 1):
        print(f"\n  Image {i} ({name}):")
        print(f"    Location: ({pred['pred_lat']:.4f} deg, {pred['pred_lng']:.4f} deg)")
        print(f"    Country:  {pred['pred_country']}")
        print(f"    Top Concept: {pred['top1_concept']} ({pred['top1_prob']*100:.1f}%)")
        print(f"    Top-5: {', '.join(pred['top5_concepts'][:3])}...")


if __name__ == "__main__":
    main()


