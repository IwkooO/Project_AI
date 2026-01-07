#!/usr/bin/env python3
"""
Evaluate model on GeoGuessr HuggingFace dataset.

Dataset: fren-gor/geoguessr-locations
Columns: panorama_360 (image), lat, lng, country (2-letter code)

Metrics:
- Mean/Median km error
- Country accuracy
- Attention visualization
- Concept prediction distribution

Usage:
    python scripts/evaluation/evaluate_geoguessr.py \
        --checkpoint checkpoints/stage3_joint_latefusion/stage3_joint_both_scratchgeo_dim256_pos_latefusion_2.5ct_0.01gatereg \
        --concept-data-dir data/concept_data_v2 \
        --output-dir results/geoguessr_evaluation \
        --num-samples 1000 \
        --visualize-samples 20
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import math
import io

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cbm.phase1.model import Phase1CBMTopKMil
from cbm.phase2.model import ConceptEmbeddingAdapter, Stage2CrossAttentionGeoHead
from cbm.phase2.geocells import latlng_to_xyz, assign_geocells
from cbm.phase2.metrics import xyz_to_latlng, haversine_km

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# ============================================================================
# COUNTRY PREDICTION (using pycountry and reverse_geocoder)
# ============================================================================

def predict_country_from_coords(lat: float, lng: float) -> Optional[str]:
    """
    Predict country code from coordinates using reverse_geocoder.
    Returns 2-letter ISO country code or None if failed.
    """
    try:
        import reverse_geocoder as rg
        result = rg.search([(lat, lng)], mode=1)
        if result and len(result) > 0:
            return result[0].get('cc', None)
    except ImportError:
        pass
    except Exception:
        pass
    return None


def batch_predict_countries(coords: List[Tuple[float, float]]) -> List[Optional[str]]:
    """
    Batch predict countries from coordinates.
    """
    try:
        import reverse_geocoder as rg
        results = rg.search(coords, mode=2)  # mode=2 for batch
        return [r.get('cc', None) if r else None for r in results]
    except ImportError:
        print("   Warning: reverse_geocoder not installed. Install with: pip install reverse_geocoder")
        return [None] * len(coords)
    except Exception as e:
        print(f"   Warning: reverse_geocoder failed: {e}")
        return [None] * len(coords)


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def load_streetclip_model(device: torch.device):
    """Load StreetCLIP model for feature extraction."""
    from transformers import CLIPModel, CLIPProcessor
    
    model_name = "geolocal/StreetCLIP"
    print(f"📦 Loading StreetCLIP model: {model_name}")
    
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    model = model.to(device)
    model.eval()
    
    return model, processor


def extract_features_from_image(
    image: Image.Image,
    clip_model: nn.Module,
    processor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract patch tokens and pooled embeddings from an image using CLIP.
    
    IMPORTANT: Must match the extraction used during training (precompute_streetclip_embeddings.py):
    - pooled_emb: Uses get_image_features() + L2 normalization
    - patch_tokens: Raw hidden states from vision model (before projection)
    
    Returns:
        patch_tokens: [1, num_patches, dim]
        pooled_emb: [1, dim] - L2 normalized
    """
    # Process image
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    
    with torch.no_grad():
        # Get vision model outputs
        vision_outputs = clip_model.vision_model(pixel_values, output_hidden_states=True)
        
        # Get last hidden state (includes CLS + patch tokens)
        last_hidden = vision_outputs.last_hidden_state  # [B, seq_len, dim]
        
        # Patch tokens: everything except CLS token (raw, before projection)
        patch_tokens = last_hidden[:, 1:, :]  # [B, num_patches, dim]
        
        # Pooled embedding: Use get_image_features() to match training exactly
        # This uses pooler_output + visual_projection (not raw CLS token!)
        pooled_emb = clip_model.get_image_features(pixel_values=pixel_values)
        
        # L2 normalize - CRITICAL: training did this!
        pooled_emb = F.normalize(pooled_emb, p=2, dim=-1)
    
    return patch_tokens, pooled_emb


# ============================================================================
# GEOGUESSR DATASET
# ============================================================================

class GeoGuessrDataset(Dataset):
    """Dataset wrapper for GeoGuessr HuggingFace dataset."""
    
    def __init__(
        self,
        hf_dataset,
        max_samples: int = None,
    ):
        self.dataset = hf_dataset
        dataset_len = len(hf_dataset)
        # Handle None, 0, or negative values as "use all samples"
        if max_samples is None or max_samples <= 0:
            self.max_samples = dataset_len
        else:
            self.max_samples = min(max_samples, dataset_len)
        
    def __len__(self):
        return self.max_samples
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Get panorama image
        image = item["panorama_360"]
        if not isinstance(image, Image.Image):
            # Convert bytes to PIL Image if needed
            if isinstance(image, bytes):
                image = Image.open(io.BytesIO(image)).convert("RGB")
            elif isinstance(image, dict) and "bytes" in image:
                image = Image.open(io.BytesIO(image["bytes"])).convert("RGB")
        
        # Get coordinates
        lat = float(item["lat"])
        lng = float(item["lng"])
        
        # Get country code
        country = str(item.get("country", ""))
        
        return {
            "image": image,
            "lat": lat,
            "lng": lng,
            "country": country,
            "idx": idx,
        }


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model_from_checkpoint(
    checkpoint_dir: Path,
    num_concepts: int,
    device: torch.device,
    args,
) -> Tuple:
    """Load phase1 model, stage2 model, and concept adapter from checkpoint."""
    
    joint_ckpt_path = checkpoint_dir / "best_joint.pt"
    if not joint_ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {joint_ckpt_path}")
    
    print(f"   Loading from: {joint_ckpt_path}")
    checkpoint = torch.load(joint_ckpt_path, map_location=device, weights_only=False)
    
    # Get Phase1 state dict
    phase1_state = checkpoint.get('phase1_model_state_dict', {})
    if not phase1_state:
        raise ValueError("Could not find Phase1 state dict in checkpoint")
    
    # Detect configuration from state dict
    has_pos_embed = any('pos_embed' in k for k in phase1_state.keys())
    has_per_concept_tau = any('concept_tau_logit' in k for k in phase1_state.keys())
    
    # Detect dimensions
    concept_dim = args.concept_dim
    patch_dim = args.patch_dim
    for k, v in phase1_state.items():
        if k == 'query' or k.endswith('.query'):
            concept_dim = v.shape[1]
        if 'patch_proj' in k and 'weight' in k and len(v.shape) == 2:
            patch_dim = v.shape[1]
    
    print(f"   Detected: patch_dim={patch_dim}, concept_dim={concept_dim}")
    
    # Create Phase1 model
    phase1_model = Phase1CBMTopKMil(
        num_concepts=num_concepts,
        patch_dim=patch_dim,
        concept_dim=concept_dim,
        dropout=args.phase1_dropout,
        mil_topk=args.mil_topk,
        mil_tau=args.mil_tau,
        mix_depth=args.mix_depth,
        mix_heads=args.mix_heads,
        mix_mlp_ratio=args.mix_mlp_ratio,
        proj_type=args.proj_type,
        use_pos_encoding=has_pos_embed,
        use_per_concept_tau=has_per_concept_tau,
    )
    
    phase1_model.load_state_dict(phase1_state)
    phase1_model = phase1_model.to(device)
    phase1_model.eval()
    
    # Load geocells
    if 'geocell_info' in checkpoint:
        geocells_data = checkpoint['geocell_info']
        centers_xyz = np.array(geocells_data["centers_xyz"])
        num_cells = len(centers_xyz)
    else:
        geocells_path = checkpoint_dir / "phase2" / "geocells.json"
        with open(geocells_path) as f:
            geocells_data = json.load(f)
        centers_xyz = np.array(geocells_data["centers_xyz"])
        num_cells = len(centers_xyz)
    
    print(f"   Loaded {num_cells} geocells")
    
    # Load Stage2
    stage2_state = checkpoint.get('stage2_model_state_dict', {})
    if not stage2_state:
        raise ValueError("Could not find Stage2 state dict in checkpoint")
    
    # Detect mode
    has_image_adapter = any(k.startswith('image_adapter') for k in stage2_state.keys())
    has_concept_adapter = any(k.startswith('concept_adapter') for k in stage2_state.keys())
    if has_image_adapter and has_concept_adapter:
        mode = "both"
    elif has_concept_adapter:
        mode = "concept_only"
    else:
        mode = "image_only"
    
    print(f"   Detected mode: {mode}")
    
    # Detect pooled_dim
    pooled_dim = 768
    for k, v in stage2_state.items():
        if k == 'pooled_proj.weight' or k == 'pooled_proj.0.weight':
            pooled_dim = v.shape[1]
            break
    
    # Create Stage2 model
    stage2_model = Stage2CrossAttentionGeoHead(
        concept_dim=concept_dim,
        patch_dim=patch_dim,
        num_cells=num_cells,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        mode=mode,
        pooled_dim=pooled_dim,
    )
    
    # Map state dict
    mapped_state = {}
    for k, v in stage2_state.items():
        if 'concept_refinement' in k:
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
    
    stage2_model.load_state_dict(mapped_state, strict=False)
    stage2_model = stage2_model.to(device)
    stage2_model.eval()
    
    # Load concept adapter
    concept_temp = getattr(args, 'concept_temperature', 2.5)
    if 'concept_adapter_state_dict' in checkpoint:
        adapter_state = checkpoint['concept_adapter_state_dict']
        concept_vectors = adapter_state['concept_vectors']
        concept_adapter = ConceptEmbeddingAdapter(concept_vectors, temperature=concept_temp)
        concept_adapter.load_state_dict(adapter_state)
    else:
        concept_vectors = torch.randn(num_concepts, concept_dim)
        concept_adapter = ConceptEmbeddingAdapter(concept_vectors, temperature=concept_temp)
    
    concept_adapter = concept_adapter.to(device)
    concept_adapter.eval()
    
    return phase1_model, stage2_model, concept_adapter, centers_xyz, mode, pooled_dim


# ============================================================================
# EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate_single_sample(
    image: Image.Image,
    clip_model: nn.Module,
    processor,
    phase1_model: Phase1CBMTopKMil,
    stage2_model: Stage2CrossAttentionGeoHead,
    concept_adapter: ConceptEmbeddingAdapter,
    centers_xyz: np.ndarray,
    device: torch.device,
) -> Dict:
    """Evaluate a single image."""
    
    # Extract features
    patch_tokens, pooled_emb = extract_features_from_image(
        image, clip_model, processor, device
    )
    
    # Phase 1: Concept prediction
    c_logits, c_hidden, attn_weights, _ = phase1_model(patch_tokens)
    c_probs = torch.softmax(c_logits, dim=1)
    c_pred = c_logits.argmax(dim=1).item()
    
    # Phase 2: Geolocation
    concept_emb = concept_adapter(c_logits)
    forward_output = stage2_model(concept_emb, patch_tokens, pooled_emb)
    
    if stage2_model.mode == "both":
        cell_logits, offset_pred, gate_info, _, _ = forward_output
        if isinstance(gate_info, dict) and 'gate' in gate_info:
            gate = gate_info['gate']
            gate_val = gate[0, 0].item() if gate is not None else 0.0
        else:
            gate_val = 0.0
    else:
        cell_logits, offset_pred, gate = forward_output
        gate_val = gate[0, 0].item() if gate is not None else 0.0
    
    # Convert to lat/lng
    pred_cell = cell_logits.argmax(dim=1).item()
    pred_cell_center = centers_xyz[pred_cell]
    pred_xyz = pred_cell_center + offset_pred[0].cpu().numpy()
    
    # Use proper conversion function (expects [batch_size, 3])
    pred_xyz_2d = pred_xyz.reshape(1, -1)  # [1, 3]
    pred_lat_arr, pred_lng_arr = xyz_to_latlng(pred_xyz_2d)
    pred_lat = float(pred_lat_arr[0])
    pred_lng = float(pred_lng_arr[0])
    
    return {
        "pred_lat": float(pred_lat),
        "pred_lng": float(pred_lng),
        "pred_cell": int(pred_cell),
        "concept_pred": int(c_pred),
        "concept_probs": c_probs[0].cpu().numpy(),
        "attention_weights": attn_weights[0].cpu().numpy(),
        "gate_value": float(gate_val),
    }


@torch.no_grad()
def extract_features_batch(
    images: List[Image.Image],
    clip_model: nn.Module,
    processor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract patch tokens and pooled embeddings from a batch of images.
    
    Returns:
        patch_tokens: [B, num_patches, dim]
        pooled_emb: [B, dim] - L2 normalized
    """
    # Process all images at once
    inputs = processor(images=images, return_tensors="pt", padding=True)
    pixel_values = inputs["pixel_values"].to(device)
    
    # Get vision model outputs
    vision_outputs = clip_model.vision_model(pixel_values, output_hidden_states=True)
    
    # Get last hidden state (includes CLS + patch tokens)
    last_hidden = vision_outputs.last_hidden_state  # [B, seq_len, dim]
    
    # Patch tokens: everything except CLS token (raw, before projection)
    patch_tokens = last_hidden[:, 1:, :]  # [B, num_patches, dim]
    
    # Pooled embedding: Use get_image_features() to match training exactly
    pooled_emb = clip_model.get_image_features(pixel_values=pixel_values)
    
    # L2 normalize - CRITICAL: training did this!
    pooled_emb = F.normalize(pooled_emb, p=2, dim=-1)
    
    return patch_tokens, pooled_emb


@torch.no_grad()
def evaluate_batch(
    images: List[Image.Image],
    clip_model: nn.Module,
    processor,
    phase1_model: Phase1CBMTopKMil,
    stage2_model: Stage2CrossAttentionGeoHead,
    concept_adapter: ConceptEmbeddingAdapter,
    centers_xyz: np.ndarray,
    device: torch.device,
) -> List[Dict]:
    """Evaluate a batch of images."""
    
    batch_size = len(images)
    
    # Extract features for all images at once
    patch_tokens, pooled_emb = extract_features_batch(
        images, clip_model, processor, device
    )
    
    # Phase 1: Concept prediction (batched)
    c_logits, c_hidden, attn_weights, _ = phase1_model(patch_tokens)
    c_probs = torch.softmax(c_logits, dim=1)
    c_preds = c_logits.argmax(dim=1)  # [B]
    
    # Phase 2: Geolocation (batched)
    concept_emb = concept_adapter(c_logits)
    forward_output = stage2_model(concept_emb, patch_tokens, pooled_emb)
    
    if stage2_model.mode == "both":
        cell_logits, offset_pred, gate_info, _, _ = forward_output
        if isinstance(gate_info, dict) and 'gate' in gate_info:
            gate = gate_info['gate']
            gate_vals = gate[:, 0].cpu().numpy() if gate is not None else np.zeros(batch_size)
        else:
            gate_vals = np.zeros(batch_size)
    else:
        cell_logits, offset_pred, gate = forward_output
        gate_vals = gate[:, 0].cpu().numpy() if gate is not None else np.zeros(batch_size)
    
    # Convert to lat/lng (batched)
    pred_cells = cell_logits.argmax(dim=1).cpu().numpy()  # [B]
    pred_cell_centers = centers_xyz[pred_cells]  # [B, 3]
    pred_xyz = pred_cell_centers + offset_pred.cpu().numpy()  # [B, 3]
    
    # Use proper conversion function
    pred_lats, pred_lngs = xyz_to_latlng(pred_xyz)
    
    # Build results list
    results = []
    for i in range(batch_size):
        results.append({
            "pred_lat": float(pred_lats[i]),
            "pred_lng": float(pred_lngs[i]),
            "pred_cell": int(pred_cells[i]),
            "concept_pred": int(c_preds[i].item()),
            "concept_probs": c_probs[i].cpu().numpy(),
            "attention_weights": attn_weights[i].cpu().numpy(),
            "gate_value": float(gate_vals[i]),
        })
    
    return results


def geoguessr_collate_fn(batch):
    """Collate function for batched evaluation."""
    images = [item["image"] for item in batch]
    lats = [item["lat"] for item in batch]
    lngs = [item["lng"] for item in batch]
    countries = [item.get("country") for item in batch]
    return {"images": images, "lats": lats, "lngs": lngs, "countries": countries}


def evaluate_dataset(
    geoguessr_ds: GeoGuessrDataset,
    clip_model: nn.Module,
    processor,
    phase1_model: Phase1CBMTopKMil,
    stage2_model: Stage2CrossAttentionGeoHead,
    concept_adapter: ConceptEmbeddingAdapter,
    centers_xyz: np.ndarray,
    device: torch.device,
    idx_to_concept: Dict[int, str],
    batch_size: int = 16,
) -> Dict:
    """Evaluate the entire dataset with batched inference."""
    
    all_results = []
    all_errors = []
    all_concept_preds = []
    
    # Create DataLoader for batched evaluation
    dataloader = DataLoader(
        geoguessr_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=geoguessr_collate_fn,
        pin_memory=True,
    )
    
    sample_idx = 0
    for batch in tqdm(dataloader, desc=f"Evaluating (batch_size={batch_size})"):
        try:
            # Evaluate batch
            batch_results = evaluate_batch(
                batch["images"],
                clip_model,
                processor,
                phase1_model,
                stage2_model,
                concept_adapter,
                centers_xyz,
                device,
            )
            
            # Compute errors for batch
            pred_lats = np.array([r["pred_lat"] for r in batch_results])
            pred_lngs = np.array([r["pred_lng"] for r in batch_results])
            true_lats = np.array(batch["lats"])
            true_lngs = np.array(batch["lngs"])
            
            errors_km = haversine_km(pred_lats, pred_lngs, true_lats, true_lngs)
            
            # Add ground truth and errors to results
            for i, result in enumerate(batch_results):
                result["true_lat"] = batch["lats"][i]
                result["true_lng"] = batch["lngs"][i]
                result["true_country"] = batch["countries"][i]
                result["error_km"] = float(errors_km[i])
                result["idx"] = sample_idx + i
                
                all_results.append(result)
                all_errors.append(errors_km[i])
                all_concept_preds.append(result["concept_pred"])
            
            sample_idx += len(batch_results)
            
        except Exception as e:
            print(f"  Warning: Failed to process batch starting at {sample_idx}: {e}")
            # Fall back to single-sample processing for this batch
            for i, img in enumerate(batch["images"]):
                try:
                    result = evaluate_single_sample(
                        img, clip_model, processor, phase1_model,
                        stage2_model, concept_adapter, centers_xyz, device,
                    )
                    error_km = haversine_km(
                        np.array([result["pred_lat"]]),
                        np.array([result["pred_lng"]]),
                        np.array([batch["lats"][i]]),
                        np.array([batch["lngs"][i]]),
                    )[0]
                    result["true_lat"] = batch["lats"][i]
                    result["true_lng"] = batch["lngs"][i]
                    result["true_country"] = batch["countries"][i]
                    result["error_km"] = float(error_km)
                    result["idx"] = sample_idx + i
                    all_results.append(result)
                    all_errors.append(error_km)
                    all_concept_preds.append(result["concept_pred"])
                except Exception as inner_e:
                    print(f"    Warning: Failed single sample {sample_idx + i}: {inner_e}")
            sample_idx += len(batch["images"])
    
    errors = np.array(all_errors)
    
    # Batch predict countries from predicted coordinates
    print("\n🌍 Computing country predictions...")
    pred_coords = [(r["pred_lat"], r["pred_lng"]) for r in all_results]
    pred_countries = batch_predict_countries(pred_coords)
    
    # Add predicted countries to results and compute accuracy
    country_correct = 0
    country_total = 0
    for i, (result, pred_country) in enumerate(zip(all_results, pred_countries)):
        result["pred_country"] = pred_country
        true_country = result.get("true_country", "")
        
        if pred_country and true_country:
            country_total += 1
            # Normalize for comparison (uppercase)
            if pred_country.upper() == true_country.upper():
                country_correct += 1
                result["country_correct"] = True
            else:
                result["country_correct"] = False
        else:
            result["country_correct"] = None
    
    country_accuracy = country_correct / country_total if country_total > 0 else 0.0
    
    # Compute continent-level accuracy (approximate)
    # Map countries to continents for relaxed accuracy
    continent_correct = 0
    continent_total = 0
    
    CONTINENT_MAP = {
        # Europe
        'AT': 'EU', 'BE': 'EU', 'BG': 'EU', 'HR': 'EU', 'CY': 'EU', 'CZ': 'EU',
        'DK': 'EU', 'EE': 'EU', 'FI': 'EU', 'FR': 'EU', 'DE': 'EU', 'GR': 'EU',
        'HU': 'EU', 'IE': 'EU', 'IT': 'EU', 'LV': 'EU', 'LT': 'EU', 'LU': 'EU',
        'MT': 'EU', 'NL': 'EU', 'PL': 'EU', 'PT': 'EU', 'RO': 'EU', 'SK': 'EU',
        'SI': 'EU', 'ES': 'EU', 'SE': 'EU', 'GB': 'EU', 'UK': 'EU', 'NO': 'EU',
        'CH': 'EU', 'IS': 'EU', 'AL': 'EU', 'AD': 'EU', 'BA': 'EU', 'BY': 'EU',
        'MD': 'EU', 'MC': 'EU', 'ME': 'EU', 'MK': 'EU', 'RS': 'EU', 'SM': 'EU',
        'UA': 'EU', 'VA': 'EU', 'XK': 'EU', 'JE': 'EU', 'GG': 'EU', 'IM': 'EU',
        # North America
        'US': 'NA', 'CA': 'NA', 'MX': 'NA', 'GT': 'NA', 'BZ': 'NA', 'SV': 'NA',
        'HN': 'NA', 'NI': 'NA', 'CR': 'NA', 'PA': 'NA', 'CU': 'NA', 'JM': 'NA',
        'HT': 'NA', 'DO': 'NA', 'PR': 'NA', 'BS': 'NA', 'BB': 'NA', 'TT': 'NA',
        # South America
        'BR': 'SA', 'AR': 'SA', 'CO': 'SA', 'PE': 'SA', 'VE': 'SA', 'CL': 'SA',
        'EC': 'SA', 'BO': 'SA', 'PY': 'SA', 'UY': 'SA', 'GY': 'SA', 'SR': 'SA',
        # Asia
        'CN': 'AS', 'JP': 'AS', 'KR': 'AS', 'IN': 'AS', 'ID': 'AS', 'TH': 'AS',
        'VN': 'AS', 'PH': 'AS', 'MY': 'AS', 'SG': 'AS', 'MM': 'AS', 'KH': 'AS',
        'LA': 'AS', 'BD': 'AS', 'PK': 'AS', 'LK': 'AS', 'NP': 'AS', 'BT': 'AS',
        'TW': 'AS', 'HK': 'AS', 'MO': 'AS', 'MN': 'AS', 'KP': 'AS', 'RU': 'AS',
        'KZ': 'AS', 'UZ': 'AS', 'TM': 'AS', 'KG': 'AS', 'TJ': 'AS', 'AF': 'AS',
        'IR': 'AS', 'IQ': 'AS', 'SA': 'AS', 'AE': 'AS', 'QA': 'AS', 'KW': 'AS',
        'BH': 'AS', 'OM': 'AS', 'YE': 'AS', 'JO': 'AS', 'LB': 'AS', 'SY': 'AS',
        'IL': 'AS', 'PS': 'AS', 'TR': 'AS', 'AZ': 'AS', 'AM': 'AS', 'GE': 'AS',
        # Africa
        'EG': 'AF', 'ZA': 'AF', 'NG': 'AF', 'KE': 'AF', 'ET': 'AF', 'TZ': 'AF',
        'UG': 'AF', 'GH': 'AF', 'MA': 'AF', 'DZ': 'AF', 'TN': 'AF', 'LY': 'AF',
        'SD': 'AF', 'AO': 'AF', 'MZ': 'AF', 'MG': 'AF', 'CM': 'AF', 'CI': 'AF',
        'NE': 'AF', 'BF': 'AF', 'ML': 'AF', 'MW': 'AF', 'ZM': 'AF', 'ZW': 'AF',
        'SN': 'AF', 'TD': 'AF', 'SO': 'AF', 'RW': 'AF', 'BI': 'AF', 'BJ': 'AF',
        'TG': 'AF', 'SL': 'AF', 'LR': 'AF', 'MR': 'AF', 'GM': 'AF', 'GW': 'AF',
        'NA': 'AF', 'BW': 'AF', 'LS': 'AF', 'SZ': 'AF', 'DJ': 'AF', 'ER': 'AF',
        'CF': 'AF', 'CG': 'AF', 'CD': 'AF', 'GA': 'AF', 'GQ': 'AF', 'ST': 'AF',
        'CV': 'AF', 'MU': 'AF', 'SC': 'AF', 'KM': 'AF', 'RE': 'AF', 'YT': 'AF',
        # Oceania
        'AU': 'OC', 'NZ': 'OC', 'PG': 'OC', 'FJ': 'OC', 'SB': 'OC', 'VU': 'OC',
        'NC': 'OC', 'PF': 'OC', 'WS': 'OC', 'TO': 'OC', 'KI': 'OC', 'FM': 'OC',
        'MH': 'OC', 'PW': 'OC', 'NR': 'OC', 'TV': 'OC', 'CK': 'OC', 'NU': 'OC',
        'GU': 'OC', 'AS': 'OC', 'MP': 'OC',
    }
    
    for result in all_results:
        pred_country = result.get("pred_country")
        true_country = result.get("true_country", "")
        
        if pred_country and true_country:
            pred_continent = CONTINENT_MAP.get(pred_country.upper())
            true_continent = CONTINENT_MAP.get(true_country.upper())
            
            if pred_continent and true_continent:
                continent_total += 1
                if pred_continent == true_continent:
                    continent_correct += 1
                    result["continent_correct"] = True
                else:
                    result["continent_correct"] = False
            else:
                result["continent_correct"] = None
        else:
            result["continent_correct"] = None
    
    continent_accuracy = continent_correct / continent_total if continent_total > 0 else 0.0
    
    return {
        "results": all_results,
        "errors": errors,
        "concept_preds": np.array(all_concept_preds),
        "metrics": {
            "mean_error_km": float(np.mean(errors)),
            "median_error_km": float(np.median(errors)),
            "std_error_km": float(np.std(errors)),
            "p25_error_km": float(np.percentile(errors, 25)),
            "p75_error_km": float(np.percentile(errors, 75)),
            "p90_error_km": float(np.percentile(errors, 90)),
            "p95_error_km": float(np.percentile(errors, 95)),
            "acc_1km": float(np.mean(errors <= 1.0)),
            "acc_10km": float(np.mean(errors <= 10.0)),
            "acc_25km": float(np.mean(errors <= 25.0)),
            "acc_100km": float(np.mean(errors <= 100.0)),
            "acc_200km": float(np.mean(errors <= 200.0)),
            "acc_750km": float(np.mean(errors <= 750.0)),
            "acc_1000km": float(np.mean(errors <= 1000.0)),
            "acc_2500km": float(np.mean(errors <= 2500.0)),
            "country_accuracy": float(country_accuracy),
            "country_correct": int(country_correct),
            "country_total": int(country_total),
            "continent_accuracy": float(continent_accuracy),
            "continent_correct": int(continent_correct),
            "continent_total": int(continent_total),
            "num_samples": len(errors),
        }
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_metrics_summary(
    eval_results: Dict,
    output_path: Path,
):
    """Plot summary metrics."""
    metrics = eval_results["metrics"]
    errors = eval_results["errors"]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Error histogram
    ax = axes[0, 0]
    ax.hist(errors, bins=50, alpha=0.7, color='#3498db', edgecolor='white')
    ax.axvline(metrics["mean_error_km"], color='red', linestyle='--', linewidth=2, label=f'Mean: {metrics["mean_error_km"]:.1f} km')
    ax.axvline(metrics["median_error_km"], color='green', linestyle='--', linewidth=2, label=f'Median: {metrics["median_error_km"]:.1f} km')
    ax.set_xlabel('Error (km)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Error Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # 2. Error CDF
    ax = axes[0, 1]
    sorted_errors = np.sort(errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    ax.plot(sorted_errors, cdf, linewidth=2, color='#2ecc71')
    
    # Add threshold markers
    thresholds = [1, 10, 25, 100, 200, 750, 1000, 2500]
    for thr in thresholds:
        acc = np.mean(errors <= thr)
        ax.axvline(thr, alpha=0.3, linestyle='--', color='gray')
        ax.scatter([thr], [acc], s=50, zorder=5)
        ax.annotate(f'{acc*100:.1f}%', (thr, acc), textcoords="offset points", xytext=(5,5), fontsize=8)
    
    ax.set_xlabel('Error (km)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cumulative Fraction', fontsize=11, fontweight='bold')
    ax.set_title('Cumulative Error Distribution', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # 3. Threshold accuracies bar chart
    ax = axes[1, 0]
    thresholds = ['1km', '10km', '25km', '100km', '200km', '750km', '1000km', '2500km']
    accuracies = [
        metrics['acc_1km'], metrics['acc_10km'], metrics['acc_25km'],
        metrics['acc_100km'], metrics['acc_200km'], metrics['acc_750km'],
        metrics['acc_1000km'], metrics['acc_2500km']
    ]
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(thresholds)))
    bars = ax.bar(thresholds, accuracies, color=colors, edgecolor='black', linewidth=0.5)
    
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Distance Threshold', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('Accuracy at Distance Thresholds', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    # Build summary text with country accuracy if available
    country_text = ""
    if 'country_accuracy' in metrics and metrics.get('country_total', 0) > 0:
        country_text = f"""
    Country/Continent Accuracy:
      • Country Accuracy:   {metrics['country_accuracy']*100:.2f}% ({metrics['country_correct']}/{metrics['country_total']})
      • Continent Accuracy: {metrics['continent_accuracy']*100:.2f}% ({metrics['continent_correct']}/{metrics['continent_total']})
    """
    
    summary_text = f"""
    GEOGUESSR EVALUATION SUMMARY
    {'─'*45}
    
    Total samples evaluated: {metrics['num_samples']:,}
    
    Error Statistics:
      • Mean Error:   {metrics['mean_error_km']:,.1f} km
      • Median Error: {metrics['median_error_km']:,.1f} km
      • Std Dev:      {metrics['std_error_km']:,.1f} km
      • P25:          {metrics['p25_error_km']:,.1f} km
      • P75:          {metrics['p75_error_km']:,.1f} km
      • P90:          {metrics['p90_error_km']:,.1f} km
      • P95:          {metrics['p95_error_km']:,.1f} km
    
    GeoGuessr Score Thresholds:
      • ≤25 km (5000 pts):    {metrics['acc_25km']*100:.2f}%
      • ≤200 km (4000 pts):   {metrics['acc_200km']*100:.2f}%
      • ≤750 km (3000 pts):   {metrics['acc_750km']*100:.2f}%
      • ≤2500 km (1000 pts):  {metrics['acc_2500km']*100:.2f}%
    {country_text}"""
    
    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', alpha=0.95, edgecolor='gray'))
    
    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_country_analysis(
    eval_results: Dict,
    output_path: Path,
):
    """Plot country accuracy analysis."""
    results = eval_results["results"]
    metrics = eval_results["metrics"]
    
    # Skip if no country data
    if metrics.get('country_total', 0) == 0:
        print("   Skipping country analysis (no country data)")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Country accuracy pie chart
    ax = axes[0, 0]
    country_correct = metrics['country_correct']
    country_incorrect = metrics['country_total'] - country_correct
    ax.pie([country_correct, country_incorrect], 
           labels=[f'Correct ({country_correct})', f'Incorrect ({country_incorrect})'],
           colors=['#27ae60', '#e74c3c'], autopct='%1.1f%%', startangle=90,
           explode=[0.02, 0.02])
    ax.set_title(f'Country Prediction Accuracy\n({metrics["country_accuracy"]*100:.1f}%)', 
                fontsize=13, fontweight='bold')
    
    # 2. Continent accuracy pie chart
    ax = axes[0, 1]
    if metrics.get('continent_total', 0) > 0:
        continent_correct = metrics['continent_correct']
        continent_incorrect = metrics['continent_total'] - continent_correct
        ax.pie([continent_correct, continent_incorrect], 
               labels=[f'Correct ({continent_correct})', f'Incorrect ({continent_incorrect})'],
               colors=['#3498db', '#e67e22'], autopct='%1.1f%%', startangle=90,
               explode=[0.02, 0.02])
        ax.set_title(f'Continent Prediction Accuracy\n({metrics["continent_accuracy"]*100:.1f}%)', 
                    fontsize=13, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No continent data', ha='center', va='center',
               transform=ax.transAxes, fontsize=11)
        ax.set_title('Continent Prediction Accuracy', fontsize=13, fontweight='bold')
    
    # 3. Error distribution by country correctness
    ax = axes[1, 0]
    correct_errors = [r['error_km'] for r in results if r.get('country_correct') == True]
    incorrect_errors = [r['error_km'] for r in results if r.get('country_correct') == False]
    
    if correct_errors and incorrect_errors:
        bp = ax.boxplot([correct_errors, incorrect_errors], 
                  tick_labels=['Country Correct', 'Country Incorrect'],
                  patch_artist=True)
        # Color the boxes using the returned boxplot dict
        colors = ['#27ae60', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_yscale('log')
        ax.set_ylabel('Error (km, log scale)', fontsize=11, fontweight='bold')
        ax.set_title('Error Distribution by Country Prediction', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add median labels
        ax.text(1, np.median(correct_errors), f'median: {np.median(correct_errors):.0f}km',
               ha='left', va='bottom', fontsize=9)
        ax.text(2, np.median(incorrect_errors), f'median: {np.median(incorrect_errors):.0f}km',
               ha='left', va='bottom', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
               transform=ax.transAxes, fontsize=11)
        ax.set_title('Error Distribution by Country Prediction', fontsize=13, fontweight='bold')
    
    # 4. Top confused country pairs
    ax = axes[1, 1]
    confusion_pairs = {}
    for r in results:
        true_c = r.get('true_country', '')
        pred_c = r.get('pred_country', '')
        if true_c and pred_c and true_c.upper() != pred_c.upper():
            pair = f"{true_c.upper()}→{pred_c.upper()}"
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
    
    if confusion_pairs:
        sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:15]
        pairs, counts = zip(*sorted_pairs)
        
        y_pos = np.arange(len(pairs))
        ax.barh(y_pos, counts, color='#e74c3c', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(pairs, fontsize=9)
        ax.set_xlabel('Count', fontsize=11, fontweight='bold')
        ax.set_title('Top Country Confusions', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    else:
        ax.text(0.5, 0.5, 'No confusion data', ha='center', va='center',
               transform=ax.transAxes, fontsize=11)
        ax.set_title('Top Country Confusions', fontsize=13, fontweight='bold')
    
    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_concept_distribution(
    eval_results: Dict,
    idx_to_concept: Dict[int, str],
    output_path: Path,
):
    """Plot concept prediction distribution."""
    concept_preds = eval_results["concept_preds"]
    
    # Count concepts
    unique_concepts, counts = np.unique(concept_preds, return_counts=True)
    
    # Get top 20 concepts
    top_indices = np.argsort(counts)[-20:]
    top_concept_names = []
    for i in top_indices:
        name = idx_to_concept.get(int(unique_concepts[i]), f'C{unique_concepts[i]}')
        if len(name) > 25:
            name = name[:22] + '...'
        top_concept_names.append(name)
    top_counts = counts[top_indices]
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # 1. Horizontal bar chart
    ax = axes[0]
    y_pos = np.arange(len(top_concept_names))
    ax.barh(y_pos, top_counts, color='#3498db', alpha=0.8, edgecolor='white')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_concept_names, fontsize=9)
    ax.set_xlabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Top 20 Predicted Concepts', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add count labels
    for i, (count, y) in enumerate(zip(top_counts, y_pos)):
        ax.text(count + max(top_counts)*0.01, y, f'{count:,}', va='center', fontsize=9)
    
    # 2. Error by concept
    ax = axes[1]
    results = eval_results["results"]
    
    # Group errors by concept
    concept_errors = {}
    for r in results:
        c = r["concept_pred"]
        if c not in concept_errors:
            concept_errors[c] = []
        concept_errors[c].append(r["error_km"])
    
    # Get concepts with most samples
    concept_counts = [(c, len(e)) for c, e in concept_errors.items()]
    concept_counts.sort(key=lambda x: x[1], reverse=True)
    top_concepts = [c for c, _ in concept_counts[:15]]
    
    error_data = []
    error_labels = []
    for c in top_concepts:
        error_data.append(concept_errors[c])
        name = idx_to_concept.get(int(c), f'C{c}')
        if len(name) > 15:
            name = name[:12] + '...'
        error_labels.append(f'{name}\n(n={len(concept_errors[c])})')
    
    bp = ax.boxplot(error_data, tick_labels=error_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#e74c3c')
        patch.set_alpha(0.7)
    
    ax.set_yscale('log')
    ax.set_ylabel('Error (km, log scale)', fontsize=11, fontweight='bold')
    ax.set_title('Error by Top Predicted Concepts', fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', labelsize=8, rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def visualize_sample_attention(
    sample_result: Dict,
    image: Image.Image,
    idx_to_concept: Dict[int, str],
    output_path: Path,
):
    """Visualize attention for a single sample."""
    
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Original image
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(image)
    ax.set_title('Original Panorama', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # 2. Attention overlay for top concept
    ax = fig.add_subplot(gs[0, 1])
    attn_weights = sample_result["attention_weights"]  # [K, P]
    concept_pred = sample_result["concept_pred"]
    
    # Get attention for predicted concept
    top_attn = attn_weights[concept_pred]  # [P]
    num_patches = len(top_attn)
    side = int(np.sqrt(num_patches))
    
    if side * side == num_patches:
        attn_map = top_attn.reshape(side, side)
    else:
        # Try to reshape to closest square
        side = int(np.ceil(np.sqrt(num_patches)))
        padded = np.zeros(side * side)
        padded[:num_patches] = top_attn
        attn_map = padded.reshape(side, side)
    
    # Upsample to image size
    attn_tensor = torch.tensor(attn_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    img_arr = np.array(image)
    attn_upsampled = F.interpolate(
        attn_tensor, 
        size=(img_arr.shape[0], img_arr.shape[1]), 
        mode='bilinear', 
        align_corners=False
    ).squeeze().numpy()
    
    ax.imshow(img_arr)
    ax.imshow(attn_upsampled, cmap='hot', alpha=0.5)
    
    concept_name = idx_to_concept.get(int(concept_pred), f'Concept {concept_pred}')
    if len(concept_name) > 30:
        concept_name = concept_name[:27] + '...'
    ax.set_title(f'Attention: {concept_name}', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # 3. Top 5 concept predictions
    ax = fig.add_subplot(gs[0, 2])
    concept_probs = sample_result["concept_probs"]
    top5_idx = np.argsort(concept_probs)[-5:][::-1]
    top5_probs = concept_probs[top5_idx]
    top5_names = []
    for i in top5_idx:
        name = idx_to_concept.get(int(i), f'C{i}')
        if len(name) > 25:
            name = name[:22] + '...'
        top5_names.append(name)
    
    colors = ['#27ae60' if i == concept_pred else '#3498db' for i in top5_idx]
    y_pos = np.arange(len(top5_names))
    ax.barh(y_pos, top5_probs, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top5_names, fontsize=9)
    ax.set_xlabel('Probability', fontsize=10, fontweight='bold')
    ax.set_title('Top 5 Concept Predictions', fontsize=12, fontweight='bold')
    ax.set_xlim(0, max(top5_probs) * 1.15)
    ax.grid(True, alpha=0.3, axis='x')
    
    # 4. World map with prediction
    ax = fig.add_subplot(gs[1, :])
    
    # Try to use cartopy for better maps
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        
        ax = fig.add_subplot(gs[1, :], projection=ccrs.Robinson())
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle='--')
        ax.add_feature(cfeature.LAND, alpha=0.3, facecolor='#e8e8e8')
        ax.add_feature(cfeature.OCEAN, alpha=0.3, facecolor='#d4e6f1')
        ax.gridlines(draw_labels=False, alpha=0.4, linewidth=0.5)
        
        # Plot true location
        ax.scatter(
            sample_result["true_lng"], sample_result["true_lat"],
            c='green', s=300, marker='*', label='True Location',
            transform=ccrs.PlateCarree(), zorder=5, edgecolors='black', linewidths=2
        )
        
        # Plot predicted location
        ax.scatter(
            sample_result["pred_lng"], sample_result["pred_lat"],
            c='red', s=200, marker='o', label='Predicted Location',
            transform=ccrs.PlateCarree(), zorder=4, edgecolors='black', linewidths=1.5
        )
        
        # Draw line between
        ax.plot(
            [sample_result["true_lng"], sample_result["pred_lng"]],
            [sample_result["true_lat"], sample_result["pred_lat"]],
            'r--', linewidth=2, alpha=0.7, transform=ccrs.PlateCarree()
        )
        
    except ImportError:
        # Fallback without cartopy
        ax.scatter(sample_result["true_lng"], sample_result["true_lat"],
                  c='green', s=300, marker='*', label='True Location',
                  zorder=5, edgecolors='black', linewidths=2)
        ax.scatter(sample_result["pred_lng"], sample_result["pred_lat"],
                  c='red', s=200, marker='o', label='Predicted Location',
                  zorder=4, edgecolors='black', linewidths=1.5)
        ax.plot([sample_result["true_lng"], sample_result["pred_lng"]],
               [sample_result["true_lat"], sample_result["pred_lat"]],
               'r--', linewidth=2, alpha=0.7)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.grid(True, alpha=0.3)
    
    error_km = sample_result["error_km"]
    ax.set_title(f'Location Prediction (Error: {error_km:.1f} km)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    
    # Add info text
    true_country = sample_result.get('true_country', 'N/A')
    pred_country = sample_result.get('pred_country', 'N/A')
    country_match = "✓" if sample_result.get('country_correct', False) else "✗"
    
    info_text = (f"True: ({sample_result['true_lat']:.4f}, {sample_result['true_lng']:.4f})\n"
                f"Pred: ({sample_result['pred_lat']:.4f}, {sample_result['pred_lng']:.4f})\n"
                f"Error: {error_km:.1f} km\n"
                f"Country: {true_country} → {pred_country} {country_match}")
    
    fig.text(0.02, 0.02, info_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate model on GeoGuessr dataset")
    
    # Required
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Checkpoint directory")
    parser.add_argument("--concept-data-dir", type=str, required=True,
                       help="Concept data directory (with concept_vocab.json)")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for results")
    
    # Dataset options
    parser.add_argument("--num-samples", type=int, default=1000,
                       help="Number of samples to evaluate (default: 1000, -1 for all)")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for evaluation (default: 16)")
    parser.add_argument("--visualize-samples", type=int, default=20,
                       help="Number of samples to visualize (default: 20)")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split to use (default: train)")
    
    # Model config (should match training)
    parser.add_argument("--concept-dim", type=int, default=256)
    parser.add_argument("--patch-dim", type=int, default=1024)
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
    parser.add_argument("--proj-type", type=str, default="simple")
    parser.add_argument("--concept-temperature", type=float, default=2.5)
    
    # Other
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("GEOGUESSR DATASET EVALUATION")
    print("="*80)
    
    # Load concept vocabulary
    concept_vocab_path = Path(args.concept_data_dir) / "concept_vocab.json"
    with open(concept_vocab_path) as f:
        concept_vocab = json.load(f)
    idx_to_concept = {int(k): v for k, v in concept_vocab["idx_to_concept"].items()}
    num_concepts = concept_vocab["num_concepts"]
    print(f"\n📂 Loaded {num_concepts} concepts")
    
    # Load StreetCLIP
    clip_model, processor = load_streetclip_model(device)
    
    # Load model
    print(f"\n📦 Loading model from: {args.checkpoint}")
    phase1_model, stage2_model, concept_adapter, centers_xyz, mode, pooled_dim = load_model_from_checkpoint(
        Path(args.checkpoint),
        num_concepts,
        device,
        args,
    )
    
    # Load GeoGuessr dataset
    print(f"\n📊 Loading GeoGuessr dataset...")
    from datasets import load_dataset
    import time
    import os
    
    # Set up HuggingFace cache
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path(os.environ["HF_HOME"]) / "datasets"))
    
    # Check if dataset is already cached
    cache_dir = Path(os.environ["HF_DATASETS_CACHE"])
    dataset_cache = cache_dir / "fren-gor___geoguessr-locations"
    
    if dataset_cache.exists():
        print(f"   ✅ Found cached dataset at: {dataset_cache}")
        print("   Loading from cache (no API calls needed)...")
    else:
        print(f"   ⚠️  Dataset not found in cache. Will download from HuggingFace.")
        print(f"   💡 Tip: Run 'python scripts/data_processing/download_geoguessr_dataset.py' first")
        print(f"          to download once and avoid rate limits.")
    
    # Try to use hf_transfer for faster downloads (if available)
    try:
        import hf_transfer
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        print("   Using hf_transfer for faster downloads")
    except ImportError:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    
    # Load dataset (will use cache if available)
    max_retries = 3  # Fewer retries since we check cache first
    retry_delay = 60
    
    for attempt in range(max_retries):
        try:
            hf_dataset = load_dataset(
                "fren-gor/geoguessr-locations", 
                split=args.split,
                trust_remote_code=True,
                streaming=False,  # Load full dataset (from cache if available)
            )
            print(f"   ✅ Loaded {len(hf_dataset):,} samples from '{args.split}' split")
            break
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate limit" in error_str.lower() or "Too Many Requests" in error_str:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"   ⚠️  Rate limited. Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"   ❌ Error loading dataset after {max_retries} attempts: {e}")
                    print("\n   💡 Solutions:")
                    print("     1. Download dataset first: python scripts/data_processing/download_geoguessr_dataset.py")
                    print("     2. Wait 5 minutes and try again (rate limit resets)")
                    print("     3. Upgrade to HuggingFace PRO: https://hf.co/pricing")
                    return
            else:
                print(f"   ❌ Error loading dataset: {e}")
                print("   Make sure you're logged in with `huggingface-cli login`")
                return
    
    geoguessr_ds = GeoGuessrDataset(hf_dataset, max_samples=args.num_samples)
    print(f"   Using {len(geoguessr_ds)} samples for evaluation")
    
    # Evaluate
    print(f"\n🔄 Evaluating model...")
    eval_results = evaluate_dataset(
        geoguessr_ds,
        clip_model,
        processor,
        phase1_model,
        stage2_model,
        concept_adapter,
        centers_xyz,
        device,
        idx_to_concept,
        batch_size=args.batch_size,
    )
    
    # Print results
    metrics = eval_results["metrics"]
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\n📊 Error Metrics:")
    print(f"   Mean Error:   {metrics['mean_error_km']:,.1f} km")
    print(f"   Median Error: {metrics['median_error_km']:,.1f} km")
    print(f"   Std Dev:      {metrics['std_error_km']:,.1f} km")
    print(f"\n📍 GeoGuessr Score Thresholds:")
    print(f"   ≤25 km (5000 pts):    {metrics['acc_25km']*100:.2f}%")
    print(f"   ≤200 km (4000 pts):   {metrics['acc_200km']*100:.2f}%")
    print(f"   ≤750 km (3000 pts):   {metrics['acc_750km']*100:.2f}%")
    print(f"   ≤2500 km (1000 pts):  {metrics['acc_2500km']*100:.2f}%")
    
    if metrics.get('country_total', 0) > 0:
        print(f"\n🌍 Country/Continent Accuracy:")
        print(f"   Country Accuracy:   {metrics['country_accuracy']*100:.2f}% ({metrics['country_correct']}/{metrics['country_total']})")
        print(f"   Continent Accuracy: {metrics['continent_accuracy']*100:.2f}% ({metrics['continent_correct']}/{metrics['continent_total']})")
    
    # Save metrics
    print(f"\n💾 Saving results...")
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create visualizations
    print(f"\n📈 Creating visualizations...")
    
    # 1. Metrics summary
    print("   1. Metrics summary...")
    plot_metrics_summary(eval_results, output_dir / "metrics_summary.png")
    
    # 2. Concept distribution
    print("   2. Concept distribution...")
    plot_concept_distribution(eval_results, idx_to_concept, output_dir / "concept_distribution.png")
    
    # 3. Country analysis
    print("   3. Country analysis...")
    plot_country_analysis(eval_results, output_dir / "country_analysis.png")
    
    # 4. Sample visualizations
    if args.visualize_samples > 0:
        print(f"   4. Visualizing {args.visualize_samples} samples...")
        samples_dir = output_dir / "sample_visualizations"
        samples_dir.mkdir(exist_ok=True)
        
        # Select diverse samples: best, worst, and random
        errors = eval_results["errors"]
        sorted_indices = np.argsort(errors)
        
        # Best 5, worst 5, random 10
        n_best = min(5, args.visualize_samples // 3)
        n_worst = min(5, args.visualize_samples // 3)
        n_random = args.visualize_samples - n_best - n_worst
        
        best_indices = sorted_indices[:n_best].tolist()
        worst_indices = sorted_indices[-n_worst:].tolist()
        random_indices = np.random.choice(
            sorted_indices[n_best:-n_worst] if len(sorted_indices) > n_best + n_worst else sorted_indices,
            min(n_random, len(sorted_indices) - n_best - n_worst),
            replace=False
        ).tolist()
        
        selected_indices = best_indices + worst_indices + random_indices
        
        for i, idx in enumerate(tqdm(selected_indices, desc="   Visualizing")):
            try:
                sample = geoguessr_ds[idx]
                result = eval_results["results"][idx]
                
                category = "best" if idx in best_indices else ("worst" if idx in worst_indices else "random")
                filename = f"{category}_{i:03d}_error_{result['error_km']:.0f}km.png"
                
                visualize_sample_attention(
                    result,
                    sample["image"],
                    idx_to_concept,
                    samples_dir / filename,
                )
            except Exception as e:
                print(f"      Warning: Failed to visualize sample {idx}: {e}")
    
    print(f"\n✅ Results saved to: {output_dir}")
    print(f"\nGenerated files:")
    print(f"   - metrics.json")
    print(f"   - metrics_summary.png")
    print(f"   - concept_distribution.png")
    print(f"   - country_analysis.png")
    if args.visualize_samples > 0:
        print(f"   - sample_visualizations/ ({args.visualize_samples} images)")


if __name__ == "__main__":
    main()

