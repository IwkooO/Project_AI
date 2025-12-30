#!/usr/bin/env python3
"""
Flask API Server for GeoGuessr Bot Inference

Provides REST API endpoint for geolocation prediction using Stage 2 cross-attention model.
Logs concept predictions with visualizations to results folder.

Usage:
    python bot/api_server.py --checkpoint /path/to/stage2_checkpoint.pt
"""

import argparse
import base64
import io
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

from src.dataset import get_transforms_from_processor
from src.models.streetclip_encoder import StreetCLIPEncoder, StreetCLIPConfig
from src.models.concept_aware_cbm import Stage2CrossAttentionGeoHead, Stage1ConceptModel
from scripts.training.train_stage2_cross_attention import (
    load_stage1_checkpoint,
    load_image_encoder_weights_from_stage0_checkpoint,
    is_missing_or_none_path,
    compute_predicted_coords,
    cell_center_to_latlng,
    latlng_to_cartesian,
)
from bot.streetclip_inference import StreetCLIPInference

# Try to import geopandas for map visualization
try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
    # Path to local Natural Earth shapefile
    WORLD_SHAPEFILE = Path("/scratch-shared/pnair/Project_AI/data/geo/ne_110m_admin_0_countries.shp")
except ImportError:
    HAS_GEOPANDAS = False
    WORLD_SHAPEFILE = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== LaTeX-Ready Plot Configuration ==========
def setup_latex_style():
    """Configure matplotlib for publication-quality, LaTeX-compatible figures."""
    plt.rcParams.update({
        # Font settings
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        # Figure settings
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        # Line and edge settings
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
        # Color settings
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        # Grid
        'axes.grid': False,
        'grid.alpha': 0.3,
    })

setup_latex_style()

# Professional color palette
COLORS = {
    'primary': '#2C3E50',      # Dark blue-gray
    'secondary': '#E74C3C',    # Red accent
    'concept': '#3498DB',      # Blue
    'image': '#E67E22',        # Orange
    'balanced': '#27AE60',     # Green
    'highlight': '#9B59B6',    # Purple
    'muted': '#95A5A6',        # Gray
    'bg_light': '#F8F9FA',     # Light background
    'bg_dark': '#2C3E50',      # Dark background
}

# Custom colormap for attention
ATTENTION_CMAP = LinearSegmentedColormap.from_list(
    'attention', ['#FFFFFF', '#FFF3E0', '#FFE0B2', '#FFCC80', '#FFB74D', '#FF9800', '#F57C00', '#E65100'], N=256
)

app = Flask(__name__)
CORS(app)

# Global variables for loaded models
model = None
image_encoder = None
stage1_model = None
cell_centers = None
concept_info = None
ckpt = None
device = None
transform = None
streetclip_inference = None  # For vanilla HuggingFace StreetCLIP
use_streetclip = False  # Flag to use StreetCLIP instead of stage2

# Logging state
log_dir = None
session_csv_path = None
prediction_count = 0
session_start_time = None

# CSV header for session results
SESSION_CSV_HEADER = "round,timestamp,model,pred_lat,pred_lng,true_lat,true_lng,distance_km,score,cell_id,cell_confidence\n"


def find_nearest_cell(lat: float, lng: float, cell_centers: torch.Tensor) -> tuple[int, float]:
    """
    Find the nearest geocell for a given lat/lng coordinate.
    
    Args:
        lat: Latitude in degrees
        lng: Longitude in degrees
        cell_centers: Tensor of cell centers [num_cells, 3] in Cartesian coordinates
    
    Returns:
        Tuple of (cell_id, confidence) where confidence is the dot product (cosine similarity)
    """
    if cell_centers is None or len(cell_centers) == 0:
        return -1, 0.0
    
    # Convert lat/lng to 3D Cartesian
    coord_tensor = torch.tensor([[lat, lng]], dtype=torch.float32)
    xyz = latlng_to_cartesian(coord_tensor)
    xyz = F.normalize(xyz, p=2, dim=1)  # Normalize to unit sphere
    
    # Normalize cell centers to unit vectors
    centers = F.normalize(cell_centers.float(), p=2, dim=1)
    
    # Compute dot products (cosine similarity) - higher is closer
    dots = torch.mv(centers, xyz.squeeze())  # [num_cells]
    
    # Find nearest cell (highest dot product = smallest angle)
    cell_id = torch.argmax(dots).item()
    confidence = dots[cell_id].item()
    
    return cell_id, confidence


def init_logging_session():
    """Initialize a new logging session with timestamp directory and CSV file."""
    global log_dir, session_csv_path, session_start_time, prediction_count
    
    session_start_time = datetime.now()
    timestamp = session_start_time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path("/scratch-shared/pnair/Project_AI/results/geoguessr_game_logs") / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    prediction_count = 0
    
    # Create session results CSV with header
    session_csv_path = log_dir / "session_results.csv"
    with open(session_csv_path, 'w') as f:
        f.write(SESSION_CSV_HEADER)
    
    logger.info(f"📊 Logging session started: {log_dir}")
    logger.info(f"📄 Session CSV: {session_csv_path}")
    return log_dir


def load_stage2_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple:
    """Load Stage 2 checkpoint."""
    logger.info(f"Loading Stage 2 checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Store the checkpoint path for API endpoint
    ckpt["checkpoint_path"] = str(checkpoint_path)

    stage1_ckpt_path = Path(ckpt["stage1_checkpoint"])
    stage1_ckpt_data = torch.load(stage1_ckpt_path, map_location="cpu", weights_only=False)
    encoder_config = StreetCLIPConfig(
        model_name=ckpt["encoder_model"],
        finetune=False,
        device=device,
    )
    image_encoder = StreetCLIPEncoder(encoder_config)

    embedded_enc = stage1_ckpt_data.get("image_encoder_state_dict")
    if embedded_enc is not None:
        logger.info("Using image_encoder_state_dict embedded in Stage1 checkpoint")
        image_encoder.load_state_dict(embedded_enc, strict=False)
    else:
        stage0_checkpoint = stage1_ckpt_data.get("stage0_checkpoint")
        if not is_missing_or_none_path(stage0_checkpoint):
            load_image_encoder_weights_from_stage0_checkpoint(Path(stage0_checkpoint), image_encoder)
        else:
            logger.info("Using base encoder weights")

    image_encoder.model.eval()
    for param in image_encoder.model.parameters():
        param.requires_grad = False

    stage1_model, concept_info = load_stage1_checkpoint(
        stage1_ckpt_path,
        image_encoder,
        device,
    )

    model = Stage2CrossAttentionGeoHead(
        patch_dim=ckpt["patch_dim"],
        concept_emb_dim=ckpt["concept_dim"],
        num_cells=ckpt["num_cells"],
        coord_output_dim=ckpt["coord_output_dim"],
        num_heads=ckpt["num_heads"],
        ablation_mode=ckpt.get("ablation_mode", "both"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    cell_centers = ckpt["cell_centers"].to(device)

    return model, image_encoder, stage1_model, cell_centers, concept_info, ckpt


@torch.no_grad()
def predict_single_image(image_data: str, save_image: bool = True) -> Dict:
    """Predict location for a single base64-encoded image."""
    global model, image_encoder, stage1_model, cell_centers, ckpt, device, transform
    global streetclip_inference, use_streetclip
    global log_dir, prediction_count

    # Decode base64 image
    if image_data.startswith('data:image'):
        header, encoded = image_data.split(',', 1)
        image_bytes = base64.b64decode(encoded)
    else:
        image_bytes = base64.b64decode(image_data)

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Use StreetCLIP if enabled
    if use_streetclip and streetclip_inference is not None:
        pred_lat, pred_lng = streetclip_inference.predict(image)
        
        # Log prediction (no concepts for vanilla StreetCLIP)
        prediction_count += 1
        if log_dir is not None:
            try:
                threading.Thread(
                    target=log_prediction_async,
                    args=(image.copy(), pred_lat, pred_lng, prediction_count, 
                          None, None, None,  # No concepts for vanilla StreetCLIP
                          None, None, None, None, None, None, None, None),
                    daemon=True
                ).start()
            except Exception as e:
                logger.warning(f"Failed to start logging thread: {e}")
        
        return {
            "results": {
                "lat": pred_lat,
                "lng": pred_lng
            }
        }
    
    # Original stage2 prediction code
    image_tensor = transform(image).unsqueeze(0).to(device)

    ablation_mode = ckpt.get("ablation_mode", "both")
    patch_dim = ckpt["patch_dim"]
    concept_dim = ckpt["concept_dim"]
    coord_output_dim = ckpt["coord_output_dim"]

    # Get image features
    if ablation_mode == "concept_only":
        img_features = image_encoder(image_tensor)
        concept_embs = stage1_model.concept_bottleneck(img_features.float())
        patch_tokens = torch.empty((1, 0, patch_dim), device=device, dtype=img_features.dtype)
    else:
        img_features, patch_tokens = image_encoder.get_features_and_patches(image_tensor)
        if ablation_mode == "image_only":
            concept_embs = torch.zeros((1, concept_dim), device=device, dtype=img_features.dtype)
        else:
            concept_embs = stage1_model.concept_bottleneck(img_features.float())

    # Stage 2 forward pass - ENABLE attention and gate for diagnostics
    outputs = model(concept_embs, patch_tokens, return_attention=True, return_gate=True)
    cell_logits = outputs["cell_logits"]
    pred_offsets = outputs["pred_offsets"]
    attn_weights = outputs.get("attn_weights")  # [1, 1, 576] or None
    gate = outputs.get("gate")  # [1, 512] or None

    pred_cells = cell_logits.argmax(dim=1)
    pred_coords = compute_predicted_coords(pred_cells, pred_offsets, cell_centers, coord_output_dim, device)

    pred_lat = pred_coords[0, 0].item()
    pred_lng = pred_coords[0, 1].item()

    # Compute cell prediction confidence
    cell_probs = F.softmax(cell_logits, dim=1)
    cell_confidence = cell_probs.max().item()
    top3_cell_probs, top3_cell_idx = cell_probs.topk(3, dim=1)

    # Compute gate statistics (if available) - shows concept vs image contribution
    gate_stats = None
    if gate is not None:
        gate_flat = gate.squeeze()
        gate_stats = {
            "mean": gate_flat.mean().item(),
            "std": gate_flat.std().item(),
            "min": gate_flat.min().item(),
            "max": gate_flat.max().item(),
        }

    # Compute attention statistics (if available)
    attn_stats = None
    if attn_weights is not None:
        attn_flat = attn_weights.squeeze()  # [576]
        attn_entropy = -(attn_flat * torch.log(attn_flat + 1e-10)).sum().item()
        attn_max_idx = attn_flat.argmax().item()
        attn_max_val = attn_flat.max().item()
        attn_stats = {
            "entropy": attn_entropy,
            "max_patch_idx": attn_max_idx,
            "max_attention": attn_max_val,
        }

    # Get Stage 1 concept predictions for logging
    meta_probs, parent_probs = None, None
    try:
        if hasattr(stage1_model, 'T_meta') and stage1_model.T_meta is not None:
            # Compute logits using the model's forward method
            stage1_outputs = stage1_model.forward_from_features(img_features.float())
            meta_probs = stage1_outputs.get("meta_probs")
            parent_probs = stage1_outputs.get("parent_probs")
            if parent_probs is None and "parent_logits" in stage1_outputs:
                parent_probs = F.softmax(stage1_outputs["parent_logits"], dim=1)
    except Exception as e:
        logger.warning(f"Could not get concept predictions: {e}")

    # Log gate statistics to console for quick debugging
    if gate_stats is not None:
        logger.info(f"🔬 Gate stats: mean={gate_stats['mean']:.4f}, std={gate_stats['std']:.4f} "
                   f"(gate>0.5 = concept-heavy, gate<0.5 = image-heavy)")

    # Prepare top cells data for geographic visualization
    top_cells_data = None
    if cell_centers is not None:
        try:
            cell_lats_all, cell_lngs_all = cell_center_to_latlng(cell_centers.cpu())
            top_cell_ids = top3_cell_idx[0].cpu().tolist()
            top_cells_data = {
                'ids': top_cell_ids,
                'probs': top3_cell_probs[0].cpu().tolist(),
                'lats': [cell_lats_all[cid].item() for cid in top_cell_ids],
                'lngs': [cell_lngs_all[cid].item() for cid in top_cell_ids],
            }
        except Exception as e:
            logger.warning(f"Could not compute cell centers for visualization: {e}")
    
    # Log prediction in background
    prediction_count += 1
    if log_dir is not None:
        try:
            threading.Thread(
                target=log_prediction_async,
                args=(image.copy(), pred_lat, pred_lng, prediction_count, 
                      meta_probs, parent_probs, concept_info,
                      gate_stats, attn_stats, cell_confidence, 
                      top3_cell_idx[0].cpu().tolist(), top3_cell_probs[0].cpu().tolist(),
                      attn_weights.cpu() if attn_weights is not None else None,
                      top_cells_data),
                daemon=True
            ).start()
        except Exception as e:
            logger.warning(f"Failed to start logging thread: {e}")

    return {
        "results": {
            "lat": pred_lat,
            "lng": pred_lng
        }
    }


def log_prediction_async(
    image: Image.Image,
    lat: float,
    lng: float,
    round_num: int,
    meta_probs: Optional[torch.Tensor],
    parent_probs: Optional[torch.Tensor],
    concept_info: Optional[Dict],
    gate_stats: Optional[Dict] = None,
    attn_stats: Optional[Dict] = None,
    cell_confidence: Optional[float] = None,
    top3_cell_idx: Optional[List[int]] = None,
    top3_cell_probs: Optional[List[float]] = None,
    attn_weights: Optional[torch.Tensor] = None,
    top_cells_data: Optional[Dict] = None,
):
    """Log prediction with concept visualization and diagnostics (runs in background thread)."""
    global log_dir
    
    try:
        if log_dir is None:
            return
        
        timestamp = datetime.now().strftime("%H%M%S")
        
        # Save input image
        image_path = log_dir / f"round_{round_num:02d}_{timestamp}_input.png"
        image.save(image_path)
        
        # Create visualization if we have concept predictions
        if meta_probs is not None and concept_info is not None:
            try:
                create_concept_visualization(
                    image=image,
                    meta_probs=meta_probs[0].cpu() if meta_probs.dim() > 1 else meta_probs.cpu(),
                    parent_probs=parent_probs[0].cpu() if parent_probs is not None and parent_probs.dim() > 1 else (parent_probs.cpu() if parent_probs is not None else None),
                    lat=lat,
                    lng=lng,
                    round_num=round_num,
                    timestamp=timestamp,
                    concept_info=concept_info,
                    output_dir=log_dir,
                    gate_stats=gate_stats,
                    attn_stats=attn_stats,
                    cell_confidence=cell_confidence,
                    attn_weights=attn_weights,
                    top_cells_data=top_cells_data,
                )
            except Exception as e:
                logger.warning(f"Failed to create visualization: {e}")
                import traceback
                traceback.print_exc()
        
        # Save text summary
        summary_path = log_dir / f"round_{round_num:02d}_{timestamp}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Round: {round_num}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Prediction: ({lat:.6f}, {lng:.6f})\n")
            f.write(f"Google Maps: https://www.google.com/maps?q={lat},{lng}\n")
            
            # Gate statistics (concept vs image contribution)
            if gate_stats is not None:
                f.write(f"\n════════════════════════════════════════\n")
                f.write(f"  CONCEPT vs IMAGE CONTRIBUTION (Gate)\n")
                f.write(f"════════════════════════════════════════\n")
                f.write(f"Gate Mean:  {gate_stats['mean']:.4f}\n")
                f.write(f"Gate Std:   {gate_stats['std']:.4f}\n")
                f.write(f"Gate Range: [{gate_stats['min']:.4f}, {gate_stats['max']:.4f}]\n")
                f.write(f"\nInterpretation:\n")
                f.write(f"  - gate > 0.5 = CONCEPT-heavy (relies on semantic concepts)\n")
                f.write(f"  - gate < 0.5 = IMAGE-heavy (relies on raw visual patches)\n")
                if gate_stats['mean'] > 0.6:
                    f.write(f"\n⚠️  Model is HEAVILY using CONCEPTS (gate mean > 0.6)\n")
                elif gate_stats['mean'] < 0.4:
                    f.write(f"\n⚠️  Model is HEAVILY using IMAGE PATCHES (gate mean < 0.4)\n")
                else:
                    f.write(f"\n✓ Model is using BALANCED mix of concepts and patches\n")
            
            # Attention statistics
            if attn_stats is not None:
                f.write(f"\n════════════════════════════════════════\n")
                f.write(f"  ATTENTION STATISTICS\n")
                f.write(f"════════════════════════════════════════\n")
                f.write(f"Attention Entropy:    {attn_stats['entropy']:.4f}\n")
                f.write(f"Max Attention Patch:  {attn_stats['max_patch_idx']} (value: {attn_stats['max_attention']:.4f})\n")
                f.write(f"\nInterpretation:\n")
                f.write(f"  - High entropy = diffuse attention (looking at many patches)\n")
                f.write(f"  - Low entropy = focused attention (looking at few patches)\n")
            
            # Cell prediction confidence
            if cell_confidence is not None:
                f.write(f"\n════════════════════════════════════════\n")
                f.write(f"  GEOCELL PREDICTION CONFIDENCE\n")
                f.write(f"════════════════════════════════════════\n")
                f.write(f"Top Cell Confidence: {cell_confidence:.4f}\n")
                if top3_cell_idx is not None and top3_cell_probs is not None:
                    f.write(f"Top 3 Cells: {top3_cell_idx}\n")
                    f.write(f"Top 3 Probs: [{', '.join([f'{p:.4f}' for p in top3_cell_probs])}]\n")
            
            # Concept predictions
            if meta_probs is not None and concept_info is not None:
                idx_to_concept = concept_info.get("idx_to_concept", {})
                idx_to_parent = concept_info.get("idx_to_parent", {})
                
                f.write(f"\n════════════════════════════════════════\n")
                f.write(f"  CONCEPT PREDICTIONS\n")
                f.write(f"════════════════════════════════════════\n")
                
                # Top 5 concepts
                probs = meta_probs[0].cpu() if meta_probs.dim() > 1 else meta_probs.cpu()
                top5_probs, top5_idx = torch.topk(probs, k=min(5, len(probs)))
                f.write(f"\nTop 5 Child Concepts:\n")
                for prob, idx in zip(top5_probs, top5_idx):
                    concept_name = idx_to_concept.get(idx.item(), f"concept_{idx.item()}")
                    f.write(f"  {prob:.4f} - {concept_name}\n")
                
                # Top 3 parents
                if parent_probs is not None:
                    p_probs = parent_probs[0].cpu() if parent_probs.dim() > 1 else parent_probs.cpu()
                    top3_probs, top3_idx = torch.topk(p_probs, k=min(3, len(p_probs)))
                    f.write(f"\nTop 3 Parent Concepts:\n")
                    for prob, idx in zip(top3_probs, top3_idx):
                        parent_name = idx_to_parent.get(idx.item(), f"parent_{idx.item()}")
                        f.write(f"  {prob:.4f} - {parent_name}\n")
        
        logger.info(f"📊 Logged round {round_num} to {log_dir.name}/")
    except Exception as e:
        logger.error(f"Error in log_prediction_async: {e}")


def create_concept_visualization(
    image: Image.Image,
    meta_probs: torch.Tensor,
    parent_probs: Optional[torch.Tensor],
    lat: float,
    lng: float,
    round_num: int,
    timestamp: str,
    concept_info: Dict,
    output_dir: Path,
    gate_stats: Optional[Dict] = None,
    attn_stats: Optional[Dict] = None,
    cell_confidence: Optional[float] = None,
    attn_weights: Optional[torch.Tensor] = None,
    top_cells_data: Optional[Dict] = None,
    true_lat: Optional[float] = None,
    true_lng: Optional[float] = None,
    distance_km: Optional[float] = None,
):
    """Create publication-quality visualization with geographic context."""
    idx_to_concept = concept_info.get("idx_to_concept", {})
    idx_to_parent = concept_info.get("idx_to_parent", {})
    
    # Get top concepts
    top5_probs, top5_indices = torch.topk(meta_probs, k=min(5, len(meta_probs)))
    top5_concepts = [idx_to_concept.get(idx.item(), f"concept_{idx.item()}") for idx in top5_indices]
    top5_probs_np = top5_probs.numpy()
    
    top3_parents, top3_parent_probs_np = [], []
    if parent_probs is not None:
        top3_parent_probs, top3_parent_indices = torch.topk(parent_probs, k=min(3, len(parent_probs)))
        top3_parents = [idx_to_parent.get(idx.item(), f"parent_{idx.item()}") for idx in top3_parent_indices]
        top3_parent_probs_np = top3_parent_probs.numpy()
    
    # Create figure with GridSpec - 2 columns, 4 rows
    # Row 0: Input image (full width, largest)
    # Row 1: Attention heatmap | Geographic map  
    # Row 2: Gate gauge | Confidence panel
    # Row 3: Child concepts | Parent concepts
    fig = plt.figure(figsize=(18, 20))
    gs = gridspec.GridSpec(4, 2, figure=fig, 
                           height_ratios=[2.2, 1.3, 1.0, 1.0],
                           hspace=0.30, wspace=0.25)
    
    # ========== ROW 0: Input Image (Full Width, Prominent) ==========
    ax_img = fig.add_subplot(gs[0, :])
    ax_img.imshow(np.array(image))
    ax_img.axis("off")
    ax_img.set_title(f"(A) Input Image — Round {round_num}", fontsize=16, fontweight='bold', pad=15)
    
    # ========== ROW 1: Attention Heatmap + Geographic Map ==========
    
    # Panel B: Attention heatmap overlay
    ax_attn = fig.add_subplot(gs[1, 0])
    if attn_weights is not None:
        attn_flat = attn_weights.squeeze().cpu().numpy()
        grid_size = int(np.sqrt(len(attn_flat)))
        attn_grid = attn_flat.reshape(grid_size, grid_size)
        
        # Resize image to match attention grid for overlay
        img_resized = image.resize((grid_size * 16, grid_size * 16))
        ax_attn.imshow(np.array(img_resized), alpha=0.4)
        
        # Overlay attention as heatmap
        attn_upscaled = np.kron(attn_grid, np.ones((16, 16)))
        im = ax_attn.imshow(attn_upscaled, cmap=ATTENTION_CMAP, alpha=0.65, vmin=0, vmax=attn_flat.max())
        
        # Add colorbar with proper spacing
        cbar = plt.colorbar(im, ax=ax_attn, fraction=0.046, pad=0.04, shrink=0.85)
        cbar.set_label('Attention Weight', fontsize=10)
        cbar.ax.tick_params(labelsize=9)
    else:
        ax_attn.imshow(np.array(image), alpha=0.4)
        ax_attn.text(0.5, 0.5, "Attention\nNot Available", transform=ax_attn.transAxes,
                     ha='center', va='center', fontsize=12, color=COLORS['muted'])
    ax_attn.axis("off")
    ax_attn.set_title("(B) Patch Attention Heatmap", fontsize=14, fontweight='bold', pad=12)
    
    # Panel C: Geographic Map with world basemap
    ax_map = fig.add_subplot(gs[1, 1])
    create_geographic_map_panel(ax_map, lat, lng, top_cells_data, round_num)
    
    # ========== ROW 2: Gate Visualization + Model Confidence ==========
    
    # Panel D: Gate Contribution Gauge
    ax_gate = fig.add_subplot(gs[2, 0])
    create_gate_gauge_panel(ax_gate, gate_stats)
    
    # Panel E: Cell Confidence + Attention Stats
    ax_conf = fig.add_subplot(gs[2, 1])
    create_confidence_panel(ax_conf, cell_confidence, attn_stats, gate_stats, top_cells_data)
    
    # ========== ROW 3: Concept Predictions ==========
    
    # Panel F: Top-5 Child Concepts
    ax_child = fig.add_subplot(gs[3, 0])
    create_concept_bar_panel(ax_child, top5_concepts, top5_probs_np, 
                             title="(F) Top-5 Child Concept Predictions",
                             color_scheme='blues')
    
    # Panel G: Top-3 Parent Concepts  
    ax_parent = fig.add_subplot(gs[3, 1])
    if parent_probs is not None and len(top3_parents) > 0:
        create_concept_bar_panel(ax_parent, top3_parents, top3_parent_probs_np,
                                 title="(G) Top-3 Parent Concept Predictions", 
                                 color_scheme='greens')
    else:
        ax_parent.text(0.5, 0.5, "Parent concepts not available", 
                       ha='center', va='center', fontsize=12, color=COLORS['muted'])
        ax_parent.axis("off")
        ax_parent.set_title("(G) Top-3 Parent Concept Predictions", fontsize=14, fontweight='bold')
    
    # Add overall figure title with prediction coordinates
    fig.suptitle(f"Prediction: ({lat:.4f}°, {lng:.4f}°)", 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # Save figure
    output_path = output_dir / f"round_{round_num:02d}_{timestamp}_concepts.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor='white', edgecolor='none')
    plt.close(fig)


def create_geographic_map_panel(ax, lat: float, lng: float, 
                                 top_cells_data: Optional[Dict], round_num: int,
                                 true_lat: Optional[float] = None, 
                                 true_lng: Optional[float] = None,
                                 distance_km: Optional[float] = None):
    """Create a geographic map showing the prediction with world basemap.
    
    Args:
        ax: Matplotlib axis
        lat: Predicted latitude
        lng: Predicted longitude
        top_cells_data: Top cell predictions data
        round_num: Current round number
        true_lat: True latitude (optional, for error visualization)
        true_lng: True longitude (optional, for error visualization)
        distance_km: Distance error in km (optional)
    """
    
    # Set ocean background FIRST
    ax.set_facecolor('#D4E8F0')
    
    # Load and plot world basemap
    if HAS_GEOPANDAS and WORLD_SHAPEFILE is not None and WORLD_SHAPEFILE.exists():
        try:
            world = gpd.read_file(WORLD_SHAPEFILE)
            world.plot(ax=ax, color='#D4E6D4', edgecolor='#707070', linewidth=0.5, zorder=2)
        except Exception as e:
            logger.warning(f"Failed to load world shapefile: {e}")
    
    # Set limits AFTER plotting world
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    
    # Draw grid lines (behind everything)
    ax.grid(True, alpha=0.4, linestyle='-', color='#90A4AE', linewidth=0.4, zorder=1)
    ax.set_axisbelow(True)
    
    # Plot top-k cell centers and show offset arrow from cell to prediction
    if top_cells_data is not None:
        cell_lats = top_cells_data.get('lats', [])
        cell_lngs = top_cells_data.get('lngs', [])
        cell_probs = top_cells_data.get('probs', [])
        cell_ids = top_cells_data.get('ids', [])
        
        if cell_lats and cell_lngs:
            # Top cell center
            top_cell_lat = cell_lats[0]
            top_cell_lng = cell_lngs[0]
            top_cell_prob = cell_probs[0] if cell_probs else 0
            top_cell_id = cell_ids[0] if cell_ids else '?'
            
            # Draw cell center as a circle
            ax.scatter(top_cell_lng, top_cell_lat, c=COLORS['concept'], s=250, marker='o',
                      edgecolors='white', linewidths=3, zorder=15, alpha=0.95,
                      label=f'Cell #{top_cell_id} (p={top_cell_prob:.3f})')
            
            # Draw offset arrow from cell center to final prediction
            ax.annotate('', xy=(lng, lat), xytext=(top_cell_lng, top_cell_lat),
                       arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], 
                                      lw=3, mutation_scale=18),
                       zorder=18)
            
            # Add offset label at midpoint
            mid_lng = (top_cell_lng + lng) / 2
            mid_lat = (top_cell_lat + lat) / 2
            offset_dist = np.sqrt((lat - top_cell_lat)**2 + (lng - top_cell_lng)**2)
            ax.text(mid_lng, mid_lat + 4, f'offset: {offset_dist:.1f}°', 
                   ha='center', va='bottom', fontsize=10, color=COLORS['secondary'],
                   fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', 
                   facecolor='white', edgecolor=COLORS['secondary'], linewidth=1.5, alpha=0.95))
            
            # Draw other candidate cells (fainter)
            for i, (clat, clng, cprob) in enumerate(zip(cell_lats[1:], cell_lngs[1:], cell_probs[1:])):
                alpha = 0.3 + 0.3 * (cprob / top_cell_prob) if top_cell_prob > 0 else 0.4
                ax.scatter(clng, clat, c=COLORS['highlight'], s=100, marker='o',
                          edgecolors='white', linewidths=1.5, zorder=10, alpha=alpha)
    
    # Plot final prediction (star)
    ax.scatter(lng, lat, c=COLORS['secondary'], s=400, marker='*', 
              edgecolors='white', linewidths=3, zorder=20,
              label=f'Prediction ({lat:.2f}°, {lng:.2f}°)')
    
    # Plot true location and error arrow if available
    if true_lat is not None and true_lng is not None:
        # Plot true location (green circle)
        ax.scatter(true_lng, true_lat, c=COLORS['balanced'], s=350, marker='o', 
                  edgecolors='white', linewidths=3, zorder=21,
                  label=f'True Location ({true_lat:.2f}°, {true_lng:.2f}°)')
        
        # Draw error arrow from prediction to true location
        ax.annotate('', xy=(true_lng, true_lat), xytext=(lng, lat),
                   arrowprops=dict(arrowstyle='->', color=COLORS['highlight'], 
                                  lw=3, mutation_scale=15, linestyle='--'),
                   zorder=19)
        
        # Add distance label at midpoint
        mid_lng = (lng + true_lng) / 2
        mid_lat = (lat + true_lat) / 2
        if distance_km is not None:
            dist_label = f'{distance_km:.0f} km' if distance_km >= 1 else f'{distance_km*1000:.0f} m'
        else:
            # Approximate distance using haversine
            from math import radians, sin, cos, sqrt, atan2
            R = 6371  # Earth radius in km
            dlat = radians(true_lat - lat)
            dlng = radians(true_lng - lng)
            a = sin(dlat/2)**2 + cos(radians(lat)) * cos(radians(true_lat)) * sin(dlng/2)**2
            dist_label = f'{R * 2 * atan2(sqrt(a), sqrt(1-a)):.0f} km'
        
        ax.text(mid_lng, mid_lat - 5, f'Error: {dist_label}', 
               ha='center', va='top', fontsize=11, color=COLORS['highlight'],
               fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', 
               facecolor='white', edgecolor=COLORS['highlight'], linewidth=2, alpha=0.95),
               zorder=25)
    
    # Crosshair at prediction
    ax.axhline(y=lat, color=COLORS['secondary'], alpha=0.4, linewidth=1.5, linestyle='--', zorder=5)
    ax.axvline(x=lng, color=COLORS['secondary'], alpha=0.4, linewidth=1.5, linestyle='--', zorder=5)
    
    # Axis labels and title
    ax.set_xlabel('Longitude (°)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude (°)', fontsize=12, fontweight='bold')
    ax.set_title("(C) Geographic Prediction Map", fontsize=14, fontweight='bold', pad=12)
    
    # Legend with better placement and visibility
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95, 
              fancybox=True, edgecolor=COLORS['muted'])
    
    # Tick formatting
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
    ax.set_yticks([-60, -30, 0, 30, 60])
    ax.tick_params(axis='both', labelsize=10)


def create_gate_gauge_panel(ax, gate_stats: Optional[Dict]):
    """Create a professional gauge visualization for concept vs image contribution."""
    if gate_stats is None:
        ax.text(0.5, 0.5, "Gate statistics not available", 
                ha='center', va='center', fontsize=12, color=COLORS['muted'])
        ax.axis("off")
        ax.set_title("(D) Concept vs Image Contribution", fontsize=12, fontweight='bold')
        return
    
    gate_mean = gate_stats['mean']
    gate_std = gate_stats['std']
    
    # Create horizontal gauge
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 1.6)
    
    # Background gradient bar (thicker)
    gradient = np.linspace(0, 1, 100).reshape(1, -1)
    cmap_gradient = LinearSegmentedColormap.from_list('gauge', 
        [COLORS['image'], '#F5F5F5', COLORS['concept']])
    ax.imshow(gradient, extent=[0, 1, -0.2, 0.2], aspect='auto', cmap=cmap_gradient, alpha=0.85)
    
    # Marker for current value
    ax.axvline(x=gate_mean, ymin=0.28, ymax=0.72, color=COLORS['primary'], linewidth=4)
    ax.scatter([gate_mean], [0], s=200, c=COLORS['primary'], zorder=10, edgecolors='white', linewidths=2.5)
    
    # Uncertainty band (±1 std)
    ax.axvspan(max(0, gate_mean - gate_std), min(1, gate_mean + gate_std), 
               ymin=0.32, ymax=0.68, alpha=0.2, color=COLORS['primary'])
    
    # Labels (larger, clearer)
    ax.text(0.0, -0.45, "IMAGE\nDominant", ha='center', va='top', fontsize=10, 
            color=COLORS['image'], fontweight='bold')
    ax.text(0.5, -0.45, "BALANCED", ha='center', va='top', fontsize=10, 
            color=COLORS['muted'], fontweight='bold')
    ax.text(1.0, -0.45, "CONCEPT\nDominant", ha='center', va='top', fontsize=10, 
            color=COLORS['concept'], fontweight='bold')
    
    # Current value annotation
    if gate_mean > 0.6:
        interp, color = "Concept-Heavy", COLORS['concept']
    elif gate_mean < 0.4:
        interp, color = "Image-Heavy", COLORS['image']
    else:
        interp, color = "Balanced", COLORS['balanced']
    
    ax.text(gate_mean, 0.55, f"{gate_mean:.3f}", ha='center', va='bottom', 
            fontsize=14, fontweight='bold', color=COLORS['primary'])
    ax.text(0.5, 1.2, f"Model is {interp}", ha='center', va='bottom', 
            fontsize=13, fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=color, linewidth=2, alpha=0.95))
    
    # Stats box
    stats_text = f"mean={gate_mean:.3f}  std={gate_std:.3f}\nrange=[{gate_stats['min']:.2f}, {gate_stats['max']:.2f}]"
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, ha='right', va='top',
            fontsize=9, fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['bg_light'], edgecolor=COLORS['muted'], alpha=0.9))
    
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis='x', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_title("(D) Concept vs Image Contribution (Gate)", fontsize=12, fontweight='bold', pad=15)


def create_confidence_panel(ax, cell_confidence: Optional[float], 
                            attn_stats: Optional[Dict], gate_stats: Optional[Dict],
                            top_cells_data: Optional[Dict]):
    """Create a panel showing model confidence metrics."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    
    # Add light background
    ax.add_patch(plt.Rectangle((0, 0), 10, 10, facecolor=COLORS['bg_light'], 
                                edgecolor=COLORS['muted'], linewidth=1, alpha=0.5))
    
    y_pos = 9.2
    line_height = 1.3
    
    # Title
    ax.text(5, y_pos, "(E) Model Confidence Metrics", ha='center', va='top',
            fontsize=12, fontweight='bold', color=COLORS['primary'])
    y_pos -= 1.4
    
    # Cell Confidence
    if cell_confidence is not None:
        conf_color = COLORS['balanced'] if cell_confidence > 0.1 else COLORS['secondary']
        ax.text(0.3, y_pos, "Cell Confidence:", ha='left', va='top', fontsize=11, fontweight='bold')
        ax.text(9.7, y_pos, f"{cell_confidence:.4f}", ha='right', va='top', fontsize=12, 
                color=conf_color, fontweight='bold')
        y_pos -= line_height
        
        # Top-3 cells
        if top_cells_data is not None:
            cell_ids = top_cells_data.get('ids', [])
            cell_probs = top_cells_data.get('probs', [])
            if cell_ids and cell_probs:
                ax.text(0.3, y_pos, "Top-3 Cells:", ha='left', va='top', fontsize=10, fontweight='bold')
                cell_str = ", ".join([f"#{cid} ({cp:.3f})" for cid, cp in zip(cell_ids[:3], cell_probs[:3])])
                ax.text(9.7, y_pos, cell_str, ha='right', va='top', fontsize=9, color=COLORS['primary'])
                y_pos -= line_height
    
    # Divider
    ax.axhline(y=y_pos + 0.4, xmin=0.03, xmax=0.97, color=COLORS['muted'], linewidth=1, alpha=0.6)
    y_pos -= 0.6
    
    # Attention Statistics
    if attn_stats is not None:
        ax.text(0.3, y_pos, "Attention Entropy:", ha='left', va='top', fontsize=11, fontweight='bold')
        ax.text(9.7, y_pos, f"{attn_stats['entropy']:.4f}", ha='right', va='top', fontsize=11, 
                color=COLORS['primary'], fontweight='bold')
        y_pos -= line_height
        
        ax.text(0.3, y_pos, "Max Patch Attention:", ha='left', va='top', fontsize=10)
        ax.text(9.7, y_pos, f"patch #{attn_stats['max_patch_idx']} ({attn_stats['max_attention']:.4f})", 
                ha='right', va='top', fontsize=10, color=COLORS['primary'])
        y_pos -= line_height
        
        # Interpretation
        if attn_stats['entropy'] > 5.5:
            attn_interp = "Diffuse attention (exploring many patches)"
        else:
            attn_interp = "Focused attention (few key patches)"
        ax.text(5, y_pos, f">> {attn_interp}", ha='center', va='top', fontsize=9, 
                style='italic', color=COLORS['muted'])
        y_pos -= line_height
    
    # Divider
    ax.axhline(y=y_pos + 0.4, xmin=0.03, xmax=0.97, color=COLORS['muted'], linewidth=1, alpha=0.6)
    y_pos -= 0.6
    
    # Interpretation summary
    if gate_stats is not None and attn_stats is not None:
        gate_mean = gate_stats['mean']
        
        if gate_mean > 0.6:
            summary = "Using semantic concepts heavily"
        elif gate_mean < 0.4:
            summary = "Relying on raw visual patterns"
        else:
            summary = "Balanced concept + visual reasoning"
        
        ax.text(5, y_pos, f"Summary: {summary}", ha='center', va='top', 
                fontsize=10, fontweight='bold', color=COLORS['primary'],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor=COLORS['primary'], alpha=0.9))
    
    ax.set_title("", pad=0)


def create_concept_bar_panel(ax, concepts: List[str], probs: np.ndarray, 
                              title: str, color_scheme: str = 'blues'):
    """Create a professional horizontal bar chart for concept predictions."""
    n_concepts = len(concepts)
    y_pos = np.arange(n_concepts)
    
    # Color schemes
    if color_scheme == 'blues':
        colors = [COLORS['concept']] * n_concepts
        edge_color = '#1976D2'
    else:  # greens
        colors = [COLORS['balanced']] * n_concepts
        edge_color = '#388E3C'
    
    # Create bars (reversed so highest is on top)
    bars = ax.barh(y_pos, probs[::-1], color=colors, edgecolor=edge_color, 
                   linewidth=0.8, height=0.65, alpha=0.9)
    
    # Labels
    ax.set_yticks(y_pos)
    labels = [c[:32] + "..." if len(c) > 32 else c for c in concepts[::-1]]
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Probability", fontsize=11, fontweight='bold')
    ax.set_xlim(0, max(0.3, probs.max() * 1.35))
    
    # Value annotations
    for bar, prob in zip(bars, probs[::-1]):
        ax.text(prob + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{prob:.3f}", va='center', fontsize=10, fontweight='bold', color=COLORS['primary'])
    
    # Clean up axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)


def create_error_map_visualization(
    round_num: int,
    pred_lat: float,
    pred_lng: float,
    true_lat: float,
    true_lng: float,
    distance_km: float,
    score: int,
    model_name: str,
    output_dir: Path,
    top_cells_data: Optional[Dict] = None,
):
    """Create a standalone error map visualization with side-by-side world and zoomed views.
    
    Args:
        round_num: Round number
        pred_lat: Predicted latitude
        pred_lng: Predicted longitude
        true_lat: True latitude
        true_lng: True longitude
        distance_km: Distance error in km
        score: GeoGuessr score
        model_name: Model identifier
        output_dir: Directory to save the visualization
        top_cells_data: Top cell predictions data (optional, for showing cell centers)
    """
    try:
        # Create figure with side-by-side layout
        fig, (ax_world, ax_zoom) = plt.subplots(1, 2, figsize=(20, 9))
        
        # ========== LEFT PANEL: World Map with Cell and Offset (NO ERROR) ==========
        # Don't show error on world view - only show cell and offset
        create_geographic_map_panel(
            ax_world, pred_lat, pred_lng, 
            top_cells_data=top_cells_data, 
            round_num=round_num,
            true_lat=None,  # Don't show error on world view
            true_lng=None,
            distance_km=None
        )
        ax_world.set_title("(A) World View: Prediction with Cell & Offset", 
                          fontsize=14, fontweight='bold', pad=12)
        
        # ========== RIGHT PANEL: Zoomed-in Error View ==========
        # Calculate zoom bounds to show both prediction and true location clearly
        # Use a buffer based on distance_km (convert to degrees, roughly 1° ≈ 111 km)
        buffer_deg = max(2.0, (distance_km / 111.0) * 2.5)  # At least 2° buffer, or 2.5x the error distance
        
        center_lat = (pred_lat + true_lat) / 2
        center_lng = (pred_lng + true_lng) / 2
        
        # Set zoom limits
        zoom_lat_min = min(pred_lat, true_lat) - buffer_deg
        zoom_lat_max = max(pred_lat, true_lat) + buffer_deg
        zoom_lng_min = min(pred_lng, true_lng) - buffer_deg
        zoom_lng_max = max(pred_lng, true_lng) + buffer_deg
        
        # Set ocean background
        ax_zoom.set_facecolor('#D4E8F0')
        
        # Load and plot world basemap (will be clipped by zoom limits)
        if HAS_GEOPANDAS and WORLD_SHAPEFILE is not None and WORLD_SHAPEFILE.exists():
            try:
                world = gpd.read_file(WORLD_SHAPEFILE)
                world.plot(ax=ax_zoom, color='#D4E6D4', edgecolor='#707070', linewidth=0.5, zorder=2)
            except Exception as e:
                logger.warning(f"Failed to load world shapefile: {e}")
        
        # Set zoom limits
        ax_zoom.set_xlim(zoom_lng_min, zoom_lng_max)
        ax_zoom.set_ylim(zoom_lat_min, zoom_lat_max)
        
        # Draw grid lines
        ax_zoom.grid(True, alpha=0.4, linestyle='-', color='#90A4AE', linewidth=0.4, zorder=1)
        ax_zoom.set_axisbelow(True)
        
        # Plot cell centers if available (only if within zoom bounds)
        if top_cells_data is not None:
            cell_lats = top_cells_data.get('lats', [])
            cell_lngs = top_cells_data.get('lngs', [])
            cell_probs = top_cells_data.get('probs', [])
            cell_ids = top_cells_data.get('ids', [])
            
            if cell_lats and cell_lngs:
                top_cell_lat = cell_lats[0]
                top_cell_lng = cell_lngs[0]
                top_cell_prob = cell_probs[0] if cell_probs else 0
                top_cell_id = cell_ids[0] if cell_ids else '?'
                
                # Only plot if cell is within zoom bounds
                if (zoom_lat_min <= top_cell_lat <= zoom_lat_max and 
                    zoom_lng_min <= top_cell_lng <= zoom_lng_max):
                    ax_zoom.scatter(top_cell_lng, top_cell_lat, c=COLORS['concept'], s=300, marker='o',
                                  edgecolors='white', linewidths=3, zorder=15, alpha=0.95,
                                  label=f'Cell #{top_cell_id} (p={top_cell_prob:.3f})')
                    
                    # Draw offset arrow if cell is visible
                    ax_zoom.annotate('', xy=(pred_lng, pred_lat), xytext=(top_cell_lng, top_cell_lat),
                                   arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], 
                                                  lw=2.5, mutation_scale=15),
                                   zorder=18)
        
        # Plot prediction (star)
        ax_zoom.scatter(pred_lng, pred_lat, c=COLORS['secondary'], s=500, marker='*', 
                      edgecolors='white', linewidths=3, zorder=20,
                      label=f'Prediction ({pred_lat:.4f}°, {pred_lng:.4f}°)')
        
        # Plot true location (green circle)
        ax_zoom.scatter(true_lng, true_lat, c=COLORS['balanced'], s=450, marker='o', 
                      edgecolors='white', linewidths=3, zorder=21,
                      label=f'True Location ({true_lat:.4f}°, {true_lng:.4f}°)')
        
        # Draw error arrow from prediction to true location
        ax_zoom.annotate('', xy=(true_lng, true_lat), xytext=(pred_lng, pred_lat),
                       arrowprops=dict(arrowstyle='->', color=COLORS['highlight'], 
                                      lw=3.5, mutation_scale=18, linestyle='--'),
                       zorder=19)
        
        # Add distance label at midpoint
        mid_lng = (pred_lng + true_lng) / 2
        mid_lat = (pred_lat + true_lat) / 2
        dist_label = f'{distance_km:.1f} km' if distance_km >= 1 else f'{distance_km*1000:.0f} m'
        
        ax_zoom.text(mid_lng, mid_lat, f'Error: {dist_label}', 
                   ha='center', va='center', fontsize=13, color=COLORS['highlight'],
                   fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', 
                   facecolor='white', edgecolor=COLORS['highlight'], linewidth=2.5, alpha=0.95),
                   zorder=25)
        
        # Crosshairs at both locations
        ax_zoom.axhline(y=pred_lat, color=COLORS['secondary'], alpha=0.3, linewidth=1.2, linestyle='--', zorder=5)
        ax_zoom.axvline(x=pred_lng, color=COLORS['secondary'], alpha=0.3, linewidth=1.2, linestyle='--', zorder=5)
        ax_zoom.axhline(y=true_lat, color=COLORS['balanced'], alpha=0.3, linewidth=1.2, linestyle='--', zorder=5)
        ax_zoom.axvline(x=true_lng, color=COLORS['balanced'], alpha=0.3, linewidth=1.2, linestyle='--', zorder=5)
        
        # Axis labels and title
        ax_zoom.set_xlabel('Longitude (°)', fontsize=12, fontweight='bold')
        ax_zoom.set_ylabel('Latitude (°)', fontsize=12, fontweight='bold')
        ax_zoom.set_title("(B) Zoomed View: Error Detail", fontsize=14, fontweight='bold', pad=12)
        
        # Legend
        ax_zoom.legend(loc='upper right', fontsize=10, framealpha=0.95, 
                      fancybox=True, edgecolor=COLORS['muted'])
        
        # Format ticks for zoomed view (more granular)
        lat_range = zoom_lat_max - zoom_lat_min
        lng_range = zoom_lng_max - zoom_lng_min
        
        # Choose appropriate tick spacing based on zoom level
        if lat_range < 5:
            lat_ticks = np.arange(np.floor(zoom_lat_min), np.ceil(zoom_lat_max) + 0.5, 0.5)
        elif lat_range < 20:
            lat_ticks = np.arange(np.floor(zoom_lat_min), np.ceil(zoom_lat_max) + 1, 1)
        else:
            lat_ticks = np.arange(np.floor(zoom_lat_min), np.ceil(zoom_lat_max) + 5, 5)
        
        if lng_range < 5:
            lng_ticks = np.arange(np.floor(zoom_lng_min), np.ceil(zoom_lng_max) + 0.5, 0.5)
        elif lng_range < 20:
            lng_ticks = np.arange(np.floor(zoom_lng_min), np.ceil(zoom_lng_max) + 1, 1)
        else:
            lng_ticks = np.arange(np.floor(zoom_lng_min), np.ceil(zoom_lng_max) + 5, 5)
        
        ax_zoom.set_xticks(lng_ticks)
        ax_zoom.set_yticks(lat_ticks)
        ax_zoom.tick_params(axis='both', labelsize=9)
        
        # Overall figure title
        fig.suptitle(f"Round {round_num} | {model_name} | Score: {score} | Error: {distance_km:.1f} km", 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Adjust spacing between subplots
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure
        timestamp = datetime.now().strftime("%H%M%S")
        output_path = output_dir / f"round_{round_num:02d}_{timestamp}_error_map.png"
        plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor='white', edgecolor='none')
        plt.close(fig)
        
        logger.info(f"📍 Saved error map: {output_path.name}")
        return output_path
    except Exception as e:
        logger.warning(f"Failed to create error map: {e}")
        import traceback
        traceback.print_exc()
        return None


@app.route('/api/v1/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(device) if device else "not initialized",
        "log_dir": str(log_dir) if log_dir else None,
        "predictions_logged": prediction_count,
    })


@app.route('/api/v1/predict', methods=['POST'])
def predict():
    """API endpoint for geolocation prediction."""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "Missing 'image' field in request"}), 400

        image_data = data['image']
        result = predict_single_image(image_data)
        logger.info(f"Prediction #{prediction_count}: lat={result['results']['lat']:.4f}, lng={result['results']['lng']:.4f}")

        return jsonify(result)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/new_session', methods=['POST'])
def new_session():
    """Start a new logging session."""
    log_path = init_logging_session()
    return jsonify({
        "status": "ok",
        "log_dir": str(log_path),
    })


@app.route('/api/v1/log_result', methods=['POST'])
def log_result():
    """
    Log a game result with prediction, ground truth, and score.
    
    Computes cell assignments server-side from true location and model checkpoint.
    
    Expected JSON payload:
    {
        "round": 1,
        "pred_lat": 48.8566,
        "pred_lng": 2.3522,
        "true_lat": 48.8584,
        "true_lng": 2.2945,
        "distance_m": 4523.5,
        "score": 4832,
        "model": "stage2_v1"  # optional, defaults to checkpoint name
    }
    """
    global session_csv_path, log_dir, cell_centers
    
    if session_csv_path is None or log_dir is None:
        init_logging_session()
    
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "No JSON data"}), 400
    
    # Required fields
    required = ['round', 'pred_lat', 'pred_lng', 'true_lat', 'true_lng', 'distance_m', 'score']
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"status": "error", "message": f"Missing fields: {missing}"}), 400
    
    # Extract fields
    round_num = data['round']
    pred_lat = data['pred_lat']
    pred_lng = data['pred_lng']
    true_lat = data['true_lat']
    true_lng = data['true_lng']
    distance_km = data['distance_m'] / 1000.0  # Convert to km
    score = data['score']
    model_name = data.get('model', 'unknown')
    
    # Compute cell assignments from true location (server-side)
    true_cell_id = ''
    true_cell_confidence = ''
    pred_cell_id = ''
    pred_cell_confidence = ''
    top_cells_data = None
    
    if cell_centers is not None:
        try:
            # Find cell for true location
            true_cell_id, true_cell_conf = find_nearest_cell(true_lat, true_lng, cell_centers.cpu())
            true_cell_id = str(true_cell_id)
            true_cell_confidence = f"{true_cell_conf:.4f}"
            
            # Find cell for predicted location
            pred_cell_id_val, pred_cell_conf = find_nearest_cell(pred_lat, pred_lng, cell_centers.cpu())
            pred_cell_id = str(pred_cell_id_val)
            pred_cell_confidence = f"{pred_cell_conf:.4f}"
            
            # Prepare top_cells_data for visualization (use true location cell as top cell)
            cell_lats_all, cell_lngs_all = cell_center_to_latlng(cell_centers.cpu())
            
            # Get top-3 cells around true location (for visualization)
            # Find cells closest to true location
            coord_tensor = torch.tensor([[true_lat, true_lng]], dtype=torch.float32)
            xyz = latlng_to_cartesian(coord_tensor)
            xyz = F.normalize(xyz, p=2, dim=1)
            centers = F.normalize(cell_centers.float().cpu(), p=2, dim=1)
            dots = torch.mv(centers, xyz.squeeze())
            top3_cell_probs, top3_cell_idx = dots.topk(3)
            
            # Convert to probabilities (softmax of dot products)
            top3_cell_probs = F.softmax(top3_cell_probs, dim=0)
            
            top_cells_data = {
                'ids': top3_cell_idx.cpu().tolist(),
                'probs': top3_cell_probs.cpu().tolist(),
                'lats': [cell_lats_all[cid].item() for cid in top3_cell_idx.cpu().tolist()],
                'lngs': [cell_lngs_all[cid].item() for cid in top3_cell_idx.cpu().tolist()],
            }
            
            logger.info(f"📍 Computed cells: true_cell={true_cell_id} (conf={true_cell_confidence}), "
                       f"pred_cell={pred_cell_id} (conf={pred_cell_confidence})")
        except Exception as e:
            logger.warning(f"Could not compute cell assignments: {e}")
    
    # Use true location cell for CSV logging (as requested)
    cell_id = true_cell_id
    cell_confidence = true_cell_confidence
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Append to session CSV
    try:
        with open(session_csv_path, 'a') as f:
            f.write(f"{round_num},{timestamp},{model_name},{pred_lat:.6f},{pred_lng:.6f},"
                    f"{true_lat:.6f},{true_lng:.6f},{distance_km:.3f},{score},{cell_id},{cell_confidence}\n")
    except Exception as e:
        logger.error(f"Failed to write to CSV: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    
    logger.info(f"📝 Logged result: Round {round_num} | {model_name} | "
                f"Distance: {distance_km:.1f}km | Score: {score} | True Cell: {cell_id}")
    
    # Optionally create error map visualization (in background)
    save_map = data.get('save_map', True)
    if save_map and log_dir is not None:
        try:
            threading.Thread(
                target=create_error_map_visualization,
                args=(round_num, pred_lat, pred_lng, true_lat, true_lng,
                      distance_km, score, model_name, log_dir, top_cells_data),
                daemon=True
            ).start()
        except Exception as e:
            logger.warning(f"Failed to start error map thread: {e}")
    
    return jsonify({
        "status": "ok",
        "logged": {
            "round": round_num,
            "distance_km": distance_km,
            "score": score,
            "csv_path": str(session_csv_path),
            "true_cell_id": true_cell_id,
            "pred_cell_id": pred_cell_id,
        }
    })


@app.route('/api/v1/checkpoints', methods=['GET'])
def get_checkpoints():
    """Get checkpoint information for logging."""
    global ckpt

    if ckpt is None:
        return jsonify({"error": "No checkpoint loaded"}), 400

    checkpoint_info = {
        "stage1_checkpoint": str(ckpt.get("stage1_checkpoint", "")),
        "stage2_checkpoint": str(ckpt.get("checkpoint_path", ckpt.get("stage2_checkpoint", "")))
    }

    return jsonify(checkpoint_info)


def main():
    parser = argparse.ArgumentParser(description="GeoGuessr Bot API Server")
    parser.add_argument("--checkpoint", type=str, help="Path to Stage 2 checkpoint")
    parser.add_argument("--default-streetclip", action="store_true", 
                       help="Use vanilla HuggingFace StreetCLIP instead of stage2 checkpoint")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--grid-step", type=float, default=5.0, 
                       help="Grid step size in degrees for StreetCLIP (default: 5.0)")

    args = parser.parse_args()

    global device, use_streetclip, streetclip_inference
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    global model, image_encoder, stage1_model, cell_centers, concept_info, ckpt, transform
    
    if args.default_streetclip:
        # Use vanilla HuggingFace StreetCLIP
        logger.info("Loading vanilla HuggingFace StreetCLIP...")
        use_streetclip = True
        streetclip_inference = StreetCLIPInference(device=device, grid_step=args.grid_step)
        
        # Set checkpoint info for logging
        ckpt = streetclip_inference.get_checkpoint_info()
        
        # No transform needed for StreetCLIP (uses processor internally)
        transform = None
        
    else:
        # Use stage2 checkpoint
        if not args.checkpoint:
            raise ValueError("Must provide --checkpoint unless using --default-streetclip")
        
        use_streetclip = False
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model, image_encoder, stage1_model, cell_centers, concept_info, ckpt = load_stage2_checkpoint(
            checkpoint_path, device
        )

        transform = get_transforms_from_processor(image_encoder.image_processor)

    # Initialize logging session
    init_logging_session()

    logger.info("Model loaded successfully. Starting Flask server...")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
