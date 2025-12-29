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
)
from bot.streetclip_inference import StreetCLIPInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
prediction_count = 0
session_start_time = None


def init_logging_session():
    """Initialize a new logging session with timestamp directory."""
    global log_dir, session_start_time, prediction_count
    
    session_start_time = datetime.now()
    timestamp = session_start_time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path("/scratch-shared/pnair/Project_AI/results/geoguessr_game_logs") / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    prediction_count = 0
    
    logger.info(f"📊 Logging session started: {log_dir}")
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
                          None, None, None),  # No concepts for vanilla StreetCLIP
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

    # Log prediction in background
    prediction_count += 1
    if log_dir is not None:
        try:
            threading.Thread(
                target=log_prediction_async,
                args=(image.copy(), pred_lat, pred_lng, prediction_count, 
                      meta_probs, parent_probs, concept_info,
                      gate_stats, attn_stats, cell_confidence, 
                      top3_cell_idx[0].cpu().tolist(), top3_cell_probs[0].cpu().tolist()),
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
                )
            except Exception as e:
                logger.warning(f"Failed to create visualization: {e}")
        
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
):
    """Create and save concept prediction visualization with contribution diagnostics."""
    idx_to_concept = concept_info.get("idx_to_concept", {})
    idx_to_parent = concept_info.get("idx_to_parent", {})
    
    # Get top-5 meta concepts
    top5_probs, top5_indices = torch.topk(meta_probs, k=min(5, len(meta_probs)))
    top5_concepts = [idx_to_concept.get(idx.item(), f"concept_{idx.item()}") for idx in top5_indices]
    top5_probs_np = top5_probs.numpy()
    
    # Get top-3 parent concepts
    top3_parents = []
    top3_parent_probs_np = []
    if parent_probs is not None:
        top3_parent_probs, top3_parent_indices = torch.topk(parent_probs, k=min(3, len(parent_probs)))
        top3_parents = [idx_to_parent.get(idx.item(), f"parent_{idx.item()}") for idx in top3_parent_indices]
        top3_parent_probs_np = top3_parent_probs.numpy()
    
    # Create figure with 2 rows: top row for diagnostics, bottom row for concepts
    fig = plt.figure(figsize=(18, 10))
    
    # ========== TOP ROW: Image + Gate Contribution ==========
    # Subplot 1: Input image with prediction
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.imshow(np.array(image))
    ax1.axis("off")
    title_text = f"Round {round_num}\n({lat:.4f}, {lng:.4f})"
    if cell_confidence is not None:
        title_text += f"\nConf: {cell_confidence:.2f}"
    ax1.set_title(title_text, fontsize=10)
    
    # Subplot 2: Gate contribution gauge (Concept vs Image)
    ax2 = fig.add_subplot(2, 4, 2)
    if gate_stats is not None:
        gate_mean = gate_stats['mean']
        # Create a horizontal bar showing concept vs image contribution
        ax2.barh([0], [gate_mean], color='#2196F3', height=0.5, label='Concept')
        ax2.barh([0], [1 - gate_mean], left=[gate_mean], color='#FF9800', height=0.5, label='Image')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(-0.5, 0.5)
        ax2.set_yticks([])
        ax2.set_xlabel("Contribution Weight")
        ax2.set_title(f"Concept vs Image Contribution\n(Gate Mean: {gate_mean:.3f})", fontsize=10, fontweight='bold')
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=9)
        
        # Add interpretation text
        if gate_mean > 0.6:
            interp = "CONCEPT-heavy"
            color = '#2196F3'
        elif gate_mean < 0.4:
            interp = "IMAGE-heavy"
            color = '#FF9800'
        else:
            interp = "BALANCED"
            color = '#4CAF50'
        ax2.text(0.5, 0.3, interp, ha='center', va='center', fontsize=12, fontweight='bold', color=color)
    else:
        ax2.text(0.5, 0.5, "Gate stats\nnot available", ha='center', va='center', fontsize=10)
        ax2.axis("off")
    
    # Subplot 3: Gate distribution details
    ax3 = fig.add_subplot(2, 4, 3)
    if gate_stats is not None:
        stats_text = (
            f"Gate Statistics:\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"Mean:  {gate_stats['mean']:.4f}\n"
            f"Std:   {gate_stats['std']:.4f}\n"
            f"Min:   {gate_stats['min']:.4f}\n"
            f"Max:   {gate_stats['max']:.4f}\n"
            f"━━━━━━━━━━━━━━━━\n\n"
            f"Gate > 0.5 = Concepts\n"
            f"Gate < 0.5 = Images"
        )
        ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax3.axis("off")
    ax3.set_title("Contribution Details", fontsize=10)
    
    # Subplot 4: Attention statistics
    ax4 = fig.add_subplot(2, 4, 4)
    if attn_stats is not None:
        attn_text = (
            f"Attention Statistics:\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"Entropy:     {attn_stats['entropy']:.4f}\n"
            f"Max Patch:   {attn_stats['max_patch_idx']}\n"
            f"Max Value:   {attn_stats['max_attention']:.4f}\n"
            f"━━━━━━━━━━━━━━━━━━━\n\n"
            f"High entropy = diffuse\n"
            f"Low entropy = focused"
        )
        ax4.text(0.1, 0.9, attn_text, transform=ax4.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    else:
        ax4.text(0.5, 0.5, "Attention stats\nnot available", ha='center', va='center', fontsize=10)
    ax4.axis("off")
    ax4.set_title("Attention Details", fontsize=10)
    
    # ========== BOTTOM ROW: Concept Predictions ==========
    # Subplot 5: Top-5 meta concepts (spanning 2 columns)
    ax5 = fig.add_subplot(2, 2, 3)
    y_pos = np.arange(len(top5_concepts))
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(top5_concepts)))[::-1]
    bars = ax5.barh(y_pos, top5_probs_np[::-1], color=colors)
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels([c[:40] + "..." if len(c) > 40 else c for c in top5_concepts[::-1]], fontsize=9)
    ax5.set_xlabel("Probability")
    ax5.set_title("Top 5 Child Concept Predictions", fontsize=10, fontweight='bold')
    ax5.set_xlim(0, 1)
    
    for bar, prob in zip(bars, top5_probs_np[::-1]):
        ax5.text(prob + 0.02, bar.get_y() + bar.get_height()/2, f"{prob:.3f}", va='center', fontsize=8)
    
    # Subplot 6: Top-3 parent concepts (spanning 2 columns)
    ax6 = fig.add_subplot(2, 2, 4)
    if parent_probs is not None and len(top3_parents) > 0:
        y_pos = np.arange(len(top3_parents))
        colors = plt.cm.Greens(np.linspace(0.4, 0.8, len(top3_parents)))[::-1]
        bars = ax6.barh(y_pos, top3_parent_probs_np[::-1], color=colors)
        ax6.set_yticks(y_pos)
        ax6.set_yticklabels([p[:40] + "..." if len(p) > 40 else p for p in top3_parents[::-1]], fontsize=9)
        ax6.set_xlabel("Probability")
        ax6.set_title("Top 3 Parent Concept Predictions", fontsize=10, fontweight='bold')
        ax6.set_xlim(0, 1)
        
        for bar, prob in zip(bars, top3_parent_probs_np[::-1]):
            ax6.text(prob + 0.02, bar.get_y() + bar.get_height()/2, f"{prob:.3f}", va='center', fontsize=8)
    else:
        ax6.text(0.5, 0.5, "Parent concepts\nnot available", ha='center', va='center', fontsize=10)
        ax6.axis("off")
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / f"round_{round_num:02d}_{timestamp}_concepts.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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


# Storage for true locations from Tampermonkey
true_location_store = {}

@app.route('/api/v1/true_location', methods=['POST', 'GET'])
def true_location():
    """
    Receive true location from Tampermonkey script (POST)
    or retrieve latest true location (GET).
    """
    global true_location_store

    if request.method == 'POST':
        data = request.get_json()
        if data and 'true_lat' in data and 'true_lng' in data:
            true_location_store = {
                'true_lat': data['true_lat'],
                'true_lng': data['true_lng'],
                'timestamp': data.get('timestamp', datetime.now().timestamp() * 1000)
            }
            logger.info(f"📍 Received true location: ({data['true_lat']:.6f}, {data['true_lng']:.6f})")
            return jsonify({"status": "ok", "received": true_location_store})
        return jsonify({"status": "error", "message": "Missing lat/lng"}), 400

    else:  # GET
        if true_location_store:
            result = true_location_store.copy()
            true_location_store = {}  # Clear after reading
            return jsonify({"status": "ok", "data": result})
        return jsonify({"status": "ok", "data": None})


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
