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
    global log_dir, prediction_count

    # Decode base64 image
    if image_data.startswith('data:image'):
        header, encoded = image_data.split(',', 1)
        image_bytes = base64.b64decode(encoded)
    else:
        image_bytes = base64.b64decode(image_data)

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
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

    # Stage 2 forward pass
    outputs = model(concept_embs, patch_tokens, return_attention=False, return_gate=False)
    cell_logits = outputs["cell_logits"]
    pred_offsets = outputs["pred_offsets"]

    pred_cells = cell_logits.argmax(dim=1)
    pred_coords = compute_predicted_coords(pred_cells, pred_offsets, cell_centers, coord_output_dim, device)

    pred_lat = pred_coords[0, 0].item()
    pred_lng = pred_coords[0, 1].item()

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

    # Log prediction in background
    prediction_count += 1
    if log_dir is not None:
        try:
            threading.Thread(
                target=log_prediction_async,
                args=(image.copy(), pred_lat, pred_lng, prediction_count, 
                      meta_probs, parent_probs, concept_info),
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
):
    """Log prediction with concept visualization (runs in background thread)."""
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
            
            if meta_probs is not None and concept_info is not None:
                idx_to_concept = concept_info.get("idx_to_concept", {})
                idx_to_parent = concept_info.get("idx_to_parent", {})
                
                # Top 5 concepts
                probs = meta_probs[0].cpu() if meta_probs.dim() > 1 else meta_probs.cpu()
                top5_probs, top5_idx = torch.topk(probs, k=min(5, len(probs)))
                f.write(f"\nTop 5 Concepts:\n")
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
):
    """Create and save concept prediction visualization."""
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
    
    # Create figure
    n_cols = 3 if parent_probs is not None else 2
    fig = plt.figure(figsize=(6 * n_cols, 6))
    
    # Subplot 1: Input image
    ax1 = fig.add_subplot(1, n_cols, 1)
    ax1.imshow(np.array(image))
    ax1.axis("off")
    ax1.set_title(f"Round {round_num}\nPrediction: ({lat:.4f}, {lng:.4f})", fontsize=10)
    
    # Subplot 2: Top-5 meta concepts
    ax2 = fig.add_subplot(1, n_cols, 2)
    y_pos = np.arange(len(top5_concepts))
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(top5_concepts)))[::-1]
    bars = ax2.barh(y_pos, top5_probs_np[::-1], color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([c[:30] + "..." if len(c) > 30 else c for c in top5_concepts[::-1]], fontsize=9)
    ax2.set_xlabel("Probability")
    ax2.set_title("Top 5 Concept Predictions", fontsize=10)
    ax2.set_xlim(0, 1)
    
    for bar, prob in zip(bars, top5_probs_np[::-1]):
        ax2.text(prob + 0.02, bar.get_y() + bar.get_height()/2, f"{prob:.3f}", va='center', fontsize=8)
    
    # Subplot 3: Top-3 parent concepts (if available)
    if parent_probs is not None and len(top3_parents) > 0:
        ax3 = fig.add_subplot(1, n_cols, 3)
        y_pos = np.arange(len(top3_parents))
        colors = plt.cm.Greens(np.linspace(0.4, 0.8, len(top3_parents)))[::-1]
        bars = ax3.barh(y_pos, top3_parent_probs_np[::-1], color=colors)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([p[:30] + "..." if len(p) > 30 else p for p in top3_parents[::-1]], fontsize=9)
        ax3.set_xlabel("Probability")
        ax3.set_title("Top 3 Parent Concepts", fontsize=10)
        ax3.set_xlim(0, 1)
        
        for bar, prob in zip(bars, top3_parent_probs_np[::-1]):
            ax3.text(prob + 0.02, bar.get_y() + bar.get_height()/2, f"{prob:.3f}", va='center', fontsize=8)
    
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


def main():
    parser = argparse.ArgumentParser(description="GeoGuessr Bot API Server")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Stage 2 checkpoint")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")

    args = parser.parse_args()

    global device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    global model, image_encoder, stage1_model, cell_centers, concept_info, ckpt, transform
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
