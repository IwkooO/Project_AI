#!/usr/bin/env python3
"""
Flask API Server for GeoGuessr Bot Inference

Provides REST API endpoint for geolocation prediction using Stage 2 cross-attention model.
Matches the API contract expected by the Chrome extension (duel.js/classic.js).

Usage:
    python bot/api_server.py --checkpoint /path/to/stage2_checkpoint.pt
"""

import argparse
import base64
import io
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
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
CORS(app)  # Enable CORS for all routes

# Global variables for loaded models
model = None
image_encoder = None
stage1_model = None
cell_centers = None
concept_info = None
ckpt = None
device = None
transform = None


def load_stage2_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple:
    """Load Stage 2 checkpoint."""
    logger.info(f"Loading Stage 2 checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load Stage 1 checkpoint
    stage1_ckpt_path = Path(ckpt["stage1_checkpoint"])
    stage1_ckpt_data = torch.load(stage1_ckpt_path, map_location="cpu", weights_only=False)
    encoder_config = StreetCLIPConfig(
        model_name=ckpt["encoder_model"],
        finetune=False,
        device=device,
    )
    image_encoder = StreetCLIPEncoder(encoder_config)

    # Load image encoder weights
    embedded_enc = stage1_ckpt_data.get("image_encoder_state_dict")
    if embedded_enc is not None:
        logger.info("Using image_encoder_state_dict embedded in Stage1 checkpoint for patch extraction")
        image_encoder.load_state_dict(embedded_enc, strict=False)
    else:
        stage0_checkpoint = stage1_ckpt_data.get("stage0_checkpoint")
        if not is_missing_or_none_path(stage0_checkpoint):
            load_image_encoder_weights_from_stage0_checkpoint(Path(stage0_checkpoint), image_encoder)
        else:
            logger.info("Stage1 indicates vanilla lineage (no Stage0 checkpoint); using base encoder weights")

    image_encoder.model.eval()
    for param in image_encoder.model.parameters():
        param.requires_grad = False

    stage1_model, concept_info = load_stage1_checkpoint(
        stage1_ckpt_path,
        image_encoder,
        device,
    )

    # Create Stage 2 model
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
def predict_single_image(image_data: str) -> Dict:
    """
    Predict location for a single base64-encoded image.

    Args:
        image_data: Base64-encoded image string (data:image/png;base64,...)

    Returns:
        Dict with prediction results: {"results": {"lat": float, "lng": float}}
    """
    global model, image_encoder, stage1_model, cell_centers, ckpt, device, transform

    # Decode base64 image
    if image_data.startswith('data:image'):
        # Remove data URL prefix
        header, encoded = image_data.split(',', 1)
        image_bytes = base64.b64decode(encoded)
    else:
        image_bytes = base64.b64decode(image_data)

    # Convert to PIL Image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Apply transforms
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Get ablation mode and dimensions
    ablation_mode = ckpt.get("ablation_mode", "both")
    patch_dim = ckpt["patch_dim"]
    concept_dim = ckpt["concept_dim"]
    coord_output_dim = ckpt["coord_output_dim"]

    # Compute features + patch tokens
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

    # Forward pass
    outputs = model(concept_embs, patch_tokens, return_attention=False, return_gate=False)
    cell_logits = outputs["cell_logits"]
    pred_offsets = outputs["pred_offsets"]

    # Get predictions
    pred_cells = cell_logits.argmax(dim=1)
    pred_coords = compute_predicted_coords(pred_cells, pred_offsets, cell_centers, coord_output_dim, device)

    # Extract lat/lng
    pred_lat = pred_coords[0, 0].item()
    pred_lng = pred_coords[0, 1].item()

    return {
        "results": {
            "lat": pred_lat,
            "lng": pred_lng
        }
    }


@app.route('/api/v1/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(device) if device else "not initialized"
    })


@app.route('/api/v1/predict', methods=['POST'])
def predict():
    """API endpoint for geolocation prediction."""
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "Missing 'image' field in request"}), 400

    image_data = data['image']
    result = predict_single_image(image_data)
    logger.info(f"Prediction: lat={result['results']['lat']:.4f}, lng={result['results']['lng']:.4f}")

    return jsonify(result)


def main():
    parser = argparse.ArgumentParser(description="GeoGuessr Bot API Server")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to Stage 2 checkpoint (.pt file)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to bind to (default: 5000)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detected if not specified.",
    )

    args = parser.parse_args()

    # Setup device
    global device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    global model, image_encoder, stage1_model, cell_centers, concept_info, ckpt, transform
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model, image_encoder, stage1_model, cell_centers, concept_info, ckpt = load_stage2_checkpoint(
        checkpoint_path, device
    )

    # Get transforms
    transform = get_transforms_from_processor(image_encoder.image_processor)

    logger.info("Model loaded successfully. Starting Flask server...")

    # Start Flask server
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
