#!/usr/bin/env python3
"""Full end-to-end test of the prediction visualization."""

import sys
sys.path.insert(0, '/scratch-shared/pnair/Project_AI')

import base64
from pathlib import Path

# Test image path
TEST_IMAGE = Path("/scratch-shared/pnair/Project_AI/results/geoguessr_game_logs/2025-12-30_08-34-21/round_06_083711_input.png")
OUTPUT_DIR = Path("/scratch-shared/pnair/Project_AI/results/geoguessr_game_logs/test_full")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load image as base64
with open(TEST_IMAGE, 'rb') as f:
    image_bytes = f.read()
image_b64 = base64.b64encode(image_bytes).decode('utf-8')
print(f"✓ Loaded test image: {TEST_IMAGE.name} ({len(image_bytes)} bytes)")

# Import and setup
import torch
print(f"✓ PyTorch available: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"✓ Using device: {device}")

# Load checkpoint
CHECKPOINT = Path("/scratch-shared/pnair/Project_AI/results/stage2_cross_attention_both/2025-12-25_09-13-54/checkpoints/best_model_stage2_xattn.pt")
print(f"Loading checkpoint: {CHECKPOINT}")

# Import after path setup
from bot.api_server import (
    load_stage2_checkpoint, 
    predict_single_image,
    init_logging_session,
    get_transforms_from_processor,
)

# Patch globals
import bot.api_server as api_server
api_server.device = device

# Load model
model, image_encoder, stage1_model, cell_centers, concept_info, ckpt = load_stage2_checkpoint(CHECKPOINT, device)
print(f"✓ Model loaded: {ckpt['num_cells']} cells, {len(concept_info.get('idx_to_concept', {}))} concepts")

# Set globals
api_server.model = model
api_server.image_encoder = image_encoder
api_server.stage1_model = stage1_model
api_server.cell_centers = cell_centers
api_server.concept_info = concept_info
api_server.ckpt = ckpt
api_server.transform = get_transforms_from_processor(image_encoder.image_processor)
api_server.use_streetclip = False

# Set custom log dir for test
api_server.log_dir = OUTPUT_DIR
api_server.prediction_count = 0

print(f"\n✓ Running prediction...")

# Run prediction
result = predict_single_image(image_b64)

print(f"\n✓ Prediction result:")
print(f"   Latitude:  {result['results']['lat']:.6f}")
print(f"   Longitude: {result['results']['lng']:.6f}")

# Wait for async logging to complete
import time
time.sleep(3)

# Check output files
output_files = list(OUTPUT_DIR.glob("*.png"))
print(f"\n✓ Output files in {OUTPUT_DIR}:")
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"   {f.name} ({f.stat().st_size} bytes)")

print(f"\n✓ Test complete!")



