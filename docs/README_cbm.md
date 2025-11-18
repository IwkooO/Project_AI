# StreetCLIP CBM Geolocation – Complete System Guide

This document provides a comprehensive explanation of the concept bottleneck model (CBM) geolocation pipeline, from data loading through training to evaluation. It includes detailed explanations of coordinate normalization, residual prediction logic, and complete code examples.

---

## Table of Contents

1. [Data Flow](#1-data-flow)
2. [Coordinate Normalization](#2-coordinate-normalization)
3. [Residual Prediction Logic](#3-residual-prediction-logic-detailed)
4. [Model Architecture](#4-model-architecture)
5. [Training Pipeline](#5-training-pipeline)
6. [Evaluation & Metrics](#6-evaluation--metrics)
7. [Configuration Guide](#7-configuration-guide)
8. [Recommended Workflow](#8-recommended-workflow)

---

## 1. Data Flow

### 1.1 Source Data Structure

Raw panorama data is organized under `data/<geoguessrId>/`:

```
data/6906237dc7731161a37282b2/
├── metas/
│   ├── __IA9LmOI2B0wNzHEDCsyw.json  # Metadata per panorama
│   └── ...
├── panorama_processed/
│   ├── image_<pano_id>.jpg          # 336×336 processed images
│   └── ...
└── panorama_original/                # Original full-resolution images
```

Each metadata JSON file contains:
```json
{
  "metaName": "ACT gap reflector",      # Concept label
  "country": "Australia",                # Country name
  "lat": -35.363,                       # Latitude in degrees
  "lng": 149.167,                       # Longitude in degrees
  "note": "...",                        # Optional notes
  "images": [...]                       # Related image IDs
}
```

### 1.2 Dataset Loading Process

The `PanoramaCBMDataset` class (`src/dataset.py`) performs the following steps:

**Step 1: Load and filter samples**
```python
# From src/dataset.py, lines 213-288
def _load_samples(self) -> List[Dict]:
    samples = []
    for meta_path in meta_folder.glob("*.json"):
        with meta_path.open() as f:
            meta = json.load(f)
        
        # Filter by country if specified
        if self.country and meta['country'] != self.country:
            continue
        
        # Extract coordinates
        lat = meta.get('lat')  # e.g., -35.363
        lng = meta.get('lng')  # e.g., 149.167
        
        sample = {
            'pano_id': meta_path.stem,
            'meta_name': meta['metaName'],
            'country': meta['country'],
            'lat': lat,
            'lng': lng,
        }
        samples.append(sample)
    return samples
```

**Step 2: Build vocabularies**
```python
# From src/dataset.py, lines 334-346
def get_concept_to_idx(samples: List[Dict]) -> Tuple[Dict[str, int], Dict[int, str]]:
    meta_names = sorted(set(s['meta_name'] for s in samples))
    concept_to_idx = {name: i for i, name in enumerate(meta_names)}
    # Example: {"ACT gap reflector": 0, "NSW guardrail ending": 1, ...}
    return concept_to_idx, idx_to_concept
```

**Step 3: Normalize coordinates** (see Section 2 for details)

**Step 4: Create train/val/test splits**
```python
# From src/dataset.py, lines 348-417
train_samples, val_samples, test_samples = create_splits_stratified(
    dataset.samples,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
)
# Returns actual sample dictionaries, not just indices
```

---

## 2. Coordinate Normalization

### 2.1 Why Normalize?

Raw GPS coordinates span large ranges:
- **Latitude**: -90° to +90° (180° total range)
- **Longitude**: -180° to +180° (360° total range)

Neural networks train more stably when inputs/outputs are in a bounded range like `[-1, 1]`.

### 2.2 Normalization Formula

```python
# From src/dataset.py, lines 571-578
def normalize_coordinates(lat: Optional[float], lng: Optional[float]) -> torch.Tensor:
    """Normalize coordinates to [-1, 1] range."""
    if lat is None or lng is None:
        return torch.tensor([float('nan'), float('nan')], dtype=torch.float32)
    
    lat_norm = float(lat) / 90.0   # Divide by max latitude
    lng_norm = float(lng) / 180.0   # Divide by max longitude
    return torch.tensor([lat_norm, lng_norm], dtype=torch.float32)
```

### 2.3 Numeric Example

**Input (degrees):**
- Latitude: `-35.363°` (Canberra, Australia)
- Longitude: `149.167°`

**Normalization:**
```python
lat_norm = -35.363 / 90.0  = -0.393
lng_norm = 149.167 / 180.0 = 0.829
```

**Result:** `[-0.393, 0.829]` (both values in `[-1, 1]`)

### 2.4 Denormalization (for evaluation)

```python
# From src/evaluation.py, lines 14-37
def denormalize_coordinates(coords: torch.Tensor) -> torch.Tensor:
    """Convert normalized coordinates back to degrees."""
    if coords.dim() == 1:
        lat = coords[0] * 90.0   # Multiply by max latitude
        lng = coords[1] * 180.0  # Multiply by max longitude
        return torch.stack([lat, lng])
    else:  # Batch
        lat = coords[:, 0] * 90.0
        lng = coords[:, 1] * 180.0
        return torch.stack([lat, lng], dim=1)
```

**Example:**
- Normalized: `[-0.393, 0.829]`
- Denormalized: `[-0.393 * 90.0, 0.829 * 180.0] = [-35.37°, 149.22°]`

---

## 3. Residual Prediction Logic (Detailed)

### 3.1 The Problem with Small Datasets

When training on only 251 Australian samples, the model can struggle to learn the full coordinate space. Predicting absolute coordinates requires the model to:
1. Learn the concept → location mapping
2. Learn the entire geographic distribution from scratch

**Residual prediction** simplifies this by:
1. Pre-computing the dataset's geographic center (centroid)
2. Having the model predict only **offsets** (deltas) from that center
3. Scaling those offsets by the dataset's spread

### 3.2 Step-by-Step Residual Computation

#### Step 1: Compute Residual Center

```python
# From scripts/training/train_cbm_geolocation.py, lines 292-301
def compute_coordinate_stats(samples: List[Dict]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    # Convert all samples to normalized coordinates
    coord_tensor = coords_tensor_from_samples(samples)
    # coord_tensor shape: (N, 2) where N = number of samples
    
    # Compute mean (centroid) of all normalized coordinates
    center = coord_tensor.mean(dim=0)  # Shape: (2,)
    # center[0] = mean normalized latitude
    # center[1] = mean normalized longitude
    
    # Compute maximum deviation from center
    max_deviation = torch.max(torch.abs(coord_tensor - center), dim=0).values
    # Clamp to minimum 1e-2 to avoid division by zero
    max_deviation = torch.clamp(max_deviation, min=1e-2)
    
    return center, max_deviation
```

**Numeric Example (Australia dataset):**

Assume we have 3 samples with normalized coordinates:
```python
sample_1: [-0.393, 0.829]  # Canberra
sample_2: [-0.378, 0.833]  # Sydney
sample_3: [-0.250, 0.750]  # Melbourne

coord_tensor = torch.tensor([
    [-0.393, 0.829],
    [-0.378, 0.833],
    [-0.250, 0.750]
])

# Compute center (mean)
center = coord_tensor.mean(dim=0)
# center = [-0.340, 0.804]

# Compute max deviation
deviations = torch.abs(coord_tensor - center)
# deviations = [
#     [0.053, 0.025],  # |sample_1 - center|
#     [0.038, 0.029],  # |sample_2 - center|
#     [0.090, 0.054]   # |sample_3 - center|
# ]
max_deviation = torch.max(deviations, dim=0).values
# max_deviation = [0.090, 0.054]
```

**Real values from Australia dataset (251 samples):**
- `center ≈ [-0.355, 0.770]` (approximately `-32°` lat, `138.6°` lng)
- `max_deviation ≈ [0.117, 0.083]` (approximately `±10.5°` lat, `±15°` lng)

#### Step 2: Register Residual Buffers in Model

```python
# From src/models/cbm_geolocation.py, lines 46-57
self.register_buffer(
    "coordinate_residual_center",
    coordinate_residual_center.view(1, -1) if coordinate_residual_center is not None else None,
)
self.register_buffer(
    "coordinate_residual_bounds",
    coordinate_residual_bounds.view(1, -1) if coordinate_residual_bounds is not None else None,
)
```

These are registered as **buffers** (not parameters), meaning:
- They're part of the model state but don't receive gradients
- They're saved/loaded with checkpoints
- They're moved to the correct device (CPU/GPU) automatically

#### Step 3: Forward Pass with Residuals

```python
# From src/models/cbm_geolocation.py, lines 84-116
def forward(self, images: torch.Tensor):
    # ... encoder and concept layers ...
    
    # Coordinate head outputs raw logits
    coord_logits = self.coordinate_head(coord_input)  # Shape: (batch, 2)
    
    if self.coordinate_loss_type == "sphere":
        coordinates = F.normalize(coord_logits, p=2, dim=1)
    else:
        # Apply tanh to constrain to [-1, 1]
        coordinate_delta = torch.tanh(coord_logits)  # Shape: (batch, 2)
        
        if (self.coordinate_residual_center is not None and 
            self.coordinate_residual_bounds is not None):
            # RESIDUAL MODE: predict offset from center, scaled by bounds
            coordinates = torch.clamp(
                self.coordinate_residual_center + 
                coordinate_delta * self.coordinate_residual_bounds,
                -1.0, 1.0
            )
        else:
            # STANDARD MODE: predict absolute coordinates
            coordinates = coordinate_delta
    
    return concept_logits, country_logits, coordinates
```

### 3.3 Complete Numeric Example

**Setup:**
- Residual center: `center = [-0.355, 0.770]`
- Residual bounds: `bounds = [0.117, 0.083]`
- Model outputs: `coord_logits = [0.5, -0.3]` (raw logits)

**Step-by-step computation:**

1. **Apply tanh to logits:**
   ```python
   coordinate_delta = torch.tanh([0.5, -0.3])
   # = [0.462, -0.291]  (tanh maps to [-1, 1])
   ```

2. **Scale delta by bounds:**
   ```python
   scaled_delta = coordinate_delta * bounds
   # = [0.462 * 0.117, -0.291 * 0.083]
   # = [0.054, -0.024]
   ```

3. **Add to center:**
   ```python
   coordinates = center + scaled_delta
   # = [-0.355 + 0.054, 0.770 + (-0.024)]
   # = [-0.301, 0.746]
   ```

4. **Clamp to [-1, 1]:**
   ```python
   coordinates = torch.clamp([-0.301, 0.746], -1.0, 1.0)
   # = [-0.301, 0.746]  (already in range)
   ```

5. **Denormalize for evaluation:**
   ```python
   lat = -0.301 * 90.0  = -27.09°
   lng = 0.746 * 180.0  = 134.28°
   ```

**Final prediction:** `(-27.09°, 134.28°)` in Australia

### 3.4 Why Residuals Help

**Without residuals:**
- Model must learn: "ACT gap reflector" → `(-35.363°, 149.167°)`
- Output space: entire `[-1, 1] × [-1, 1]` range
- With 251 samples, model struggles to cover the full space

**With residuals:**
- Model learns: "ACT gap reflector" → small offset from center `[-0.355, 0.770]`
- Output space: `center ± bounds` = `[-0.355 ± 0.117, 0.770 ± 0.083]`
- Effectively a smaller, more constrained space
- Model only needs to learn **relative** positions, not absolute geography

### 3.5 Training Configuration

```python
# From scripts/training/train_cbm_geolocation.py, lines 976-995
if args.use_coordinate_residuals:
    if args.residual_stats_source == "train":
        stats_samples = train_samples  # Use only training split
        stats_name = "train split"
    else:
        stats_samples = dataset.samples  # Use full dataset
        stats_name = "full dataset"
    
    stats = compute_coordinate_stats(stats_samples)
    if stats is None:
        logger.warning("Unable to compute coordinate residual stats")
    else:
        residual_center, residual_bounds = stats
        logger.info(f"Coordinate residual center ({stats_name}): "
                   f"lat={residual_center[0]:.3f}, lng={residual_center[1]:.3f}")
        logger.info(f"Coordinate residual bounds: "
                   f"lat<=±{residual_bounds[0]:.3f}, lng<=±{residual_bounds[1]:.3f}")
```

**CLI flags:**
- `--use_coordinate_residuals`: Enable residual mode
- `--residual_stats_source {train,dataset}`: Compute stats from train split or full dataset

---

## 4. Model Architecture

### 4.1 Complete Forward Pass

```python
# Simplified version of src/models/cbm_geolocation.py forward()
def forward(self, images: torch.Tensor):
    # 1. Encode image
    features = self.encoder(images)  # Shape: (batch, feature_dim)
    
    # 2. Concept prediction
    concept_logits = self.concept_layer(features)  # Shape: (batch, num_concepts)
    concept_probs = F.softmax(concept_logits, dim=1)
    
    # 3. Country prediction (from concept logits)
    country_logits = self.country_head(concept_logits)  # Shape: (batch, num_countries)
    
    # 4. Coordinate prediction
    # Option A: Use concept probabilities (default)
    coord_input = concept_probs if self.coordinate_input == "probs" else concept_logits
    
    # Option B: Detach to prevent gradients flowing back
    if self.detach_concepts_for_prediction:
        coord_input = coord_input.detach()
    
    # Option C: Add encoder feature skip connection
    if self.feature_skip is not None:
        projected_features = self.feature_skip(features)  # Shape: (batch, skip_dim)
        coord_input = torch.cat([coord_input, projected_features], dim=1)
    
    # 5. Coordinate head
    coord_logits = self.coordinate_head(coord_input)  # Shape: (batch, 2)
    
    # 6. Apply residual logic (if enabled)
    # ... (see Section 3.3)
    
    return concept_logits, country_logits, coordinates
```

### 4.2 Architecture Diagram

```
Input Image (336×336×3)
    ↓
[Encoder: StreetCLIP/DINO]  ← frozen by default, can be fine-tuned
    ↓
Features (batch, 1024)  ← feature_dim
    ↓
┌─────────────────────────────────────┐
│  Concept Layer (Linear)             │
│  (1024 → 32 concepts)                │
└─────────────────────────────────────┘
    ↓
Concept Logits (batch, 32)
    ↓
    ├─→ [Softmax] → Concept Probs (batch, 32)
    │
    ├─→ Country Head (Linear: 32 → 1) → Country Logits
    │
    └─→ Coordinate Head Input
            ↓
        [Optional: Feature Skip]
        (1024 → 256 via GELU+LayerNorm)
            ↓
        Concatenate: [Concept Probs, Projected Features]
            ↓
        Coordinate Head (MLP: 32+256 → 256 → 128 → 2)
            ↓
        Coord Logits (batch, 2)
            ↓
        [Tanh] → Coordinate Delta [-1, 1]
            ↓
        [Residual: center + delta * bounds] → Final Coordinates
```

### 4.3 Stage-Aware Training

The model supports **sequential training stages** where different components are trained at different times:

```python
# From src/models/cbm_geolocation.py, lines 147-190
def set_stage(self, stage: str, finetune_encoder: bool = False,
              train_prediction_head: bool = False,
              train_country_head: bool = False):
    if stage == "concept":
        # Train only concept layer (and optionally encoder)
        self.concept_layer.requires_grad = True
        self.country_head.requires_grad = train_country_head
        self.coordinate_head.requires_grad = train_prediction_head
        
    elif stage == "prediction":
        # Freeze concept layer, train prediction heads
        self.concept_layer.requires_grad = False
        self.country_head.requires_grad = True
        self.coordinate_head.requires_grad = True
        
    elif stage == "finetune":
        # Train everything end-to-end
        for param in self.parameters():
            param.requires_grad = True
```

**Stage schedule:**
1. **Concept stage** (epochs 1-5): Train concept layer to recognize visual concepts
2. **Prediction stage** (epochs 6-10): Freeze concepts, train country/coordinate heads
3. **Finetune stage** (optional): Joint training of all components

---

## 5. Training Pipeline

### 5.1 Complete Training Script Flow

```python
# From scripts/training/train_cbm_geolocation.py main()
def main():
    # 1. Parse arguments
    args = parse_args()
    
    # 2. Setup: seeds, checkpoint dir, logging, W&B
    # ...
    
    # 3. Load dataset
    dataset = PanoramaCBMDataset(
        country=args.country_filter,
        require_coordinates=args.require_coordinates,
        encoder_model=args.encoder_model,
    )
    
    # 4. Create splits
    train_loader, val_loader, test_loader, train_samples, val_samples, test_samples = \
        build_dataloaders(...)
    
    # 5. Compute residual stats (if enabled)
    if args.use_coordinate_residuals:
        residual_center, residual_bounds = compute_coordinate_stats(...)
    
    # 6. Create model
    model = CBMGeolocationModel(
        encoder=encoder,
        num_concepts=len(dataset.concept_to_idx),
        num_countries=len(dataset.country_to_idx),
        coordinate_residual_center=residual_center,
        coordinate_residual_bounds=residual_bounds,
        ...
    )
    
    # 7. Training loop over stages
    for stage_name, epochs in stages:
        model.set_stage(stage_name, ...)
        optimizer = optimizer_for_stage(model, stage_name, args)
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss = train_one_epoch(...)
            
            # Validate
            val_loss, val_metrics = evaluate(...)
            
            # Log to W&B
            wandb.log({f"{stage_name}/val_loss": val_loss, ...})
            
            # Visualize (every 5 epochs)
            if epoch % 5 == 0:
                visualize_predictions(...)
            
            # Diagnostics (every N epochs)
            if epoch % args.diagnostics_interval == 0:
                dump_coordinate_diagnostics(...)
    
    # 8. Final test evaluation
    test_loss, test_metrics = evaluate(test_loader, ...)
```

### 5.2 Loss Computation

```python
# From src/losses.py, lines 61-82
def combined_loss(concept_logits, country_logits, predicted_coords,
                  concept_targets, country_targets, coordinate_targets,
                  weights: LossWeights, coordinate_loss_type: str):
    losses = {}
    
    # Concept classification loss
    losses["concept"] = F.cross_entropy(concept_logits, concept_targets) * weights.concept
    
    # Country classification loss
    losses["country"] = F.cross_entropy(country_logits, country_targets) * weights.country
    
    # Coordinate regression loss
    if coordinate_loss_type == "haversine":
        distances = haversine_distance(predicted_coords, coordinate_targets)
        losses["distance"] = distances.mean() * weights.distance
    elif coordinate_loss_type == "mse":
        losses["distance"] = F.mse_loss(predicted_coords, coordinate_targets) * weights.distance
    
    total = sum(losses.values())
    return total, losses
```

**Stage-specific weights:**
```python
# From scripts/training/train_cbm_geolocation.py, lines 308-326
def resolve_stage_loss_weights(stage, base_weights, 
                                concept_stage_distance_weight,
                                concept_stage_country_weight):
    if stage == "concept":
        # Only concept loss (or optionally small coordinate loss)
        return LossWeights(
            concept=base_weights.concept,
            distance=concept_stage_distance_weight,  # Usually 0.0 or 0.2
            country=concept_stage_country_weight,   # Usually 0.0
        )
    elif stage == "prediction":
        # Only prediction losses
        return LossWeights(
            concept=0.0,
            distance=base_weights.distance,
            country=base_weights.country,
        )
    else:  # finetune
        return base_weights  # All losses active
```

---

## 6. Evaluation & Metrics

### 6.1 Automatic Metrics

```python
# From src/evaluation.py, lines 103-153
def compute_geolocation_metrics(...):
    metrics = {}
    
    # Classification metrics
    metrics["concept_accuracy"] = (concept_logits.argmax(dim=1) == concept_targets).float().mean()
    metrics["country_accuracy"] = (country_logits.argmax(dim=1) == country_targets).float().mean()
    
    # Coordinate error metrics
    mask = ~torch.isnan(coordinate_targets).any(dim=1)
    if mask.sum() > 0:
        # Normalized MSE/MAE
        mse = torch.mean((predicted_coords[mask] - coordinate_targets[mask]) ** 2)
        mae = torch.mean(torch.abs(predicted_coords[mask] - coordinate_targets[mask]))
        metrics["coord_mse"] = mse.item()
        metrics["coord_mae"] = mae.item()
        
        # Haversine distance (great-circle distance in km)
        distances = haversine_distance(predicted_coords, coordinate_targets)
        metrics["median_km"] = distances.median().item()
        metrics["mean_km"] = distances.mean().item()
        metrics["p90_km"] = distances.quantile(0.9).item()
        
        # Accuracy at thresholds
        for threshold in [1, 10, 100, 1000]:
            metrics[f"acc@{threshold}km"] = (distances <= threshold).float().mean().item()
        
        # Bias statistics
        pred_deg = denormalize_coordinates(predicted_coords[mask])
        true_deg = denormalize_coordinates(coordinate_targets[mask])
        lat_bias = (pred_deg[:, 0] - true_deg[:, 0]).mean().item()
        lng_bias = (pred_deg[:, 1] - true_deg[:, 1]).mean().item()
        lat_std = (pred_deg[:, 0] - true_deg[:, 0]).std().item()
        lng_std = (pred_deg[:, 1] - true_deg[:, 1]).std().item()
        metrics["lat_bias_deg"] = lat_bias
        metrics["lng_bias_deg"] = lng_bias
        metrics["lat_std_deg"] = lat_std
        metrics["lng_std_deg"] = lng_std
    
    return metrics
```

### 6.2 Metric Interpretation

**Classification metrics:**
- `concept_accuracy`: Fraction of samples where top predicted concept matches ground truth
- `country_accuracy`: Fraction of samples where predicted country matches (usually 1.0 for single-country datasets)

**Distance metrics:**
- `median_km`: Median Haversine distance (robust to outliers)
- `mean_km`: Mean Haversine distance
- `p90_km`: 90th percentile distance (captures worst-case performance)
- `acc@1km`: Fraction of predictions within 1 km of ground truth

**Bias metrics:**
- `lat_bias_deg`: Average error in latitude (positive = predicted too far north)
- `lng_bias_deg`: Average error in longitude (positive = predicted too far east)
- `lat_std_deg` / `lng_std_deg`: Standard deviation of errors (measures consistency)

**Example interpretation:**
```
concept_accuracy: 0.3341  → 33.4% of concepts correctly predicted
median_km: 13135.2920      → Median error is 13,135 km (very poor!)
lat_bias_deg: -2.2        → Predictions are 2.2° too far south on average
lng_std_deg: 10.4         → Longitude predictions vary by ±10.4° (high uncertainty)
```

### 6.3 Diagnostics CSV

```python
# From scripts/training/train_cbm_geolocation.py, lines 718-817
def dump_coordinate_diagnostics(...):
    rows = []
    for batch in dataloader:
        # ... get predictions ...
        for i in range(len(images)):
            rows.append({
                "pano_id": metadata[i]["pano_id"],
                "pred_lat": float(pred_deg[0].item()),
                "pred_lng": float(pred_deg[1].item()),
                "true_lat": float(true_deg[0].item()),
                "true_lng": float(true_deg[1].item()),
                "distance_km": distance_km,
                "pred_country": pred_country,
                "true_country": true_country,
                "top_concept": top_concept,
                "true_concept": true_concept,
                "concept_correct": bool(top_concept_idx == concept_idx[i].item()),
            })
    
    # Write to CSV
    with output_path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
```

**Example CSV row:**
```csv
pano_id,pred_lat,pred_lng,true_lat,true_lng,distance_km,pred_country,true_country,top_concept,true_concept,concept_correct
__IA9LmOI2B0wNzHEDCsyw,-8.003,3.212,-35.363,149.167,14016.6,Australia,Australia,ACT gap reflector,ACT gap reflector,True
```

---

## 7. Configuration Guide

### 7.1 Key Arguments

**Data selection:**
```bash
--country_filter Australia          # Filter to specific country
--require_coordinates               # Skip samples without GPS
--max_samples 100                   # Limit for debugging
```

**Coordinate representation:**
```bash
--coordinate_loss_type haversine    # Loss: mse, haversine, or sphere
--concept_to_coord_input probs      # Input: logits or probs
--coordinate_feature_skip_dim 256  # Encoder skip connection size (0 to disable)
```

**Residual mode:**
```bash
--use_coordinate_residuals          # Enable residual prediction
--residual_stats_source train       # Compute stats from: train or dataset
```

**Training stages:**
```bash
--sequential                        # Use sequential training (default)
--concept_epochs 5                  # Epochs for concept stage
--prediction_epochs 5               # Epochs for prediction stage
--finetune_epochs 0                 # Epochs for finetune stage
```

**Joint training (concept stage):**
```bash
--concept_stage_distance_weight 0.2  # Coordinate loss weight during concept stage
--concept_stage_country_weight 0.0   # Country loss weight during concept stage
```

**Learning rates:**
```bash
--cbm_lr 1e-3                       # Default LR for CBM layers
--coordinate_head_lr 5e-4           # Override for coordinate head
--country_head_lr 5e-4              # Override for country head
--encoder_lr 1e-5                   # LR for encoder (if fine-tuning)
```

**Regularization:**
```bash
--grad_clip_norm 1.0                # Gradient clipping threshold
--retain_coord_grad_through_concepts # Allow gradients through concept layer
```

**Logging:**
```bash
--diagnostics_interval 2             # Dump diagnostics every N epochs
--diagnostics_samples 128            # Max samples per diagnostics dump
--checkpoint_interval 1              # Save checkpoint every N epochs
--no_wandb                          # Disable W&B logging
```

### 7.2 Example Job Configuration

```bash
# From jobs/train_cbm_geolocation.job
python scripts/training/train_cbm_geolocation.py \
  --batch_size 16 \
  --concept_epochs 5 \
  --prediction_epochs 5 \
  --encoder_model "geolocal/StreetCLIP" \
  --country_filter "Australia" \
  --coordinate_loss_type "haversine" \
  --use_coordinate_residuals \
  --residual_stats_source "train" \
  --concept_stage_distance_weight 0.2 \
  --coordinate_head_lr 5e-4 \
  --grad_clip_norm 1.0 \
  --diagnostics_interval 2 \
  --diagnostics_samples 128 \
  --stratified_concept_sampling \
  --sequential
```

---

## 8. Recommended Workflow

### 8.1 Pre-Training Validation

```bash
# Activate environment
module load 2025
module load Anaconda3/2025.06-1
cd /scratch-shared/pnair/Project_AI/
module load CUDA/12.8.0
source activate streetview_pnair
export PYTHONNOUSERSITE=1

# Validate dataset
PYTHONPATH=/scratch-shared/pnair/Project_AI python \
  scripts/tests/validate_australia_coords.py \
  --output-dir data/export

# Check outputs
# - data/export/australia_coords.csv: Coordinate table
# - data/export/australia_coords.png: Scatter plot
```

### 8.2 Launch Training

```bash
# Submit job
sbatch jobs/train_cbm_geolocation.job

# Monitor logs
tail -f jobs/outputs/train_cbm_geolocation_streetclip.log
```

### 8.3 Post-Training Analysis

1. **Check W&B dashboard:**
   - `concept/val_concept_accuracy`: Should increase during concept stage
   - `prediction/val_median_km`: Should decrease during prediction stage
   - `prediction/val_lat_bias_deg`, `prediction/val_lng_bias_deg`: Check for systematic errors

2. **Inspect diagnostics CSVs:**
   ```bash
   # Find worst predictions
   cat results/.../diagnostics/prediction/epoch_5.csv | \
     sort -t, -k6 -rn | head -20
   ```

3. **Review visualizations:**
   ```bash
   # View prediction samples
   ls results/.../visualizations/prediction/
   ```

### 8.4 Iteration

Based on diagnostics:
- **High coordinate errors**: Try increasing `--concept_stage_distance_weight` or `--prediction_epochs`
- **Concept accuracy low**: Increase `--concept_epochs` or check dataset quality
- **Systematic bias**: Residuals may need adjustment; check `residual_stats_source`
- **Overfitting**: Reduce learning rates or add more data

---

## Appendix: Quick Reference

### Coordinate Transformations

| Operation | Formula | Example |
|-----------|---------|---------|
| Normalize | `lat_norm = lat / 90.0`<br>`lng_norm = lng / 180.0` | `(-35.363°, 149.167°)` → `(-0.393, 0.829)` |
| Denormalize | `lat = lat_norm * 90.0`<br>`lng = lng_norm * 180.0` | `(-0.393, 0.829)` → `(-35.37°, 149.22°)` |
| Residual (with) | `coord = center + tanh(logits) * bounds` | `center=[-0.355, 0.770]`, `bounds=[0.117, 0.083]` |
| Residual (without) | `coord = tanh(logits)` | Direct prediction |

### File Locations

| Component | Path |
|-----------|------|
| Training script | `scripts/training/train_cbm_geolocation.py` |
| Model definition | `src/models/cbm_geolocation.py` |
| Dataset loader | `src/dataset.py` |
| Evaluation utils | `src/evaluation.py` |
| Loss functions | `src/losses.py` |
| Job script | `jobs/train_cbm_geolocation.job` |
| Checkpoints | `results/<experiment>/checkpoints/` |
| Diagnostics | `results/<experiment>/diagnostics/<stage>/epoch_*.csv` |
| Visualizations | `results/<experiment>/visualizations/<stage>/epoch_*_sample_*.png` |

---

This guide should provide a complete understanding of the CBM geolocation system. For dataset collection procedures, see `docs/README_dataset.md`. For project structure, see `docs/README_project.md`.
