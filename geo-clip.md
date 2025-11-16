# DINOv3 CBM Geolocation Model Implementation

## Architecture Overview

```
Image (3, 224, 224) 
  → DINOv3 Vision Encoder (pretrained, frozen or fine-tunable)
  → Features (batch, 768 for ViT-B)
  → Concept Layer: Linear(768 → num_concepts)
  → Concepts (batch, num_concepts) [logits]
  → Country Head: Linear(num_concepts → num_countries)
  → Coordinate Head: Linear(num_concepts → 2) → normalized lat/lng
```

## Components to Implement

### 1. Dataset Updates (`src/dataset.py`)

- Update `__getitem__` to return coordinates as tensor: `(lat, lng)` normalized to [-1, 1]
- Add coordinate normalization: `lat_norm = lat / 90.0`, `lng_norm = lng / 180.0`
- Handle missing coordinates (skip samples without coordinates or use NaN mask)
- Add DINOv3-compatible transforms (resize to 224x224, ImageNet normalization)
- Return format: `(image_tensor, concept_idx, country_idx, coordinates_tensor, metadata)`
- Filter for South America samples (optional flag)

### 2. DINOv3 Encoder (`src/models/dinov3_encoder.py`)

- Load pretrained DINOv3-ViT-B model (facebook/dinov3-vitb14)
- Extract CLS token features: `features = model.get_intermediate_layers(images, n=1)[0][:, 0]`
- Handle image preprocessing (resize to 518x518, ImageNet normalization)
- Option to freeze encoder or allow fine-tuning
- Feature dimension: 768 for ViT-B

### 3. CBM Model (`src/models/cbm_geolocation.py`)

- **Vision Encoder**: DINOv3 wrapper (frozen or fine-tunable)
- **Concept Layer**: Linear(768, num_concepts) - maps DINOv3 features to concept logits
- **Country Head**: Linear(num_concepts, num_countries) - maps concepts to country logits
- **Coordinate Head**: Linear(num_concepts, 2) - maps concepts to normalized lat/lng
- Forward pass returns: `(concept_logits, country_logits, coordinates)`
- Use tanh activation on coordinate head to ensure [-1, 1] range
- Sequential training support: train concept layer first, then prediction heads

### 4. Loss Functions (`src/losses.py`)

- **Concept Loss**: Cross-entropy (single-label: one concept per image)
- **Distance Loss**: MSE on normalized coordinates (lat/lng both in [-1, 1]) - MAIN LOSS
- **Country Loss**: Cross-entropy for country classification (auxiliary)
- **Combined Loss**: `λ_concept * L_concept + λ_distance * L_distance + λ_country * L_country`
- Default weights: concept=1.0, distance=1.0 (main), country=0.5
- Handle missing coordinates with masking

### 5. Training Script (`scripts/training/train_cbm_geolocation.py`)

- Load dataset with train/val/test splits (filter South America if specified)
- Initialize DINOv3 encoder (frozen or fine-tunable)
- Initialize CBM model (concept layer + prediction heads)
- **Sequential CBM Training**:
  - Stage 1: Train concept layer only (freeze encoder, train concept layer)
  - Stage 2: Train prediction heads (freeze encoder + concept layer, train country + coordinate heads)
  - Stage 3 (optional): End-to-end fine-tuning (all layers trainable)
- Training loop with:
  - Concept loss + distance loss + country loss
  - Separate learning rates for DINOv3 (if fine-tuning, e.g., 1e-5) vs CBM layers (e.g., 1e-3)
  - Validation metrics: MSE, Haversine distance (km), concept accuracy, country accuracy
  - Checkpoint saving (full model + optimizer states)
  - Learning rate scheduling (ReduceLROnPlateau)
- Command-line arguments:
  - DINOv3 variant: `--dinov3_model` (vitb14 default)
  - Fine-tune encoder: `--finetune_encoder` (bool, default False)
  - Loss weights: `--concept_weight`, `--distance_weight`, `--country_weight`
  - Learning rates: `--encoder_lr`, `--cbm_lr`
  - Training: `--batch_size`, `--epochs`, `--data_root`, `--south_america_only`
  - Sequential training: `--sequential` (bool, default True)

### 6. Evaluation Utilities (`src/evaluation.py`)

- **Haversine distance**: Calculate great-circle distance in km between predicted and true coordinates
- **Coordinate denormalization**: Convert from [-1, 1] back to lat/lng degrees
- **Metrics**:
  - MSE, MAE on coordinates
  - Median error (km)
  - Accuracy within X km (e.g., within 1km, 10km, 100km, 1000km)
  - Concept prediction accuracy
  - Country prediction accuracy

### 7. Configuration (`src/config.py`)

- Default hyperparameters
- DINOv3 model variants and feature dimensions mapping
- Default loss weights
- Default learning rates
- Training stage configurations

### 8. SLURM Job File (`jobs/train_cbm_geolocation.job`)

- GPU allocation (DINOv3 needs GPU)
- Environment setup
- Install dependencies if needed (torch, transformers)
- Run training script

## Implementation Details

### Sequential CBM Training Strategy

**Key Principle**: Train concept predictor first, then train task predictor on PREDICTED concepts (not ground truth) to reduce train/test mismatch.

1. **Stage 1 - Concept Predictor Training (g: x → ĉ)**:

   - Train: DINOv3 encoder + Concept layer
   - Objective: Learn g(x) = ĉ that predicts concepts from images
   - Loss: `L_concept = CrossEntropy(g(x^(i)), c^(i))` where c^(i) are ground truth concept labels
   - Goal: Make concept layer as accurate as possible in predicting human-interpretable concepts
   - Use concept labels from dataset (metaName field)
   - After training: **FREEZE** g (DINOv3 encoder + concept layer)

2. **Stage 2 - Task Predictor Training (f: ĉ → y)**:

   - Train: Country head + Coordinate head ONLY
   - Freeze: DINOv3 encoder + Concept layer (g is frozen)
   - **CRITICAL**: Use PREDICTED concepts ĉ = g(x) from Stage 1, NOT ground truth concepts c
   - Objective: Learn f(ĉ) that maps predicted concepts to targets
   - Loss: 
     - `L_coord = MSE(f_coord(g(x^(i))), lat/lng^(i))` - MAIN LOSS
     - `L_country = CrossEntropy(f_country(g(x^(i))), country^(i))` - Auxiliary loss
   - Goal: Learn to predict country and coordinates from predicted concepts
   - **Why predicted concepts?**: Reduces train/test mismatch - f sees same input type at inference

3. **Stage 3 - Fine-tuning** (optional):

   - Unfreeze all layers (g + f)
   - Lower learning rates (e.g., 1e-5 for encoder, 1e-4 for CBM layers)
   - Loss: Combined `L_concept + L_coord + L_country`
   - Goal: End-to-end optimization for final performance boost

### Coordinate Handling

- Normalize: `lat_norm = lat / 90.0`, `lng_norm = lng / 180.0` → range [-1, 1]
- Denormalize: `lat = lat_norm * 90.0`, `lng = lng_norm * 180.0`
- Apply tanh activation on coordinate head output to ensure [-1, 1] range
- Mask loss for samples without coordinates

### Model Outputs

- **Concepts**: `[batch_size, num_concepts]` logits → apply softmax for probabilities
- **Coordinates**: `[batch_size, 2]` normalized (lat, lng) in [-1, 1] range
- **Country**: `[batch_size, num_countries]` logits