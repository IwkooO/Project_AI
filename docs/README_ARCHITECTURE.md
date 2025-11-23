# Concept-Aware Global Image-GPS Alignment Architecture

## Overview

This document describes the architecture of a **Concept-Aware Global Image-GPS Alignment Model** that learns to geolocate images by understanding semantic concepts (e.g., "urban street", "rural landscape", "coastal area") and their relationship to geographic locations. The model uses a Concept Bottleneck Model (CBM) approach where images are first mapped to a concept space, then concepts are used to predict geographic coordinates.

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Data Pipeline](#data-pipeline)
3. [Model Architecture](#model-architecture)
4. [Training Process](#training-process)
5. [Loss Functions](#loss-functions)
6. [Semantic Geocells](#semantic-geocells)
7. [Dataset Structure](#dataset-structure)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                                │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────┐
│   Dataset    │  (PanoramaCBMDataset or CBMDataset)
│  - Images    │  - Each sample: (image, concept_name, country, lat, lng, note)
│  - Concepts  │
│  - Coords    │
└──────┬───────┘
       │
       ├─────────────────────────────────────────────────────┐
       │                                                       │
       ▼                                                       ▼
┌──────────────┐                                    ┌──────────────────┐
│ Concept      │                                    │ Semantic         │
│ Extraction   │                                    │ Geocell          │
│              │                                    │ Generation       │
│ - Extract    │                                    │                  │
│   unique     │                                    │ - Per-country    │
│   concepts   │                                    │   K-Means        │
│   (meta_name)│                                    │ - Cell centers   │
│ - Map to     │                                    │   in 3D space    │
│   notes      │                                    │ - Assign samples │
│              │                                    │   to cells       │
└──────┬───────┘                                    └────────┬─────────┘
       │                                                     │
       ▼                                                     ▼
┌──────────────┐                                    ┌──────────────────┐
│ Concept      │                                    │ Cell Centers     │
│ Encoding     │                                    │ [N_cells, 3]     │
│              │                                    │ Sample-to-Cell   │
│ - Text       │                                    │ Mapping          │
│   encoder    │                                    │                  │
│   (StreetCLIP)│                                   └──────────────────┘
│ - Encode     │
│   concept    │
│   notes      │
│ - E_concept  │
│   [k, d]     │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    ConceptAwareGeoModel                                  │
│                                                                          │
│  ┌──────────────┐              ┌──────────────┐                         │
│  │ Image Path   │              │ Location Path│                         │
│  │              │              │              │                         │
│  │ Image [B,3,H,W]             │ GPS [B,2]   │                         │
│  │      │                       │      │       │                         │
│  │      ▼                       │      ▼       │                         │
│  │ StreetCLIP   │              │ LocationEncoder│                        │
│  │ Encoder      │              │ (GeoCLIP)     │                         │
│  │      │                       │      │       │                         │
│  │      ▼                       │      ▼       │                         │
│  │ x_img [B,d]  │              │ x_loc [B,512] │                         │
│  │      │                       │      │       │                         │
│  │      ├───────────────────────┼──────┘       │                         │
│  │      │                       │               │                         │
│  │      ▼                       │               │                         │
│  │ Image        │              │ Location       │                         │
│  │ Projector    │              │ Adapter        │                         │
│  │      │                       │      │         │                         │
│  │      ▼                       │      ▼         │                         │
│  │ z_img [B,k]  │              │ x_loc [B,d]    │                         │
│  │      │                       │      │         │                         │
│  │      │                       │      ▼         │                         │
│  │      │                       │ Concept Basis  │                         │
│  │      │                       │ B = E + Δ     │                         │
│  │      │                       │      │         │                         │
│  │      │                       │      ▼         │                         │
│  │      │                       │ z_loc [B,k]   │                         │
│  │      │                       │               │                         │
│  │      ├───────────────────────┘               │                         │
│  │      │                                       │                         │
│  │      ▼                                       │                         │
│  │ Fused Features [B, k+d]                      │                         │
│  │      │                                       │                         │
│  │      ├──────────┬──────────┬──────────────┐ │                         │
│  │      │          │          │              │ │                         │
│  │      ▼          ▼          ▼              ▼ │                         │
│  │  Cell Head  Offset Head Country Head    │  │                         │
│  │  [B,N_cells][B,2/3]    [B,C]           │  │                         │
│  └──────────────┴──────────┴──────────────┴──┘                         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         LOSS COMPUTATION                                 │
│                                                                          │
│  - Contrastive Loss (z_img, z_loc)                                      │
│  - Concept Divergence Loss                                              │
│  - Concept Classification Loss                                          │
│  - Country Classification Loss                                          │
│  - Semantic Reconstruction Loss                                         │
│  - Cell Classification Loss                                              │
│  - Offset Regression Loss                                               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Pipeline

### Dataset Classes

#### 1. PanoramaCBMDataset (`src/dataset.py`)

Loads panorama images from GeoGuessr data structure:

```
data/
└── {geoguessr_id}/
    ├── panorama/          (or panorama_processed/)
    │   ├── image_123.jpg
    │   └── ...
    └── metas/
        ├── 123.json
        └── ...
```

**Sample Structure:**
```python
{
    'pano_id': str,
    'image_path': Path,
    'meta_name': str,      # Concept name (e.g., "Urban Street")
    'country': str,        # Country name
    'lat': float,          # Latitude
    'lng': float,          # Longitude
    'note': str,           # Text description of concept
    'images': List[str]    # Additional image URLs
}
```

**Returns:**
- `image`: `[3, H, W]` - Preprocessed image tensor
- `concept_idx`: `int` - Index into concept vocabulary
- `target_idx`: `int` - Index into country vocabulary
- `coordinates`: `[2]` - (lat, lng) in degrees
- `metadata`: `Dict` - Original sample information

#### 2. CBMDataset (`src/iwo_dataset.py`)

Loads from CSV file with columns:
- `image_path`: Path to image file
- `meta_name`: Concept name
- `country`: Country name
- `lat`, `lng`: Coordinates
- `note`: Concept description
- `pano_id`: Unique identifier

**Same return format as PanoramaCBMDataset**

### Data Preprocessing

```
Image [H, W, 3] (RGB)
    │
    ▼
Resize to (336, 336)  [or from processor]
    │
    ▼
RandomHorizontalFlip (training only)
    │
    ▼
ToTensor (normalize to [0, 1])
    │
    ▼
Normalize (mean, std)
    │
    ▼
Image Tensor [3, 336, 336]
```

**Normalization (StreetCLIP defaults):**
- Mean: `[0.48145466, 0.4578275, 0.40821073]`
- Std: `[0.26862954, 0.26130258, 0.27577711]`

### Train/Val/Test Splits

Uses **stratified splitting** by concept (`meta_name`) to ensure:
- Every concept appears in training set (at least 1 sample)
- Concept distribution preserved across splits
- No concept leakage between splits

**Default ratios:** 70% train, 20% val, 10% test

---

## Model Architecture

### ConceptAwareGeoModel (`src/models/concept_aware_cbm.py`)

#### Component Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ConceptAwareGeoModel                             │
│                                                                     │
│  Inputs:                                                            │
│    - images: [B, 3, H, W]  (batch of images)                       │
│    - gps_coords: [B, 2]    (lat, lng) - optional for training      │
│                                                                     │
│  Outputs:                                                           │
│    - z_img: [B, k]         (image concept activations)              │
│    - z_loc: [B, k]         (location concept activations) - if GPS │
│    - country_logits: [B, C] (country classification)                │
│    - cell_logits: [B, N_cells] (geocell classification)            │
│    - pred_offsets: [B, 2] or [B, 3] (coordinate offsets)           │
│    - fused_features: [B, k+d] (concatenated features)               │
└─────────────────────────────────────────────────────────────────────┘
```

Where:
- `B` = batch size
- `k` = number of concepts
- `d` = StreetCLIP feature dimension (typically 768)
- `C` = number of countries
- `N_cells` = number of semantic geocells

#### Detailed Component Breakdown

##### 1. Image Encoder (StreetCLIP)

```
Image [B, 3, H, W]
    │
    ▼
StreetCLIP Vision Encoder
    │
    ▼
x_img [B, d]  (d = 768 for StreetCLIP)
```

- **Input:** `[B, 3, 336, 336]` (preprocessed images)
- **Output:** `[B, 768]` (image embeddings)
- **Frozen or Finetuned:** Controlled by `finetune_encoder` flag

##### 2. Concept Basis (Learnable)

```
E_concept [k, d]  (Initial concept embeddings from text encoder)
    │
    ├─── Register as buffer (frozen)
    │
    └─── Delta [k, d]  (Learnable parameter, initialized to zeros)
         │
         ▼
    B = E_concept + Delta  (Concept Basis)
    B^T [d, k]  (Transposed for matrix multiplication)
```

- **E_concept**: Pre-computed by encoding concept descriptions (notes) using StreetCLIP text encoder
- **Delta**: Learnable offset allowing concept embeddings to adapt during training
- **B**: Final concept basis matrix `[d, k]` used for projection

##### 3. Image Projector

```
x_img [B, d]
    │
    ▼
Linear(d → 512)
    │
    ▼
LayerNorm(512)
    │
    ▼
GELU()
    │
    ▼
Dropout(0.3)
    │
    ▼
Linear(512 → k)
    │
    ▼
z_img [B, k]  (Concept activations)
```

- **Purpose:** Maps image embeddings to concept space
- **Architecture:** 2-layer MLP with LayerNorm and GELU
- **Output:** Logits over `k` concepts (before softmax)

##### 4. Location Encoder (GeoCLIP)

```
GPS Coords [B, 2]  (lat, lng in degrees)
    │
    ▼
LocationEncoder (GeoCLIP)
    │
    ▼
x_loc_raw [B, 512]
    │
    ▼
Location Adapter (Linear 512 → d)
    │
    ▼
x_loc [B, d]
    │
    ▼
MatMul with B^T [d, k]
    │
    ▼
z_loc [B, k]  (Location concept activations)
```

- **LocationEncoder**: GeoCLIP's LocationEncoder (frozen)
- **Location Adapter**: Projects 512-dim location features to StreetCLIP dimension `d`
- **Concept Projection**: Uses same concept basis `B` as image path

##### 5. Feature Fusion

```
z_img [B, k]  ──┐
                ├─── Concatenate
x_img [B, d]  ──┘
                │
                ▼
    fused_features [B, k+d]
```

- **Purpose:** Combines concept activations with raw image features
- **Used by:** Cell head and Offset head for hierarchical prediction

##### 6. Cell Head (Coarse Location)

```
fused_features [B, k+d]
    │
    ▼
Linear(k+d → 1024)
    │
    ▼
LayerNorm(1024)
    │
    ▼
GELU()
    │
    ▼
Dropout(0.3)
    │
    ▼
Linear(1024 → N_cells)
    │
    ▼
cell_logits [B, N_cells]
```

- **Purpose:** Predicts which semantic geocell the image belongs to
- **Output:** Logits over `N_cells` geocells

##### 7. Offset Head (Fine Location)

```
fused_features [B, k+d]
    │
    ▼
Linear(k+d → 512)
    │
    ▼
LayerNorm(512)
    │
    ▼
GELU()
    │
    ▼
Dropout(0.3)
    │
    ▼
Linear(512 → coord_output_dim)
    │
    ▼
pred_offsets [B, 2] or [B, 3]
```

- **Purpose:** Predicts fine-grained offset from cell center
- **Output dimension:**
  - `2` for lat/lng offsets (degrees)
  - `3` for 3D Cartesian offsets (on unit sphere)

##### 8. Country Head (Auxiliary)

```
z_img [B, k]
    │
    ▼
Linear(k → 256)
    │
    ▼
LayerNorm(256)
    │
    ▼
GELU()
    │
    ▼
Dropout(0.3)
    │
    ▼
Linear(256 → C)
    │
    ▼
country_logits [B, C]
```

- **Purpose:** Predicts country from concept activations (interpretability check)
- **Output:** Logits over `C` countries

---

## Training Process

### Forward Pass Flow

```
Batch: (images, concept_idx, target_idx, coords, metadata, cell_labels, note_embs)
    │
    ├─── images [B, 3, H, W] ──────────────────────────────┐
    │                                                       │
    ├─── coords [B, 2] ────────────────────────────────────┤
    │                                                       │
    └─── note_embs [B, d_note] ───────────────────────────┤
                                                            │
                                                            ▼
                                    ┌───────────────────────────────┐
                                    │   ConceptAwareGeoModel        │
                                    │                               │
                                    │   Image Path:                 │
                                    │     images → x_img → z_img    │
                                    │                               │
                                    │   Location Path:              │
                                    │     coords → x_loc → z_loc    │
                                    │                               │
                                    │   Heads:                      │
                                    │     fused → cell_logits       │
                                    │     fused → pred_offsets      │
                                    │     z_img → country_logits    │
                                    └───────────┬───────────────────┘
                                                │
                                                ▼
                    ┌───────────────────────────────────────────┐
                    │         Outputs                            │
                    │  - z_img [B, k]                            │
                    │  - z_loc [B, k]                            │
                    │  - country_logits [B, C]                   │
                    │  - cell_logits [B, N_cells]                │
                    │  - pred_offsets [B, 2/3]                  │
                    └───────────┬───────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────────────────────────┐
                    │         Loss Computation                  │
                    │                                            │
                    │  1. Contrastive Loss                      │
                    │     L_contrastive = contrastive(z_img, z_loc)│
                    │                                            │
                    │  2. Concept Divergence Loss                │
                    │     L_divergence = divergence(z_img, z_loc)│
                    │                                            │
                    │  3. Concept Classification Loss           │
                    │     L_concept = CE(z_img, concept_idx)    │
                    │                                            │
                    │  4. Country Classification Loss            │
                    │     L_country = CE(country_logits, target_idx)│
                    │                                            │
                    │  5. Semantic Reconstruction Loss           │
                    │     pred_note = softmax(z_img) @ B^T      │
                    │     L_semantic = MSE(pred_note, note_embs)│
                    │                                            │
                    │  6. Cell Classification Loss               │
                    │     L_cell = CE(cell_logits, cell_labels) │
                    │                                            │
                    │  7. Offset Regression Loss                 │
                    │     cell_center = cell_centers[cell_labels]│
                    │     pred_coords = cell_center + pred_offsets│
                    │     L_offset = loss(pred_coords, coords)  │
                    │                                            │
                    │  Total Loss:                               │
                    │    L = λ_contrastive * L_contrastive       │
                    │      + λ_divergence * L_divergence         │
                    │      + λ_concept * L_concept                │
                    │      + λ_country * L_country                │
                    │      + λ_semantic * L_semantic              │
                    │      + λ_cell * L_cell                      │
                    │      + λ_offset * L_offset                 │
                    └───────────┬───────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────────────────────────┐
                    │         Backward Pass                      │
                    │  - Gradient accumulation                   │
                    │  - AMP (Automatic Mixed Precision)        │
                    │  - Optimizer step (AdamW)                 │
                    └───────────────────────────────────────────┘
```

### Training Hyperparameters

**Default Values:**
- Batch size: 32
- Learning rate: 1e-4
- Weight decay: 0.1
- Epochs: 20
- Gradient accumulation steps: 1
- AMP: Optional (flag)

**Loss Weights (λ):**
- `λ_contrastive`: 0.1
- `λ_divergence`: 0.1
- `λ_concept`: 10.0
- `λ_country`: 0.1
- `λ_semantic`: 1.0
- `λ_cell`: 1.0
- `λ_offset`: 10.0

---

## Loss Functions

### 1. Contrastive Alignment Loss

**Purpose:** Aligns image and location concept activations in the same space.

```
z_img [B, k]  ──┐
                ├─── Normalize (L2)
z_loc [B, k]  ──┘
                │
                ▼
    similarity = z_img_norm @ z_loc_norm^T  [B, B]
                │
                ▼
    L_contrastive = InfoNCE(similarity, temperature=0.07)
```

**Formula:**
```
L_contrastive = -log(exp(sim(z_img_i, z_loc_i) / τ) / Σ_j exp(sim(z_img_i, z_loc_j) / τ))
```

Where `τ` is the temperature parameter (default: 0.07).

### 2. Concept Divergence Loss

**Purpose:** Encourages image and location concept activations to be similar.

```
L_divergence = ||z_img - z_loc||^2 / (2 * σ^2)
```

Where `σ` is a scaling parameter (default: 1.0).

### 3. Concept Classification Loss

**Purpose:** Supervised learning of concept predictions from images.

```
L_concept = CrossEntropy(z_img, concept_idx)
```

With label smoothing (default: 0.1).

### 4. Country Classification Loss

**Purpose:** Auxiliary task to predict country from concepts.

```
L_country = CrossEntropy(country_logits, target_idx)
```

### 5. Semantic Reconstruction Loss

**Purpose:** Neuro-symbolic loss ensuring concept activations reconstruct semantic embeddings.

```
concept_probs = softmax(z_img)  [B, k]
basis = B^T  [d, k]
pred_note_embs = concept_probs @ basis  [B, d]

# Normalize for cosine distance
pred_note_norm = normalize(pred_note_embs)
target_note_norm = normalize(note_embs)

L_semantic = MSE(pred_note_norm, target_note_norm)
```

**Note:** If dimension mismatch, pred_note_embs is sliced/padded to match note_embs.

### 6. Cell Classification Loss

**Purpose:** Predicts coarse location (semantic geocell).

```
L_cell = CrossEntropy(cell_logits, cell_labels)
```

### 7. Offset Regression Loss

**Purpose:** Predicts fine-grained offset from cell center.

**For 3D Cartesian (coord_output_dim=3):**
```
cell_centers = cell_centers[cell_labels]  [B, 3]
true_cart = latlon_to_cartesian(coords)  [B, 3]
target_offsets = true_cart - cell_centers  [B, 3]
L_offset = MSE(pred_offsets, target_offsets)
```

**For 2D Lat/Lng (coord_output_dim=2):**
```
cell_latlng = cartesian_to_latlng(cell_centers)  [B, 2]
target_offsets = coords - cell_latlng  [B, 2]
# Handle longitude wraparound
target_offsets[:, 1] = (target_offsets[:, 1] + 180) % 360 - 180

# If coordinate_loss_type == "haversine":
pred_latlng = cell_latlng + pred_offsets
L_offset = haversine_distance(pred_latlng, coords)
# Else (MSE):
L_offset = MSE(pred_offsets, target_offsets)
```

---

## Semantic Geocells

### Generation Process

Semantic geocells are generated using **per-country K-Means clustering**:

```
For each country:
    │
    ├─── If samples > min_samples_per_cell (default: 500):
    │       │
    │       ├─── k = samples // min_samples_per_cell
    │       │
    │       ├─── Convert lat/lng to 3D Cartesian
    │       │    (x, y, z) = (cos(lat)*cos(lng), cos(lat)*sin(lng), sin(lat))
    │       │
    │       ├─── K-Means clustering (k clusters)
    │       │
    │       └─── Store cluster centers (normalized to unit sphere)
    │
    └─── Else:
            │
            └─── Single cell (mean of all country samples)
```

**Output:**
- `cell_centers`: `[N_cells, 3]` - Cartesian coordinates on unit sphere
- `sample_to_cell`: `[N_samples]` - Mapping from sample index to cell ID

### Visualization

Geocells are visualized on a world map:
- Samples colored by cell ID
- Cell centers marked as red stars
- Saved to `visualizations/geocells_map.png`

---

## Dataset Structure

### Sample Format

Each sample in the dataset contains:

```python
{
    'pano_id': str,           # Unique identifier
    'image_path': Path,       # Path to image file
    'meta_name': str,          # Concept name (e.g., "Urban Street")
    'country': str,            # Country name
    'lat': float,              # Latitude in degrees
    'lng': float,              # Longitude in degrees
    'note': str,               # Text description of concept
    'images': List[str],       # Additional image URLs (optional)
    'cell_label': int,         # Semantic geocell ID (added during training)
    'note_embedding': Tensor   # Pre-computed note embedding [d] (added during training)
}
```

### Label Mappings

**Concepts:**
- `concept_to_idx`: `Dict[str, int]` - Maps concept name to index
- `idx_to_concept`: `Dict[int, str]` - Maps index to concept name
- Concepts are sorted alphabetically for determinism

**Countries:**
- `country_to_idx`: `Dict[str, int]` - Maps country name to index
- `idx_to_country`: `Dict[int, str]` - Maps index to country name
- Countries are sorted alphabetically

### Data Loading

**Training:**
- Shuffled batches
- `drop_last=True` for contrastive loss stability
- Gradient accumulation support

**Validation/Test:**
- No shuffling
- Full dataset evaluation

---

## Key Design Decisions

### 1. Concept Basis Learning

The concept basis `B = E_concept + Δ` allows:
- **Initialization:** Concepts start with semantic meaning from text descriptions
- **Adaptation:** Learnable `Δ` allows concepts to adapt to visual patterns
- **Interpretability:** Concepts remain grounded in semantic descriptions

### 2. Hierarchical Location Prediction

Two-stage prediction:
- **Coarse:** Cell classification (semantic geocells)
- **Fine:** Offset regression (from cell center)

This mimics human geolocation: first identify region, then refine location.

### 3. Multi-Task Learning

Multiple losses ensure:
- **Concept alignment:** Images and locations share concept space
- **Semantic consistency:** Concept activations reconstruct text embeddings
- **Geographic accuracy:** Direct coordinate prediction
- **Auxiliary supervision:** Country prediction for interpretability

### 4. Stratified Splits

Splitting by concept ensures:
- All concepts seen during training
- No concept leakage between splits
- Fair evaluation of generalization

---

## File Structure

```
Project_AI/
├── scripts/training/
│   └── train_concept_aware.py      # Main training script
├── src/
│   ├── models/
│   │   └── concept_aware_cbm.py   # Model architecture
│   ├── dataset.py                  # PanoramaCBMDataset
│   ├── iwo_dataset.py              # CBMDataset (CSV-based)
│   └── concepts/
│       └── utils.py                # Concept extraction utilities
└── README_ARCHITECTURE.md          # This file
```

---

## Usage Example

```python
# Initialize dataset
dataset = CBMDataset(
    dataframe=pd.read_csv("data/dataset-43k.csv"),
    encoder_model="geolocal/StreetCLIP",
    country=None  # Global training
)

# Extract concepts
concept_names, concept_map = extract_concepts_from_dataset(dataset)

# Encode concepts
base_encoder = StreetCLIPEncoder(...)
E_concept = encode_concepts(concept_names, concept_map, base_encoder)

# Generate geocells
cell_centers, sample_to_cell = generate_semantic_geocells(
    dataset, min_samples_per_cell=500
)

# Initialize model
model = ConceptAwareGeoModel(
    image_encoder=image_encoder,
    concept_features=E_concept,
    num_concepts=len(concept_names),
    num_countries=len(dataset.country_to_idx),
    num_cells=len(cell_centers),
    streetclip_dim=768,
    location_encoder_dim=512,
    coord_output_dim=2,  # or 3 for sphere
    text_encoder=base_encoder
)

# Training loop (see train_concept_aware.py)
```

---

## Evaluation Metrics

**Concept Accuracy:**
- Top-1 accuracy of concept predictions

**Country Accuracy:**
- Top-1 accuracy of country predictions

**Cell Accuracy:**
- Top-1 accuracy of geocell predictions

**Distance Metrics:**
- Median error (km)
- Threshold accuracies:
  - Street: 1 km
  - City: 25 km
  - Region: 200 km
  - Country: 750 km
  - Continent: 2500 km

---

## Notes

- The model uses **frozen text encoder** for semantic alignment (note embeddings)
- **Image encoder** can be frozen or finetuned (controlled by `finetune_encoder`)
- **Location encoder** (GeoCLIP) is always frozen
- Concept basis `Δ` is learnable, allowing concept adaptation
- All losses are weighted and can be tuned via hyperparameters

