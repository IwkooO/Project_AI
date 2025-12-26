# Concept-Aware Global Image-GPS Alignment Architecture

## Overview

This document describes the **3-Stage Training Pipeline** for a Concept-Aware Global Image-GPS Alignment Model that learns to geolocate images by understanding semantic concepts and their relationship to geographic locations. The system uses a hierarchical concept structure with Concept Bottleneck Model (CBM) architecture, progressing through domain pretraining → concept learning → geolocation prediction with cross-attention interpretability.

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Three-Stage Training Pipeline](#three-stage-training-pipeline)
3. [Hierarchical Concept Structure](#hierarchical-concept-structure)
4. [Dataset Structure](#dataset-structure)
5. [Stage 0: Domain Contrastive Pretraining](#stage-0-domain-contrastive-pretraining)
6. [Stage 1: Text-Prototype Concept Learning](#stage-1-text-prototype-concept-learning)
7. [Stage 2: Cross-Attention Geolocation](#stage-2-cross-attention-geolocation)
8. [Loss Functions](#loss-functions)
9. [Semantic Geocells](#semantic-geocells)
10. [Key Design Decisions](#key-design-decisions)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    THREE-STAGE TRAINING PIPELINE                      │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────┐
│   Dataset    │  (PanoramaCBMDataset)
│  - Images    │  - Each sample: (image, meta_concept, parent_concept, country, lat, lng, note)
│  - Concepts  │  - Hierarchical: meta_name (fine-grained) → parent_concept (coarse)
│  - Coords    │
└──────┬───────┘
       │
       ├──────────────────────────────────────────────────────────────────┐
       │                                                               │
       ▼                                                               ▼
┌──────────────────┐                                      ┌──────────────────┐
│   Concepts     │                                      │ Semantic         │
│   Extraction   │                                      │ Geocell          │
│                │                                      │ Generation       │
│ - Extract      │                                      │                  │
│   unique       │                                      │ - Per-country    │
│   meta concepts│                                      │   K-Means        │
│ - Extract      │                                      │ - Cell centers   │
│   unique parent │                                      │   in 3D space    │
│   concepts     │                                      │ - Assign samples │
│ - Build       │                                      │   to cells       │
│   hierarchy    │                                      │                  │
└──────────────────┘                                      └──────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                      STAGE 0: PRETRAINING                          │
│                                                                      │
│  ┌──────────────┐              ┌──────────────────┐                │
│  │ Image Encoder │              │ Text Prototypes  │                │
│  │ (StreetCLIP) │              │                  │                │
│  │              │              │ T_meta [k_m,d]  │                │
│  │              │              │ T_parent [k_p,d] │                │
│  │ x_img [B,768]│              └────────┬─────────┘                │
│  │              │                       │                             │
│  │              │                       ▼                             │
│  │              │              ┌──────────────────┐                │
│  │              │              │ Concept          │                │
│  │              │              │ Bottleneck       │                │
│  │              │              │ (768→512)        │                │
│  │              │              └────────┬─────────┘                │
│  │              │                       │                             │
│  │              │                       ▼                             │
│  │              │              concept_emb [B,512]                  │
│  └──────────────┘                                                    │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │ Losses:                                                  │          │
│  │  1. Image-GPS Contrastive (image features vs GPS)    │          │
│  │  2. Image-Meta Concept Contrastive                    │          │
│  │  3. Image-Parent Concept Contrastive                  │          │
│  │  4. Hierarchy Consistency (intra-batch supervised)    │          │
│  │  5. Patch-GPS Contrastive (Stage2-aligned)           │          │
│  │  6. Anchor Loss (optional: keep encoder close)        │          │
│  └──────────────────────────────────────────────────────────┘          │
│                                                                      │
│  Trainable: Top N vision layers, text encoder, bottleneck, GPS adapter │
│  Frozen: GeoCLIP location encoder                                    │
└─────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼ (frozen encoder)
┌─────────────────────────────────────────────────────────────────────────┐
│                  STAGE 1: CONCEPT LEARNING                          │
│                                                                      │
│  ┌──────────────────┐    ┌──────────────────────────────────┐    │
│  │ Image Encoder   │    │ Text-Anced Prototypes            │    │
│  │ (frozen)       │    │                                  │    │
│  │ x_img [B,768] │───▶│ T_meta = T_meta_base + Δ_meta   │    │
│  │              │    │ T_parent = T_parent_base + Δ_parent│    │
│  │              │    │                                  │    │
│  │              │    │ Projected to concept_emb_dim:    │    │
│  │              │    │   T_meta_proj, T_parent_proj    │    │
│  └──────────────┘    └──────────────────────────────────┘    │
│                           │                                     │
│                           ▼                                     │
│              ┌──────────────────┐                             │
│              │ Concept          │                             │
│              │ Bottleneck       │                             │
│              │ (MLP or         │                             │
│              │  Transformer)    │                             │
│              └────────┬─────────┘                             │
│                       │                                       │
│                       ▼                                       │
│              concept_emb [B,512]                               │
│                       │                                       │
│                       ▼                                       │
│              ┌──────────────────┐                             │
│              │ Cosine Similarity│                             │
│              │ to Prototypes    │                             │
│              │                  │                             │
│              │ meta_logits = scale * (emb @ T_meta^T) + bias │    │
│              │ parent_logits = scale * (emb @ T_parent^T) + bias │
│              └──────────────────┘                             │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │ Losses:                                                  │          │
│  │  1. Meta Classification (Focal Loss)                │          │
│  │  2. Parent Classification (Focal Loss)              │          │
│  │  3. Hierarchical Consistency (KL divergence)          │          │
│  │  4. Parent-Guided Meta Loss (soft constraint)       │          │
│  │  5. Inter-Parent Contrastive                       │          │
│  │  6. Prototype Contrastive (concept ↔ prototypes)     │          │
│  │  7. Prototype Regularization (L2 on residuals)      │          │
│  │  8. Intra-Parent Consistency (soft weight sharing)   │          │
│  │  9. Semantic Soft Cross-Entropy (anti-overfitting)  │          │
│  └──────────────────────────────────────────────────────────┘          │
│                                                                      │
│  Trainable: Bottleneck, prototype residuals, biases, logit scales    │
│  Frozen: Image encoder, text prototypes (base)                         │
└─────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼ (frozen concept_emb)
┌─────────────────────────────────────────────────────────────────────────┐
│              STAGE 2: CROSS-ATTENTION GEOLOCATION                   │
│                                                                      │
│  ┌──────────────────┐    ┌──────────────────┐    ┌────────────┐  │
│  │ concept_emb     │    │ Image Encoder   │    │ Geocells   │  │
│  │ [B,512] (froz)│    │ (frozen)        │    │            │  │
│  │                 │    │ patch_tokens    │    │ centers[N,3]│  │
│  │                 │    │ [B,576,1024]   │    └─────┬──────┘  │
│  │                 │    │                 │          │             │
│  └────────┬────────┘    └────────┬────────┘          │             │
│           │                     │                    │             │
│           ▼                     ▼                    │             │
│  concept_proj         patch_proj [B,576,512]     │             │
│  (as query)           (as keys/values)          │             │
│           │                     │                    │             │
│           └──────────┬──────────┘                    │             │
│                      │                               │             │
│                      ▼                               │             │
│           ┌──────────────────┐                      │             │
│           │ Cross-Attention  │                      │             │
│           │ (multi-head)     │                      │             │
│           │ query=concept     │                      │             │
│           │ keys/values=patches│                      │             │
│           └────────┬─────────┘                      │             │
│                    │                                │             │
│                    ▼                                │             │
│           attn_output [B,512]                      │             │
│                    │                                │             │
│                    ▼                                │             │
│           ┌──────────────────┐                      │             │
│           │ Residual + Norm  │                      │             │
│           │ + FFN            │                      │             │
│           └────────┬─────────┘                      │             │
│                    │                                │             │
│                    ▼                                │             │
│           fused_emb [B,512]                         │             │
│                    │                                │             │
│           ┌────────┴────────┐                       │             │
│           │                 │                       │             │
│           ▼                 ▼                       │             │
│    ┌──────────┐     ┌──────────┐               │             │
│    │ Cell Head │     │ Offset   │               │             │
│    │           │     │ Head     │               │             │
│    │ [B,N]    │     │ [B,2/3]  │               │             │
│    └──────────┘     └──────────┘               │             │
│                                                      │             │
│                                                      ▼             │
│                                             pred_coords [B,2]  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │ Losses:                                                  │          │
│  │  1. Cell Classification (CrossEntropy)                  │          │
│  │  2. Offset Regression (MSE to cell center)            │          │
│  └──────────────────────────────────────────────────────────┘          │
│                                                                      │
│  Trainable: Patch projection, cross-attn, FFN, fusion, heads     │
│  Frozen: Image encoder, Stage 1 concept bottleneck                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Three-Stage Training Pipeline

The training pipeline consists of three sequential stages, each building on the previous:

### Stage 0: Domain Contrastive Pretraining
**Purpose:** Align image encoder with GPS and concept domains

- **Partially unfreezes** top N layers of StreetCLIP vision encoder
- **Trains:**
  - Image ↔ GPS alignment (global features)
  - Image ↔ Meta (child) concept alignment
  - Image ↔ Parent concept alignment
  - Intra-batch hierarchical consistency
  - Optional anchor loss to keep encoder close to vanilla
  - Patch-level GPS alignment (Stage2-aligned)
- **Frozen:** GeoCLIP location encoder, base text prototypes

### Stage 1: Text-Prototype Concept Learning
**Purpose:** Learn concept representations with hierarchical supervision

- **Fully freezes** StreetCLIP image encoder (from Stage 0)
- **Trains:**
  - Concept bottleneck (MLP or Transformer)
  - Learnable prototype residuals (Δ)
  - Per-concept bias and logit scale
  - Multiple auxiliary losses for consistency
- **Key Innovation:** Text-anchored classification via cosine similarity to prototypes
- **Frozen:** Image encoder, base text prototypes

### Stage 2: Cross-Attention Geolocation
**Purpose:** Interpretable geolocation prediction with attention visualization

- **Frozen:** Image encoder, Stage 1 concept bottleneck
- **Trains:**
  - Patch projection layer
  - Cross-attention mechanism (concept queries patches)
  - Fusion and gating mechanisms
  - Cell classification head
  - Offset regression head
- **Key Innovation:** Cross-attention provides patch-level interpretability
- **Three Ablation Modes:**
  - `both`: Concept + image fusion (default)
  - `concept_only`: Only concept embeddings
  - `image_only`: Only image patches

---

## Hierarchical Concept Structure

The system uses a two-level concept hierarchy:

### Meta Concepts (Fine-Grained)
- Examples: "Urban Street", "Suburban Road", "Rural Landscape", "Coastal Area"
- Number: ~100+ concepts
- Extracted from `meta_name` column in dataset

### Parent Concepts (Coarse Categories)
- Examples: "Urban", "Rural", "Natural", "Coastal"
- Number: ~10-20 concepts
- Extracted from `parent_concept` column in dataset

### Hierarchical Relationships
- Each meta concept belongs to exactly one parent concept
- Mapping: `meta_name → parent_concept`
- Used for:
  - Hierarchical supervision losses (consistency, parent-guided)
  - Intra-parent prototype consistency (soft weight sharing)
  - Multi-scale concept predictions

### Concept Templates

**Meta Concept Templates** (for text encoding):
```
"A street view showing {}"
"A photo of {} in scene"
"An area characterized by {}"
"{} visible from road"
"A location with {}"
```

**Parent Concept Templates**:
```
"A {} area"
"A scene showing {} features"
"An environment with {} characteristics"
"{} landscape"
"A {} region"
```

---

## Dataset Structure

### Sample Format

Each sample in the dataset contains:

```python
{
    'pano_id': str,              # Unique identifier
    'image_path': Path,          # Path to image file
    'meta_name': str,            # Fine-grained concept (e.g., "Urban Street")
    'parent_concept': str,       # Coarse concept (e.g., "Urban")
    'country': str,              # Country name
    'lat': float,                # Latitude in degrees
    'lng': float,                # Longitude in degrees
    'note': str,                 # Text description of concept (HTML)
    'cell_label': int,           # Semantic geocell ID (added during training)
}
```

### Label Mappings

**Meta Concepts:**
- `concept_to_idx`: Maps `meta_name` → index
- `idx_to_concept`: Maps index → `meta_name`
- Alphabetically sorted for determinism

**Parent Concepts:**
- `parent_to_idx`: Maps `parent_concept` → index
- `idx_to_parent`: Maps index → `parent_concept`

**Hierarchy:**
- `meta_to_parent`: Maps `meta_name` → `parent_concept`
- `meta_to_parent_idx`: Tensor mapping `concept_idx` → `parent_idx`

### Train/Val/Test Splits

- **Ratio:** 70% train, 15% val, 15% test
- **Stratified:** By `meta_name` to ensure all concepts seen in training
- **Consistency:** Same splits used across all stages (loaded from `splits.json`)
- **Stage 0 Split:** Further 90/10 split of train for pretrain_train/pretrain_val

---

## Stage 0: Domain Contrastive Pretraining

### Stage0PretrainingModel

**Purpose:** Pretrain image encoder for domain alignment

#### Components

```
Image [B, 3, 336, 336]
    │
    ▼
StreetCLIP Vision Encoder (partially frozen)
    │
    ├─── Top N layers trainable (unfreeze_layers parameter)
    │
    ▼
x_img [B, 768]  (global features)
    │
    ├─── patch_tokens [B, 576, 1024] (ViT patch embeddings)
    │
    ▼
┌──────────────────┐
│ GPS Adapter     │
│ 512 → 768      │
└──────┬───────────┘
       │
       ▼
gps_emb_768 [B, 768] (projected GPS features)

┌──────────────────┐
│ Concept        │
│ Bottleneck     │
│ 768 → 512      │
└──────┬───────────┘
       │
       ▼
concept_emb [B, 512]

Text Prototypes (frozen base):
- T_meta [k_m, 768]
- T_parent [k_p, 768]
```

#### Loss Functions

**1. Image-GPS Contrastive:**
```
L_gps = InfoNCE(
    img_features_norm [B, 768],
    gps_emb_768_norm [B, 768],
    temperature=0.07
)
```

**2. Image-Child Concept Contrastive:**
```
L_child = InfoNCE(
    concept_emb_norm [B, 512],
    T_meta_projected [k_m, 512],
    concept_idx [B],
    temperature=0.07
)
```

**3. Image-Parent Concept Contrastive:**
```
L_parent = InfoNCE(
    concept_emb_norm [B, 512],
    T_parent_projected [k_p, 512],
    parent_idx [B],
    temperature=0.07
)
```

**4. Hierarchy Consistency (intra-batch supervised contrastive):**
```
L_hierarchy = -log(exp(sim(z_i, z_i_same_parent) / τ) / Σ_j exp(sim(z_i, z_j) / τ))
```

**5. Patch-GPS Contrastive (Stage2-aligned):**
```
patch_emb = patch_projection(patch_tokens)  # [B, 576] → pooled [B, 512]
L_patch_gps = InfoNCE(
    patch_emb_norm [B, 512],
    gps_emb [B, 512],  # Original 512-dim GPS features
    temperature=0.07
)
```

**6. Anchor Loss (optional):**
```
L_anchor = MSE(
    img_features,
    vanilla_image_features  # From frozen vanilla encoder
)
```

**Total Loss:**
```
L = λ_gps * L_gps 
  + λ_child * L_child 
  + λ_parent * L_parent 
  + λ_hierarchy * L_hierarchy
  + λ_patch_gps * L_patch_gps
  + λ_anchor * L_anchor
```

#### Default Hyperparameters

- **Batch size:** 128
- **Learning rate:** Encoder 3e-5, Non-encoder 1e-4
- **Epochs:** 20
- **Unfreeze layers:** 2 (top vision layers)
- **Loss weights:**
  - λ_gps: 1.0
  - λ_child: 1.0
  - λ_parent: 0.5
  - λ_hierarchy: 0.3
  - λ_patch_gps: 0.2
  - λ_anchor: 0.01 (optional)

---

## Stage 1: Text-Prototype Concept Learning

### Stage1ConceptModel

**Purpose:** Learn concept representations with text-anchored classification

#### Components

```
Image [B, 3, 336, 336]
    │
    ▼
StreetCLIP Vision Encoder (frozen)
    │
    ▼
x_img [B, 768]
    │
    ▼
┌──────────────────┐
│ Concept         │
│ Bottleneck      │
│                  │
│ Option 1: MLP  │
│ 768 → 1024     │
│   → 512        │
│                  │
│ Option 2:       │
│ Transformer      │
│ (with CLS token │
│  + attn pool)   │
└────────┬─────────┘
         │
         ▼
concept_emb [B, 512]
    │
    │
    ├───┐
    │   ▼
    │  Normalize
    │    │
    │    ▼
    │  concept_emb_norm [B, 512]
    │
    ├───┐───────────────────────────────────┐
    │   │                               │
    │   ▼                               ▼
    │  Cosine Similarity           Cosine Similarity
    │   │                               │
    │   ▼                               ▼
    │  meta_logits [B, k_m]          parent_logits [B, k_p]
    │   │                               │
    │   ▼                               ▼
    │  meta_probs [B, k_m]          parent_probs [B, k_p]
```

#### Prototype Construction

```
T_meta_base = encode_text(meta_names, templates)  # [k_m, 768]
T_parent_base = encode_text(parent_names, templates)  # [k_p, 768]

# Learnable residuals (fine-tune prototypes)
Δ_meta ~ N(0, 0.01)  # [k_m, 768]
Δ_parent ~ N(0, 0.01)  # [k_p, 768]

# Project to concept embedding space
T_meta_projected = normalize(proj(T_meta_base + Δ_meta))  # [k_m, 512]
T_parent_projected = normalize(proj(T_parent_base + Δ_parent))  # [k_p, 512]
```

#### Classification via Cosine Similarity

```
# Learnable temperature/logit scale (initialized to 14.0)
meta_scale = logit_scale_meta.clamp(max=20.0)
parent_scale = logit_scale_parent.clamp(max=20.0)

# Per-concept bias for calibration
meta_logits = meta_scale * (concept_emb_norm @ T_meta_projected.T) + meta_bias
parent_logits = parent_scale * (concept_emb_norm @ T_parent_projected.T) + parent_bias

meta_probs = softmax(meta_logits)
parent_probs = softmax(parent_logits)
```

#### Loss Functions

**1. Meta Concept Classification (Focal Loss):**
```
L_meta = FocalLoss(
    meta_logits,
    meta_labels,
    gamma=2.0,
    alpha=class_weights,
    label_smoothing=0.2
)
```

**2. Parent Concept Classification (Focal Loss):**
```
L_parent = FocalLoss(
    parent_logits,
    parent_labels,
    gamma=2.0,
    alpha=parent_class_weights,
    label_smoothing=0.2
)
```

**3. Hierarchical Consistency Loss:**
```
# Expected parent distribution from meta predictions
expected_parent_probs = meta_probs @ M_hier  # M_hier maps meta→parent

# KL divergence
L_consistency = KL(expected_parent_probs || parent_probs)
```

**4. Parent-Guided Meta Loss:**
```
# Soft constraint: meta prediction should be consistent with parent
L_meta_guided = weighted_sum(
    -log(P(meta_i | parent_i))  # Guide meta towards parent's children
)
```

**5. Inter-Parent Contrastive Loss:**
```
# Push apart concept embeddings from different parents
L_parent_contrastive = InfoNCE(
    concept_emb,
    parent_labels,
    temperature=0.1
)
```

**6. Prototype Contrastive Loss:**
```
L_contrastive = InfoNCE(
    concept_emb,
    T_meta_projected,
    meta_labels,
    temperature=0.07
)
```

**7. Prototype Regularization:**
```
L_reg = λ_reg * (||Δ_meta||^2 + ||Δ_parent||^2)
```

**8. Intra-Parent Consistency:**
```
# Encourage meta prototypes within same parent to be similar
L_intra_parent = Σ_parent Σ_{i,j∈children(parent)} ||T_i - T_j||^2
```

**9. Semantic Soft Cross-Entropy (Anti-Overfitting):**
```
# Build similarity matrix between all meta prototypes
S = T_meta_base @ T_meta_base.T  # [k_m, k_m]

# Soft targets for wrong but similar concepts
L_semantic = -log(Σ_{j≠label} S[i,j] * softmax(logits)[j])
```

**Total Loss:**
```
L = λ_meta * L_meta 
  + λ_parent * L_parent 
  + λ_consistency * L_consistency 
  + λ_parent_contrastive * L_parent_contrastive 
  + λ_contrastive * L_contrastive 
  + L_reg 
  + L_intra_parent 
  + λ_semantic * L_semantic
```

#### Default Hyperparameters

- **Batch size:** 256 (with precomputed embeddings)
- **Learning rate:** 3e-4
- **Epochs:** 50
- **Bottleneck:** MLP with 0.4 dropout or Transformer (2 layers, 8 heads)
- **Loss weights:**
  - λ_meta: 1.0
  - λ_parent: 0.5
  - λ_consistency: 0.3
  - λ_parent_contrastive: 0.2
  - λ_contrastive: 0.5
  - λ_reg: 0.001
  - λ_intra_parent: 0.01
  - λ_semantic: 0.15

---

## Stage 2: Cross-Attention Geolocation

### Stage2CrossAttentionGeoHead

**Purpose:** Interpretable geolocation prediction with attention visualization

#### Components

```
┌──────────────────┐    ┌──────────────────┐
│ concept_emb     │    │ patch_tokens     │
│ [B, 512]      │    │ [B, 576, 1024]  │
│ (frozen)        │    │ (frozen)         │
└────────┬─────────┘    └────────┬─────────┘
         │                     │
         │                     ▼
         │              ┌──────────────────┐
         │              │ Patch Projection │
         │              │ 1024 → 512      │
         │              └────────┬─────────┘
         │                     │
         │                     ▼
         │              patch_proj [B, 576, 512]
         │                     │
         └──────────┬──────────┘
                    │
                    ▼
         ┌──────────────────┐
         │ Query: concept_emb.unsqueeze(1)  │ [B, 1, 512]
         │ Key/Value: patch_proj              │ [B, 576, 512]
         └──────────────────┘
                    │
                    ▼
         ┌──────────────────┐
         │ Cross-Attention │
         │ (multi-head)     │
         └────────┬─────────┘
                  │
                  ├─▶ attn_weights [B, 1, 576] (for viz)
                  │
                  ▼
         attn_output [B, 512]
                  │
                  ▼
         ┌──────────────────┐
         │ Residual + Norm │
         │               │
         │ fused = norm(concept_emb + attn_output)
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │ FFN            │
         │ 512 → 1024     │
         │   → 512        │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │ Residual + Norm │
         └────────┬─────────┘
                  │
                  ▼
         fused_emb [B, 512]
                  │
         ┌────────┴────────┐
         │                 │
         ▼                 ▼
    ┌──────────┐     ┌──────────┐
    │ Cell Head │     │ Offset   │
    │           │     │ Head     │
    │           │     │           │
    │ 512 → N  │     │ 512 → 2/3│
    └──────────┘     └──────────┘
         │                 │
         ▼                 ▼
    cell_logits      pred_offsets
    [B, N_cells]     [B, 2/3]
```

#### Ablation Modes

**Mode `both` (default):**
- Full cross-attention with concept + image fusion
- Explicit fusion: `concat([concept_emb, fused_emb])`
- Optional gating: `gate * concept_emb + (1-gate) * fused_emb`

**Mode `concept_only`:**
- Only concept embedding contributes to prediction
- Skips cross-attention entirely
- `final_emb = concept_only_proj(concept_emb)`

**Mode `image_only`:**
- Only image patches contribute to prediction
- Pools patch tokens and projects
- `final_emb = image_pool(mean(patch_proj))`

#### Loss Functions

**1. Cell Classification:**
```
L_cell = CrossEntropy(cell_logits, cell_labels)
```

**2. Offset Regression:**

For 3D Cartesian (coord_output_dim=3):
```
cell_center = cell_centers[cell_label]  # [B, 3]
gt_cart = latlon_to_cartesian(coords)  # [B, 3]
target_offsets = gt_cart - cell_center
L_offset = MSE(pred_offsets, target_offsets)
```

For 2D Lat/Lng (coord_output_dim=2):
```
cell_center_lat, cell_center_lng = cartesian_to_latlng(cell_centers)
target_lat_offset = lat - cell_center_lat
target_lng_offset = ((lng - cell_center_lng + 180) % 360) - 180
target_offsets = [target_lat_offset, target_lng_offset]
L_offset = MSE(pred_offsets, target_offsets)
```

**Total Loss:**
```
L = λ_cell * L_cell + λ_offset * L_offset
```

#### Default Hyperparameters

- **Batch size:** 32
- **Learning rate:** 1e-4
- **Epochs:** 30
- **Loss weights:**
  - λ_cell: 1.0
  - λ_offset: 5.0
- **Ablation mode:** `both` (default)

---

## Loss Functions

### Focal Loss

Handles class imbalance by focusing on hard examples:

```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```

Where:
- `p_t`: Predicted probability for true class
- `α_t`: Class weight (inverse frequency)
- `γ`: Focusing parameter (default: 2.0)

### InfoNCE (Contrastive) Loss

```
L = -log(exp(sim(z_i, z_pos_i) / τ) / Σ_j exp(sim(z_i, z_j) / τ))
```

Where:
- `sim(z_a, z_b) = z_a · z_b / (||z_a|| * ||z_b||)` (cosine similarity)
- `τ`: Temperature (default: 0.07)

### Hierarchical Consistency Loss

KL divergence between expected parent from meta predictions and actual parent predictions:

```
P_expected(parent_j) = Σ_{i: meta_i → parent_j} P(meta_i)
L_consistency = Σ_i KL(P_expected(parent) || P_actual(parent))
```

### Semantic Soft Cross-Entropy

Anti-overfitting loss that allows predictions to be "close" to semantically similar concepts:

```
# Build similarity matrix from base text prototypes
S_ij = cosine_similarity(T_i_base, T_j_base)

# For each sample, compute soft target over similar concepts
soft_target_j = S[i,j] / Σ_{k≠i} S[i,k]

L_semantic = -Σ_{j≠label} soft_target_j * log(P(meta_j))
```

---

## Semantic Geocells

### Generation Process

Semantic geocells are generated using **per-country K-Means clustering in 3D**:

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

## Key Design Decisions

### 1. Three-Stage Curriculum

**Stage 0:** Domain alignment without concept supervision
- Learns general visual-geographic correspondence
- Aligns encoder with GPS and concept domains
- Optional anchor loss prevents catastrophic forgetting

**Stage 1:** Concept-focused learning
- Text-anchored classification ensures semantic grounding
- Hierarchical supervision provides multi-scale signals
- Frozen encoder prevents overfitting to concept distribution

**Stage 2:** Geolocation with interpretability
- Cross-attention provides patch-level explanations
- Enforced concept usage prevents bypassing bottleneck
- Hierarchical prediction (coarse cell + fine offset)

### 2. Text-Anchored Prototypes

**Benefits:**
- Semantic initialization from CLIP's multimodal space
- Learnable residuals allow adaptation to visual patterns
- Per-concept bias and logit scale provide calibration

**Alternative:** Could use pure MLP classifiers, but loses semantic interpretability

### 3. Strict CBM Architecture

**All downstream tasks operate on concept embeddings only:**

**Benefits:**
- Enforces interpretability: predictions traceable to concepts
- Prevents shortcut learning via raw image features
- Compact 512-dim bottleneck for Stage 2

**Implementation:**
- Stage 1: Only bottleneck and downstream heads trained
- Stage 2: Only cross-attn and prediction heads trained

### 4. Cross-Attention Interpretability

**Concept as Query, Patches as Key/Value:**
- Attention weights show which image regions support concept-based prediction
- Spatial attention map (24×24) for visualization

**Gating Mechanism:**
- Learns balance between concept and spatial information
- Prevents model from ignoring concepts

### 5. Hierarchical Concept Structure

**Fine → Coarse:**
- Meta: ~100 fine-grained concepts
- Parent: ~10-20 coarse categories

**Uses:**
- Parent-guided meta loss
- Hierarchical consistency loss
- Intra-parent prototype consistency (soft weight sharing)

**Benefits:**
- Multi-scale supervision improves generalization
- Coarse parent concepts stabilize fine-grained learning
- Better handling of rare meta concepts

### 6. Regularization Strategies

**Stage 0:**
- Anchor loss to prevent forgetting
- Early stopping on validation

**Stage 1:**
- Focal loss for class imbalance
- Label smoothing (0.2)
- Heavy dropout (0.4 in bottleneck)
- Prototype regularization (L2 on residuals)
- Semantic soft cross-entropy (anti-overfitting)

**Stage 2:**
- Cross-attention dropout
- Residual connections
- Layer normalization

---

## File Structure

```
Project_AI/
├── scripts/training/
│   ├── train_stage0_prototype.py      # Stage 0 pretraining
│   ├── train_stage1_prototype.py      # Stage 1 concept learning
│   └── train_stage2_cross_attention.py  # Stage 2 geolocation
├── src/
│   ├── models/
│   │   ├── concept_aware_cbm.py     # All model definitions
│   │   ├── streetclip_encoder.py      # StreetCLIP wrapper
│   │   └── baseline.py              # Baseline models
│   ├── dataset.py                    # Dataset class
│   ├── losses.py                     # Loss functions
│   └── concepts/
│       └── utils.py                 # Concept extraction
├── docs/
│   └── README_ARCHITECTURE.md        # This file
└── README.md
```

---

## Usage Example

```python
# Stage 0: Pretraining
python scripts/training/train_stage0_prototype.py \
    --csv_path data/dataset-43k-mapped.csv \
    --stage0_epochs 20 \
    --unfreeze_layers 2 \
    --use_wandb

# Stage 1: Concept Learning
python scripts/training/train_stage1_prototype.py \
    --csv_path data/dataset-43k-mapped.csv \
    --resume_from_checkpoint results/stage0-.../best_model_stage0.pt \
    --stage1_epochs 50 \
    --use_wandb

# Stage 2: Geolocation
python scripts/training/train_stage2_cross_attention.py \
    --csv_path data/dataset-43k-mapped.csv \
    --stage1_checkpoint results/stage1-.../best_model_stage1.pt \
    --epochs 30 \
    --ablation_mode both \
    --use_wandb
```

---

## Evaluation Metrics

### Stage 1 (Concept)

**Meta Concept Accuracy:**
- Top-1 accuracy: P(pred_meta == gt_meta)
- Top-5 accuracy: P(gt_meta ∈ top-5 predictions)
- Recall (macro average)

**Parent Concept Accuracy:**
- Top-1 accuracy
- Top-5 accuracy
- Recall (macro average)

**Semantic-Close Accuracy:**
- Accuracy within semantic similarity threshold (e.g., 0.7)

### Stage 2 (Geolocation)

**Cell Accuracy:**
- Top-1 accuracy of geocell classification

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

- **Strict CBM:** All downstream predictions use concept embeddings, not raw image features
- **Frozen Encoders:** Image encoder frozen after Stage 0; concept bottleneck frozen after Stage 1
- **Learnable Prototypes:** Residuals allow adaptation while preserving semantic grounding
- **Multi-Task Losses:** Each stage has multiple complementary losses for robust learning
- **Ablation Support:** Stage 2 supports concept_only, image_only, and both modes for analysis
