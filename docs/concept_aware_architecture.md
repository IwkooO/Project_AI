# Concept-Aware Global Image-GPS Alignment Framework

This document describes the implementation of the Concept-Aware Global Image-GPS Alignment Framework, based on the paper "Towards Interpretable Geo-localization: a Concept-Aware Global Image-GPS Alignment Framework" (arXiv:2509.01910v1).

## Overview

The framework integrates geographic concepts into contrastive learning for image-GPS alignment. It introduces a **Concept Bottleneck Model (CBM)** that forces both image and location embeddings to pass through a shared, interpretable concept space defined by human-readable geographic descriptions.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PRE-TRAINING PHASE                                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. CONCEPT SPACE CONSTRUCTION                                               │
└─────────────────────────────────────────────────────────────────────────────┘

    GeoGuessr Meta Descriptions (note field)
              │
              │ HTML cleaning
              ▼
    ┌─────────────────────────┐
    │  Concept Text List      │  e.g., ["B Type with a cap on top...",
    │  C = {c₁, c₂, ..., cₖ} │        "Follow car in Kenya...",
    │                         │        ...]
    └─────────────────────────┘
              │
              │ StreetCLIP Text Encoder (frozen)
              ▼
    ┌─────────────────────────┐
    │  E_concept ∈ R^(k×d)    │  Text embeddings of concepts
    │  [t₁; t₂; ...; tₖ]       │  Each tᵢ ∈ R^d
    └─────────────────────────┘
              │
              │ + Learnable Offset Δ
              ▼
    ┌─────────────────────────┐
    │  B = E_concept + Δ      │  Concept Basis Matrix
    │  B ∈ R^(d×k)            │  Defines k concept axes
    └─────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. DUAL-MODAL ENCODING                                                      │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐                    ┌──────────────┐
    │   Image I    │                    │  GPS (lat,   │
    │              │                    │      lon)    │
    └──────┬───────┘                    └──────┬───────┘
           │                                   │
           │ StreetCLIP                       │ GeoCLIP LocationEncoder
           │ Vision Encoder                    │ (Fourier features + MLP)
           │                                   │
           ▼                                   ▼
    ┌──────────────┐                    ┌──────────────┐
    │  x_img       │                    │  x_loc_raw   │
    │  ∈ R^d       │                    │  ∈ R^512     │
    └──────┬───────┘                    └──────┬───────┘
           │                                   │
           │                                   │ Location Adapter
           │                                   │ (Linear: 512 → d)
           │                                   │
           │                                   ▼
           │                            ┌──────────────┐
           │                            │  x_loc       │
           │                            │  ∈ R^d       │
           │                            └──────┬───────┘
           │                                   │
           └───────────────────────────────────┘
                       │
                       ▼


┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. CONCEPT SPACE PROJECTION (BOTTLENECK)                                    │
└─────────────────────────────────────────────────────────────────────────────┘

                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
┌──────────────┐              ┌──────────────┐
│  f_img MLP   │              │  x_loc @ B   │
│  (d → k)     │              │  (matrix     │
│              │              │   multiply)  │
└──────┬───────┘              └──────┬───────┘
       │                            │
       ▼                            ▼
┌──────────────┐              ┌──────────────┐
│  z_img       │              │  z_loc       │
│  ∈ R^k       │              │  ∈ R^k         │
│              │              │               │
│  Concept     │              │  Concept     │
│  Activations │              │  Activations │
└──────┬───────┘              └──────┬───────┘
       │                            │
       └────────────┬───────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  SHARED CONCEPT SPACE │
        │  Both z_img and z_loc │
        │  live in R^k          │
        └───────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. ALIGNMENT LOSSES                                                         │
└─────────────────────────────────────────────────────────────────────────────┘

                    │
        ┌───────────┴───────────┐
        │                      │
        ▼                      ▼
┌──────────────┐      ┌──────────────────────┐
│  L_img-gps   │      │  L_concept           │
│              │      │  (Divergence Loss)   │
│  Contrastive │      │                      │
│  Loss        │      │  MMD with Gaussian   │
│              │      │  Kernel              │
│  InfoNCE:    │      │                      │
│  Pull matching│     │  Aligns distributions│
│  pairs close  │     │  of z_img and z_loc  │
│  Push others │     │                      │
│  apart       │     │                      │
└──────┬───────┘      └──────┬───────────────┘
       │                     │
       └──────────┬──────────┘
                  │
                  ▼
         ┌─────────────────┐
         │  L_total =      │
         │  L_img-gps +    │
         │  λ·L_concept    │
         └─────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                           INFERENCE PHASE                                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. QUERY IMAGE PROCESSING                                                   │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │  Query Image │
    │      I       │
    └──────┬───────┘
           │
           │ StreetCLIP
           │ Vision Encoder
           ▼
    ┌──────────────┐
    │  x_img       │
    │  ∈ R^d       │
    └──────┬───────┘
           │
           │ f_img MLP
           ▼
    ┌──────────────┐
    │  z_img       │
    │  ∈ R^k       │
    └──────┬───────┘
           │
           ▼


┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. GPS GALLERY (Pre-computed)                                                │
└─────────────────────────────────────────────────────────────────────────────┘

    Global Grid of GPS Points
              │
              │ GeoCLIP LocationEncoder
              │ + Location Adapter
              │ + Basis B
              ▼
    ┌─────────────────────────┐
    │  Gallery:               │
    │  {L₁, L₂, ..., Lₙ}     │
    │                         │
    │  Each Lⱼ encoded to:    │
    │  z_loc,ⱼ ∈ R^k          │
    └─────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. RETRIEVAL                                                                │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │  z_img       │
    │  ∈ R^k       │
    └──────┬───────┘
           │
           │ Similarity Search
           │ (Cosine / Dot Product)
           ▼
    ┌─────────────────────────┐
    │  Find:                    │
    │  argmaxⱼ Sim(z_img,       │
    │            z_loc,ⱼ)       │
    └──────┬────────────────────┘
           │
           ▼
    ┌──────────────┐
    │  Predicted   │
    │  Location    │
    │  (lat, lon)   │
    └──────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                           KEY COMPONENTS                                    │
└─────────────────────────────────────────────────────────────────────────────┘

1. ENCODERS
   ├── Image Encoder: StreetCLIP (CLIP-based, specialized for street-level images)
   └── Location Encoder: GeoCLIP's LocationEncoder (Fourier features + MLP)

2. CONCEPT SPACE
   ├── Source: GeoGuessr meta descriptions (note field)
   ├── Encoding: StreetCLIP text encoder (frozen)
   ├── Basis: B = E_concept + Δ (learnable offset)
   └── Dimension: k concepts, each in R^d

3. PROJECTION LAYERS
   ├── Image Projector: f_img (MLP: d → k)
   └── Location Adapter: Linear (512 → d)

4. LOSSES
   ├── Contrastive Loss (L_img-gps): InfoNCE alignment
   └── Divergence Loss (L_concept): MMD with Gaussian kernel


┌─────────────────────────────────────────────────────────────────────────────┐
│                           FILE STRUCTURE                                    │
└─────────────────────────────────────────────────────────────────────────────┘

src/
├── models/
│   ├── streetclip_encoder.py      # StreetCLIP wrapper with text encoding
│   └── concept_aware_cbm.py       # Main ConceptAwareGeoModel
├── concepts/
│   └── utils.py                   # Concept extraction from dataset
├── losses.py                      # Contrastive + Divergence losses
└── dataset.py                     # PanoramaCBMDataset with note field

scripts/training/
└── train_concept_aware.py         # Training script

jobs/
└── train_concept_aware.job        # SLURM job file


┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING WORKFLOW                                 │
└─────────────────────────────────────────────────────────────────────────────┘

1. Load dataset and extract unique meta_name → note mappings
2. Encode all concept descriptions using StreetCLIP text encoder → E_concept
3. Initialize ConceptAwareGeoModel with:
   - StreetCLIP image encoder
   - GeoCLIP location encoder
   - Concept basis B = E_concept + Δ (Δ is learnable)
4. For each batch:
   - Forward: images → z_img, (gps_coords) → z_loc
   - Compute: L_img-gps + λ·L_concept
   - Backward: Update Δ, f_img, location_adapter, location_encoder
5. Validation: Monitor alignment quality


┌─────────────────────────────────────────────────────────────────────────────┐
│                           INFERENCE WORKFLOW                                │
└─────────────────────────────────────────────────────────────────────────────┘

1. Load trained model
2. Build GPS Gallery:
   - Generate grid of GPS coordinates (e.g., global 1km grid)
   - Encode each: Lⱼ → z_loc,ⱼ using model.encode_location()
   - Store gallery: {(Lⱼ, z_loc,ⱼ)}
3. For query image:
   - Encode: I → z_img using model(image, gps_coords=None)
   - Search: Find Lⱼ with max similarity to z_img
   - Return: Predicted location (lat, lon)


┌─────────────────────────────────────────────────────────────────────────────┐
│                           KEY DIFFERENCES FROM BASELINE                     │
└─────────────────────────────────────────────────────────────────────────────┘

Baseline (GeoCLIP):
- Direct image ↔ location alignment in embedding space
- No explicit concept layer
- Less interpretable

Our Method:
- Image → Concepts → Location (via shared concept space)
- Explicit geographic concepts (e.g., "eucalyptus trees", "tuk-tuk")
- Interpretable: Can see which concepts drive predictions
- Better alignment through concept-level supervision


┌─────────────────────────────────────────────────────────────────────────────┐
│                           MATHEMATICAL FORMULATION                          │
└─────────────────────────────────────────────────────────────────────────────┘

Given:
- Image I, GPS coordinates L = (lat, lon)
- Concept set C = {c₁, ..., cₖ} with descriptions
- Encoders: E_img, E_loc
- Concept basis: B = E_concept + Δ ∈ R^(d×k)

Image Path:
  x_img = E_img(I) ∈ R^d
  z_img = f_img(x_img) ∈ R^k

Location Path:
  x_loc_raw = E_loc(L) ∈ R^512
  x_loc = Adapter(x_loc_raw) ∈ R^d
  z_loc = x_loc^T · B ∈ R^k

Training Objective:
  L = L_img-gps + λ·L_concept

  where:
  L_img-gps = -log(exp(z_img · z_loc / τ) / Σⱼ exp(z_img · z_loc,ⱼ / τ))
  
  L_concept = (1/N²) Σᵢ,ⱼ [log K(z_img,ᵢ, z_img,ⱼ) + log K(z_loc,ᵢ, z_loc,ⱼ) 
                          - 2 log K(z_img,ᵢ, z_loc,ⱼ)]
  
  with Gaussian kernel: K(x, y) = exp(-||x - y||² / 2σ²)

Inference:
  Given query image I:
    z_img = f_img(E_img(I))
    L̂ = argmax_{Lⱼ} Sim(z_img, z_loc,ⱼ)

