#!/usr/bin/env python3
"""
Compute concept text embeddings, zero-shot scores, and class priors.

This script:
1. Extracts unique meta_name values from dataset → K concepts
2. For each concept k:
   - Extracts note text (strips HTML, cleans) → T_k^text
   - Computes text embedding: v_k = E_text(T_k^text) / ||E_text(T_k^text)||_2
   - Stores in matrix V ∈ R^(K×768)
3. Computes zero-shot scores z_k(x) for all images and concepts
4. Estimates class priors: π_k = max(π_k^annot, π_k^clip)
5. Saves: concept vocabulary, text embeddings V, priors π_k, zero-shot scores
"""

import argparse
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import re
from html import unescape
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple


def clean_html_text(text: str) -> str:
    """Clean HTML text: remove tags, decode entities, strip whitespace."""
    if pd.isna(text) or text == "":
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', str(text))
    # Decode HTML entities
    text = unescape(text)
    # Clean up whitespace
    text = ' '.join(text.split())
    return text.strip()


def compute_text_embeddings(model, tokenizer, texts: List[str], device: torch.device) -> torch.Tensor:
    """
    Compute normalized text embeddings for a list of texts.
    
    Args:
        model: CLIP/StreetCLIP model with text encoder
        tokenizer: Text tokenizer
        texts: List of text strings
        device: Device to run on
    
    Returns:
        embeddings: Normalized embeddings [N, 768]
    """
    embeddings = []
    
    with torch.no_grad():
        for text in texts:
            # Tokenize
            inputs = tokenizer(text, padding=True, truncation=True, max_length=77, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get text embedding
            if hasattr(model, 'text_model'):
                # CLIP-style model
                text_outputs = model.text_model(**inputs)
                text_features = text_outputs.last_hidden_state
                # Use CLS token or mean pooling
                if hasattr(model, 'text_projection'):
                    text_features = text_features[:, 0, :]  # CLS token
                    text_emb = model.text_projection(text_features)
                else:
                    text_emb = text_features.mean(dim=1)
            elif hasattr(model, 'get_text_features'):
                # Direct method
                text_emb = model.get_text_features(**inputs)
            else:
                # Try encode_text or similar
                text_emb = model.encode_text(**inputs)
            
            # Normalize
            text_emb = F.normalize(text_emb, p=2, dim=-1)
            embeddings.append(text_emb.cpu())
    
    return torch.cat(embeddings, dim=0)


def compute_zero_shot_scores(
    patch_tokens: torch.Tensor,
    text_embeddings: torch.Tensor,
    device: torch.device,
    use_pooled: bool = False,
    pooled_embeddings: torch.Tensor = None
) -> np.ndarray:
    """
    Compute zero-shot similarity scores z_k(x) for all images and concepts.
    
    Args:
        patch_tokens: Patch tokens [N, P, 768] for N images
        text_embeddings: Concept text embeddings [K, 768] for K concepts
        device: Device to run on
        use_pooled: If True, use pooled embeddings instead of max over patches
        pooled_embeddings: Pooled embeddings [N, 768] (required if use_pooled=True)
    
    Returns:
        scores: Zero-shot scores [N, K]
    """
    N, P, D = patch_tokens.shape
    K = text_embeddings.shape[0]
    
    scores = []
    
    # Move to device
    text_emb = text_embeddings.to(device)
    
    if use_pooled and pooled_embeddings is not None:
        # Use pooled embeddings: z_k(x) = z(x)^T · v_k
        pooled = pooled_embeddings.to(device)  # [N, 768]
        # Compute similarity: [N, 768] @ [768, K] = [N, K]
        scores_tensor = torch.matmul(pooled, text_emb.T)  # [N, K]
        scores = scores_tensor.cpu().numpy()
    else:
        # Use max over patches: z_k(x) = max_p v_k^T · t_p(x)
        patch_tokens_gpu = patch_tokens.to(device)  # [N, P, 768]
        
        # Compute similarity for each concept
        batch_scores = []
        batch_size = 100  # Process in batches to avoid memory issues
        
        for i in range(0, N, batch_size):
            batch_patches = patch_tokens_gpu[i:i+batch_size]  # [B, P, 768]
            # Compute similarity: [B, P, 768] @ [768, K] = [B, P, K]
            sim = torch.matmul(batch_patches, text_emb.T)  # [B, P, K]
            # Max over patches: [B, K]
            batch_max = torch.max(sim, dim=1)[0]
            batch_scores.append(batch_max.cpu())
        
        scores = torch.cat(batch_scores, dim=0).numpy()
    
    return scores


def estimate_class_priors(
    df: pd.DataFrame,
    concept_to_idx: Dict[str, int],
    zero_shot_scores: np.ndarray,
    clip_percentile: float = 98.0
) -> np.ndarray:
    """
    Estimate class priors: π_k = max(π_k^annot, π_k^clip)
    
    Args:
        df: DataFrame with 'meta_name' column
        concept_to_idx: Mapping from concept name to index
        zero_shot_scores: Zero-shot scores [N, K]
        clip_percentile: Percentile threshold for CLIP prior (default 98 = top 2%)
    
    Returns:
        priors: Class priors [K]
    """
    K = len(concept_to_idx)
    N = len(df)
    priors = np.zeros(K)
    
    # Compute annotated frequencies: π_k^annot = |P_k| / N
    concept_counts = df['meta_name'].value_counts()
    for concept_name, count in concept_counts.items():
        if concept_name in concept_to_idx:
            k = concept_to_idx[concept_name]
            priors[k] = count / N
    
    # Compute CLIP thresholds: π_k^clip
    clip_priors = np.zeros(K)
    for k in range(K):
        # Get scores for concept k
        scores_k = zero_shot_scores[:, k]  # [N]
        # Compute threshold at percentile
        threshold = np.percentile(scores_k, clip_percentile)
        # Fraction above threshold
        clip_priors[k] = np.mean(scores_k >= threshold)
    
    # Final prior: max of annotated and CLIP
    final_priors = np.maximum(priors, clip_priors)
    
    return final_priors, priors, clip_priors


def main():
    parser = argparse.ArgumentParser(description="Compute concept embeddings and priors")
    parser.add_argument("--dataset-csv", type=str, required=True, help="Path to full dataset CSV")
    parser.add_argument("--cached-embeddings-dir", type=str, required=True, help="Directory with cached embeddings")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for concept data")
    parser.add_argument("--model-name", type=str, default="geolocal/StreetCLIP", help="StreetCLIP model name")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Which split to use for computing priors")
    parser.add_argument("--clip-percentile", type=float, default=98.0, help="Percentile for CLIP prior threshold (default: 98 = top 2%)")
    parser.add_argument("--use-pooled", action="store_true", help="Use pooled embeddings for zero-shot scores instead of max over patches")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset CSV
    print(f"Loading dataset from {args.dataset_csv}")
    df = pd.read_csv(args.dataset_csv)
    df = df.dropna(subset=['meta_name', 'note'])
    print(f"Loaded {len(df)} samples")
    
    # Extract unique concepts
    unique_concepts = sorted(df['meta_name'].unique().tolist())
    K = len(unique_concepts)
    concept_to_idx = {concept: idx for idx, concept in enumerate(unique_concepts)}
    idx_to_concept = {idx: concept for concept, idx in concept_to_idx.items()}
    
    print(f"\nFound {K} unique concepts")
    
    # Load StreetCLIP model for text encoding
    print(f"\nLoading StreetCLIP model: {args.model_name}")
    model = AutoModel.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    
    # Extract and clean text descriptions
    print("\nExtracting and cleaning concept text descriptions...")
    concept_texts = []
    for concept in unique_concepts:
        # Get first occurrence of this concept's note
        concept_df = df[df['meta_name'] == concept]
        if len(concept_df) > 0:
            note = concept_df.iloc[0]['note']
            cleaned_note = clean_html_text(note)
            
            # Combine concept name with note in a format suitable for StreetCLIP
            # Format: "A street view showing [concept name]. [note]"
            # This works well with CLIP-style models trained on street view imagery
            if cleaned_note and cleaned_note.strip():
                # Use both concept name and note
                formatted_text = f"A street view showing {concept}. {cleaned_note}"
            else:
                # Fallback to just concept name if note is empty
                formatted_text = f"A street view showing {concept}"
            concept_texts.append(formatted_text)
        else:
            # Fallback to just concept name
            concept_texts.append(f"A street view showing {concept}")
    
    # Compute text embeddings
    print(f"\nComputing text embeddings for {K} concepts...")
    text_embeddings = compute_text_embeddings(model, tokenizer, concept_texts, device)
    print(f"Text embeddings shape: {text_embeddings.shape}")  # [K, 768]
    
    # Load cached patch tokens
    cached_dir = Path(args.cached_embeddings_dir)
    split = args.split
    patch_tokens_path = cached_dir / f"{split}_patch_tokens.pt"
    
    if not patch_tokens_path.exists():
        raise FileNotFoundError(f"Patch tokens not found: {patch_tokens_path}")
    
    print(f"\nLoading patch tokens from {patch_tokens_path}")
    patch_tokens = torch.load(patch_tokens_path)  # [N, P, 768]
    print(f"Patch tokens shape: {patch_tokens.shape}")
    
    # Load pooled embeddings if available and requested
    pooled_embeddings = None
    if args.use_pooled:
        pooled_path = cached_dir / f"{split}_pooled_embeddings.pt"
        if pooled_path.exists():
            print(f"Loading pooled embeddings from {pooled_path}")
            pooled_embeddings = torch.load(pooled_path)
            print(f"Pooled embeddings shape: {pooled_embeddings.shape}")
        else:
            print("Warning: Pooled embeddings not found, using max over patches")
            args.use_pooled = False
    
    # Compute zero-shot scores
    print(f"\nComputing zero-shot scores...")
    zero_shot_scores = compute_zero_shot_scores(
        patch_tokens,
        text_embeddings,
        device,
        use_pooled=args.use_pooled,
        pooled_embeddings=pooled_embeddings
    )
    print(f"Zero-shot scores shape: {zero_shot_scores.shape}")  # [N, K]
    
    # Load split CSV for prior estimation
    split_csv = cached_dir.parent / "splits" / f"dataset_{split}.csv"
    if not split_csv.exists():
        # Try alternative location
        split_csv = Path(args.dataset_csv).parent / "splits" / f"dataset_{split}.csv"
    
    if split_csv.exists():
        print(f"\nLoading {split} split for prior estimation from {split_csv}")
        split_df = pd.read_csv(split_csv)
        split_df = split_df.dropna(subset=['meta_name'])
    else:
        print(f"Warning: Split CSV not found, using full dataset for priors")
        split_df = df
    
    # Estimate class priors
    print(f"\nEstimating class priors (CLIP percentile: {args.clip_percentile})...")
    priors, priors_annot, priors_clip = estimate_class_priors(
        split_df,
        concept_to_idx,
        zero_shot_scores,
        clip_percentile=args.clip_percentile
    )
    
    print(f"\nClass prior statistics:")
    print(f"  Mean annotated prior: {priors_annot.mean():.4f}")
    print(f"  Mean CLIP prior: {priors_clip.mean():.4f}")
    print(f"  Mean final prior: {priors.mean():.4f}")
    
    # Save everything
    print(f"\nSaving concept data to {output_dir}...")
    
    # Save concept vocabulary
    vocab_path = output_dir / "concept_vocabulary.json"
    with open(vocab_path, 'w') as f:
        json.dump({
            'concepts': unique_concepts,
            'concept_to_idx': concept_to_idx,
            'idx_to_concept': idx_to_concept,
            'num_concepts': K,
            'concept_texts': concept_texts
        }, f, indent=2)
    print(f"Saved concept vocabulary to {vocab_path}")
    
    # Save text embeddings
    embeddings_path = output_dir / "concept_text_embeddings.pt"
    torch.save(text_embeddings, embeddings_path)
    print(f"Saved text embeddings to {embeddings_path}")
    
    # Save priors
    priors_path = output_dir / "class_priors.json"
    with open(priors_path, 'w') as f:
        json.dump({
            'priors': priors.tolist(),
            'priors_annotated': priors_annot.tolist(),
            'priors_clip': priors_clip.tolist(),
            'clip_percentile': args.clip_percentile
        }, f, indent=2)
    print(f"Saved class priors to {priors_path}")
    
    # Save zero-shot scores (as numpy array for efficiency)
    scores_path = output_dir / f"{split}_zero_shot_scores.npy"
    np.save(scores_path, zero_shot_scores)
    print(f"Saved zero-shot scores to {scores_path}")
    
    print(f"\n{'='*60}")
    print("Concept embedding computation complete!")
    print(f"Output saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

