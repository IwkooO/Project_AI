#!/usr/bin/env python3
"""
Train Concept Bottleneck Model - Phase 1 (Concept Prediction Only).

Patch-only training using a hard Top-K MIL pooling model (canonical Phase-1 CBM).
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
import sys
from datetime import datetime

from cbm.phase1.model import Phase1CBMTopKMil
from cbm.phase1.data import ConceptDataset, collate_fn
from cbm.viz.attention import visualize_predictions_summary

try:
    import wandb
except ImportError:
    wandb = None

# Hyperparameters
GRAD_CLIP_NORM = 5.0
WARMUP_EPOCHS = 5
LABEL_SMOOTHING = 0.1
CONCEPT_DIM_DEFAULT = 256
DEFAULT_DROPOUT = 0.3
DEFAULT_WEIGHT_DECAY = 0.02


def _normalize_concept_for_prompt(name: str) -> str:
    """Convert a concept label into something prompt-friendly."""
    return str(name).strip().replace("_", " ")


@torch.no_grad()
def _compute_text_embeddings_for_concepts(
    concepts: list[str],
    *,
    model_name: str,
    device: torch.device,
    prompt_template: str,
    batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Compute StreetCLIP text embeddings for each concept using a single prompt template.

    Args:
        model_name: HuggingFace model ID (default: "geolocal/StreetCLIP")
    
    Returns:
        (text_embeds, visual_proj_weight, projection_dim)
          - text_embeds: [K, T] on CPU, L2-normalized
          - visual_proj_weight: [T, patch_dim] on CPU (StreetCLIP visual_projection.weight)
          - projection_dim: int (T)
    """
    from transformers import CLIPModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    clip = CLIPModel.from_pretrained(model_name).to(device)
    clip.eval()

    # One prompt per concept (no prompt ensembling).
    prompts = [prompt_template.format(_normalize_concept_for_prompt(c)) for c in concepts]

    feats_batches: list[torch.Tensor] = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(batch_prompts, padding=True, truncation=True, return_tensors="pt").to(device)

        feats = clip.get_text_features(**inputs)

        feats = torch.nn.functional.normalize(feats.float(), p=2, dim=-1)
        feats_batches.append(feats.detach().cpu())

    feats_all = torch.cat(feats_batches, dim=0)  # [K, T]
    text_embeds = feats_all.contiguous()

    # Visual projection weight maps vision hidden size -> projection_dim (shared StreetCLIP space).
    if not hasattr(clip, "visual_projection") or not hasattr(clip.visual_projection, "weight"):
        raise AttributeError(
            f"Expected CLIPModel to have visual_projection.weight, but it was not found for {model_name}"
        )
    visual_proj_weight = clip.visual_projection.weight.detach().cpu().contiguous()
    projection_dim = int(visual_proj_weight.shape[0])

    return text_embeds, visual_proj_weight, projection_dim


def compute_concept_weights(dataset, num_concepts, device):
    """Compute inverse frequency weights for concept loss."""
    print("Computing concept class weights...")
    
    concept_names = dataset.df['meta_name']
    counts_dict = concept_names.value_counts().to_dict()
    
    counts = torch.zeros(num_concepts)
    for name, count in counts_dict.items():
        if name in dataset.concept_to_idx:
            idx = dataset.concept_to_idx[name]
            counts[idx] = count
    
    # Smooth inverse frequency: 1 / sqrt(count)
    weights = 1.0 / torch.sqrt(counts + 1.0)
    weights = torch.clamp(weights, min=0.1, max=10.0)
    weights = weights / weights.mean()
    return weights.to(device)


def print_param_counts(model):
    """Print parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: total={total/1e6:.2f}M, trainable={trainable/1e6:.2f}M")


def check_for_nan(tensor, name="tensor"):
    """Check if tensor contains NaN or Inf."""
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        raise ValueError(f"Inf detected in {name}")


def attention_diversity_loss(
    attn_selected: torch.Tensor,
    *,
    mode: str = "offdiag",
) -> torch.Tensor:
    """
    Attention diversity penalty inspired by Lin et al. (2017):
      P = || A A^T - I ||_F^2

    Where A contains one attention distribution per concept (row), and we want different
    concepts to attend to different patches (low redundancy).

    Args:
        attn_selected: [B, M, P] attention weights for M selected concepts per sample.
        mode:
          - "offdiag": only penalize off-diagonal redundancy (recommended default).
          - "full": penalize (A A^T - I) directly.
    """
    if attn_selected is None:
        raise ValueError("attn_selected is None")
    if attn_selected.dim() != 3:
        raise ValueError(f"attn_selected must be [B,M,P], got {tuple(attn_selected.shape)}")

    b, m, _p = attn_selected.shape
    if m <= 1:
        # Nothing to diversify if there's 0/1 concept.
        return attn_selected.new_zeros(())

    # L2-normalize over patches so dot-products in AA^T are cosine similarities.
    A = F.normalize(attn_selected, p=2, dim=-1)  # [B, M, P]
    G = torch.bmm(A, A.transpose(1, 2))  # [B, M, M]

    mode = str(mode).lower().strip()
    if mode == "offdiag":
        # Zero the diagonal and penalize redundancy only.
        G = G - torch.diag_embed(torch.diagonal(G, dim1=1, dim2=2))
        return (G ** 2).mean()
    if mode == "full":
        I = torch.eye(m, device=G.device, dtype=G.dtype).unsqueeze(0).expand(b, -1, -1)
        return ((G - I) ** 2).mean()

    raise ValueError(f"Unknown diversity mode: {mode!r} (expected 'offdiag' or 'full')")


def train_epoch(
    model,
    loader,
    device,
    optimizer,
    criterion,
    epoch: int,
    total_epochs: int,
    anneal_attn_tau: bool = False,
    attn_tau_start: float = 0.5,
    attn_tau_end: float = 0.2,
    attn_diversity_weight: float = 0.0,
    attn_diversity_topm: int = 8,
    attn_diversity_warmup_epochs: int = 5,
    attn_diversity_mode: str = "offdiag",
):
    """Train for one epoch."""
    model.train()
    
    # Update attention tau (annealing)
    if anneal_attn_tau:
        progress = epoch / max(1, total_epochs - 1)
        current_tau = attn_tau_start + (attn_tau_end - attn_tau_start) * progress
        # QuerySparse uses attn_tau; QueryTopK uses mil_tau.
        if hasattr(model.concept_head, "attn_tau"):
            model.concept_head.attn_tau = current_tau
        elif hasattr(model.concept_head, "mil_tau"):
            model.concept_head.mil_tau = current_tau
    
    total_loss = 0.0
    total_ce_loss = 0.0
    total_div_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0
    grad_norm_accum = 0.0
    grad_norm_count = 0
    
    pbar = tqdm(loader, desc=f"Train Epoch {epoch+1}")
    
    for batch_input, c_labels, coords, cell_labels, country_labels, offsets in pbar:
        batch_input = batch_input.to(device)
        c_labels = c_labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward (handles both cached patches and raw images)
        c_logits, c_hidden, attn_w, _ = model(batch_input)
        
        # Check for NaN
        try:
            check_for_nan(c_logits, "c_logits")
        except ValueError as e:
            print(f"WARNING: {e} - skipping batch")
            continue
        
        # CE loss
        ce_loss = criterion(c_logits, c_labels)
        loss = ce_loss

        # Optional attention diversity loss (regularizer).
        div_loss = None
        if (
            float(attn_diversity_weight) > 0.0
            and epoch >= int(attn_diversity_warmup_epochs)
            and attn_w is not None
            and attn_w.dim() == 3
        ):
            # Select the top-M concepts PER SAMPLE by logits (more meaningful than attention peakiness).
            M = min(int(attn_diversity_topm), int(c_logits.size(1)))
            if M > 1:
                top_idx = c_logits.topk(M, dim=1).indices  # [B, M]
                idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, attn_w.size(-1))  # [B, M, P]
                attn_sel = torch.gather(attn_w, dim=1, index=idx_exp)  # [B, M, P]
                div_loss = attention_diversity_loss(attn_sel, mode=attn_diversity_mode)
                loss = loss + float(attn_diversity_weight) * div_loss
        
        # Check for NaN in loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: NaN/Inf loss detected - skipping batch")
            continue
        
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        grad_norm_accum += grad_norm.item()
        grad_norm_count += 1
        
        optimizer.step()
        
        # Metrics
        preds = c_logits.argmax(dim=1)
        correct_top1 += (preds == c_labels).sum().item()
        
        k = min(5, c_logits.size(1))
        topk = c_logits.topk(k, dim=1).indices
        correct_top5 += (topk == c_labels.unsqueeze(1)).any(dim=1).sum().item()
        
        batch_size = c_labels.size(0)
        total_loss += loss.item() * batch_size
        total_ce_loss += ce_loss.item() * batch_size
        if div_loss is not None:
            total_div_loss += float(div_loss.item()) * batch_size
        total_samples += batch_size
        
        postfix = {
            "Loss": f"{loss.item():.4f}",
            "Acc@1": f"{correct_top1/total_samples:.3f}",
            "Acc@5": f"{correct_top5/total_samples:.3f}",
        }
        if div_loss is not None:
            postfix["Div"] = f"{float(div_loss.item()):.4f}"
        pbar.set_postfix(postfix)
    
    avg_grad_norm = grad_norm_accum / (grad_norm_count + 1e-8) if grad_norm_count > 0 else 0.0
    
    return (
        total_loss / total_samples,
        total_ce_loss / total_samples,
        correct_top1 / total_samples,
        correct_top5 / total_samples,
        avg_grad_norm,
        (total_div_loss / total_samples) if total_samples > 0 else 0.0,
    )


@torch.no_grad()
def eval_epoch(model, loader, device, criterion):
    """Evaluate for one epoch."""
    model.eval()
    
    total_ce_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0
    
    # Debug: track label distribution and prediction stats
    label_counts = {}
    pred_counts = {}
    first_batch_debug = True
    
    for batch_input, c_labels, coords, cell_labels, country_labels, offsets in tqdm(loader, desc="Eval"):
        batch_input = batch_input.to(device)
        c_labels = c_labels.to(device)
        
        # Forward (handles both cached patches and raw images)
        c_logits, c_hidden, attn_w, _ = model(batch_input)
        
        # CE loss
        ce_loss = criterion(c_logits, c_labels)
        
        # Metrics
        preds = c_logits.argmax(dim=1)
        correct_top1 += (preds == c_labels).sum().item()
        
        k = min(5, c_logits.size(1))
        topk = c_logits.topk(k, dim=1).indices
        correct_top5 += (topk == c_labels.unsqueeze(1)).any(dim=1).sum().item()
        
        # Debug: check first batch for suspicious patterns
        if first_batch_debug:
            first_batch_debug = False
            unique_labels = torch.unique(c_labels).cpu().tolist()
            unique_preds = torch.unique(preds).cpu().tolist()
            batch_size = c_labels.size(0)
            print(f"DEBUG first batch: batch_size={batch_size}, unique_labels={len(unique_labels)}, unique_preds={len(unique_preds)}")
            print(f"DEBUG first batch: label_range=[{c_labels.min().item()}, {c_labels.max().item()}], pred_range=[{preds.min().item()}, {preds.max().item()}]")
            print(f"DEBUG first batch: logits_range=[{c_logits.min().item():.2f}, {c_logits.max().item():.2f}], logits_std={c_logits.std().item():.2f}")
        
        # Track label distribution
        for label in c_labels.cpu().tolist():
            label_counts[label] = label_counts.get(label, 0) + 1
        for pred in preds.cpu().tolist():
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
        
        batch_size = c_labels.size(0)
        total_ce_loss += ce_loss.item() * batch_size
        total_samples += batch_size
    
    # Debug: print label distribution if accuracy is suspiciously high
    acc1 = correct_top1 / total_samples
    if acc1 > 0.95:
        print(f"\nWARNING: Very high accuracy ({acc1:.4f}). Debug info:")
        print(f"  Total samples: {total_samples}")
        print(f"  Unique labels in val set: {len(label_counts)}")
        print(f"  Unique predictions: {len(pred_counts)}")
        top_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top 5 most common labels: {top_labels}")
        top_preds = sorted(pred_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top 5 most common predictions: {top_preds}")
    
    return (
        total_ce_loss / total_samples,
        correct_top1 / total_samples,
        correct_top5 / total_samples,
    )


def main():
    parser = argparse.ArgumentParser(description="Train CBM Phase 1 (Concept Prediction)")
    # Canonical Phase-1 model only (legacy model variants removed).
    parser.add_argument("--train-csv", required=True, help="Training CSV path")
    parser.add_argument("--val-csv", required=True, help="Validation CSV path")
    parser.add_argument("--cached-dir", default=None, help="Directory with cached patch tokens (required if --trainable-backbone=False)")
    parser.add_argument("--concept-data-dir", required=True, help="Directory with concept vocab and S2 cells")
    parser.add_argument(
        "--trainable-backbone",
        action="store_true",
        help="Use trainable StreetCLIP vision encoder instead of cached embeddings. Requires --vision-model-name.",
    )
    parser.add_argument(
        "--vision-model-name",
        type=str,
        default="geolocal/StreetCLIP",
        help="HuggingFace model ID for vision encoder (used with --trainable-backbone)",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for checkpoints")
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--concept-dim", type=int, default=CONCEPT_DIM_DEFAULT, help="Must match StreetCLIP projection_dim when using text init.")
    parser.add_argument("--attn-tau", type=float, default=0.25, help="Temperature for LogSumExp aggregation")
    parser.add_argument("--anneal-attn-tau", action="store_true", help="Anneal attn_tau over epochs")
    parser.add_argument("--attn-tau-start", type=float, default=0.5, help="Start attn_tau for annealing")
    parser.add_argument("--attn-tau-end", type=float, default=0.2, help="End attn_tau for annealing")
    parser.add_argument("--mix-depth", type=int, default=1, help="Patch mixer depth")
    parser.add_argument("--mix-heads", type=int, default=4, help="Patch mixer heads")
    parser.add_argument("--mix-mlp-ratio", type=float, default=4.0, help="Patch mixer MLP ratio")
    parser.add_argument(
        "--mix-local-kernel-size",
        type=int,
        default=0,
        help="Neighborhood Attention kernel size for patch mixing (0 disables; must be odd, e.g. 3/5/7).",
    )
    parser.add_argument("--mil-topk", type=int, default=8, help="Hard top-K selection for MIL-style aggregation")
    parser.add_argument("--stk-mask-prob", type=float, default=0.0, help="STKIM: probability of masking top patches (0.0 = disabled)")
    parser.add_argument("--stk-k-mask", type=int, default=1, help="STKIM: number of top patches to mask (per concept)")
    parser.add_argument("--stk-mask-fill", type=str, default="min", choices=["min", "zero"], help="STKIM: fill value for masked patches ('min' or 'zero')")
    parser.add_argument(
        "--proj-type",
        type=str,
        default="simple",
        choices=["simple", "two_stage", "bottleneck"],
        help="Patch projection architecture: 'simple' (direct compression), 'two_stage' (full-res then compress), 'bottleneck' (compress-expand-compress)",
    )
    parser.add_argument(
        "--init-concepts-from-text",
        action="store_true",
        help="Initialize concept query weights from StreetCLIP text embeddings. concept_dim must match the text embedding dimension (projection_dim).",
    )
    parser.add_argument(
        "--text-model-name",
        type=str,
        default="geolocal/StreetCLIP",
        help="HF model ID for text embeddings.",
    )
    parser.add_argument(
        "--concept-prompt-template",
        type=str,
        default="a street view photo containing {}",
        help="Prompt template used to compute text embeddings. Must include a '{}' placeholder for the concept name.",
    )
    parser.add_argument("--selection-metric", type=str, default="acc5", choices=["acc1", "acc5"], help="Best checkpoint selection")
    parser.add_argument(
        "--attn-diversity-weight",
        type=float,
        default=0.0,
        help="Optional attention diversity regularizer weight (0 disables).",
    )
    parser.add_argument(
        "--attn-diversity-topm",
        type=int,
        default=8,
        help="When diversity is enabled, regularize only the top-M concepts per sample (selected by logits).",
    )
    parser.add_argument(
        "--attn-diversity-warmup-epochs",
        type=int,
        default=5,
        help="Enable attention diversity only after this many epochs (stabilizes early training).",
    )
    parser.add_argument(
        "--attn-diversity-mode",
        type=str,
        default="offdiag",
        choices=["offdiag", "full"],
        help="Diversity penalty type: 'offdiag' penalizes redundancy only; 'full' matches (AA^T - I).",
    )
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="cbm_concept_bottleneck")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS, help="LR warmup epochs")
    parser.add_argument("--early-stop-patience", type=int, default=10, help="Early stopping patience (epochs without improvement). Set to 0 or negative to disable.")
    args = parser.parse_args()
    
    # Disable early stopping if patience <= 0
    if args.early_stop_patience <= 0:
        args.early_stop_patience = None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_output_dir = output_dir / "phase1"
    phase_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Run ID: {run_id}")
    print(f"Using device: {device}")
    
    # W&B setup
    use_wandb = args.wandb and wandb is not None
    if args.wandb and wandb is None:
        print("WARNING: wandb requested but not installed; skipping W&B.")
    
    wandb_run = None
    if use_wandb:
        wandb_config = {k: v for k, v in vars(args).items()}
        wandb_config["run_id"] = run_id
        wandb_config["grad_clip_norm"] = GRAD_CLIP_NORM
        wandb_config["label_smoothing"] = LABEL_SMOOTHING
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or f"phase1_{run_id}",
            config=wandb_config,
            reinit=True,
        )
    
    # Load vocabularies
    concept_vocab = Path(args.concept_data_dir) / "concept_vocab.json"
    s2_vocab = Path(args.concept_data_dir) / "s2_cells.json"
    
    # Load datasets (patch-only)
    print("Loading datasets...")
    train_ds = ConceptDataset(
        args.train_csv,
        args.cached_dir,
        concept_vocab,
        str(s2_vocab),
        split="train",
        allow_unsafe_index_fallback=False,
    )
    val_ds = ConceptDataset(
        args.val_csv,
        args.cached_dir,
        concept_vocab,
        str(s2_vocab),
        split="val",
        allow_unsafe_index_fallback=False,
    )
    
    if wandb_run:
        wandb_run.config.update({
            "num_train_samples": len(train_ds),
            "num_val_samples": len(val_ds),
            "num_concepts": train_ds.num_concepts,
            "num_cells": train_ds.num_cells,
            "run_id": run_id,
        }, allow_val_change=True)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    # Detect patch dimension
    patch_dim = train_ds.patch_tokens.shape[2]
    print(f"Detected patch dimension: {patch_dim}")

    # Initialize model (patch-only)
    mix_local_kernel_size = int(args.mix_local_kernel_size)
    if mix_local_kernel_size == 0:
        mix_local_kernel_size = None
    print("Initializing Phase1CBMTopKMil (canonical Phase-1 model)...")
    model = Phase1CBMTopKMil(
        num_concepts=train_ds.num_concepts,
        patch_dim=patch_dim,
        concept_dim=args.concept_dim,
        dropout=args.dropout,
        mil_topk=args.mil_topk,
        mil_tau=args.attn_tau_start if args.anneal_attn_tau else args.attn_tau,
        mix_depth=args.mix_depth,
        mix_heads=args.mix_heads,
        mix_mlp_ratio=args.mix_mlp_ratio,
        mix_local_kernel_size=mix_local_kernel_size,
        stk_mask_prob=args.stk_mask_prob,
        stk_k_mask=args.stk_k_mask,
        stk_mask_fill=args.stk_mask_fill,
        proj_type=args.proj_type,
    )
    model = model.to(device)
    
    # Initialize concept queries from text embeddings if requested
    if args.init_concepts_from_text:
        print("Initializing concept queries from StreetCLIP text embeddings...")
        # Get concept names in sorted order by index
        concept_names = [train_ds.idx_to_concept[i] for i in sorted(train_ds.idx_to_concept.keys())]
        if len(concept_names) != train_ds.num_concepts:
            raise ValueError(f"Concept name count mismatch: {len(concept_names)} != {train_ds.num_concepts}")
        
        # Compute text embeddings (+ StreetCLIP visual projection weight for space alignment).
        text_embeds, visual_proj_weight, projection_dim = _compute_text_embeddings_for_concepts(
            concept_names,
            model_name=args.text_model_name,
            device=device,
            prompt_template=args.concept_prompt_template,
        )
        
        # Check if concept_dim matches projection_dim
        projection_layer = None
        if args.concept_dim != projection_dim:
            print(f"Warning: concept_dim={args.concept_dim} != projection_dim={projection_dim}")
            print(f"Projecting text embeddings from {projection_dim} to {args.concept_dim}...")
            # Create a simple linear projection
            projection_layer = nn.Linear(projection_dim, args.concept_dim, bias=False).to(device)
            # Initialize with small weights to preserve semantic structure
            nn.init.xavier_uniform_(projection_layer.weight, gain=0.1)
            with torch.no_grad():
                text_embeds = projection_layer(text_embeds.to(device))
            text_embeds = text_embeds.cpu()
        else:
            print(f"Using text embeddings directly (concept_dim={args.concept_dim} == projection_dim={projection_dim})")
        
        # Align spaces: initialize patch projection from StreetCLIP visual_projection.weight.
        # This makes patches and text-initialized queries start in the same CLIP projection space.
        try:
            if projection_layer is None:
                patch_proj_init_weight = visual_proj_weight.to(device=device)  # [concept_dim, patch_dim]
            else:
                # Compose the same text projection with the CLIP visual projection:
                #   patches -> (visual_proj) -> projection_dim -> (projection_layer) -> concept_dim
                patch_proj_init_weight = projection_layer.weight @ visual_proj_weight.to(device=device)

            # Find a Linear inside the concept head patch projection whose weight matches.
            target_linear = None
            linear_shapes = []
            for m in model.concept_head.patch_proj.modules():
                if isinstance(m, nn.Linear):
                    linear_shapes.append(tuple(m.weight.shape))
                    if tuple(m.weight.shape) == tuple(patch_proj_init_weight.shape):
                        target_linear = m  # if multiple match, last one wins (final projection)

            if target_linear is None:
                print(
                    "WARNING: Could not initialize patch_proj from StreetCLIP visual projection "
                    f"(wanted weight shape={tuple(patch_proj_init_weight.shape)}). "
                    f"Found Linear weights: {linear_shapes}"
                )
            else:
                with torch.no_grad():
                    target_linear.weight.copy_(patch_proj_init_weight.to(dtype=target_linear.weight.dtype))
                print(
                    "Initialized concept_head.patch_proj Linear weights from StreetCLIP visual_projection.weight "
                    f"(shape={tuple(target_linear.weight.shape)})"
                )
        except Exception as e:
            print(f"WARNING: Failed to init patch projection from StreetCLIP visual projection: {e}")

        # Initialize concept query weights with text embeddings
        if text_embeds.shape != (train_ds.num_concepts, args.concept_dim):
            raise ValueError(
                f"Text embedding shape mismatch: {text_embeds.shape} != ({train_ds.num_concepts}, {args.concept_dim})"
            )
        
        with torch.no_grad():
            model.concept_head.query.data.copy_(text_embeds.to(device))
        
        print(f"Initialized {train_ds.num_concepts} concept queries from text embeddings.")
    
    print_param_counts(model)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_metric = 0.0
    best_val_acc5 = 0.0
    epochs_without_improvement = 0
    if args.resume_checkpoint:
        print(f"Loading checkpoint from {args.resume_checkpoint}...")
        # weights_only=False needed for PyTorch 2.6+ compatibility with checkpoints containing numpy scalars
        checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_metric = checkpoint.get('best_val_metric', 0.0)
        best_val_acc5 = checkpoint.get('best_val_acc5', checkpoint.get('val_acc5', 0.0))
        epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
        print(f"Resumed from epoch {start_epoch}, best_val_metric={best_val_metric:.4f}, best_val_acc5={best_val_acc5:.4f}, epochs_without_improvement={epochs_without_improvement}")
    
    # Loss and optimizer
    weights = compute_concept_weights(train_ds, train_ds.num_concepts, device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # LR scheduler with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            # Cosine decay after warmup
            progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_ce, train_acc1, train_acc5, grad_norm, train_div = train_epoch(
            model,
            train_loader,
            device,
            optimizer,
            criterion,
            epoch,
            args.epochs,
            anneal_attn_tau=args.anneal_attn_tau,
            attn_tau_start=args.attn_tau_start,
            attn_tau_end=args.attn_tau_end,
            attn_diversity_weight=args.attn_diversity_weight,
            attn_diversity_topm=args.attn_diversity_topm,
            attn_diversity_warmup_epochs=args.attn_diversity_warmup_epochs,
            attn_diversity_mode=args.attn_diversity_mode,
        )
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Validate
        val_ce, val_acc1, val_acc5 = eval_epoch(model, val_loader, device, criterion)
        
        # Select best metric
        val_metric = val_acc5 if args.selection_metric == "acc5" else val_acc1
        
        # Print metrics
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(
            f"  Train: Loss={train_loss:.4f}, CE={train_ce:.4f}, Acc@1={train_acc1:.4f}, "
            f"Acc@5={train_acc5:.4f}, GradNorm={grad_norm:.2f}, Div={train_div:.4f}"
        )
        print(f"  Val:   CE={val_ce:.4f}, Acc@1={val_acc1:.4f}, Acc@5={val_acc5:.4f}")
        if hasattr(model.concept_head, "attn_tau"):
            tau_str = f"{float(model.concept_head.attn_tau):.3f}"
        elif hasattr(model.concept_head, "mil_tau"):
            tau_str = f"{float(model.concept_head.mil_tau):.3f}"
        else:
            tau_str = "N/A"
        print(f"  LR: {current_lr:.6f}, Tau={tau_str}")
        
        # Log to W&B
        if wandb_run:
            if hasattr(model.concept_head, "attn_tau"):
                tau_val = float(model.concept_head.attn_tau)
            elif hasattr(model.concept_head, "mil_tau"):
                tau_val = float(model.concept_head.mil_tau)
            else:
                tau_val = None
            wandb_run.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/ce": train_ce,
                "train/acc1": train_acc1,
                "train/acc5": train_acc5,
                "train/grad_norm": grad_norm,
                "train/div": train_div,
                "val/ce": val_ce,
                "val/acc1": val_acc1,
                "val/acc5": val_acc5,
                "lr": current_lr,
                "tau": tau_val,
            })
        
        # Save checkpoint (selection_metric for best model selection)
        is_best = val_metric > best_val_metric
        if is_best:
            best_val_metric = val_metric
            print(f"  *** New best {args.selection_metric}: {val_metric:.4f} ***")
        
        # Early stopping: monitor val_acc5 (top-5 accuracy) separately
        if args.early_stop_patience is not None:
            # Track best val_acc5 for early stopping
            if val_acc5 > best_val_acc5:
                best_val_acc5 = val_acc5
                epochs_without_improvement = 0
                print(f"  *** New best val_acc5: {val_acc5:.4f} ***")
            else:
                epochs_without_improvement += 1
                print(f"  No improvement in val_acc5 for {epochs_without_improvement} epochs (best: {best_val_acc5:.4f})")
            
            # Check early stopping (based on val_acc5)
            if epochs_without_improvement >= args.early_stop_patience:
                print(f"\n*** Early stopping triggered! ***")
                print(f"  No improvement in val_acc5 for {args.early_stop_patience} epochs.")
                print(f"  Best val_acc5: {best_val_acc5:.4f} (achieved at epoch {epoch - epochs_without_improvement + 1})")
                break
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_metric': best_val_metric,
            'best_val_acc5': best_val_acc5 if args.early_stop_patience is not None else val_acc5,
            'val_acc1': val_acc1,
            'val_acc5': val_acc5,
            'epochs_without_improvement': epochs_without_improvement if args.early_stop_patience is not None else 0,
        }
        
        # Save latest
        torch.save(checkpoint, phase_output_dir / "latest.pt")
        
        # Save best
        if is_best:
            torch.save(checkpoint, phase_output_dir / "best_phase1.pt")
            
            # Generate visualizations for best checkpoint
            print("  Generating visualizations...")
            visualize_predictions_summary(
                model=model,
                dataset=val_ds,
                device=device,
                epoch=epoch,
                output_dir=phase_output_dir,
                phase=1,
                run_id=run_id,
                reason="best_val_acc",
                num_samples=5,
            )
        
        # Periodic visualizations (every 5 epochs)
        if (epoch + 1) % 5 == 0:
            print("  Generating periodic visualizations...")
            visualize_predictions_summary(
                model=model,
                dataset=val_ds,
                device=device,
                epoch=epoch,
                output_dir=phase_output_dir,
                phase=1,
                run_id=run_id,
                reason="periodic_every5",
                num_samples=3,
            )
    
    print(f"\nTraining complete! Best {args.selection_metric}: {best_val_metric:.4f}")
    print(f"Checkpoints saved to: {phase_output_dir}")


if __name__ == "__main__":
    main()
