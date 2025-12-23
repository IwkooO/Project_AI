#!/usr/bin/env python3
"""
Train Stage 2 geolocation model.

Stage 2 consumes:
- Cached StreetCLIP embeddings (pooled + patch tokens)
- Frozen Phase1 concept model (Phase1CBMTopKMil)
- Concept embeddings from Phase1 logits

Outputs:
- Cell classification (semantic geocell)
- Offset regression (fine coordinates)
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime

from cbm.phase1.model import Phase1CBMTopKMil
from cbm.phase2.model import ConceptEmbeddingAdapter, Stage2CrossAttentionGeoHead
from cbm.phase2.data import Stage2Dataset, collate_fn_stage2
from cbm.phase2.geocells import fit_semantic_geocells, assign_geocells, compute_offsets, latlng_to_xyz
from cbm.phase2.metrics import haversine_km, threshold_accuracies_km, cell_accuracy, xyz_to_latlng
from cbm.viz.maps import visualize_predictions_map, dump_predictions, visualize_geocell_centers

try:
    import wandb
except ImportError:
    wandb = None


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
) -> torch.Tensor:
    """
    Compute CLIP-style text embeddings for each concept.
    
    Returns:
        text_embeds: [K, D] on CPU, L2-normalized
    """
    from transformers import CLIPModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    clip = CLIPModel.from_pretrained(model_name).to(device)
    clip.eval()

    # One prompt per concept
    prompts = [prompt_template.format(_normalize_concept_for_prompt(c)) for c in concepts]

    feats_batches: list[torch.Tensor] = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(batch_prompts, padding=True, truncation=True, return_tensors="pt").to(device)

        feats = clip.get_text_features(**inputs)
        feats = torch.nn.functional.normalize(feats.float(), p=2, dim=-1)
        feats_batches.append(feats.detach().cpu())

    feats_all = torch.cat(feats_batches, dim=0)  # [K, D]
    return feats_all.contiguous()


def load_phase1_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    num_concepts: int,
    patch_dim: int,
    concept_dim: int = 256,
    dropout: float = 0.3,
    mil_topk: int = 8,
    mil_tau: float = 0.25,
    mix_depth: int = 1,
    mix_heads: int = 4,
    mix_mlp_ratio: float = 4.0,
    mix_dropout: float | None = None,
    mix_local_kernel_size: int | None = None,
) -> Phase1CBMTopKMil:
    """Load and freeze Phase1 checkpoint."""
    print(f"Loading Phase1 checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = Phase1CBMTopKMil(
        num_concepts=num_concepts,
        patch_dim=patch_dim,
        concept_dim=concept_dim,
        dropout=dropout,
        mil_topk=mil_topk,
        mil_tau=mil_tau,
        mix_depth=mix_depth,
        mix_heads=mix_heads,
        mix_mlp_ratio=mix_mlp_ratio,
        mix_dropout=mix_dropout,
        mix_local_kernel_size=mix_local_kernel_size,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    model = model.to(device)
    print("Phase1 model loaded and frozen.")
    return model


def build_concept_vectors(
    phase1_model: Phase1CBMTopKMil,
    concept_names: list[str],
    device: torch.device,
    text_model_name: str = "geolocal/StreetCLIP",
    prompt_template: str = "a street view photo containing {}",
) -> torch.Tensor:
    """
    Build concept vectors E [K, D] from Phase1 checkpoint or CLIP text embeddings.
    
    Priority:
    1. Check if Phase1 checkpoint stored concept vectors
    2. Fallback: Compute CLIP text embeddings
    """
    # Try to get concept vectors from Phase1 model
    if hasattr(phase1_model.concept_head, 'query'):
        # Use Phase1 learned queries as concept vectors
        concept_vectors = phase1_model.concept_head.query.detach().cpu()  # [K, concept_dim]
        print(f"Using Phase1 learned concept queries as concept vectors: {concept_vectors.shape}")
        return concept_vectors
    
    # Fallback: compute CLIP text embeddings
    print("Phase1 checkpoint doesn't have stored concept vectors.")
    print(f"Computing CLIP text embeddings for {len(concept_names)} concepts...")
    concept_vectors = _compute_text_embeddings_for_concepts(
        concept_names,
        model_name=text_model_name,
        device=device,
        prompt_template=prompt_template,
    )
    print(f"Computed CLIP text embeddings: {concept_vectors.shape}")
    return concept_vectors


def train_epoch(
    model: Stage2CrossAttentionGeoHead,
    concept_adapter: ConceptEmbeddingAdapter,
    phase1_model: Phase1CBMTopKMil,
    loader: DataLoader,
    device: torch.device,
    cell_criterion: nn.Module,
    offset_criterion: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    cell_loss_weight: float = 1.0,
    offset_loss_weight: float = 1.0,
):
    """Train for one epoch."""
    model.train()
    concept_adapter.train()
    
    total_loss = 0.0
    total_cell_loss = 0.0
    total_offset_loss = 0.0
    total_samples = 0
    
    correct_cells = 0
    
    pbar = tqdm(loader, desc=f"Train Epoch {epoch+1}")
    
    for batch in pbar:
        pooled_emb = batch['pooled_emb'].to(device)  # [B, D]
        patch_tokens = batch['patch_tokens']
        if patch_tokens is not None:
            patch_tokens = patch_tokens.to(device)  # [B, P, D]
        
        cell_labels = batch['cell_labels'].to(device)  # [B]
        offset_targets = batch['offset_targets'].to(device)  # [B, 3]
        
        optimizer.zero_grad()
        
        # Forward through Phase1 (frozen) to get concept logits
        with torch.no_grad():
            if patch_tokens is not None:
                phase1_logits, _, _, _ = phase1_model(patch_tokens)  # [B, K]
            else:
                # If no patch tokens, we can't use Phase1 - skip this batch
                print("Warning: No patch tokens available, skipping batch")
                continue
        
        # Convert logits to concept embeddings
        concept_emb = concept_adapter(phase1_logits)  # [B, concept_dim]
        
        # Forward through Stage2
        cell_logits, offset_pred, gate = model(concept_emb, patch_tokens, pooled_emb)  # [B, num_cells], [B, 3], [B, hidden_dim] | None
        
        # Losses
        cell_loss = cell_criterion(cell_logits, cell_labels)
        offset_loss = offset_criterion(offset_pred, offset_targets)
        loss = cell_loss_weight * cell_loss + offset_loss_weight * offset_loss
        
        loss.backward()
        optimizer.step()
        
        # Metrics
        pred_cells = cell_logits.argmax(dim=1)
        correct_cells += (pred_cells == cell_labels).sum().item()
        
        batch_size = cell_labels.size(0)
        total_loss += loss.item() * batch_size
        total_cell_loss += cell_loss.item() * batch_size
        total_offset_loss += offset_loss.item() * batch_size
        total_samples += batch_size
        
        weighted_cell_loss = cell_loss_weight * cell_loss.item()
        weighted_offset_loss = offset_loss_weight * offset_loss.item()
        
        # Print mean gate weight if available
        gate_mean = gate.mean().item() if gate is not None else None
        
        postfix_dict = {
            "Loss": f"{loss.item():.4f}",
            "Cell": f"{cell_loss.item():.4f}",
            "Offset": f"{offset_loss.item():.4f}",
            "WCell": f"{weighted_cell_loss:.4f}",
            "WOffset": f"{weighted_offset_loss:.4f}",
            "CellAcc": f"{correct_cells/total_samples:.3f}",
        }
        if gate_mean is not None:
            postfix_dict["GateMean"] = f"{gate_mean:.4f}"
        pbar.set_postfix(postfix_dict)
    
    return (
        total_loss / total_samples,
        total_cell_loss / total_samples,
        total_offset_loss / total_samples,
        correct_cells / total_samples,
    )


@torch.no_grad()
def eval_epoch(
    model: Stage2CrossAttentionGeoHead,
    concept_adapter: ConceptEmbeddingAdapter,
    phase1_model: Phase1CBMTopKMil,
    loader: DataLoader,
    device: torch.device,
    cell_criterion: nn.Module,
    offset_criterion: nn.Module,
    centers_xyz: np.ndarray,
    cell_loss_weight: float = 1.0,
    offset_loss_weight: float = 1.0,
):
    """Evaluate for one epoch."""
    model.eval()
    concept_adapter.eval()
    
    total_loss = 0.0
    total_cell_loss = 0.0
    total_offset_loss = 0.0
    total_samples = 0
    
    all_pred_cells = []
    all_true_cells = []
    all_pred_lat = []
    all_pred_lng = []
    all_true_lat = []
    all_true_lng = []
    
    for batch in tqdm(loader, desc="Eval"):
        pooled_emb = batch['pooled_emb'].to(device)
        patch_tokens = batch['patch_tokens']
        if patch_tokens is not None:
            patch_tokens = patch_tokens.to(device)
        
        cell_labels = batch['cell_labels'].to(device)
        offset_targets = batch['offset_targets'].to(device)
        true_coords = batch['coords'].cpu().numpy()  # [B, 2] (lat, lng)
        
        # Forward
        if patch_tokens is not None:
            phase1_logits, _, _, _ = phase1_model(patch_tokens)
        else:
            continue
        
        concept_emb = concept_adapter(phase1_logits)
        cell_logits, offset_pred, gate = model(concept_emb, patch_tokens, pooled_emb)
        
        # Losses
        cell_loss = cell_criterion(cell_logits, cell_labels)
        offset_loss = offset_criterion(offset_pred, offset_targets)
        loss = cell_loss_weight * cell_loss + offset_loss_weight * offset_loss
        
        # Predictions
        pred_cells = cell_logits.argmax(dim=1).cpu().numpy()
        
        # Convert offset + cell center to lat/lng
        pred_cell_centers = centers_xyz[pred_cells]  # [B, 3]
        pred_xyz = pred_cell_centers + offset_pred.cpu().numpy()  # [B, 3]
        pred_lat, pred_lng = xyz_to_latlng(pred_xyz)
        
        # Collect
        all_pred_cells.append(pred_cells)
        all_true_cells.append(cell_labels.cpu().numpy())
        all_pred_lat.append(pred_lat)
        all_pred_lng.append(pred_lng)
        all_true_lat.append(true_coords[:, 0])
        all_true_lng.append(true_coords[:, 1])
        
        batch_size = cell_labels.size(0)
        total_loss += loss.item() * batch_size
        total_cell_loss += cell_loss.item() * batch_size
        total_offset_loss += offset_loss.item() * batch_size
        total_samples += batch_size
    
    # Aggregate
    all_pred_cells = np.concatenate(all_pred_cells)
    all_true_cells = np.concatenate(all_true_cells)
    all_pred_lat = np.concatenate(all_pred_lat)
    all_pred_lng = np.concatenate(all_pred_lng)
    all_true_lat = np.concatenate(all_true_lat)
    all_true_lng = np.concatenate(all_true_lng)
    
    # Metrics
    cell_acc = cell_accuracy(all_pred_cells, all_true_cells)
    distances_km = haversine_km(all_pred_lat, all_pred_lng, all_true_lat, all_true_lng)
    mean_error = np.mean(distances_km)
    median_error = np.median(distances_km)
    threshold_accs = threshold_accuracies_km(all_pred_lat, all_pred_lng, all_true_lat, all_true_lng)
    
    return (
        total_loss / total_samples,
        total_cell_loss / total_samples,
        total_offset_loss / total_samples,
        cell_acc,
        mean_error,
        median_error,
        threshold_accs,
        {
            'pred_lat': all_pred_lat,
            'pred_lng': all_pred_lng,
            'true_lat': all_true_lat,
            'true_lng': all_true_lng,
            'distances_km': distances_km,
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Train Stage 2 Geolocation Model")
    parser.add_argument("--train-csv", required=True, help="Training CSV path")
    parser.add_argument("--val-csv", required=True, help="Validation CSV path")
    parser.add_argument("--test-csv", default=None, help="Test CSV path (optional)")
    parser.add_argument("--cached-dir", required=True, help="Directory with cached StreetCLIP embeddings")
    parser.add_argument("--phase1-checkpoint", required=True, help="Phase1 checkpoint path (Phase1CBMTopKMil)")
    parser.add_argument("--concept-data-dir", required=True, help="Directory with concept vocab")
    parser.add_argument("--output-dir", required=True, help="Output directory for checkpoints")
    
    # Model args
    parser.add_argument("--num-cells", type=int, default=1000, help="Number of semantic geocells")
    parser.add_argument("--per-country-cells", action="store_true", help="Fit cells per country")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension for Stage2")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of cross-attention layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--mode", type=str, default="both", choices=["both", "concept_only", "image_only"], help="Stage2 mode")
    
    # Training args
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--cell-loss-weight", type=float, default=1.0, help="Weight for cell classification loss")
    parser.add_argument("--offset-loss-weight", type=float, default=1.0, help="Weight for offset regression loss")
    
    # Concept vector args
    parser.add_argument("--text-model-name", type=str, default="geolocal/StreetCLIP", help="HF model for text embeddings (fallback)")
    parser.add_argument("--concept-prompt-template", type=str, default="a street view photo containing {}", help="Prompt template")
    
    # Phase1 model args (must match checkpoint)
    parser.add_argument("--concept-dim", type=int, default=256, help="Concept dimension (must match Phase1)")
    parser.add_argument("--patch-dim", type=int, default=1024, help="Patch dimension (must match cached embeddings)")
    parser.add_argument("--phase1-dropout", type=float, default=0.3, help="Phase1 dropout (must match checkpoint)")
    parser.add_argument("--phase1-mil-topk", type=int, default=8, help="Phase1 MIL top-K (must match checkpoint)")
    parser.add_argument("--phase1-mil-tau", type=float, default=0.25, help="Phase1 MIL tau (must match checkpoint)")
    parser.add_argument("--phase1-mix-depth", type=int, default=1, help="Phase1 mix depth (must match checkpoint)")
    parser.add_argument("--phase1-mix-heads", type=int, default=4, help="Phase1 mix heads (must match checkpoint)")
    parser.add_argument("--phase1-mix-mlp-ratio", type=float, default=4.0, help="Phase1 mix MLP ratio (must match checkpoint)")
    parser.add_argument("--phase1-mix-dropout", type=float, default=None, help="Phase1 mix dropout (must match checkpoint)")
    parser.add_argument("--phase1-mix-local-kernel-size", type=int, default=None, help="Phase1 mix local kernel size (must match checkpoint, 0=None)")
    
    # Other
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="cbm_stage2")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=10, help="Early stopping patience (epochs without improvement). Set to 0 to disable.")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phase2_output_dir = output_dir / "phase2"
    phase2_output_dir.mkdir(parents=True, exist_ok=True)
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Run ID: {run_id}")
    print(f"Using device: {device}")
    
    # W&B setup
    use_wandb = args.wandb and wandb is not None
    wandb_run = None
    if use_wandb:
        wandb_config = {k: v for k, v in vars(args).items()}
        wandb_config["run_id"] = run_id
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or f"stage2_{run_id}",
            config=wandb_config,
            reinit=True,
        )
    
    # Load concept vocabulary
    concept_vocab_path = Path(args.concept_data_dir) / "concept_vocab.json"
    if not concept_vocab_path.exists():
        raise FileNotFoundError(f"Concept vocab not found: {concept_vocab_path}")
    
    with open(concept_vocab_path, 'r') as f:
        concept_vocab = json.load(f)
    
    concept_to_idx = concept_vocab.get('concept_to_idx', {})
    idx_to_concept = {int(k): v for k, v in concept_vocab.get('idx_to_concept', {}).items()}
    concept_names = [idx_to_concept[i] for i in sorted(idx_to_concept.keys())]
    num_concepts = len(concept_names)
    print(f"Loaded {num_concepts} concepts")
    
    # Load datasets
    print("Loading datasets...")
    train_ds = Stage2Dataset(args.train_csv, args.cached_dir, split="train")
    val_ds = Stage2Dataset(args.val_csv, args.cached_dir, split="val")
    
    test_ds = None
    if args.test_csv:
        test_ds = Stage2Dataset(args.test_csv, args.cached_dir, split="test")
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    if test_ds:
        print(f"Test: {len(test_ds)}")
    
    # Detect patch dimension from dataset (for Phase1)
    if train_ds.patch_tokens is not None:
        detected_patch_dim = train_ds.patch_tokens.shape[2]
        if args.patch_dim != detected_patch_dim:
            print(f"Warning: --patch-dim={args.patch_dim} != detected={detected_patch_dim}, using detected")
            args.patch_dim = detected_patch_dim
    else:
        print("Warning: No patch tokens in dataset, Stage2 mode must be 'concept_only'")
        if args.mode != "concept_only":
            raise ValueError(f"Stage2 mode must be 'concept_only' when patch tokens are unavailable")
    
    # Detect pooled dimension from dataset (for Stage2)
    detected_pooled_dim = train_ds.pooled_embeddings.shape[1]
    print(f"Detected pooled_dim: {detected_pooled_dim}")
    
    # Fit geocells on train split only
    print("\nFitting semantic geocells on train split...")
    train_coords = train_ds.coords
    train_countries = train_ds.countries if args.per_country_cells else None
    
    centers_xyz, geocell_metadata = fit_semantic_geocells(
        train_coords,
        train_countries,
        num_cells=args.num_cells,
        per_country=args.per_country_cells,
    )
    
    # Assign geocells to all splits
    print("Assigning geocells to train split...")
    train_cell_labels = assign_geocells(
        train_ds.coords,
        train_ds.countries,
        centers_xyz,
        geocell_metadata,
    )
    
    print("Assigning geocells to val split...")
    val_cell_labels = assign_geocells(
        val_ds.coords,
        val_ds.countries,
        centers_xyz,
        geocell_metadata,
    )
    
    if test_ds:
        print("Assigning geocells to test split...")
        test_cell_labels = assign_geocells(
            test_ds.coords,
            test_ds.countries,
            centers_xyz,
            geocell_metadata,
        )
    
    # Compute offsets
    print("Computing offsets...")
    train_offsets = compute_offsets(train_ds.coords, train_cell_labels, centers_xyz)
    val_offsets = compute_offsets(val_ds.coords, val_cell_labels, centers_xyz)
    if test_ds:
        test_offsets = compute_offsets(test_ds.coords, test_cell_labels, centers_xyz)
    
    # Add cell labels and offsets to datasets (modify __getitem__ return)
    # We'll create wrapper functions or modify the dataset
    # For now, let's create a simple wrapper that adds these to batches
    
    # Save geocell info
    geocell_info = {
        'centers_xyz': centers_xyz.tolist(),
        'metadata': geocell_metadata,
        'num_cells': len(centers_xyz),
    }
    with open(phase2_output_dir / "geocells.json", 'w') as f:
        json.dump(geocell_info, f, indent=2)
    print(f"Saved geocell info to {phase2_output_dir / 'geocells.json'}")
    
    # Visualize geocell centers on a map
    print("Visualizing geocell centers on world map...")
    visualize_geocell_centers(
        centers_xyz,
        phase2_output_dir / "geocell_centers_map.png",
        title="Semantic Geocell Centers (Fitted on Train Split)",
    )
    
    # Load Phase1 checkpoint
    mix_local_kernel_size = args.phase1_mix_local_kernel_size
    if mix_local_kernel_size == 0:
        mix_local_kernel_size = None
    
    phase1_model = load_phase1_checkpoint(
        args.phase1_checkpoint,
        device,
        num_concepts,
        args.patch_dim,
        args.concept_dim,
        dropout=args.phase1_dropout,
        mil_topk=args.phase1_mil_topk,
        mil_tau=args.phase1_mil_tau,
        mix_depth=args.phase1_mix_depth,
        mix_heads=args.phase1_mix_heads,
        mix_mlp_ratio=args.phase1_mix_mlp_ratio,
        mix_dropout=args.phase1_mix_dropout,
        mix_local_kernel_size=mix_local_kernel_size,
    )
    
    # Build concept vectors
    concept_vectors = build_concept_vectors(
        phase1_model,
        concept_names,
        device,
        args.text_model_name,
        args.concept_prompt_template,
    )
    concept_vectors = concept_vectors.to(device)
    
    # Create concept adapter
    concept_adapter = ConceptEmbeddingAdapter(concept_vectors, temperature=1.0).to(device)
    
    # Create Stage2 model
    num_cells_actual = len(centers_xyz)
    stage2_model = Stage2CrossAttentionGeoHead(
        concept_dim=args.concept_dim,
        patch_dim=args.patch_dim,
        num_cells=num_cells_actual,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        mode=args.mode,
        pooled_dim=detected_pooled_dim,
    ).to(device)
    
    print(f"Stage2 model parameters: {sum(p.numel() for p in stage2_model.parameters())/1e6:.2f}M")
    
    # Create data loaders (cell labels and offsets are already in dataset)
    
    # Set cell labels and offsets in datasets
    train_ds.set_labels(train_cell_labels, train_offsets)
    val_ds.set_labels(val_cell_labels, val_offsets)
    if test_ds:
        test_ds.set_labels(test_cell_labels, test_offsets)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_stage2, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_stage2, num_workers=4)
    if test_ds:
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_stage2, num_workers=4)
    
    # Loss functions
    cell_criterion = nn.CrossEntropyLoss()
    offset_criterion = nn.MSELoss()
    
    # Optimizer (concept_adapter has no trainable parameters, only buffers)
    optimizer = optim.AdamW(
        stage2_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # LR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_val_error = float('inf')
    best_val_metric = 0.0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    # Disable early stopping if patience <= 0
    early_stop_enabled = args.early_stop_patience > 0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    if early_stop_enabled:
        print(f"Early stopping enabled: patience={args.early_stop_patience} epochs")
    else:
        print("Early stopping disabled")
    for epoch in range(args.epochs):
        # Train
        train_loss, train_cell_loss, train_offset_loss, train_cell_acc = train_epoch(
            stage2_model,
            concept_adapter,
            phase1_model,
            train_loader,
            device,
            cell_criterion,
            offset_criterion,
            optimizer,
            epoch,
            args.cell_loss_weight,
            args.offset_loss_weight,
        )
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Validate
        val_loss, val_cell_loss, val_offset_loss, val_cell_acc, val_mean_error, val_median_error, val_threshold_accs, val_preds = eval_epoch(
            stage2_model,
            concept_adapter,
            phase1_model,
            val_loader,
            device,
            cell_criterion,
            offset_criterion,
            centers_xyz,
            args.cell_loss_weight,
            args.offset_loss_weight,
        )
        
        # Print metrics
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train: Loss={train_loss:.4f}, Cell={train_cell_loss:.4f}, Offset={train_offset_loss:.4f}, CellAcc={train_cell_acc:.4f}")
        print(f"  Val:   Loss={val_loss:.4f}, Cell={val_cell_loss:.4f}, Offset={val_offset_loss:.4f}, CellAcc={val_cell_acc:.4f}")
        print(f"  Val Error: Mean={val_mean_error:.2f}km, Median={val_median_error:.2f}km")
        for thresh, acc in val_threshold_accs.items():
            print(f"    {thresh}: {acc:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        # Log to W&B
        if wandb_run:
            # Calculate weighted losses
            train_weighted_cell = args.cell_loss_weight * train_cell_loss
            train_weighted_offset = args.offset_loss_weight * train_offset_loss
            val_weighted_cell = args.cell_loss_weight * val_cell_loss
            val_weighted_offset = args.offset_loss_weight * val_offset_loss
            
            log_dict = {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/cell_loss": train_cell_loss,
                "train/offset_loss": train_offset_loss,
                "train/weighted_cell_loss": train_weighted_cell,
                "train/weighted_offset_loss": train_weighted_offset,
                "train/cell_acc": train_cell_acc,
                "val/loss": val_loss,
                "val/cell_loss": val_cell_loss,
                "val/offset_loss": val_offset_loss,
                "val/weighted_cell_loss": val_weighted_cell,
                "val/weighted_offset_loss": val_weighted_offset,
                "val/cell_acc": val_cell_acc,
                "val/mean_error_km": val_mean_error,
                "val/median_error_km": val_median_error,
                "lr": current_lr,
            }
            log_dict.update({f"val/{k}": v for k, v in val_threshold_accs.items()})
            wandb_run.log(log_dict)
        
        # Save checkpoint
        is_best = val_mean_error < best_val_error
        if is_best:
            best_val_error = val_mean_error
            best_val_metric = val_cell_acc
        
        # Early stopping: track validation loss
        if early_stop_enabled:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                print(f"  *** New best val loss: {val_loss:.4f} ***")
            else:
                epochs_without_improvement += 1
                print(f"  No improvement in val loss for {epochs_without_improvement} epochs (best: {best_val_loss:.4f})")
            
            # Check early stopping
            if epochs_without_improvement >= args.early_stop_patience:
                print(f"\n*** Early stopping triggered! ***")
                print(f"  No improvement in val loss for {args.early_stop_patience} epochs.")
                print(f"  Best val loss: {best_val_loss:.4f} (achieved at epoch {epoch - epochs_without_improvement + 1})")
                print(f"  Best val error: {best_val_error:.2f}km")
                break
        
        checkpoint = {
            'epoch': epoch,
            'stage2_model_state_dict': stage2_model.state_dict(),
            'concept_adapter_state_dict': concept_adapter.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_error': best_val_error,
            'best_val_metric': best_val_metric,
            'best_val_loss': best_val_loss if early_stop_enabled else None,
            'epochs_without_improvement': epochs_without_improvement if early_stop_enabled else 0,
            'val_mean_error': val_mean_error,
            'val_median_error': val_median_error,
            'val_cell_acc': val_cell_acc,
            'val_threshold_accs': val_threshold_accs,
            'geocell_info': geocell_info,
        }
        
        # Save latest
        torch.save(checkpoint, phase2_output_dir / "latest.pt")
        
        # Save best
        if is_best:
            torch.save(checkpoint, phase2_output_dir / "best_phase2.pt")
            print(f"  *** New best val error: {val_mean_error:.2f}km ***")
            
            # Save visualizations
            vis_dir = phase2_output_dir / "visualizations" / f"epoch_{epoch+1:03d}"
            vis_dir.mkdir(parents=True, exist_ok=True)
            
            # Prediction map
            visualize_predictions_map(
                val_preds['pred_lat'],
                val_preds['pred_lng'],
                val_preds['true_lat'],
                val_preds['true_lng'],
                vis_dir / "predictions_map.png",
            )
            
            # Dump predictions
            val_pano_ids = [val_ds.df.iloc[i]['pano_id'] for i in range(len(val_ds))]
            dump_predictions(
                val_pano_ids,
                val_preds['pred_lat'],
                val_preds['pred_lng'],
                val_preds['true_lat'],
                val_preds['true_lng'],
                val_preds['distances_km'],
                vis_dir / "predictions.txt",
            )
    
    print(f"\nTraining complete! Best val error: {best_val_error:.2f}km")
    print(f"Checkpoints saved to: {phase2_output_dir}")
    
    # Evaluate on test set if provided
    if test_ds:
        print("\n" + "="*80)
        print("Evaluating on test set...")
        print("="*80)
        
        # Load best checkpoint for test evaluation
        best_checkpoint_path = phase2_output_dir / "best_phase2.pt"
        if best_checkpoint_path.exists():
            print(f"Loading best checkpoint from {best_checkpoint_path}...")
            checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
            stage2_model.load_state_dict(checkpoint['stage2_model_state_dict'])
            concept_adapter.load_state_dict(checkpoint['concept_adapter_state_dict'])
            print("Best checkpoint loaded for test evaluation.")
        
        test_loss, test_cell_loss, test_offset_loss, test_cell_acc, test_mean_error, test_median_error, test_threshold_accs, test_preds = eval_epoch(
            stage2_model,
            concept_adapter,
            phase1_model,
            test_loader,
            device,
            cell_criterion,
            offset_criterion,
            centers_xyz,
            args.cell_loss_weight,
            args.offset_loss_weight,
        )
        
        print(f"\nTest Results:")
        print(f"  Loss: {test_loss:.4f}, Cell={test_cell_loss:.4f}, Offset={test_offset_loss:.4f}, CellAcc={test_cell_acc:.4f}")
        print(f"  Error: Mean={test_mean_error:.2f}km, Median={test_median_error:.2f}km")
        for thresh, acc in test_threshold_accs.items():
            print(f"    {thresh}: {acc:.4f}")
        
        # Save test visualizations
        test_vis_dir = phase2_output_dir / "visualizations" / "test_set"
        test_vis_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving test set visualizations to {test_vis_dir}...")
        
        # Prediction map
        visualize_predictions_map(
            test_preds['pred_lat'],
            test_preds['pred_lng'],
            test_preds['true_lat'],
            test_preds['true_lng'],
            test_vis_dir / "predictions_map.png",
        )
        
        # Dump predictions
        test_pano_ids = [test_ds.df.iloc[i]['pano_id'] for i in range(len(test_ds))]
        dump_predictions(
            test_pano_ids,
            test_preds['pred_lat'],
            test_preds['pred_lng'],
            test_preds['true_lat'],
            test_preds['true_lng'],
            test_preds['distances_km'],
            test_vis_dir / "predictions.txt",
        )
        
        # Save test results to JSON
        test_results = {
            'loss': float(test_loss),
            'cell_loss': float(test_cell_loss),
            'offset_loss': float(test_offset_loss),
            'cell_acc': float(test_cell_acc),
            'mean_error_km': float(test_mean_error),
            'median_error_km': float(test_median_error),
            'threshold_accuracies': {k: float(v) for k, v in test_threshold_accs.items()},
        }
        with open(phase2_output_dir / "test_results.json", 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"Test results saved to {phase2_output_dir / 'test_results.json'}")
        
        # Log to W&B
        if wandb_run:
            # Calculate weighted losses
            test_weighted_cell = args.cell_loss_weight * test_cell_loss
            test_weighted_offset = args.offset_loss_weight * test_offset_loss
            
            wandb_run.log({
                'test/loss': test_loss,
                'test/cell_loss': test_cell_loss,
                'test/offset_loss': test_offset_loss,
                'test/weighted_cell_loss': test_weighted_cell,
                'test/weighted_offset_loss': test_weighted_offset,
                'test/cell_acc': test_cell_acc,
                'test/mean_error_km': test_mean_error,
                'test/median_error_km': test_median_error,
            })
            wandb_run.log({f'test/{k}': v for k, v in test_threshold_accs.items()})
        
        print("="*80)


if __name__ == "__main__":
    main()
