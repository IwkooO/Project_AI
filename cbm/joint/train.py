#!/usr/bin/env python3
"""
Joint training script for Phase 1 (concept prediction) and Phase 2 (geolocation).

Trains both models together with a combined loss:
- Phase 1: Concept prediction loss
- Phase 2: Geolocation loss (cell classification + offset regression)
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
from cbm.phase1.data import ConceptDataset, collate_fn
from cbm.phase2.model import ConceptEmbeddingAdapter, Stage2CrossAttentionGeoHead
from cbm.phase2.data import Stage2Dataset, collate_fn_stage2
from cbm.phase2.geocells import fit_semantic_geocells, assign_geocells, compute_offsets
from cbm.phase2.metrics import haversine_km, threshold_accuracies_km, cell_accuracy, xyz_to_latlng
from cbm.viz.maps import visualize_predictions_map, dump_predictions, visualize_geocell_centers
from cbm.viz.attention import visualize_predictions_summary


class JointDataset(ConceptDataset):
    """Wrapper around ConceptDataset that adds geocell labels and offsets for joint training."""
    
    def __init__(self, base_dataset: ConceptDataset, cell_labels: np.ndarray, offsets: np.ndarray):
        """
        Args:
            base_dataset: The underlying ConceptDataset
            cell_labels: Assigned geocell labels [N]
            offsets: Computed offsets [N, 3]
        """
        # Copy all attributes from base dataset
        for attr in dir(base_dataset):
            if not attr.startswith('_') or attr in ['_labels', '_coords', '_cell_labels', '_country_labels', '_cache_idxs']:
                setattr(self, attr, getattr(base_dataset, attr))
        
        # Store assigned labels and offsets
        self._assigned_cell_labels = torch.tensor(cell_labels, dtype=torch.long)
        self._assigned_offsets = torch.tensor(offsets, dtype=torch.float32)
    
    def __getitem__(self, idx: int):
        """Return data with assigned cell labels and offsets."""
        patches, c_label, coords, _, country_label, cache_idx = super().__getitem__(idx)
        # Replace with assigned cell label and add offset
        cell_label = self._assigned_cell_labels[idx]
        offset = self._assigned_offsets[idx]
        return patches, c_label, coords, cell_label, country_label, cache_idx, offset

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
    """Compute CLIP-style text embeddings for each concept."""
    from transformers import CLIPModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    clip = CLIPModel.from_pretrained(model_name).to(device)
    clip.eval()

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


def build_concept_vectors(
    phase1_model: Phase1CBMTopKMil,
    concept_names: list[str],
    device: torch.device,
    text_model_name: str = "geolocal/StreetCLIP",
    prompt_template: str = "a street view photo containing {}",
) -> torch.Tensor:
    """Build concept vectors from Phase1 model or CLIP text embeddings."""
    # Try to get concept vectors from Phase1 model
    if hasattr(phase1_model.concept_head, 'query'):
        concept_vectors = phase1_model.concept_head.query.detach().cpu()  # [K, concept_dim]
        print(f"Using Phase1 learned concept queries as concept vectors: {concept_vectors.shape}")
        return concept_vectors
    
    # Fallback: compute CLIP text embeddings
    print("Computing CLIP text embeddings for concept vectors...")
    concept_vectors = _compute_text_embeddings_for_concepts(
        concept_names,
        model_name=text_model_name,
        device=device,
        prompt_template=prompt_template,
    )
    print(f"Computed CLIP text embeddings: {concept_vectors.shape}")
    return concept_vectors


def collate_fn_joint(batch, pooled_embeddings):
    """Collate function for joint training that includes pooled embeddings."""
    # batch is a list of tuples: (patches, c_label, coords, cell_label, country_label, cache_idx, offset)
    patches = torch.stack([b[0] for b in batch], dim=0)  # [B,P,D]
    c_labels = torch.stack([b[1] for b in batch], dim=0)  # [B]
    # Handle coords - could be tensor or numpy array
    coords_list = []
    for b in batch:
        coords_val = b[2]
        if isinstance(coords_val, torch.Tensor):
            coords_list.append(coords_val)
        else:
            coords_list.append(torch.tensor(coords_val, dtype=torch.float32))
    coords = torch.stack(coords_list, dim=0)  # [B,2]
    # Handle cell_labels - should be tensors now
    cell_labels = torch.stack([b[3] if isinstance(b[3], torch.Tensor) else torch.tensor(b[3], dtype=torch.long) for b in batch], dim=0)  # [B]
    country_labels = torch.stack([b[4] if isinstance(b[4], torch.Tensor) else torch.tensor(b[4], dtype=torch.long) for b in batch], dim=0)  # [B]
    cache_idxs = torch.tensor([b[5] for b in batch], dtype=torch.long)  # [B]
    
    # Get pooled embeddings using cache indices
    pooled_emb = pooled_embeddings[cache_idxs]  # [B, D]
    
    # Get offsets (7th element) - should always be present now
    offsets_list = []
    for b in batch:
        if len(b) >= 7:
            offset_val = b[6]
            if isinstance(offset_val, torch.Tensor):
                offsets_list.append(offset_val)
            else:
                offsets_list.append(torch.tensor(offset_val, dtype=torch.float32))
        else:
            # This shouldn't happen with JointDataset, but handle gracefully
            print(f"WARNING: Batch element has {len(b)} elements, expected 7. Using zero offset.")
            offsets_list.append(torch.zeros(3, dtype=torch.float32))
    offsets = torch.stack(offsets_list, dim=0)  # [B, 3]
    
    # Validate cell labels are in valid range
    num_cells = pooled_embeddings.shape[0]  # This won't work, we need to pass num_cells
    # Actually, we'll validate in the training loop instead
    
    return {
        'patches': patches,
        'c_labels': c_labels,
        'coords': coords,
        'cell_labels': cell_labels,
        'country_labels': country_labels,
        'pooled_emb': pooled_emb,
        'offsets': offsets,
    }


class VisualizationDataset:
    """Wrapper to convert JointDataset (7 values) to format expected by visualization (6 values)."""
    def __init__(self, joint_dataset):
        self.joint_dataset = joint_dataset
        self.df = joint_dataset.df
        # Expose concept mapping from underlying dataset
        self.idx_to_concept = joint_dataset.idx_to_concept
        self.concept_to_idx = joint_dataset.concept_to_idx
    
    def __len__(self):
        return len(self.joint_dataset)
    
    def __getitem__(self, idx):
        patches, c_label, coords, cell_label, country_label, cache_idx, offset = self.joint_dataset[idx]
        # Return 6 values as expected by visualization function (drop cache_idx)
        return patches, c_label, coords, cell_label, country_label, offset


def compute_concept_weights(dataset, num_concepts, device):
    """Compute inverse frequency weights for concept loss."""
    print("Computing concept class weights...")
    
    concept_names = dataset.df['generalized'] if 'generalized' in dataset.df.columns else dataset.df['meta_name']
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


def train_epoch(
    phase1_model: Phase1CBMTopKMil,
    stage2_model: Stage2CrossAttentionGeoHead,
    concept_adapter: ConceptEmbeddingAdapter,
    loader: DataLoader,
    device: torch.device,
    concept_criterion: nn.Module,
    cell_criterion: nn.Module,
    offset_criterion: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    concept_loss_weight: float = 1.0,
    cell_loss_weight: float = 1.0,
    offset_loss_weight: float = 1.0,
):
    """Train for one epoch with joint loss from single dataset."""
    phase1_model.train()
    stage2_model.train()
    concept_adapter.train()
    
    total_loss = 0.0
    total_concept_loss = 0.0
    total_cell_loss = 0.0
    total_offset_loss = 0.0
    total_samples = 0
    
    correct_concept_top1 = 0
    correct_concept_top5 = 0
    correct_cells = 0
    
    pbar = tqdm(loader, desc=f"Train Epoch {epoch+1}")
    
    for batch in pbar:
        patches = batch['patches'].to(device)  # [B, P, D]
        c_labels = batch['c_labels'].to(device)  # [B]
        cell_labels = batch['cell_labels'].to(device)  # [B]
        pooled_emb = batch['pooled_emb'].to(device)  # [B, D]
        offset_targets = batch['offsets'].to(device)  # [B, 3]
        
        # Validate cell labels are in valid range (should be [0, num_cells))
        num_cells = stage2_model.num_cells
        if cell_labels.max() >= num_cells or cell_labels.min() < 0:
            print(f"ERROR: Invalid cell labels! min={cell_labels.min()}, max={cell_labels.max()}, num_cells={num_cells}")
            # Clamp to valid range
            cell_labels = torch.clamp(cell_labels, 0, num_cells - 1)
        
        optimizer.zero_grad()
        
        # Forward through Phase1
        c_logits, c_hidden, attn_w, _ = phase1_model(patches)
        
        # Concept loss
        concept_loss = concept_criterion(c_logits, c_labels)
        
        # Convert logits to concept embeddings for Stage2
        concept_emb = concept_adapter(c_logits)
        
        # Forward through Stage2
        cell_logits, offset_pred, gate = stage2_model(concept_emb, patches, pooled_emb)
        
        # Geo losses
        cell_loss = cell_criterion(cell_logits, cell_labels)
        offset_loss = offset_criterion(offset_pred, offset_targets)
        
        # Combined loss
        batch_loss = concept_loss_weight * concept_loss + cell_loss_weight * cell_loss + offset_loss_weight * offset_loss
        
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(phase1_model.parameters()) + list(stage2_model.parameters()),
            5.0
        )
        optimizer.step()
        
        # Metrics
        preds = c_logits.argmax(dim=1)
        correct_concept_top1 += (preds == c_labels).sum().item()
        k = min(5, c_logits.size(1))
        topk = c_logits.topk(k, dim=1).indices
        correct_concept_top5 += (topk == c_labels.unsqueeze(1)).any(dim=1).sum().item()
        
        pred_cells = cell_logits.argmax(dim=1)
        correct_cells += (pred_cells == cell_labels).sum().item()
        
        batch_size = c_labels.size(0)
        total_loss += batch_loss.item() * batch_size
        total_concept_loss += concept_loss.item() * batch_size
        total_cell_loss += cell_loss.item() * batch_size
        total_offset_loss += offset_loss.item() * batch_size
        total_samples += batch_size
        
        pbar.set_postfix({
            "Loss": f"{batch_loss.item():.4f}",
            "CAcc1": f"{correct_concept_top1/total_samples:.3f}",
            "CellAcc": f"{correct_cells/total_samples:.3f}",
        })
    
    return (
        total_loss / total_samples if total_samples > 0 else 0.0,
        total_concept_loss / total_samples if total_samples > 0 else 0.0,
        total_cell_loss / total_samples if total_samples > 0 else 0.0,
        total_offset_loss / total_samples if total_samples > 0 else 0.0,
        correct_concept_top1 / total_samples if total_samples > 0 else 0.0,
        correct_concept_top5 / total_samples if total_samples > 0 else 0.0,
        correct_cells / total_samples if total_samples > 0 else 0.0,
    )


@torch.no_grad()
def eval_epoch(
    phase1_model: Phase1CBMTopKMil,
    stage2_model: Stage2CrossAttentionGeoHead,
    concept_adapter: ConceptEmbeddingAdapter,
    loader: DataLoader,
    device: torch.device,
    concept_criterion: nn.Module,
    cell_criterion: nn.Module,
    offset_criterion: nn.Module,
    centers_xyz: np.ndarray,
    concept_loss_weight: float = 1.0,
    cell_loss_weight: float = 1.0,
    offset_loss_weight: float = 1.0,
):
    """Evaluate for one epoch with single dataset."""
    phase1_model.eval()
    stage2_model.eval()
    concept_adapter.eval()
    
    total_loss = 0.0
    total_concept_loss = 0.0
    total_cell_loss = 0.0
    total_offset_loss = 0.0
    total_samples = 0
    
    correct_concept_top1 = 0
    correct_concept_top5 = 0
    correct_cells = 0
    
    all_pred_cells = []
    all_true_cells = []
    all_pred_lat = []
    all_pred_lng = []
    all_true_lat = []
    all_true_lng = []
    
    for batch in tqdm(loader, desc="Eval"):
        patches = batch['patches'].to(device)
        c_labels = batch['c_labels'].to(device)
        cell_labels = batch['cell_labels'].to(device)
        pooled_emb = batch['pooled_emb'].to(device)
        offset_targets = batch['offsets'].to(device)
        coords = batch['coords'].cpu().numpy()
        
        # Forward through Phase1
        c_logits, _, _, _ = phase1_model(patches)
        concept_loss = concept_criterion(c_logits, c_labels)
        
        # Concept metrics
        preds = c_logits.argmax(dim=1)
        correct_concept_top1 += (preds == c_labels).sum().item()
        k = min(5, c_logits.size(1))
        topk = c_logits.topk(k, dim=1).indices
        correct_concept_top5 += (topk == c_labels.unsqueeze(1)).any(dim=1).sum().item()
        
        # Convert to concept embeddings for Stage2
        concept_emb = concept_adapter(c_logits)
        
        # Forward through Stage2
        cell_logits, offset_pred, gate = stage2_model(concept_emb, patches, pooled_emb)
        
        cell_loss = cell_criterion(cell_logits, cell_labels)
        offset_loss = offset_criterion(offset_pred, offset_targets)
        
        # Geo predictions
        pred_cells = cell_logits.argmax(dim=1).cpu().numpy()
        pred_cell_centers = centers_xyz[pred_cells]
        pred_xyz = pred_cell_centers + offset_pred.cpu().numpy()
        pred_lat, pred_lng = xyz_to_latlng(pred_xyz)
        
        all_pred_cells.append(pred_cells)
        all_true_cells.append(cell_labels.cpu().numpy())
        all_pred_lat.append(pred_lat)
        all_pred_lng.append(pred_lng)
        all_true_lat.append(coords[:, 0])
        all_true_lng.append(coords[:, 1])
        
        correct_cells += (pred_cells == cell_labels.cpu().numpy()).sum()
        
        batch_size = c_labels.size(0)
        total_concept_loss += concept_loss.item() * batch_size
        total_cell_loss += cell_loss.item() * batch_size
        total_offset_loss += offset_loss.item() * batch_size
        total_samples += batch_size
    
    # Aggregate
    if len(all_pred_cells) > 0:
        all_pred_cells = np.concatenate(all_pred_cells)
        all_true_cells = np.concatenate(all_true_cells)
        all_pred_lat = np.concatenate(all_pred_lat)
        all_pred_lng = np.concatenate(all_pred_lng)
        all_true_lat = np.concatenate(all_true_lat)
        all_true_lng = np.concatenate(all_true_lng)
        
        cell_acc = cell_accuracy(all_pred_cells, all_true_cells)
        distances_km = haversine_km(all_pred_lat, all_pred_lng, all_true_lat, all_true_lng)
        mean_error = np.mean(distances_km)
        median_error = np.median(distances_km)
        threshold_accs = threshold_accuracies_km(all_pred_lat, all_pred_lng, all_true_lat, all_true_lng)
    else:
        cell_acc = 0.0
        mean_error = 0.0
        median_error = 0.0
        threshold_accs = {}
    
    total_loss = concept_loss_weight * (total_concept_loss / total_samples if total_samples > 0 else 0.0) + \
                 cell_loss_weight * (total_cell_loss / total_samples if total_samples > 0 else 0.0) + \
                 offset_loss_weight * (total_offset_loss / total_samples if total_samples > 0 else 0.0)
    
    return (
        total_loss,
        total_concept_loss / total_samples if total_samples > 0 else 0.0,
        total_cell_loss / total_samples if total_samples > 0 else 0.0,
        total_offset_loss / total_samples if total_samples > 0 else 0.0,
        correct_concept_top1 / total_samples if total_samples > 0 else 0.0,
        correct_concept_top5 / total_samples if total_samples > 0 else 0.0,
        cell_acc,
        mean_error,
        median_error,
        threshold_accs,
        {
            'pred_lat': all_pred_lat if len(all_pred_cells) > 0 else np.array([]),
            'pred_lng': all_pred_lng if len(all_pred_cells) > 0 else np.array([]),
            'true_lat': all_true_lat if len(all_pred_cells) > 0 else np.array([]),
            'true_lng': all_true_lng if len(all_pred_cells) > 0 else np.array([]),
            'distances_km': distances_km if len(all_pred_cells) > 0 else np.array([]),
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Joint Training: Phase 1 + Phase 2")
    
    # Data args
    parser.add_argument("--train-csv", required=True, help="Training CSV path")
    parser.add_argument("--val-csv", required=True, help="Validation CSV path")
    parser.add_argument("--test-csv", default=None, help="Test CSV path (optional)")
    parser.add_argument("--cached-dir", required=True, help="Directory with cached StreetCLIP embeddings")
    parser.add_argument("--concept-data-dir", required=True, help="Directory with concept vocab")
    parser.add_argument("--output-dir", required=True, help="Output directory for checkpoints")
    
    # Model args
    parser.add_argument("--num-cells", type=int, default=1000, help="Number of semantic geocells")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension for Stage2")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of cross-attention layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for Stage2")
    parser.add_argument("--phase1-dropout", type=float, default=0.3, help="Dropout rate for Phase1")
    parser.add_argument("--mode", type=str, default="both", choices=["both", "concept_only", "image_only"], help="Stage2 mode")
    
    # Phase1 model args
    parser.add_argument("--concept-dim", type=int, default=256, help="Concept dimension")
    parser.add_argument("--patch-dim", type=int, default=1024, help="Patch dimension")
    parser.add_argument("--mil-topk", type=int, default=8, help="MIL top-K")
    parser.add_argument("--mil-tau", type=float, default=0.25, help="MIL tau")
    parser.add_argument("--mix-depth", type=int, default=1, help="Mix depth")
    parser.add_argument("--mix-heads", type=int, default=4, help="Mix heads")
    parser.add_argument("--mix-mlp-ratio", type=float, default=4.0, help="Mix MLP ratio")
    parser.add_argument("--mix-local-kernel-size", type=int, default=5, help="Mix local kernel size (0=None)")
    
    # Training args
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (reduced default for joint training to avoid OOM)")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of DataLoader workers (reduced default for joint training)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--phase1-lr", type=float, default=None, help="Separate LR for Phase1 (default: same as --lr)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--concept-loss-weight", type=float, default=1.0, help="Weight for concept prediction loss")
    parser.add_argument("--cell-loss-weight", type=float, default=1.0, help="Weight for cell classification loss")
    parser.add_argument("--offset-loss-weight", type=float, default=6371.0, help="Weight for offset regression loss")
    
    # Checkpoint loading args
    parser.add_argument("--phase1-checkpoint", type=str, default=None, help="Path to Phase1 checkpoint to load (optional)")
    parser.add_argument("--phase2-checkpoint", type=str, default=None, help="Path to Phase2 checkpoint to load (optional)")
    parser.add_argument("--resume-joint-checkpoint", type=str, default=None, help="Path to joint checkpoint directory (loads both phase1/latest.pt and phase2/latest.pt)")
    
    # Other
    parser.add_argument("--text-model-name", type=str, default="geolocal/StreetCLIP")
    parser.add_argument("--concept-prompt-template", type=str, default="a street view photo containing {}")
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="cbm_joint")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=10, help="Early stopping patience")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phase1_output_dir = output_dir / "phase1"
    phase2_output_dir = output_dir / "phase2"
    phase1_output_dir.mkdir(parents=True, exist_ok=True)
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
            name=args.wandb_run_name or f"joint_{run_id}",
            config=wandb_config,
            reinit=True,
        )
    
    # Load concept vocabulary
    concept_vocab_path = Path(args.concept_data_dir) / "concept_vocab.json"
    s2_vocab_path = Path(args.concept_data_dir) / "s2_cells.json"
    
    with open(concept_vocab_path, 'r') as f:
        concept_vocab = json.load(f)
    
    concept_to_idx = concept_vocab.get('concept_to_idx', {})
    idx_to_concept = {int(k): v for k, v in concept_vocab.get('idx_to_concept', {}).items()}
    concept_names = [idx_to_concept[i] for i in sorted(idx_to_concept.keys())]
    num_concepts = len(concept_names)
    print(f"Loaded {num_concepts} concepts")
    
    # Load datasets (use ConceptDataset for both - it has everything we need).
    # For joint training we also need pooled embeddings, so we load them via the dataset
    # to avoid separate duplicate tensor loads.
    print("Loading datasets...")
    train_ds = ConceptDataset(
        args.train_csv,
        args.cached_dir,
        concept_vocab_path,
        s2_vocab_path,
        split="train",
        allow_unsafe_index_fallback=False,
        load_pooled_embeddings=True,
    )
    val_ds = ConceptDataset(
        args.val_csv,
        args.cached_dir,
        concept_vocab_path,
        s2_vocab_path,
        split="val",
        allow_unsafe_index_fallback=False,
        load_pooled_embeddings=True,
    )
    
    test_ds = None
    if args.test_csv:
        test_ds = ConceptDataset(
            args.test_csv,
            args.cached_dir,
            concept_vocab_path,
            s2_vocab_path,
            split="test",
            allow_unsafe_index_fallback=False,
            load_pooled_embeddings=True,
        )
    
    # Detect dimensions
    patch_dim = train_ds.patch_tokens.shape[2]
    if args.patch_dim != patch_dim:
        print(f"Warning: --patch-dim={args.patch_dim} != detected={patch_dim}, using detected")
        args.patch_dim = patch_dim
    
    if train_ds.pooled_embeddings is None:
        raise RuntimeError("Expected train_ds.pooled_embeddings to be loaded (got None).")
    detected_pooled_dim = train_ds.pooled_embeddings.shape[1]
    print(f"Detected patch_dim: {patch_dim}, pooled_dim: {detected_pooled_dim}")
    
    # Fit geocells
    print("\nFitting semantic geocells on train split...")
    train_coords = train_ds._coords.numpy()  # Convert to numpy for geocell fitting
    centers_xyz, geocell_metadata = fit_semantic_geocells(
        train_coords,
        None,
        num_cells=args.num_cells,
        per_country=False,
    )
    
    # Assign geocells
    train_countries = None
    if hasattr(train_ds, '_country_labels') and train_ds.num_countries > 0:
        train_countries = train_ds._country_labels.numpy()
    val_countries = None
    if hasattr(val_ds, '_country_labels') and val_ds.num_countries > 0:
        val_countries = val_ds._country_labels.numpy()
    test_countries = None
    if test_ds and hasattr(test_ds, '_country_labels') and test_ds.num_countries > 0:
        test_countries = test_ds._country_labels.numpy()
    
    train_cell_labels = assign_geocells(train_ds._coords.numpy(), train_countries, centers_xyz, geocell_metadata)
    val_cell_labels = assign_geocells(val_ds._coords.numpy(), val_countries, centers_xyz, geocell_metadata)
    if test_ds:
        test_cell_labels = assign_geocells(test_ds._coords.numpy(), test_countries, centers_xyz, geocell_metadata)
    
    # Validate cell labels are in valid range
    num_cells_actual = len(centers_xyz)
    print(f"Validating cell labels (num_cells={num_cells_actual})...")
    if train_cell_labels.max() >= num_cells_actual or train_cell_labels.min() < 0:
        print(f"WARNING: Invalid train cell labels! min={train_cell_labels.min()}, max={train_cell_labels.max()}, clipping to [0, {num_cells_actual-1}]")
        train_cell_labels = np.clip(train_cell_labels, 0, num_cells_actual - 1)
    if val_cell_labels.max() >= num_cells_actual or val_cell_labels.min() < 0:
        print(f"WARNING: Invalid val cell labels! min={val_cell_labels.min()}, max={val_cell_labels.max()}, clipping to [0, {num_cells_actual-1}]")
        val_cell_labels = np.clip(val_cell_labels, 0, num_cells_actual - 1)
    if test_ds and (test_cell_labels.max() >= num_cells_actual or test_cell_labels.min() < 0):
        print(f"WARNING: Invalid test cell labels! min={test_cell_labels.min()}, max={test_cell_labels.max()}, clipping to [0, {num_cells_actual-1}]")
        test_cell_labels = np.clip(test_cell_labels, 0, num_cells_actual - 1)
    print(f"Cell label ranges: train=[{train_cell_labels.min()}, {train_cell_labels.max()}], val=[{val_cell_labels.min()}, {val_cell_labels.max()}]")
    
    # Compute offsets
    train_offsets = compute_offsets(train_ds._coords.numpy(), train_cell_labels, centers_xyz)
    val_offsets = compute_offsets(val_ds._coords.numpy(), val_cell_labels, centers_xyz)
    if test_ds:
        test_offsets = compute_offsets(test_ds._coords.numpy(), test_cell_labels, centers_xyz)
    
    # Wrap datasets with JointDataset to add geocell labels and offsets
    train_ds = JointDataset(train_ds, train_cell_labels, train_offsets)
    val_ds = JointDataset(val_ds, val_cell_labels, val_offsets)
    if test_ds:
        test_ds = JointDataset(test_ds, test_cell_labels, test_offsets)
    
    # Verify the wrapper works
    test_sample = train_ds[0]
    if len(test_sample) != 7:
        raise RuntimeError(f"Expected 7 elements from __getitem__, got {len(test_sample)}")
    print(f"Verified JointDataset returns {len(test_sample)} elements")
    
    # Save geocell info
    geocell_info = {
        'centers_xyz': centers_xyz.tolist(),
        'metadata': geocell_metadata,
        'num_cells': len(centers_xyz),
    }
    with open(phase2_output_dir / "geocells.json", 'w') as f:
        json.dump(geocell_info, f, indent=2)
    
    # Initialize models
    mix_local_kernel_size = args.mix_local_kernel_size if args.mix_local_kernel_size > 0 else None
    
    phase1_model = Phase1CBMTopKMil(
        num_concepts=num_concepts,
        patch_dim=patch_dim,
        concept_dim=args.concept_dim,
        dropout=args.phase1_dropout,
        mil_topk=args.mil_topk,
        mil_tau=args.mil_tau,
        mix_depth=args.mix_depth,
        mix_heads=args.mix_heads,
        mix_mlp_ratio=args.mix_mlp_ratio,
        mix_local_kernel_size=mix_local_kernel_size,
    ).to(device)
    
    # Build concept vectors
    concept_vectors = build_concept_vectors(
        phase1_model,
        concept_names,
        device,
        args.text_model_name,
        args.concept_prompt_template,
    ).to(device)
    
    concept_adapter = ConceptEmbeddingAdapter(concept_vectors, temperature=1.0).to(device)
    
    num_cells_actual = len(centers_xyz)
    stage2_model = Stage2CrossAttentionGeoHead(
        concept_dim=args.concept_dim,
        patch_dim=patch_dim,
        num_cells=num_cells_actual,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        mode=args.mode,
        pooled_dim=detected_pooled_dim,
    ).to(device)
    
    print(f"Phase1 params: {sum(p.numel() for p in phase1_model.parameters())/1e6:.2f}M")
    print(f"Stage2 params: {sum(p.numel() for p in stage2_model.parameters())/1e6:.2f}M")
    
    # Load checkpoints if provided
    start_epoch = 0
    best_val_concept_acc1 = 0.0
    best_val_error = float('inf')
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    # Handle resume from joint checkpoint directory
    if args.resume_joint_checkpoint:
        resume_dir = Path(args.resume_joint_checkpoint)
        phase1_checkpoint_path = resume_dir / "phase1" / "latest.pt"
        phase2_checkpoint_path = resume_dir / "phase2" / "latest.pt"
        
        if phase1_checkpoint_path.exists():
            args.phase1_checkpoint = str(phase1_checkpoint_path)
            print(f"Resuming Phase1 from joint checkpoint: {phase1_checkpoint_path}")
        if phase2_checkpoint_path.exists():
            args.phase2_checkpoint = str(phase2_checkpoint_path)
            print(f"Resuming Phase2 from joint checkpoint: {phase2_checkpoint_path}")
    
    # Load Phase1 checkpoint (weights only, not training state)
    if args.phase1_checkpoint:
        print(f"Loading Phase1 checkpoint from {args.phase1_checkpoint}...")
        checkpoint = torch.load(args.phase1_checkpoint, map_location=device, weights_only=False)
        phase1_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Update concept vectors from loaded model
        concept_vectors = build_concept_vectors(
            phase1_model,
            concept_names,
            device,
            args.text_model_name,
            args.concept_prompt_template,
        ).to(device)
        concept_adapter = ConceptEmbeddingAdapter(concept_vectors, temperature=1.0).to(device)
        
        # Only extract training state if resuming from joint checkpoint (not when loading separate checkpoints)
        # When loading separate checkpoints, we only use weights as initialization
        if args.resume_joint_checkpoint:
            if 'epoch' in checkpoint:
                start_epoch = max(start_epoch, checkpoint.get('epoch', 0) + 1)
            if 'best_val_concept_acc1' in checkpoint:
                best_val_concept_acc1 = checkpoint.get('best_val_concept_acc1', 0.0)
            print(f"Phase1 checkpoint loaded. Starting from epoch {start_epoch}, best_val_concept_acc1={best_val_concept_acc1:.4f}")
        else:
            print(f"Phase1 checkpoint loaded (weights only). Starting from epoch 0.")
    
    # Load Phase2 checkpoint (weights only, not training state)
    if args.phase2_checkpoint:
        print(f"Loading Phase2 checkpoint from {args.phase2_checkpoint}...")
        checkpoint = torch.load(args.phase2_checkpoint, map_location=device, weights_only=False)
        stage2_model.load_state_dict(checkpoint['stage2_model_state_dict'])
        if 'concept_adapter_state_dict' in checkpoint:
            concept_adapter.load_state_dict(checkpoint['concept_adapter_state_dict'])
        
        # Only extract training state if resuming from joint checkpoint (not when loading separate checkpoints)
        # When loading separate checkpoints, we only use weights as initialization
        if args.resume_joint_checkpoint:
            if 'epoch' in checkpoint:
                start_epoch = max(start_epoch, checkpoint.get('epoch', 0) + 1)
            if 'best_val_error' in checkpoint:
                best_val_error = checkpoint.get('best_val_error', float('inf'))
            if 'best_val_loss' in checkpoint and checkpoint['best_val_loss'] is not None:
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            if 'epochs_without_improvement' in checkpoint:
                epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
            print(f"Phase2 checkpoint loaded. Starting from epoch {start_epoch}, best_val_error={best_val_error:.2f}km")
        else:
            print(f"Phase2 checkpoint loaded (weights only). Starting from epoch 0.")
    
    # Create collate functions with pooled embeddings bound
    def make_collate_fn(pooled_emb):
        return lambda b: collate_fn_joint(b, pooled_emb)
    
    # Data loaders (single dataset for both tasks)
    # Can use multiprocessing now since JointDataset is properly pickleable
    num_workers = args.num_workers if hasattr(args, 'num_workers') else 2
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=make_collate_fn(train_ds.pooled_embeddings),
        num_workers=num_workers, 
        pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=make_collate_fn(val_ds.pooled_embeddings),
        num_workers=num_workers, 
        pin_memory=False
    )
    test_loader = None
    if test_ds:
        if test_ds.pooled_embeddings is None:
            raise RuntimeError("Expected test_ds.pooled_embeddings to be loaded (got None).")
        test_loader = DataLoader(
            test_ds, 
            batch_size=args.batch_size, 
            shuffle=False, 
            collate_fn=make_collate_fn(test_ds.pooled_embeddings),
            num_workers=num_workers, 
            pin_memory=False
        )
    
    # Loss functions
    concept_weights = compute_concept_weights(train_ds, num_concepts, device)
    concept_criterion = nn.CrossEntropyLoss(weight=concept_weights, label_smoothing=0.1)
    cell_criterion = nn.CrossEntropyLoss()
    offset_criterion = nn.MSELoss()
    
    # Optimizer (can use separate LRs)
    phase1_lr = args.phase1_lr if args.phase1_lr is not None else args.lr
    optimizer = optim.AdamW(
        [
            {'params': phase1_model.parameters(), 'lr': phase1_lr},
            {'params': stage2_model.parameters(), 'lr': args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    
    # Load optimizer state if resuming
    # Note: When loading from separate Phase1/Phase2 checkpoints, optimizer state won't match
    # (separate training has different parameter groups), so we skip optimizer loading
    # Only load optimizer state when resuming from a joint checkpoint (which has matching structure)
    if args.resume_joint_checkpoint and start_epoch > 0:
        # Load optimizer state from Phase2 checkpoint (it contains the full optimizer state)
        phase2_checkpoint_path = Path(args.resume_joint_checkpoint) / "phase2" / "latest.pt"
        if phase2_checkpoint_path.exists():
            checkpoint = torch.load(phase2_checkpoint_path, map_location=device, weights_only=False)
            if 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("Loaded optimizer state from joint checkpoint")
                except (ValueError, KeyError) as e:
                    print(f"Warning: Could not load optimizer state (parameter groups mismatch): {e}")
                    print("Starting with fresh optimizer state")
    # When loading from separate Phase1/Phase2 checkpoints (not joint), skip optimizer loading
    # because the optimizer structure is different (separate training has 1 param group, joint has 2)
    
    # LR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Load scheduler state only if resuming from joint checkpoint
    # When loading separate checkpoints, scheduler starts fresh
    if args.resume_joint_checkpoint and start_epoch > 0:
        phase2_checkpoint_path = Path(args.resume_joint_checkpoint) / "phase2" / "latest.pt"
        if phase2_checkpoint_path.exists():
            checkpoint = torch.load(phase2_checkpoint_path, map_location=device, weights_only=False)
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("Loaded scheduler state from joint checkpoint")
    
    # Training loop
    
    early_stop_enabled = args.early_stop_patience > 0
    
    print(f"\nStarting joint training for {args.epochs} epochs...")
    print(f"Resuming from epoch {start_epoch}")
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_concept_loss, train_cell_loss, train_offset_loss, train_concept_acc1, train_concept_acc5, train_cell_acc = train_epoch(
            phase1_model,
            stage2_model,
            concept_adapter,
            train_loader,
            device,
            concept_criterion,
            cell_criterion,
            offset_criterion,
            optimizer,
            epoch,
            args.concept_loss_weight,
            args.cell_loss_weight,
            args.offset_loss_weight,
        )
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        val_loss, val_concept_loss, val_cell_loss, val_offset_loss, val_concept_acc1, val_concept_acc5, val_cell_acc, val_mean_error, val_median_error, val_threshold_accs, val_preds = eval_epoch(
            phase1_model,
            stage2_model,
            concept_adapter,
            val_loader,
            device,
            concept_criterion,
            cell_criterion,
            offset_criterion,
            centers_xyz,
            args.concept_loss_weight,
            args.cell_loss_weight,
            args.offset_loss_weight,
        )
        
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train: Loss={train_loss:.4f}, Concept={train_concept_loss:.4f} (Acc1={train_concept_acc1:.4f}), Geo={train_cell_loss:.4f}+{train_offset_loss:.4f} (CellAcc={train_cell_acc:.4f})")
        print(f"  Val:   Loss={val_loss:.4f}, Concept={val_concept_loss:.4f} (Acc1={val_concept_acc1:.4f}), Geo={val_cell_loss:.4f}+{val_offset_loss:.4f} (CellAcc={val_cell_acc:.4f})")
        print(f"  Val Error: Mean={val_mean_error:.2f}km, Median={val_median_error:.2f}km")
        print(f"  LR: {current_lr:.6f}")
        
        if wandb_run:
            wandb_run.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/concept_loss": train_concept_loss,
                "train/cell_loss": train_cell_loss,
                "train/offset_loss": train_offset_loss,
                "train/concept_acc1": train_concept_acc1,
                "train/concept_acc5": train_concept_acc5,
                "train/cell_acc": train_cell_acc,
                "val/loss": val_loss,
                "val/concept_loss": val_concept_loss,
                "val/cell_loss": val_cell_loss,
                "val/offset_loss": val_offset_loss,
                "val/concept_acc1": val_concept_acc1,
                "val/concept_acc5": val_concept_acc5,
                "val/cell_acc": val_cell_acc,
                "val/mean_error_km": val_mean_error,
                "val/median_error_km": val_median_error,
                "lr": current_lr,
            })
            wandb_run.log({f"val/{k}": v for k, v in val_threshold_accs.items()})
        
        # Save checkpoints
        is_best_concept = val_concept_acc1 > best_val_concept_acc1
        is_best_geo = val_mean_error < best_val_error
        
        if is_best_concept:
            best_val_concept_acc1 = val_concept_acc1
        if is_best_geo:
            best_val_error = val_mean_error
        
        # Early stopping
        if early_stop_enabled:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= args.early_stop_patience:
                print(f"\n*** Early stopping triggered! ***")
                break
        
        # Save Phase1 checkpoint
        phase1_checkpoint = {
            'epoch': epoch,
            'model_state_dict': phase1_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_concept_acc1': best_val_concept_acc1,
            'val_concept_acc1': val_concept_acc1,
            'val_concept_acc5': val_concept_acc5,
        }
        torch.save(phase1_checkpoint, phase1_output_dir / "latest.pt")
        if is_best_concept:
            torch.save(phase1_checkpoint, phase1_output_dir / "best_phase1.pt")
            
            # Generate attention map visualizations for best checkpoint
            vis_dataset = VisualizationDataset(val_ds)
            print("  Generating Phase1 attention visualizations...")
            visualize_predictions_summary(
                model=phase1_model,
                dataset=vis_dataset,
                device=device,
                epoch=epoch,
                output_dir=phase1_output_dir,
                phase=1,
                run_id=run_id,
                reason="best_val_acc",
                num_samples=5,
            )
        
        # Periodic attention visualizations (every 5 epochs)
        if (epoch + 1) % 5 == 0:
            vis_dataset = VisualizationDataset(val_ds)
            print("  Generating periodic Phase1 attention visualizations...")
            visualize_predictions_summary(
                model=phase1_model,
                dataset=vis_dataset,
                device=device,
                epoch=epoch,
                output_dir=phase1_output_dir,
                phase=1,
                run_id=run_id,
                reason="periodic_every5",
                num_samples=3,
            )
        
        # Save Phase2 checkpoint
        phase2_checkpoint = {
            'epoch': epoch,
            'stage2_model_state_dict': stage2_model.state_dict(),
            'concept_adapter_state_dict': concept_adapter.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_error': best_val_error,
            'val_mean_error': val_mean_error,
            'val_median_error': val_median_error,
            'val_cell_acc': val_cell_acc,
            'geocell_info': geocell_info,
        }
        torch.save(phase2_checkpoint, phase2_output_dir / "latest.pt")
        if is_best_geo:
            torch.save(phase2_checkpoint, phase2_output_dir / "best_phase2.pt")
            print(f"  *** New best val error: {val_mean_error:.2f}km ***")
            
            # Save Phase2 geolocation visualizations
            if len(val_preds['pred_lat']) > 0:
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
    
    print(f"\nTraining complete!")
    print(f"Best val concept Acc1: {best_val_concept_acc1:.4f}")
    print(f"Best val geo error: {best_val_error:.2f}km")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()

