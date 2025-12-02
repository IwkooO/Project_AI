#!/usr/bin/env python3
"""
Training script for Concept-Aware Global Image-GPS Alignment.
"""

import argparse
import logging
import csv
from typing import Dict, List, Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from datetime import datetime
from src.dataset import (
    PanoramaCBMDataset,
    create_splits_stratified,
    get_transforms_from_processor,
)
from src.models.streetclip_encoder import StreetCLIPEncoder, StreetCLIPConfig
from src.models.concept_aware_cbm import ConceptAwareGeoModel
from src.losses import (
    coordinate_loss,
    clip_contrastive_loss,
    geocell_contrastive_loss,
)
from src.concepts.utils import extract_concepts_from_dataset
from src.evaluation import (
    denormalize_coordinates,
    haversine_distance,
    sphere_to_latlng,
    accuracy_within_threshold,
)
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Constants ----------
ENCODER_MODEL_TO_NAME = {
    "geolocal/StreetCLIP": "streetclip",
    "facebook/dinov3-vit7b16-pretrain-lvd1689m": "dinov3",
    "facebook/dinov2-base": "dinov2",
}


def generate_semantic_geocells(dataset, min_samples_per_cell=500, output_dir=None):
    """Generate semantic geocells using per-country K-Means clustering."""
    logger.info("Generating Semantic Geocells...")

    all_coords = []
    all_countries = []

    if hasattr(dataset, "samples"):
        samples = dataset.samples
    else:
        raise ValueError("Dataset format not recognized for cell generation")

    for s in samples:
        all_coords.append([s["lat"], s["lng"]])
        all_countries.append(s["country"])

    all_coords = np.array(all_coords)
    all_countries = np.array(all_countries)

    unique_countries = np.unique(all_countries)

    cell_centers_list = []
    sample_to_cell_map = np.zeros(len(samples), dtype=int)

    current_cell_id_offset = 0

    for country in tqdm(unique_countries, desc="Clustering Countries"):
        country_mask = all_countries == country
        country_indices = np.where(country_mask)[0]
        country_coords = all_coords[country_indices]
        n_samples = len(country_coords)

        lat_rad = np.deg2rad(country_coords[:, 0])
        lng_rad = np.deg2rad(country_coords[:, 1])
        x = np.cos(lat_rad) * np.cos(lng_rad)
        y = np.cos(lat_rad) * np.sin(lng_rad)
        z = np.sin(lat_rad)

        if n_samples > min_samples_per_cell:
            k = max(1, n_samples // min_samples_per_cell)
            cart_coords = np.stack([x, y, z], axis=1)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(cart_coords)
            centers = kmeans.cluster_centers_
            centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
            global_labels = kmeans.labels_ + current_cell_id_offset
            sample_to_cell_map[country_indices] = global_labels
            for center in centers:
                cell_centers_list.append(center)
            current_cell_id_offset += k
        else:
            center = np.array([np.mean(x), np.mean(y), np.mean(z)])
            center = center / np.linalg.norm(center)
            cell_centers_list.append(center)
            sample_to_cell_map[country_indices] = current_cell_id_offset
            current_cell_id_offset += 1

    cell_centers = torch.tensor(np.stack(cell_centers_list), dtype=torch.float32)
    sample_to_cell = torch.tensor(sample_to_cell_map, dtype=torch.long)
    logger.info(f"Generated {len(cell_centers)} Semantic Geocells.")

    # Visualization
    if output_dir:
        from matplotlib.patches import Rectangle
        cx, cy, cz = cell_centers[:, 0], cell_centers[:, 1], cell_centers[:, 2]
        clat = np.rad2deg(np.arcsin(cz.numpy()))
        clng = np.rad2deg(np.arctan2(cy.numpy(), cx.numpy()))
        
        png_path = Path(output_dir) / "geocells_map.png"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(16, 9))
        max_viz = min(len(all_coords), 10000)
        indices = np.random.choice(len(all_coords), size=max_viz, replace=False) if len(all_coords) > max_viz else np.arange(len(all_coords))
        scatter = ax.scatter(all_coords[indices, 1], all_coords[indices, 0], c=sample_to_cell_map[indices], s=2, alpha=0.5, cmap='tab20')
        ax.scatter(clng, clat, c='red', s=100, marker='*', edgecolors='black', linewidths=0.5, label='Cell Centers', zorder=10)
        plt.colorbar(scatter, ax=ax, label='Cell ID')
        ax.set_xlim([-180, 180])
        ax.set_ylim([-90, 90])
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Semantic Geocells (K={len(cell_centers)})')
        ax.add_patch(Rectangle((-180, -90), 360, 180, fill=False, edgecolor='black', linewidth=1.5))
        plt.savefig(str(png_path), dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved geocell visualization to {png_path}")

    return cell_centers, sample_to_cell


def collate_batch(batch):
    """Custom collate for variable-length metadata."""
    images = torch.stack([item[0] for item in batch])
    concept_indices = torch.tensor([item[1] for item in batch], dtype=torch.long)
    target_indices = torch.tensor([item[2] for item in batch], dtype=torch.long)
    coordinates = torch.stack([item[3] for item in batch])
    metadata = [item[4] for item in batch]
    return images, concept_indices, target_indices, coordinates, metadata


@torch.no_grad()
def visualize_predictions(model, val_loader, concept_names, idx_to_country, device, args, checkpoint_dir, epoch, cell_centers, num_samples=4):
    """Visualize top predicted concepts and location predictions."""
    model.eval()
    viz_dir = checkpoint_dir / "visualizations" / f"epoch_{epoch}"
    viz_dir.mkdir(parents=True, exist_ok=True)
    wandb_images = []

    for batch in val_loader:
        images, concept_indices, _, coords, metadata, cell_labels, note_embs = batch
        images = images.to(device)
        coords = coords.to(device)
        concept_indices = concept_indices.to(device)

        outputs = model(images, coords)
        concept_logits = outputs["concept_logits"]
        country_logits = outputs["country_logits"]
        cell_logits = outputs["cell_logits"]
        pred_offsets = outputs["pred_offsets"]

        pred_cells = cell_logits.argmax(dim=1)
        batch_cell_centers = cell_centers[pred_cells]

        if model.coord_output_dim == 3:
            pred_cart = batch_cell_centers + pred_offsets
            pred_cart = torch.nn.functional.normalize(pred_cart, p=2, dim=1)
            pred_coords = sphere_to_latlng(pred_cart)
        else:
            c_x, c_y, c_z = batch_cell_centers[:, 0], batch_cell_centers[:, 1], batch_cell_centers[:, 2]
            c_lat = torch.rad2deg(torch.asin(c_z))
            c_lng = torch.rad2deg(torch.atan2(c_y, c_x))
            pred_coords = torch.stack([c_lat, c_lng], dim=1) + pred_offsets

        n_display = min(len(images), num_samples)

        for i in range(n_display):
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
            ax_img, ax_bar = axes[0], axes[1]

            probs = concept_logits[i]
            top_scores, top_indices = torch.topk(probs, k=5)

            img_cpu = images[i].cpu().permute(1, 2, 0).numpy()
            mean, std = np.array([0.48145466, 0.4578275, 0.40821073]), np.array([0.26862954, 0.26130258, 0.27577711])
            img_disp = np.clip(std * img_cpu + mean, 0, 1)
            ax_img.imshow(img_disp)
            ax_img.axis("off")

            gt_lat, gt_lng, gt_country = metadata[i]["lat"], metadata[i]["lng"], metadata[i]["country"]
            pred_coords_raw = pred_coords[i].detach().cpu()
            
            if pred_coords_raw.shape[0] == 3:
                pred_coords_deg = sphere_to_latlng(pred_coords_raw.unsqueeze(0)).squeeze(0)
                pred_lat, pred_lng = pred_coords_deg[0].item(), pred_coords_deg[1].item()
            else:
                pred_lat, pred_lng = pred_coords_raw.numpy()

            gt_coord_tensor = coords[i].unsqueeze(0)
            pred_coord_tensor = torch.tensor([pred_lat, pred_lng], device=device).unsqueeze(0) if pred_coords_raw.shape[0] == 3 else pred_coords[i].unsqueeze(0)
            distance_km = haversine_distance(pred_coord_tensor, gt_coord_tensor).item()

            pred_country_idx = country_logits[i].argmax().item()
            pred_country_cls = idx_to_country[pred_country_idx]
            gt_concept_idx = concept_indices[i].item()
            gt_concept_name = concept_names[gt_concept_idx]
            top_concept_name = concept_names[top_indices[0].item()]

            title = f"Epoch {epoch} | ID: {metadata[i]['pano_id']}\n"
            title += f"Country: {pred_country_cls} | True: {gt_country}\n"
            title += f"Coords: ({pred_lat:.3f}, {pred_lng:.3f}) | True: ({gt_lat:.3f}, {gt_lng:.3f})\n"
            title += f"Concept: {top_concept_name} | True: {gt_concept_name} | Error: {distance_km:.1f}km"
            ax_img.set_title(title, fontsize=10)

            top_indices_cpu = top_indices.cpu().numpy()
            bar_colors = ["orange" if idx == gt_concept_idx else "steelblue" for idx in top_indices_cpu]
            ax_bar.barh(np.arange(5), top_scores.cpu().numpy(), color=bar_colors)
            ax_bar.set_yticks(np.arange(5))
            ax_bar.set_yticklabels([concept_names[idx.item()] for idx in top_indices])
            ax_bar.invert_yaxis()
            ax_bar.set_xlabel("Activation Score")
            ax_bar.set_title("Top 5 Concepts" + ("" if gt_concept_idx in top_indices_cpu else f" | GT: {gt_concept_name}"))

            plt.tight_layout()
            save_path = viz_dir / f"sample_{i}.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            if args.use_wandb:
                wandb_images.append(wandb.Image(str(save_path), caption=f"Epoch {epoch} Sample {i}"))
            plt.close(fig)

        break

    if args.use_wandb and wandb_images:
        wandb.log({f"predictions/epoch_{epoch}": wandb_images})


@torch.no_grad()
def dump_diagnostics(model, dataloader, device, output_path: Path, concept_names: List[str], idx_to_country: Dict[int, str], cell_centers: torch.Tensor, max_samples: int = 64, log_to_wandb: bool = False, wandb_step: Optional[int] = None):
    """Dump prediction diagnostics to CSV."""
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for batch in dataloader:
        if len(rows) >= max_samples:
            break

        images, concept_idx, target_idx, coords, metadata, cell_labels, note_embs = batch
        images, coords = images.to(device), coords.to(device)
        concept_idx, target_idx = concept_idx.to(device), target_idx.to(device)

        outputs = model(images, coords)
        concept_logits, country_logits = outputs["concept_logits"], outputs["country_logits"]
        cell_logits, pred_offsets = outputs["cell_logits"], outputs["pred_offsets"]

        pred_cells = cell_logits.argmax(dim=1)
        batch_cell_centers = cell_centers[pred_cells]

        if model.coord_output_dim == 3:
            pred_cart = torch.nn.functional.normalize(batch_cell_centers + pred_offsets, p=2, dim=1)
            pred_coords = sphere_to_latlng(pred_cart)
        else:
            c_x, c_y, c_z = batch_cell_centers[:, 0], batch_cell_centers[:, 1], batch_cell_centers[:, 2]
            pred_coords = torch.stack([torch.rad2deg(torch.asin(c_z)), torch.rad2deg(torch.atan2(c_y, c_x))], dim=1) + pred_offsets

        concept_probs = torch.softmax(concept_logits, dim=1)
        country_probs = torch.softmax(country_logits, dim=1)

        for i in range(len(images)):
            if len(rows) >= max_samples:
                break

            top_concept_idx = concept_probs[i].argmax().item()
            pred_country_idx = country_probs[i].argmax().item()
            pred_coords_raw = pred_coords[i].detach().cpu()
            
            if pred_coords_raw.shape[0] == 3:
                pred_coords_deg = sphere_to_latlng(pred_coords_raw.unsqueeze(0)).squeeze(0)
                pred_lat, pred_lng = pred_coords_deg[0].item(), pred_coords_deg[1].item()
            else:
                pred_lat, pred_lng = pred_coords_raw.numpy()

            gt_coord_tensor = coords[i].unsqueeze(0)
            pred_coord_tensor = torch.tensor([pred_lat, pred_lng], device=device).unsqueeze(0) if pred_coords_raw.shape[0] == 3 else pred_coords[i].unsqueeze(0)
            distance_km = haversine_distance(pred_coord_tensor, gt_coord_tensor).item()

            row = {
                "pred_lat": float(pred_lat), "pred_lng": float(pred_lng),
                "true_lat": float(metadata[i]["lat"]), "true_lng": float(metadata[i]["lng"]),
                "distance_km": distance_km,
                "top_concept": concept_names[top_concept_idx], "true_concept": concept_names[concept_idx[i].item()],
                "pred_country": idx_to_country[pred_country_idx], "true_country": idx_to_country[target_idx[i].item()],
                "concept_correct": bool(top_concept_idx == concept_idx[i].item()),
                "country_correct": bool(pred_country_idx == target_idx[i].item()),
            }
            if "pano_id" in metadata[i]:
                row["pano_id"] = metadata[i]["pano_id"]
            rows.append(row)

    if not rows:
        return

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Dumped {len(rows)} diagnostic samples to {output_path}")

    if log_to_wandb:
        table = wandb.Table(columns=list(rows[0].keys()))
        for row in rows:
            table.add_data(*row.values())
        wandb.log({f"diagnostics/{output_path.stem}": table}, step=wandb_step)


def create_checkpoint_dir(
    encoder_model: str = None,
    country_filter: str = None,
    coordinate_loss_type: str = "haversine",
) -> Path:
    """Create checkpoint directory with format:
    results/concept-aware/<encoder_name>/<country>/<loss_type>/<timestamp>/
    where:
    - encoder_name: sanitized encoder model name (replace "/" with "-")
    - country: country name or "global" if no country filter
    - loss_type: coordinate loss type (haversine, mse, sphere, etc.)
    - timestamp: formatted date and time (YYYY-MM-DD_HH-MM-SS)

    Creates subdirectories: checkpoints/, logs/, visualizations/, diagnostics/
    """
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Encoder name: sanitize model name (replace "/" with "-")
    if encoder_model:
        encoder_name = ENCODER_MODEL_TO_NAME.get(encoder_model, encoder_model)
        encoder_name = encoder_name.replace("/", "-")
    else:
        encoder_name = "unknown"

    # Country: use filter or "global" if None
    country = country_filter if country_filter else "global"

    # Loss type: use provided coordinate loss type
    loss_type = coordinate_loss_type.lower()

    # Create full directory structure
    timestamp_dir = (
        Path("results")
        / "concept-aware"
        / encoder_name
        / country
        / loss_type
        / timestamp
    )
    timestamp_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (timestamp_dir / "checkpoints").mkdir(exist_ok=True)
    (timestamp_dir / "logs").mkdir(exist_ok=True)
    (timestamp_dir / "visualizations").mkdir(exist_ok=True)
    (timestamp_dir / "diagnostics").mkdir(exist_ok=True)

    return timestamp_dir


# ---------- Main Training Function ----------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ---------- Setup Data & Image Encoder ----------
    logger.info("Initializing Dataset...")

    # Use StreetCLIP transforms
    base_encoder = StreetCLIPEncoder(StreetCLIPConfig(model_name=args.encoder_model))
    transforms = get_transforms_from_processor(base_encoder.image_processor)

    # Dataset to use - PanoramaCBMDataset with optional CSV path
    full_dataset = PanoramaCBMDataset(
        transform=transforms,
        require_coordinates=True,
        country=args.country_filter,
        use_normalized_coordinates=False,
        encoder_model=args.encoder_model,
        geoguessr_id=args.geoguessr_id,
        data_root=args.data_root,
        csv_path=args.csv_path,
    )

    # Diagnostic: Check concept distribution
    all_concepts = [s["meta_name"] for s in full_dataset.samples]
    concept_counts = Counter(all_concepts)
    logger.info(
        f"Total samples: {len(full_dataset.samples)}, Concepts: {len(concept_counts)}"
    )
    logger.info(
        f"Samples per concept - Min: {min(concept_counts.values())}, Max: {max(concept_counts.values())}, Avg: {len(full_dataset.samples)/len(concept_counts):.1f}"
    )
    logger.info("Concept distribution (Top 10):")
    for name, count in concept_counts.most_common(10):
        logger.info(f"  {name}: {count}")

    # Use more balanced split: 70/20/10 to get more validation samples
    # With only 251 samples, 10% validation gives only ~21 samples which is too small
    train_samples, val_samples, test_samples = create_splits_stratified(
        full_dataset.samples, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1
    )

    # ---------- Compute Class Weights for Concept Imbalance ----------
    def compute_concept_weights(train_samples, concept_to_idx, device):
        """
        Compute class weights using inverse frequency weighting.
        Formula: weight[i] = total_samples / (num_concepts * count[i])
        This gives rare concepts higher weight, common concepts lower weight.
        
        This function works for ANY distribution:
        - If concept has 1 sample: weight = total_samples / num_concepts (highest)
        - If concept has many samples: weight approaches 0 (lowest)
        - Normalized so weights sum to num_concepts (maintains loss scale)
        """
        concept_counts = Counter(s['meta_name'] for s in train_samples)
        num_concepts = len(concept_to_idx)
        total_samples = len(train_samples)
        
        weights = torch.ones(num_concepts, device=device)
        for concept_name, idx in concept_to_idx.items():
            count = concept_counts.get(concept_name, 1)  # Avoid division by zero
            weights[idx] = total_samples / (num_concepts * count)
        
        # Normalize so weights sum to num_concepts (keeps loss scale similar)
        weights = weights * (num_concepts / weights.sum())
        
        return weights

    # Compute concept weights if enabled
    concept_weights = None
    if args.use_class_weights:
        concept_weights = compute_concept_weights(train_samples, full_dataset.concept_to_idx, device)
        logger.info(f"Computed concept weights - Min: {concept_weights.min():.4f}, "
                   f"Max: {concept_weights.max():.4f}, Mean: {concept_weights.mean():.4f}, "
                   f"Std: {concept_weights.std():.4f}")
    else:
        logger.info("Class weights disabled - using uniform weighting")

    # Create subset datasets (need to implement wrapper or just list sampling)
    # Re-using the logic from dataset.py main block
    from src.dataset import SubsetDataset

    train_dataset = SubsetDataset(full_dataset, train_samples)
    val_dataset = SubsetDataset(full_dataset, val_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,  # Important for contrastive loss stability
        collate_fn=collate_batch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_batch,
    )

    test_dataset = SubsetDataset(full_dataset, test_samples)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_batch,
    )

    logger.info(
        f"Split sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}"
    )
    logger.info(
        f"Batches per epoch: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}"
    )

    # Check train concept distribution
    train_concepts = [s["meta_name"] for s in train_samples]
    train_concept_counts = Counter(train_concepts)
    logger.info(
        f"Train set: {len(train_concepts)} samples across {len(train_concept_counts)} concepts"
    )

    # Precompute Note Embeddings for Training
    logger.info("Pre-computing Note Embeddings for Semantic Alignment...")
    all_notes = [s["note"] for s in full_dataset.samples]

    note_embeddings_list = []
    with torch.no_grad():
        for i in range(0, len(all_notes), 32):
            batch_notes = all_notes[i : i + 32]
            # Replace empty notes with " " to avoid tokenizer errors
            batch_notes = [n if n and n.strip() else " " for n in batch_notes]
            feats = base_encoder.get_text_features(text=batch_notes)
            note_embeddings_list.append(feats.cpu())
    all_note_embeddings = torch.cat(note_embeddings_list, dim=0)

    # Inject into samples
    for i, sample in enumerate(full_dataset.samples):
        sample["note_embedding"] = all_note_embeddings[i]

    # Re-define collate to handle cells and note embeddings
    def collate_batch_v2(batch):
        images = torch.stack([item[0] for item in batch])
        concept_indices = torch.tensor([item[1] for item in batch], dtype=torch.long)
        target_indices = torch.tensor([item[2] for item in batch], dtype=torch.long)
        coordinates = torch.stack([item[3] for item in batch])
        metadata = [item[4] for item in batch]

        cell_labels = []
        note_embs = []

        for m in metadata:
            pid = m["pano_id"]
            cell_labels.append(pano_to_cell[pid])
            note_embs.append(pano_to_note_emb[pid])

        cell_labels = torch.tensor(cell_labels, dtype=torch.long)
        note_embs = torch.stack(note_embs)

        return (
            images,
            concept_indices,
            target_indices,
            coordinates,
            metadata,
            cell_labels,
            note_embs,
        )

    # Build lookup maps
    pano_to_note_emb = {s["pano_id"]: s["note_embedding"] for s in full_dataset.samples}
    # pano_to_cell already built

    # Update DataLoaders to use v2 collate
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_batch_v2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_batch_v2,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_batch_v2,
    )

    logger.info(
        f"  Samples per concept - Min: {min(train_concept_counts.values())}, Max: {max(train_concept_counts.values())}, Avg: {len(train_concepts)/len(train_concept_counts):.1f}"
    )

    # ---------- Setup Output Directory ----------
    if args.output_dir is None:
        checkpoint_dir = create_checkpoint_dir(
            encoder_model=args.encoder_model,
            country_filter=args.country_filter,
            coordinate_loss_type=args.coordinate_loss_type,
        )
    else:
        checkpoint_dir = Path(args.output_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Ensure subdirectories exist
        (checkpoint_dir / "checkpoints").mkdir(exist_ok=True)
        (checkpoint_dir / "logs").mkdir(exist_ok=True)
        (checkpoint_dir / "visualizations").mkdir(exist_ok=True)
        (checkpoint_dir / "diagnostics").mkdir(exist_ok=True)

    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    # Setup file logging
    log_file = checkpoint_dir / "logs" / "training.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    logger.info(f"Logging to {log_file}")

    # ---------- Initialize WandB ----------
    if args.use_wandb:
        # Create tags
        tags = [
            args.encoder_model,
            args.country_filter if args.country_filter else "global",
            "finetuned" if args.finetune_encoder else "frozen",
            "concept_aware",
            args.coordinate_loss_type,
            args.csv_path if args.csv_path else args.geoguessr_id,
        ]

        wandb.init(
            project="concept-aware-geolocation",
            name=f"concept-aware-geolocation-{args.country_filter if args.country_filter else 'global'}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=args,
            tags=tags,
            dir=str(checkpoint_dir),
        )

    # ---------- Semantic Geocell Generation ----------
    viz_dir = checkpoint_dir / "visualizations"
    cell_centers, sample_to_cell = generate_semantic_geocells(
        full_dataset, min_samples_per_cell=args.min_samples_per_cell, output_dir=viz_dir
    )

    # Log geocell map to wandb if enabled
    geocell_png_path = viz_dir / "geocells_map.png"
    if args.use_wandb and geocell_png_path.exists():
        wandb.log(
            {
                "geocells_map": wandb.Image(
                    str(geocell_png_path), caption="Semantic Geocells Distribution"
                )
            }
        )

    cell_centers = cell_centers.to(device)
    num_cells = len(cell_centers)

    # Assign cell labels to datasets
    # We need to be careful mapping back to train/val/test splits
    # SubsetDataset doesn't easily support adding new attributes, so we'll create a lookup tensor/dict
    # sample_to_cell is aligned with full_dataset.samples

    for i, sample in enumerate(full_dataset.samples):
        sample["cell_label"] = sample_to_cell[i].item()

    pano_to_cell = {s["pano_id"]: s["cell_label"] for s in full_dataset.samples}

    # ---------- Concept Extraction & Encoding ----------
    logger.info("Extracting and encoding concepts...")
    # We need concepts from the ENTIRE dataset to build the basis, not just training set
    concept_names, concept_map = extract_concepts_from_dataset(full_dataset)

    logger.info(f"Found {len(concept_names)} unique concepts.")

    # Verify concept alignment
    # concept_names should match keys in full_dataset.concept_to_idx (both sorted by name)
    dataset_concepts = sorted(full_dataset.concept_to_idx.keys())
    if concept_names != dataset_concepts:
        logger.error(
            "Concept mismatch between extract_concepts_from_dataset and dataset.concept_to_idx!"
        )
        logger.error(f"Extract: {concept_names[:5]}...")
        logger.error(f"Dataset: {dataset_concepts[:5]}...")
        raise ValueError("Concept alignment failed")

    # Further verification: check index mapping
    for i, name in enumerate(concept_names):
        idx = full_dataset.concept_to_idx[name]
        if idx != i:
            raise ValueError(f"Index mismatch for {name}: expected {i}, got {idx}")
    logger.info("Concept alignment verified.")

    # Encode concepts to get E_concept
    # Move encoder to device for inference - ensure model is explicitly moved
    base_encoder.model.to(device)
    base_encoder = base_encoder.to(device)  # Also move any other components
    # Verify model is on the correct device (compare device types, not exact strings)
    model_device = next(base_encoder.model.parameters()).device
    if model_device.type != device.type:
        raise RuntimeError(
            f"Model is on {model_device.type} but expected {device.type}"
        )
    logger.info(f"Base encoder device: {model_device}")

    # ---------- Initialize Concept-Aware Model ----------
    logger.info("Initializing ConceptAwareGeoModel (Strict CBM Architecture)...")
    # Configure StreetCLIP for training (finetune=False usually for encoder)
    encoder_config = StreetCLIPConfig(
        model_name=args.encoder_model, finetune=args.finetune_encoder, device=device
    )
    # Re-create encoder to ensure clean state / config application
    image_encoder = StreetCLIPEncoder(encoder_config)
    actual_feature_dim = image_encoder.feature_dim
    logger.info(f"StreetCLIP vision encoder dimension: {actual_feature_dim}")

    # Determine coordinate output dimension
    coord_output_dim = 3 if args.coordinate_loss_type == "sphere" else 2
    logger.info(f"Coordinate output dimension: {coord_output_dim}")

    # CBM Architecture: Image → Concept Bottleneck → 512d Concept Embeddings
    # All downstream heads operate on concept embeddings only
    model = ConceptAwareGeoModel(
        image_encoder=image_encoder,
        num_concepts=len(concept_names),
        num_countries=len(full_dataset.country_to_idx),
        num_cells=num_cells,  # Semantic Cells
        streetclip_dim=actual_feature_dim,
        concept_emb_dim=512,  # Matches StreetCLIP text encoder output
        coord_output_dim=coord_output_dim,
        text_encoder=base_encoder,  # Pass frozen encoder for text embedding
    )
    model.to(device)

    # ---------- Three-Stage Training Setup ----------
    # Stage 0: Domain contrastive pretraining (image-text alignment)
    # Stage 1: Concept bottleneck + global alignment (concept_bottleneck, concept_head, country_head, location_encoder)
    # Stage 2: Geolocation head training (cell_head, offset_head) with frozen Stage 0+1 params
    
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    # AMP scaler (shared across stages)
    scaler = torch.amp.GradScaler('cuda', enabled=args.use_amp)
    
    logger.info(f"=== THREE-STAGE TRAINING PIPELINE ===")
    logger.info(f"Stage 0: {args.stage0_epochs} epochs - Domain Contrastive Pretraining (Image-Text)")
    logger.info(f"Stage 1: {args.stage1_epochs} epochs - Concept Bottleneck + Global Alignment")
    logger.info(f"Stage 2: {args.stage2_epochs} epochs - Geolocation Head Training")
    total_epochs = args.stage0_epochs + args.stage1_epochs + args.stage2_epochs
    logger.info(f"Total epochs: {total_epochs}")

    # ========================================================================
    # STAGE 0: Domain Contrastive Pretraining (Image-Text Alignment)
    # ========================================================================
    logger.info(f"\n{'='*60}")
    logger.info(f"STAGE 0: Domain Contrastive Pretraining")
    logger.info(f"{'='*60}")
    
    # Unfreeze top layers of image encoder and text encoder
    model.image_encoder.unfreeze_top_layers(args.unfreeze_layers)
    model.image_encoder.unfreeze_text_encoder()
    
    # Get trainable parameters for Stage 0
    stage0_params = model.image_encoder.get_trainable_params()
    optimizer = torch.optim.AdamW(stage0_params, lr=args.stage0_lr, weight_decay=args.stage0_weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.stage0_epochs, eta_min=args.stage0_lr * 0.01)
    logger.info(f"Stage 0 optimizer: {len(stage0_params)} params, lr={args.stage0_lr:.2e}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    global_epoch = 0
    
    for epoch in range(args.stage0_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.stage0_epochs} [Stage 0]")
        
        for batch in pbar:
            images, concept_idx, target_idx, coords, metadata, cell_labels, note_embs = batch
            images = images.to(device)
            
            # Get raw notes from metadata for dynamic text encoding
            notes = [m["note"] for m in metadata]
            
            with torch.amp.autocast('cuda', enabled=args.use_amp):
                # Get image features from the encoder (through the model)
                img_features = model.image_encoder(images)
                
                # Get text features with gradient (trainable)
                text_features = model.image_encoder.get_text_features_trainable(notes)
                
                # Image-Text contrastive loss
                loss = clip_contrastive_loss(img_features, text_features, temperature=args.temperature)
                loss = loss / args.gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            if (pbar.n + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item() * args.gradient_accumulation_steps
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            if args.use_wandb:
                wandb.log({"batch_loss": loss.item(), "stage": 0, "batch_contrastive_loss": loss.item()})
        
        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} [Stage 0] Train Loss: {avg_train_loss:.4f}")
        
        if args.use_wandb:
            wandb.log({"train_loss": avg_train_loss, "epoch": global_epoch + 1, "stage": 0})
        
        # Validation for Stage 0
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images, _, _, _, metadata, _, _ = batch
                images = images.to(device)
                notes = [m["note"] for m in metadata]
                
                img_features = model.image_encoder(images)
                text_features = model.image_encoder.get_text_features(notes)
                loss = clip_contrastive_loss(img_features, text_features, temperature=args.temperature)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        logger.info(f"Epoch {epoch+1} [Stage 0] Val Loss: {val_loss:.4f}")
        
        if args.use_wandb:
            wandb.log({"val_loss": val_loss, "val_contrastive_loss": val_loss})
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path = checkpoint_dir / "checkpoints" / "best_model_stage0.pt"
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best Stage 0 model (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
                logger.info(f"Stage 0 early stopping at epoch {epoch+1}")
                break
        
        global_epoch += 1
    
    # Freeze image and text encoders permanently after Stage 0
    logger.info("Freezing image encoder and text encoder after Stage 0...")
    model.image_encoder.freeze_encoder()
    model.image_encoder.freeze_text_encoder()
    
    # ========================================================================
    # STAGE 1: Concept Bottleneck + Global Alignment Training
    # ========================================================================
    logger.info(f"\n{'='*60}")
    logger.info(f"STAGE 1: Concept Bottleneck + Global Alignment Training")
    logger.info(f"{'='*60}")
    
    # Get Stage 1 parameters
    stage1_params = model.get_stage1_params()
    optimizer = torch.optim.AdamW(stage1_params, lr=args.stage1_lr, weight_decay=args.stage1_weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.stage1_epochs, eta_min=args.stage1_lr * 0.01)
    logger.info(f"Stage 1 optimizer: {len(stage1_params)} params, lr={args.stage1_lr:.2e}")
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(args.stage1_epochs):
        model.train()
        total_loss = 0
        total_concept_correct = 0
        total_concept_count = 0
        total_country_correct = 0
        total_country_count = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.stage1_epochs} [Stage 1]")
        
        for batch in pbar:
            images, concept_idx, target_idx, coords, _, cell_labels, note_embs = batch
            images = images.to(device)
            coords = coords.to(device)
            concept_idx = concept_idx.to(device)
            target_idx = target_idx.to(device)
            cell_labels = cell_labels.to(device)
            
            with torch.amp.autocast('cuda', enabled=args.use_amp):
                outputs = model(images, coords)
                concept_emb = outputs["concept_emb"]
                concept_logits = outputs["concept_logits"]
                country_logits = outputs["country_logits"]
                gps_emb = outputs["gps_emb"]
                
                # Metrics
                pred_concepts = concept_logits.argmax(dim=1)
                concept_correct = (pred_concepts == concept_idx).sum().item()
                total_concept_correct += concept_correct
                total_concept_count += len(concept_idx)
                
                pred_countries = country_logits.argmax(dim=1)
                country_correct = (pred_countries == target_idx).sum().item()
                total_country_correct += country_correct
                total_country_count += len(target_idx)
                
                # Stage 1 losses: Concept + Country + C-GPS (NO text!)
                loss_concept_gps = geocell_contrastive_loss(concept_emb, gps_emb, cell_labels, temperature=args.temperature)
                loss_concept = nn.functional.cross_entropy(
                    concept_logits, concept_idx,
                    weight=concept_weights if args.use_class_weights else None,
                    label_smoothing=args.label_smoothing
                )
                loss_country = nn.functional.cross_entropy(country_logits, target_idx, label_smoothing=args.label_smoothing)
                
                loss = (
                    # TODO: Location Encoder is frozen. Should we unfreeze it?
                    args.lambda_concept_gps * loss_concept_gps
                    + args.lambda_concept * loss_concept
                    + args.lambda_country * loss_country
                )
                loss = loss / args.gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            if (pbar.n + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item() * args.gradient_accumulation_steps
            batch_concept_acc = concept_correct / len(concept_idx)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "c_gps": f"{loss_concept_gps.item():.4f}", "concept_acc": f"{batch_concept_acc:.3f}"})
            
            if args.use_wandb:
                wandb.log({
                    "batch_loss": loss.item(), "stage": 1,
                    "batch_concept_gps_loss": loss_concept_gps.item(),
                    "batch_concept_loss": loss_concept.item(),
                    "batch_country_loss": loss_country.item(),
                    "batch_concept_accuracy": batch_concept_acc,
                })
        
        avg_train_loss = total_loss / len(train_loader)
        train_concept_acc = total_concept_correct / total_concept_count if total_concept_count > 0 else 0.0
        train_country_acc = total_country_correct / total_country_count if total_country_count > 0 else 0.0
        
        logger.info(f"Epoch {epoch+1} [Stage 1] Train Loss: {avg_train_loss:.4f}, Concept Acc: {train_concept_acc:.4f}, Country Acc: {train_country_acc:.4f}")
        
        if args.use_wandb:
            wandb.log({"train_loss": avg_train_loss, "train_concept_accuracy": train_concept_acc, "train_country_accuracy": train_country_acc, "epoch": global_epoch + 1, "stage": 1})
        
        val_metrics = validate(model, val_loader, device, args, cell_centers, concept_weights, current_stage=1)
        
        scheduler.step()
        
        val_metric = val_metrics["concept_acc"]
        if val_metric > best_val_acc:
            best_val_acc = val_metric
            patience_counter = 0
            best_model_path = checkpoint_dir / "checkpoints" / "best_model_stage1.pt"
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best Stage 1 model (Concept Acc: {val_metric:.4f})")
        else:
            patience_counter += 1
            if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
                logger.info(f"Stage 1 early stopping at epoch {epoch+1}")
                break
        
        # Visualization & Diagnostics
        if (epoch + 1) % args.save_interval == 0:
            visualize_predictions(model, val_loader, concept_names, full_dataset.idx_to_country, device, args, checkpoint_dir, global_epoch + 1, cell_centers=cell_centers)
            diag_path = checkpoint_dir / "diagnostics" / f"stage1_epoch_{epoch+1}.csv"
            dump_diagnostics(model, val_loader, device, diag_path, concept_names, full_dataset.idx_to_country, cell_centers=cell_centers, log_to_wandb=args.use_wandb, wandb_step=global_epoch + 1)
        
        global_epoch += 1
    
    # Freeze Stage 1 parameters
    logger.info("Freezing Stage 1 parameters...")
    model.freeze_stage1()
    
    # ========================================================================
    # STAGE 2: Geolocation Head Training
    # ========================================================================
    logger.info(f"\n{'='*60}")
    logger.info(f"STAGE 2: Geolocation Head Training")
    logger.info(f"{'='*60}")
    
    # Get Stage 2 parameters
    stage2_params = model.get_stage2_params()
    optimizer = torch.optim.AdamW(stage2_params, lr=args.stage2_lr, weight_decay=args.stage2_weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.stage2_epochs, eta_min=args.stage2_lr * 0.01)
    logger.info(f"Stage 2 optimizer: {len(stage2_params)} params, lr={args.stage2_lr:.2e}")
    
    best_val_metric = float('inf')  # Lower median error is better
    patience_counter = 0
    
    for epoch in range(args.stage2_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.stage2_epochs} [Stage 2]")
        
        for batch in pbar:
            images, concept_idx, target_idx, coords, _, cell_labels, note_embs = batch
            images = images.to(device)
            coords = coords.to(device)
            cell_labels = cell_labels.to(device)
            
            with torch.amp.autocast('cuda', enabled=args.use_amp):
                outputs = model(images, coords)
                cell_logits = outputs["cell_logits"]
                pred_offsets = outputs["pred_offsets"]
                
                # Stage 2 losses: Cell classification + Offset regression
                loss_cell = nn.functional.cross_entropy(cell_logits, cell_labels, label_smoothing=args.label_smoothing)
                
                batch_cell_centers = cell_centers[cell_labels]
                if model.coord_output_dim == 3:
                    lat_rad = torch.deg2rad(coords[:, 0])
                    lng_rad = torch.deg2rad(coords[:, 1])
                    x = torch.cos(lat_rad) * torch.cos(lng_rad)
                    y = torch.cos(lat_rad) * torch.sin(lng_rad)
                    z = torch.sin(lat_rad)
                    true_cart = torch.stack([x, y, z], dim=1)
                    target_offsets = true_cart - batch_cell_centers
                    loss_offset = nn.functional.mse_loss(pred_offsets, target_offsets)
                else:
                    c_x, c_y, c_z = batch_cell_centers[:, 0], batch_cell_centers[:, 1], batch_cell_centers[:, 2]
                    c_lat = torch.rad2deg(torch.asin(c_z))
                    c_lng = torch.rad2deg(torch.atan2(c_y, c_x))
                    batch_cell_latlng = torch.stack([c_lat, c_lng], dim=1)
                    if args.coordinate_loss_type == "haversine":
                        pred_latlng = batch_cell_latlng + pred_offsets
                        loss_offset = coordinate_loss(pred_latlng, coords, loss_type="haversine")
                    else:
                        target_offsets = coords - batch_cell_latlng
                        target_offsets[:, 1] = (target_offsets[:, 1] + 180) % 360 - 180
                        loss_offset = nn.functional.mse_loss(pred_offsets, target_offsets)
                
                loss = args.lambda_cell * loss_cell + args.lambda_offset * loss_offset
                loss = loss / args.gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            if (pbar.n + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item() * args.gradient_accumulation_steps
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "cell": f"{loss_cell.item():.4f}", "offset": f"{loss_offset.item():.4f}"})
            
            if args.use_wandb:
                wandb.log({
                    "batch_loss": loss.item(), "stage": 2,
                    "batch_cell_loss": loss_cell.item(),
                    "batch_offset_loss": loss_offset.item(),
                })
        
        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} [Stage 2] Train Loss: {avg_train_loss:.4f}")
        
        if args.use_wandb:
            wandb.log({"train_loss": avg_train_loss, "epoch": global_epoch + 1, "stage": 2})
        
        val_metrics = validate(model, val_loader, device, args, cell_centers, concept_weights, current_stage=2)
        
        scheduler.step()
        
        val_metric = val_metrics["median_error_km"]
        if val_metric < best_val_metric:
            best_val_metric = val_metric
            patience_counter = 0
            best_model_path = checkpoint_dir / "checkpoints" / "best_model_stage2.pt"
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best Stage 2 model (Median Error: {val_metric:.1f}km)")
        else:
            patience_counter += 1
            if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
                logger.info(f"Stage 2 early stopping at epoch {epoch+1}")
                break
        
        # Visualization & Diagnostics
        if (epoch + 1) % args.save_interval == 0:
            visualize_predictions(model, val_loader, concept_names, full_dataset.idx_to_country, device, args, checkpoint_dir, global_epoch + 1, cell_centers=cell_centers)
            diag_path = checkpoint_dir / "diagnostics" / f"stage2_epoch_{epoch+1}.csv"
            dump_diagnostics(model, val_loader, device, diag_path, concept_names, full_dataset.idx_to_country, cell_centers=cell_centers, log_to_wandb=args.use_wandb, wandb_step=global_epoch + 1)
        
        global_epoch += 1

    # ---------- Final Test Evaluation ----------
    logger.info(f"\n{'='*60}")
    logger.info("FINAL TEST EVALUATION")
    logger.info(f"{'='*60}")
    
    # Load best Stage 2 model if saved
    best_model_path = checkpoint_dir / "checkpoints" / "best_model_stage2.pt"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path))
        logger.info("Loaded best Stage 2 model for testing.")
    else:
        logger.warning("No best Stage 2 model found, using current model state.")

    test_metrics = validate(model, test_loader, device, args, cell_centers, concept_weights, current_stage=2)
    logger.info(f"Test Metrics: {test_metrics}")

    if args.use_wandb:
        wandb.log({f"test/{k}": v for k, v in test_metrics.items()})

        # Dump test diagnostics
        test_diag_path = checkpoint_dir / "diagnostics" / "test_diagnostics.csv"
        dump_diagnostics(
            model,
            test_loader,
            device,
            test_diag_path,
            concept_names,
            full_dataset.idx_to_country,
            cell_centers=cell_centers,
            max_samples=len(test_dataset),  # Dump all test samples
            log_to_wandb=True,
        )


# ---------- Validation Function ----------
@torch.no_grad()
def validate(model, val_loader, device, args, cell_centers, concept_weights=None, current_stage=1):
    model.eval()
    total_loss = 0
    total_concept_correct = 0
    total_concept_count = 0
    total_country_correct = 0
    total_country_count = 0

    # New metrics
    total_cell_correct = 0

    # Collect all distances for threshold accuracy computation
    all_distances = []

    for batch in val_loader:
        images, concept_idx, target_idx, coords, _, cell_labels, note_embs = batch
        images = images.to(device)
        coords = coords.to(device)
        concept_idx = concept_idx.to(device)
        target_idx = target_idx.to(device)
        cell_labels = cell_labels.to(device)
        note_embs = note_embs.to(device)

        # CBM Architecture outputs
        outputs = model(images, coords)
        concept_emb = outputs["concept_emb"]  # [B, 512] - the bottleneck
        concept_logits = outputs["concept_logits"]  # [B, num_concepts]
        country_logits = outputs["country_logits"]
        cell_logits = outputs["cell_logits"]
        pred_offsets = outputs["pred_offsets"]
        gps_emb = outputs["gps_emb"]  # [B, 512]

        # Compute concept accuracy
        pred_concepts = concept_logits.argmax(dim=1)
        concept_correct = (pred_concepts == concept_idx).sum().item()
        total_concept_correct += concept_correct
        total_concept_count += len(concept_idx)

        # Compute country accuracy
        pred_countries = country_logits.argmax(dim=1)
        country_correct = (pred_countries == target_idx).sum().item()
        total_country_correct += country_correct
        total_country_count += len(target_idx)

        # Compute cell accuracy
        pred_cells = cell_logits.argmax(dim=1)
        cell_correct = (pred_cells == cell_labels).sum().item()
        total_cell_correct += cell_correct

        # Compute Final Coordinate Prediction & Distance
        # Pred = Cell_Center[Pred_Cell] + Pred_Offset
        # We rely on predicted cell, not ground truth cell for validation metric!
        # 1. Get predicted cell centers
        pred_cell_centers = cell_centers[pred_cells]  # [B, 3]

        # 2. Get predicted Lat/Lng
        if model.coord_output_dim == 3:
            # Prediction is in 3D Cartesian space
            pred_cart = pred_cell_centers + pred_offsets
            # Convert to Lat/Lng for haversine
            pred_cart = torch.nn.functional.normalize(pred_cart, p=2, dim=1)
            pred_latlng_deg = sphere_to_latlng(pred_cart)  # [B, 2]
            pred_coords = pred_latlng_deg
        else:
            # Prediction is 2D Lat/Lng offset
            # Convert cell center to Lat/Lng
            c_x, c_y, c_z = (
                pred_cell_centers[:, 0],
                pred_cell_centers[:, 1],
                pred_cell_centers[:, 2],
            )
            c_lat = torch.rad2deg(torch.asin(c_z))
            c_lng = torch.rad2deg(torch.atan2(c_y, c_x))
            pred_cell_latlng = torch.stack([c_lat, c_lng], dim=1)

            pred_coords = pred_cell_latlng + pred_offsets
            # Normalize Lng to [-180, 180]? Haversine usually handles it but cleaner to normalize.

        # Compute distances for threshold accuracy
        batch_distances = haversine_distance(pred_coords, coords)
        if batch_distances.numel() > 0:
            all_distances.append(batch_distances.cpu())

        # Compute Loss (Validation) - Stage-aware
        if current_stage == 1:
            # Stage 1: Concept learning losses (NO concept-text alignment)
            loss_concept_gps = geocell_contrastive_loss(concept_emb, gps_emb, cell_labels, temperature=args.temperature)
            loss_concept = nn.functional.cross_entropy(
                concept_logits, concept_idx, 
                weight=concept_weights if args.use_class_weights else None,
                label_smoothing=args.label_smoothing
            )
            loss_country = nn.functional.cross_entropy(country_logits, target_idx, label_smoothing=args.label_smoothing)
            loss = (
                args.lambda_concept_gps * loss_concept_gps
                + args.lambda_concept * loss_concept
                + args.lambda_country * loss_country
            )
            loss_cell = torch.tensor(0.0, device=device)
            loss_offset = torch.tensor(0.0, device=device)
        else:
            # Stage 2: Location prediction losses
            loss_cell = nn.functional.cross_entropy(cell_logits, cell_labels, label_smoothing=args.label_smoothing)
            
            batch_cell_centers = cell_centers[cell_labels]
            if model.coord_output_dim == 3:
                lat_rad = torch.deg2rad(coords[:, 0])
                lng_rad = torch.deg2rad(coords[:, 1])
                x = torch.cos(lat_rad) * torch.cos(lng_rad)
                y = torch.cos(lat_rad) * torch.sin(lng_rad)
                z = torch.sin(lat_rad)
                true_cart = torch.stack([x, y, z], dim=1)
                target_offsets = true_cart - batch_cell_centers
                loss_offset = nn.functional.mse_loss(pred_offsets, target_offsets)
            else:
                c_x, c_y, c_z = batch_cell_centers[:, 0], batch_cell_centers[:, 1], batch_cell_centers[:, 2]
                c_lat = torch.rad2deg(torch.asin(c_z))
                c_lng = torch.rad2deg(torch.atan2(c_y, c_x))
                batch_cell_latlng = torch.stack([c_lat, c_lng], dim=1)
                if args.coordinate_loss_type == "haversine":
                    pred_latlng = batch_cell_latlng + pred_offsets
                    loss_offset = coordinate_loss(pred_latlng, coords, loss_type="haversine")
                else:
                    target_offsets = coords - batch_cell_latlng
                    target_offsets[:, 1] = (target_offsets[:, 1] + 180) % 360 - 180
                    loss_offset = nn.functional.mse_loss(pred_offsets, target_offsets)
            
            loss = args.lambda_cell * loss_cell + args.lambda_offset * loss_offset
            loss_concept_gps = torch.tensor(0.0, device=device)
            loss_concept = torch.tensor(0.0, device=device)
            loss_country = torch.tensor(0.0, device=device)

        total_loss += loss.item()

    # Compute threshold accuracies
    threshold_accuracies = {}
    if all_distances:
        all_distances_tensor = torch.cat(all_distances)
        thresholds = {
            "street": 1.0,  # 1 km
            "city": 25.0,  # 25 km
            "region": 200.0,  # 200 km
            "country": 750.0,  # 750 km
            "continent": 2500.0,  # 2500 km
        }
        for level, threshold_km in thresholds.items():
            acc = accuracy_within_threshold(all_distances_tensor, threshold_km)
            threshold_accuracies[f"acc_{level}"] = acc

        median_error = torch.median(all_distances_tensor).item()
        threshold_accuracies["median_error_km"] = median_error
    else:
        threshold_accuracies = {
            "acc_street": 0.0,
            "acc_city": 0.0,
            "acc_region": 0.0,
            "acc_country": 0.0,
            "acc_continent": 0.0,
            "median_error_km": 0.0,
        }

    avg_loss = total_loss / len(val_loader)
    val_concept_acc = (
        total_concept_correct / total_concept_count if total_concept_count > 0 else 0.0
    )
    val_country_acc = (
        total_country_correct / total_country_count if total_country_count > 0 else 0.0
    )
    val_cell_acc = (
        total_cell_correct / total_concept_count if total_concept_count > 0 else 0.0
    )  # Denom same as batch size

    # Log threshold accuracies
    log_msg = f"Validation Loss: {avg_loss:.4f}, Val Concept Acc: {val_concept_acc:.4f}, Val Country Acc: {val_country_acc:.4f}, Val Cell Acc: {val_cell_acc:.4f}"
    log_msg += f", Median Error: {threshold_accuracies['median_error_km']:.1f}km"
    logger.info(log_msg)
    logger.info(
        f"Thresholds: Street={threshold_accuracies['acc_street']:.3f}, City={threshold_accuracies['acc_city']:.3f}, Region={threshold_accuracies['acc_region']:.3f}"
    )

    if args.use_wandb:
        wandb.log(
            {
                "val_loss": avg_loss,
                "val_concept_accuracy": val_concept_acc,
                "val_country_accuracy": val_country_acc,
                "val_cell_accuracy": val_cell_acc,
                "val_median_error_km": threshold_accuracies["median_error_km"],
                "val_acc_street_1km": threshold_accuracies["acc_street"],
                "val_acc_city_25km": threshold_accuracies["acc_city"],
                "val_acc_region_200km": threshold_accuracies["acc_region"],
                "val_acc_country_750km": threshold_accuracies["acc_country"],
                "val_acc_continent_2500km": threshold_accuracies["acc_continent"],
            }
        )

    return {
        "loss": avg_loss,
        "concept_acc": val_concept_acc,
        "country_acc": val_country_acc,
        "cell_acc": val_cell_acc,
        **threshold_accuracies,
    }


# ---------- Main Entry Point ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Concept-Aware CBM")

    # ---------- Dataset Arguments ----------
    parser.add_argument(
        "--geoguessr_id",
        type=str,
        default="6906237dc7731161a37282b2",
        help="Geoguessr ID",
    )
    parser.add_argument(
        "--data_root", type=str, default="data", help="Data root directory"
    )
    parser.add_argument(
        "--csv_path", type=str, default=None, help="Optional path to CSV file. If provided, loads data from CSV instead of folder structure."
    )

    # ---------- Data & Model Arguments ----------
    parser.add_argument(
        "--encoder_model",
        type=str,
        default="geolocal/StreetCLIP",
        help="Image Encoder model to use",
    )
    parser.add_argument(
        "--finetune_encoder",
        action="store_true",
        help="Whether to finetune the encoder",
    )
    parser.add_argument(
        "--country_filter", type=str, default=None, help="Filter for country"
    )

    # ---------- Training Hyperparameters ----------
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    
    # ---------- Stage 0: Domain Contrastive Pretraining ----------
    parser.add_argument(
        "--stage0_epochs", type=int, default=5, help="Number of epochs for Stage 0 (domain contrastive pretraining)"
    )
    parser.add_argument(
        "--stage0_lr", type=float, default=3e-5, help="Learning rate for Stage 0"
    )
    parser.add_argument(
        "--stage0_weight_decay", type=float, default=0.05, help="Weight decay for Stage 0"
    )
    parser.add_argument(
        "--unfreeze_layers", type=int, default=2, help="Number of top vision encoder layers to unfreeze in Stage 0"
    )
    
    # ---------- Stage 1: Concept Bottleneck Training ----------
    parser.add_argument(
        "--stage1_epochs", type=int, default=15, help="Number of epochs for Stage 1 (concept bottleneck training)"
    )
    parser.add_argument(
        "--stage1_lr", type=float, default=1e-4, help="Learning rate for Stage 1"
    )
    parser.add_argument(
        "--stage1_weight_decay", type=float, default=0.01, help="Weight decay for Stage 1"
    )
    
    # ---------- Stage 2: Geolocation Head Training ----------
    parser.add_argument(
        "--stage2_epochs", type=int, default=15, help="Number of epochs for Stage 2 (geolocation head training)"
    )
    parser.add_argument(
        "--stage2_lr", type=float, default=3e-4, help="Learning rate for Stage 2"
    )
    parser.add_argument(
        "--stage2_weight_decay", type=float, default=0.01, help="Weight decay for Stage 2"
    )
    
    # ---------- Legacy/General Training Hyperparameters ----------
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Default learning rate (overridden by stage-specific lr)"
    )
    parser.add_argument(
        "--concept_lr_multiplier",
        type=float,
        default=3.0,
        help="Multiplier for concept head learning rate (default: 3.0)",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Fraction of epochs for learning rate warmup (default: 0.1)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.1, help="Default weight decay (overridden by stage-specific)"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help="Number of epochs to wait before early stopping",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of update steps to accumulate before performing a backward/update pass",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Whether to use Automatic Mixed Precision (AMP)",
    )

    parser.add_argument(
        "--min_samples_per_cell",
        type=int,
        default=500,
        help="Minimum number of samples to form a semantic geocell",
    )

    # ---------- Loss Hyperparameters ----------
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Temperature for contrastive alignment loss",
    )
    parser.add_argument(
        "--lambda_concept_gps",
        type=float,
        default=1.0,
        help="Weight for Concept-GPS contrastive alignment loss (geocell-aware)",
    )
    parser.add_argument(
        "--lambda_concept", type=float, default=1.0, help="Weight for concept classification loss"
    )
    parser.add_argument(
        "--lambda_country",
        type=float,
        default=0.5,
        help="Weight for country classification loss",
    )
    parser.add_argument(
        "--lambda_cell",
        type=float,
        default=1.0,
        help="Weight for cell classification loss",
    )
    parser.add_argument(
        "--lambda_offset",
        type=float,
        default=5.0,
        help="Weight for offset regression loss",
    )
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument(
        "--use_class_weights",
        action="store_true",
        help="Use class weights for concept loss to handle imbalance (default: True)",
    )
    parser.add_argument(
        "--no_class_weights",
        dest="use_class_weights",
        action="store_false",
        help="Disable class weights for concept loss",
    )
    parser.set_defaults(use_class_weights=True)
    parser.add_argument(
        "--coordinate_loss_type",
        type=str,
        default="haversine",
        choices=["haversine", "mse", "sphere"],
        help="Type of coordinate loss: haversine, mse, or sphere",
    )

    # ---------- Miscellaneous Arguments ----------
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (auto-generated if not provided)",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=5,
        help="When to save checkpoints and run visualizations and diagnostics",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=True,
        help="Whether to use Weights & Biases for logging",
    )

    args = parser.parse_args()
    train(args)
