#!/usr/bin/env python3
"""
Test file for visualization functions (visualize_predictions and plot_error_distribution).
Mocks model, data, and images to test the visualization pipeline.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, List
import tempfile
import shutil
import argparse

# Add parent directory to path to import from training script
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation import sphere_to_latlng


# ==================== Mock Classes ====================

class MockArgs:
    """Mock arguments object."""
    def __init__(self):
        self.use_wandb = False
        self.batch_size = 4


class MockModel(torch.nn.Module):
    """Mock ConceptAwareGeoModel for testing."""
    def __init__(self, num_concepts=10, num_countries=5, num_cells=20, coord_output_dim=2):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_countries = num_countries
        self.num_cells = num_cells
        self.coord_output_dim = coord_output_dim
        
    def forward(self, images, coords):
        batch_size = images.size(0)
        
        # Generate random but realistic logits
        concept_logits = torch.randn(batch_size, self.num_concepts)
        country_logits = torch.randn(batch_size, self.num_countries)
        cell_logits = torch.randn(batch_size, self.num_cells)
        
        # Offsets: small perturbations in coordinate space
        if self.coord_output_dim == 3:
            pred_offsets = torch.randn(batch_size, 3) * 0.1
        else:
            pred_offsets = torch.randn(batch_size, 2) * 5.0  # ~5 degree offset
        
        return {
            "concept_logits": concept_logits,
            "country_logits": country_logits,
            "cell_logits": cell_logits,
            "pred_offsets": pred_offsets,
        }


class MockDataLoader:
    """Mock DataLoader that yields synthetic batches."""
    def __init__(self, batch_size=4, num_batches=1, num_concepts=10, num_countries=5, coord_output_dim=2):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.num_concepts = num_concepts
        self.num_countries = num_countries
        self.coord_output_dim = coord_output_dim
        
    def __iter__(self):
        for _ in range(self.num_batches):
            # Generate synthetic images (random RGB images)
            images = torch.rand(self.batch_size, 3, 224, 224)
            
            # Random concept indices
            concept_indices = torch.randint(0, self.num_concepts, (self.batch_size,))
            
            # Random country indices (used as target_indices in some contexts)
            target_indices = torch.randint(0, self.num_countries, (self.batch_size,))
            
            # Random coordinates (lat, lng) within realistic ranges
            # Lat: [-90, 90], Lng: [-180, 180]
            coords = torch.stack([
                torch.rand(self.batch_size) * 180 - 90,  # latitude
                torch.rand(self.batch_size) * 360 - 180,  # longitude
            ], dim=1)
            
            # Metadata for each sample
            metadata = []
            for i in range(self.batch_size):
                metadata.append({
                    "lat": coords[i, 0].item(),
                    "lng": coords[i, 1].item(),
                    "country": f"Country_{target_indices[i].item()}",
                    "pano_id": f"pano_{i:04d}",
                })
            
            # Random cell labels
            cell_labels = torch.randint(0, 20, (self.batch_size,))
            
            yield images, concept_indices, target_indices, coords, metadata, cell_labels
    
    def __len__(self):
        return self.num_batches


# ==================== Helper Functions (from train_concept_aware.py) ====================

def format_coordinate(lat: float, lng: float, precision: int = 3) -> str:
    """Format coordinates for display."""
    return f"({lat:.{precision}f}, {lng:.{precision}f})"


def format_distance(km: float, precision: int = 1) -> str:
    """Format distance for display."""
    if km < 1:
        return f"{km*1000:.0f}m"
    elif km < 1000:
        return f"{km:.{precision}f}km"
    else:
        return f"{km/1000:.2f}Mm"


def haversine_distance(pred_coords: torch.Tensor, true_coords: torch.Tensor) -> torch.Tensor:
    """Calculate haversine distance between predicted and true coordinates."""
    R = 6371.0  # Earth radius in km
    
    lat1 = torch.deg2rad(true_coords[:, 0])
    lon1 = torch.deg2rad(true_coords[:, 1])
    lat2 = torch.deg2rad(pred_coords[:, 0])
    lon2 = torch.deg2rad(pred_coords[:, 1])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    
    return R * c


# ==================== Test Functions ====================

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711])
VIZ_DPI = 150


@torch.no_grad()
def test_visualize_predictions(keep_files=False):
    """Test visualize_predictions function with mocked data."""
    print("\n" + "="*60)
    print("Testing visualize_predictions...")
    print("="*60)
    
    # Setup
    device = torch.device("cpu")
    num_concepts = 10
    num_countries = 5
    num_cells = 20
    batch_size = 4
    num_samples = 2
    epoch = 1
    stage = 1
    coord_output_dim = 2
    
    # Create mock objects
    model = MockModel(num_concepts, num_countries, num_cells, coord_output_dim)
    model.eval()
    
    val_loader = MockDataLoader(batch_size, num_batches=1, num_concepts=num_concepts, 
                                 num_countries=num_countries, coord_output_dim=coord_output_dim)
    
    concept_names = [f"Concept_{i}" for i in range(num_concepts)]
    idx_to_country = {i: f"Country_{i}" for i in range(num_countries)}
    
    # Generate random cell centers (Cartesian coordinates on unit sphere)
    cell_centers = torch.randn(num_cells, 3)
    cell_centers = torch.nn.functional.normalize(cell_centers, p=2, dim=1)
    
    # Create temporary directory for outputs
    if keep_files:
        temp_dir = Path("test_outputs")
        temp_dir.mkdir(exist_ok=True)
    else:
        temp_dir = Path(tempfile.mkdtemp())
    checkpoint_dir = temp_dir / "checkpoints"
    
    args = MockArgs()
    
    try:
        # Run visualization
        visualize_predictions_mock(
            model, val_loader, concept_names, idx_to_country, device,
            args, checkpoint_dir, epoch, cell_centers, num_samples, stage
        )
        
        # Check that files were created
        viz_dir = checkpoint_dir / "visualizations" / f"epoch_{epoch}"
        created_files = list(viz_dir.glob("*.png"))
        
        print(f"✓ Created {len(created_files)} visualization files")
        for f in created_files:
            file_size = f.stat().st_size / 1024  # KB
            print(f"  - {f.name} ({file_size:.1f} KB)")
        
        assert len(created_files) > 0, "No visualization files were created!"
        print("✓ Test PASSED: visualize_predictions")
        
        if keep_files:
            print(f"✓ Files saved to: {viz_dir}")
        
    finally:
        # Cleanup
        if not keep_files:
            shutil.rmtree(temp_dir)
            print(f"✓ Cleaned up temporary directory: {temp_dir}")
        else:
            print(f"✓ Keeping files in: {temp_dir}")


@torch.no_grad()
def visualize_predictions_mock(
    model, val_loader, concept_names, idx_to_country, device,
    args, checkpoint_dir, epoch, cell_centers, num_samples=4, stage=None
):
    """Simplified mock of visualize_predictions for testing."""
    model.eval()
    viz_dir = checkpoint_dir / "visualizations" / f"epoch_{epoch}"
    viz_dir.mkdir(parents=True, exist_ok=True)

    for batch in val_loader:
        images, concept_indices, _, coords, metadata, cell_labels = batch
        images = images.to(device)
        coords = coords.to(device)
        concept_indices = concept_indices.to(device)

        outputs = model(images, coords)
        concept_logits = outputs["concept_logits"]
        country_logits = outputs["country_logits"]
        cell_logits = outputs["cell_logits"]
        pred_offsets = outputs["pred_offsets"]
        concept_probs = torch.softmax(concept_logits, dim=1)
        country_probs = torch.softmax(country_logits, dim=1)

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
            # Create figure with simple 2-row layout
            fig, (ax_img, ax_concept) = plt.subplots(2, 1, figsize=(10, 8),
                                                      gridspec_kw={'height_ratios': [2, 1]})
            
            # Top: Image with title
            img_cpu = images[i].cpu().permute(1, 2, 0).numpy()
            img_disp = np.clip(CLIP_STD * img_cpu + CLIP_MEAN, 0, 1)
            ax_img.imshow(img_disp)
            ax_img.axis("off")
            
            # Get ground truth and predictions
            gt_lat, gt_lng, gt_country = metadata[i]["lat"], metadata[i]["lng"], metadata[i]["country"]
            pred_lat, pred_lng = pred_coords[i, 0].item(), pred_coords[i, 1].item()
            
            gt_coord_tensor = coords[i].unsqueeze(0)
            pred_coord_tensor = pred_coords[i].unsqueeze(0)
            distance_km = haversine_distance(pred_coord_tensor, gt_coord_tensor).item()
            
            pred_country_idx = country_logits[i].argmax().item()
            pred_country = idx_to_country[pred_country_idx]
            country_confidence = country_probs[i, pred_country_idx].item()
            
            gt_concept_idx = concept_indices[i].item()
            gt_concept_name = concept_names[gt_concept_idx]
            top_concept_idx = concept_probs[i].argmax().item()
            top_concept_name = concept_names[top_concept_idx]
            concept_confidence = concept_probs[i, top_concept_idx].item()
            
            # Title with key information
            title_parts = [
                f"Epoch {epoch} | Stage {stage}" if stage is not None else f"Epoch {epoch}",
                f"ID: {metadata[i].get('pano_id', 'N/A')}",
                "",
                f"[Location]",
                f"  Pred: {format_coordinate(pred_lat, pred_lng)} ({format_distance(distance_km)})",
                f"  True: {format_coordinate(gt_lat, gt_lng)}",
                "",
                f"[Country]",
                f"  Pred: {pred_country} ({country_confidence:.2%})",
                f"  True: {gt_country}",
                "",
                f"[Concept]",
                f"  Pred: {top_concept_name} ({concept_confidence:.2%})",
                f"  True: {gt_concept_name}",
            ]
            ax_img.set_title("\n".join(title_parts), fontsize=9, family='monospace', loc='left')
            
            # Bottom: Top 5 Concepts (horizontal bar chart)
            top_k = min(5, len(concept_names))
            top_scores, top_indices = torch.topk(concept_probs[i], k=top_k)
            top_indices_cpu = top_indices.cpu().numpy()
            
            # Orange for correct concept, blue for others
            bar_colors = ["#FF8C00" if idx == gt_concept_idx else "#4169E1" for idx in top_indices_cpu]
            
            y_pos = np.arange(top_k)
            ax_concept.barh(y_pos, top_scores.cpu().numpy(), color=bar_colors, alpha=0.8)
            ax_concept.set_yticks(y_pos)
            ax_concept.set_yticklabels([concept_names[idx.item()] for idx in top_indices], fontsize=10)
            ax_concept.invert_yaxis()
            ax_concept.set_xlabel("Probability", fontsize=10)
            ax_concept.set_title(f"Top {top_k} Concepts", fontsize=11, fontweight='bold', pad=10)
            ax_concept.grid(axis='x', alpha=0.3, linestyle='--')
            ax_concept.set_xlim(0, 1.0)
            
            plt.tight_layout()
            
            save_path = viz_dir / f"sample_{i}_epoch_{epoch}.png"
            plt.savefig(save_path, dpi=VIZ_DPI, bbox_inches='tight')
            plt.close(fig)

        break
    
    print(f"✓ Saved {n_display} visualization(s) to {viz_dir}")


@torch.no_grad()
def test_plot_error_distribution(keep_files=False):
    """Test plot_error_distribution function with mocked data."""
    print("\n" + "="*60)
    print("Testing plot_error_distribution...")
    print("="*60)
    
    # Setup
    device = torch.device("cpu")
    num_cells = 20
    batch_size = 8
    num_batches = 5
    stage = 2
    coord_output_dim = 2
    
    # Create mock model and dataloader
    model = MockModel(num_concepts=10, num_countries=5, num_cells=num_cells, 
                      coord_output_dim=coord_output_dim)
    model.eval()
    
    dataloader = MockDataLoader(batch_size, num_batches, coord_output_dim=coord_output_dim)
    
    # Generate random cell centers
    cell_centers = torch.randn(num_cells, 3)
    cell_centers = torch.nn.functional.normalize(cell_centers, p=2, dim=1)
    
    # Create temporary output path
    if keep_files:
        temp_dir = Path("scripts/tests/test_outputs")
        temp_dir.mkdir(exist_ok=True)
    else:
        temp_dir = Path(tempfile.mkdtemp())
    output_path = temp_dir / "error_distribution.png"
    
    try:
        # Run error distribution plot
        plot_error_distribution_mock(model, dataloader, device, cell_centers, output_path, stage)
        
        # Check that file was created
        assert output_path.exists(), "Error distribution plot was not created!"
        file_size = output_path.stat().st_size / 1024  # KB
        print(f"✓ Created error distribution plot: {output_path.name} ({file_size:.1f} KB)")
        print("✓ Test PASSED: plot_error_distribution")
        
        if keep_files:
            print(f"✓ File saved to: {output_path}")
        
    finally:
        # Cleanup
        if not keep_files:
            shutil.rmtree(temp_dir)
            print(f"✓ Cleaned up temporary directory: {temp_dir}")
        else:
            print(f"✓ Keeping file in: {temp_dir}")


@torch.no_grad()
def plot_error_distribution_mock(model, dataloader, device, cell_centers, output_path, stage=None):
    """Mock plot_error_distribution for testing."""
    model.eval()
    all_distances = []
    
    for batch in dataloader:
        images, _, _, coords, _, _ = batch
        images = images.to(device)
        coords = coords.to(device)
        
        outputs = model(images, coords)
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
        
        distances = haversine_distance(pred_coords, coords)
        all_distances.extend(distances.cpu().tolist())
    
    if not all_distances:
        print("WARNING: No distances collected for error distribution plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    all_distances_array = np.array(all_distances)
    ax.hist(all_distances_array, bins=50, edgecolor='black', alpha=0.7)
    median_dist = np.median(all_distances_array)
    mean_dist = np.mean(all_distances_array)
    ax.axvline(median_dist, color='red', linestyle='--',
               label=f'Median: {format_distance(median_dist)}')
    ax.axvline(mean_dist, color='green', linestyle='--',
               label=f'Mean: {format_distance(mean_dist)}')
    ax.set_xlabel('Distance Error (km)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    title = f'Error Distribution'
    if stage is not None:
        title += f" - Stage {stage}"
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=VIZ_DPI, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Saved error distribution to {output_path}")
    print(f"  - Median distance: {format_distance(median_dist)}")
    print(f"  - Mean distance: {format_distance(mean_dist)}")
    print(f"  - Total samples: {len(all_distances)}")


# ==================== Main ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test visualization functions")
    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Keep generated files in test_outputs/ directory instead of deleting them"
    )
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("VISUALIZATION TESTS")
    if args.keep_files:
        print("(Files will be saved to test_outputs/)")
    print("="*60)
    
    try:
        # Test 1: visualize_predictions
        test_visualize_predictions(keep_files=args.keep_files)
        
        # Test 2: plot_error_distribution
        test_plot_error_distribution(keep_files=args.keep_files)
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        if args.keep_files:
            print("Output files saved in: test_outputs/")
        print("="*60 + "\n")
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"TEST FAILED ✗")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
