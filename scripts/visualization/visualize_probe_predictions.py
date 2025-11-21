#!/usr/bin/env python3
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import random
import numpy as np
from PIL import Image

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.dataset_probe import ConceptProbeDataset, create_stratified_splits, get_transforms
from src.models.probe import StreetClipProbe

def visualize_predictions(model, dataset, device, num_images=10, output_dir=Path("visualizations/probe")):
    model.eval()
    
    # Get random indices
    indices = random.sample(range(len(dataset)), min(num_images, len(dataset)))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Unnormalize transform for visualization
    # Assuming default CLIP normalization
    # mean = (0.48145466, 0.4578275, 0.40821073)
    # std = (0.26862954, 0.26130258, 0.27577711)
    # We will just load the original image again for visualization to avoid artifacts
    
    for i, idx in enumerate(indices):
        # Get data
        image_tensor, label_idx = dataset[idx]
        
        # Get original image path for clean visualization
        row = dataset.data.iloc[idx]
        image_path = row['image_path']
        true_label = row['meta_name']
        
        try:
            original_img = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Could not load original image {image_path}: {e}")
            continue

        # Inference
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(image_tensor)
            probs = F.softmax(logits, dim=1)
            
        # Get top 5
        top5_probs, top5_indices = torch.topk(probs, 5)
        top5_probs = top5_probs.squeeze().cpu().numpy()
        top5_indices = top5_indices.squeeze().cpu().numpy()
        
        top5_concepts = [dataset.idx_to_label[idx] for idx in top5_indices]
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        # Image
        plt.subplot(2, 1, 1)
        plt.imshow(original_img)
        plt.title(f"True Concept: {true_label}")
        plt.axis('off')
        
        # Top 5 Concepts
        plt.subplot(2, 1, 2)
        y_pos = np.arange(len(top5_concepts))
        plt.barh(y_pos, top5_probs, align='center')
        plt.yticks(y_pos, top5_concepts)
        plt.xlabel('Probability')
        plt.title('Top 5 Predicted Concepts')
        plt.gca().invert_yaxis()  # Highest probability at top
        plt.xlim(0, 1.0)
        
        # Add value labels
        for j, v in enumerate(top5_probs):
            plt.text(v + 0.01, j, f"{v:.2%}", va='center')
            
        plt.tight_layout()
        
        save_path = output_dir / f"prediction_{i}_{true_label.replace(' ', '_')}.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Saved visualization to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize Concept Probe Predictions on Test Set")
    parser.add_argument("--csv-path", type=str, 
                       default="/home/igodzwon/Project_AI/data/691df1ee911f74393c53af8c/dataset.csv",
                       help="Path to dataset CSV")
    parser.add_argument("--model-path", type=str, 
                       default="/home/igodzwon/Project_AI/checkpoints/probe/best_probe_model.pth",
                       help="Path to trained model checkpoint")
    parser.add_argument("--output-dir", type=str, default="visualizations/probe", help="Directory to save visualizations")
    parser.add_argument("--num-images", type=int, default=10, help="Number of images to visualize")
    parser.add_argument("--model-name", type=str, default="geolocal/StreetCLIP", help="Backbone model name")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create splits (to get test set)
    print("Creating stratified splits to isolate Test Set...")
    _, _, test_df = create_stratified_splits(args.csv_path)
    print(f"Test Set size: {len(test_df)}")
    
    # Get transforms
    transform = get_transforms(args.model_name)
    
    # Create dataset
    test_dataset = ConceptProbeDataset(test_df, transform=transform)
    
    num_classes = len(test_dataset.labels)
    print(f"Number of concept classes: {num_classes}")
    
    # Initialize model
    model = StreetClipProbe(num_classes=num_classes, model_name=args.model_name)
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    # Visualize
    print(f"Visualizing {args.num_images} random predictions...")
    visualize_predictions(model, test_dataset, device, num_images=args.num_images, output_dir=Path(args.output_dir))
    
    print("Done.")

if __name__ == "__main__":
    main()

