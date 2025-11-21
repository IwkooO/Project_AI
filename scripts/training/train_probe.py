#!/usr/bin/env python3
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import os
import time
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, **kwargs):
        return iterable

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.data.dataset_probe import ConceptProbeDataset, create_stratified_splits, get_transforms
from src.models.probe import StreetClipProbe

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    # Ensure backbone stays frozen (redundant safety check)
    model.backbone.eval() 
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    # Use tqdm for progress bar (update every 5 batches)
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", miniters=5)
    
    for i, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    duration = time.time() - start_time
    
    return epoch_loss, epoch_acc, duration

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Use tqdm for evaluation too (update every 5 batches)
    pbar = tqdm(loader, desc="Evaluating", miniters=5)
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})
        
    loss = running_loss / total
    acc = 100. * correct / total
    
    return loss, acc

def main():
    parser = argparse.ArgumentParser(description="Train Concept Probe")
    parser.add_argument("--csv-path", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="checkpoints/probe", help="Directory to save models")
    parser.add_argument("--model-name", type=str, default="geolocal/StreetCLIP", help="Backbone model name")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of dataloader workers")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create splits
    print("Creating stratified splits...")
    train_df, val_df, test_df = create_stratified_splits(args.csv_path)
    print(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Get transforms
    transform = get_transforms(args.model_name)
    
    # Create datasets
    train_dataset = ConceptProbeDataset(train_df, transform=transform)
    val_dataset = ConceptProbeDataset(val_df, transform=transform)
    test_dataset = ConceptProbeDataset(test_df, transform=transform)
    
    num_classes = len(train_dataset.labels)
    print(f"Number of concept classes: {num_classes}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # Initialize model
    model = StreetClipProbe(num_classes=num_classes, model_name=args.model_name)
    model = model.to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    # Only optimize head parameters
    optimizer = optim.Adam(model.head.parameters(), lr=args.lr)
    
    # Training Loop
    best_val_acc = 0.0
    
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, duration = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch}/{args.epochs}] ({duration:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = output_dir / "best_probe_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'labels': train_dataset.labels
            }, save_path)
            print(f"  Saved best model to {save_path}")
            
    print("Training complete.")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # Final Test Evaluation
    print("Running final evaluation on Test Set...")
    # Load best model
    checkpoint = torch.load(output_dir / "best_probe_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

if __name__ == "__main__":
    main()

