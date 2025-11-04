# Panorama CBM Dataset

This repository contains a PyTorch dataset implementation for training Concept Bottleneck Models (CBMs) on Google Street View panorama images. The dataset is derived from GeoGuessr meta information scraped from learnablemeta.com, providing a unique resource for geospatial concept learning.

## Overview

The dataset consists of:
- **Input**: Cropped and resized panorama images (224x224 pixels)
- **Concepts**: Meta names describing visual features (e.g., "B Type - cap", "Insulators - black double UFO-shaped")
- **Targets**: Country labels (e.g., "India", "Vietnam", "Argentina")
- **Metadata**: Geographic coordinates, notes, and additional information

## Dataset Creation Pipeline

### 1. Data Collection

#### GeoGuessr Map Scraping
The raw data originates from a personal GeoGuessr map with meta information:
```bash
# Scrape meta information from learnablemeta.com
python scrape_lm.py

# Download panorama images for each location
python download_pano.py
```

#### Image Preprocessing
Panorama images are cropped to remove black borders and resized to uniform dimensions:
```bash
# Preprocess panorama images
python preprocess_panorama.py --panorama-folder data/{geoguessrId}/panorama
```

### 2. Meta Enrichment

Meta JSON files contain concept and target information but lack geographic coordinates. The enrichment script adds coordinate information from the locations file:

```bash
# Enrich meta files with coordinates
python enrich_metas_with_coords.py
```

This adds `lat` and `lng` fields to each meta JSON file, enabling geographic analysis and visualization.

### 3. Dataset Construction

The `PanoramaCBMDataset` class loads the processed data:

```python
from src.dataset import PanoramaCBMDataset, create_splits, get_statistics, print_statistics

# Load full dataset
dataset = PanoramaCBMDataset()

# Get dataset statistics
stats = get_statistics(dataset.samples)
print_statistics(stats)

# Create train/val/test splits
train_samples, val_samples, test_samples = create_splits(dataset.samples)
```

## Data Structure

### File Organization
```
data/{geoguessrId}/
├── locations_{geoguessrId}.json    # Location coordinates and panoIds
├── metas/                          # Meta information per location
│   ├── {panoId}.json              # Concept, country, coordinates
│   └── ...
├── panorama_processed/             # Processed images (224x224)
│   ├── image_{panoId}.jpg         # Cropped panorama
│   └── ...
└── metadata_{geoguessrId}.json     # Map metadata
```

### Sample Meta JSON Structure
```json
{
  "country": "Brazil",
  "metaName": "Insulators - black double UFO-shaped",
  "note": "<p>Poles with these <strong>black double UFO-shaped insulators in a trident formation</strong> are common in Maranh\u00e3o. The insulator itself is common across many states, but not in this configuration.</p>",
  "images": ["https://learnablemeta.com/images/1449/..."],
  "footer": "Description and images taken from:  <a href=\"https://www.plonkit.net/brazil\"...",
  "lat": -2.055167,
  "lng": -45.449177
}
```

### Dataset Output Format

Each sample returns a tuple:
```python
image, concept_idx, target_idx, metadata = dataset[0]
```

- `image`: torch.Tensor of shape (3, 224, 224) - RGB image
- `concept_idx`: int - Encoded concept index
- `target_idx`: int - Encoded country index
- `metadata`: dict - Original strings and coordinates

## Usage Examples

### Basic Dataset Loading
```python
from src.dataset import PanoramaCBMDataset
from torch.utils.data import DataLoader

dataset = PanoramaCBMDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    images, concept_indices, target_indices, metadata = batch
    # Training logic here
```

### Train/Validation/Test Splits
```python
from src.dataset import create_splits, SubsetDataset

# Create splits
train_samples, val_samples, test_samples = create_splits(dataset.samples)

# Create subset datasets
train_dataset = SubsetDataset(dataset, train_samples)
val_dataset = SubsetDataset(dataset, val_samples)
test_dataset = SubsetDataset(dataset, test_samples)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

### Concept and Country Decoding
```python
# Access encoders
concept_to_idx = dataset.concept_to_idx
country_to_idx = dataset.country_to_idx

# Decode predictions
predicted_concept = dataset.idx_to_concept[concept_idx]
predicted_country = dataset.idx_to_country[target_idx]
```

## CBM Training Setup

This dataset is designed for Concept Bottleneck Model training where:
- **Bottleneck layer**: Predicts concept presence (57-dimensional vector)
- **Final layer**: Predicts country from concepts (33 classes)
- **Interpretability**: Concept predictions provide explainable intermediate representations

### Example Training Loop
```python
import torch
import torch.nn as nn

class SimpleCBM(nn.Module):
    def __init__(self, num_concepts, num_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )
        self.concept_layer = nn.Linear(64, num_concepts)
        self.classifier = nn.Linear(num_concepts, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        concepts = torch.sigmoid(self.concept_layer(features))
        outputs = self.classifier(concepts)
        return concepts, outputs

model = SimpleCBM(num_concepts=57, num_classes=33)
```

## Dependencies

- torch
- torchvision
- numpy
- PIL (Pillow)
- tqdm
- pathlib

## Data Sources

- **GeoGuessr**: Street View panoramas
- **LearnableMeta**: Meta annotations and location data
- **PlonkIt**: Additional meta descriptions and images

## Citation

If you use this dataset in your research, please cite the original data sources and acknowledge the GeoGuessr and LearnableMeta communities.

## License

This dataset is provided for research purposes. Please respect the terms of service of the original data providers (Google Street View, GeoGuessr, LearnableMeta).
