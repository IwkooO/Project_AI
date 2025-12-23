# Project AI: GeoGuessr Meta Learning

This repository contains tools for collecting, processing, and analyzing GeoGuessr meta data for training Concept Bottleneck Models (CBMs) and Vision Transformers (ViTs).

## Repository Structure

```
├── scripts/
│   ├── data_collection/     # Scripts for scraping and downloading data
│   │   ├── scrape_lm.py
│   │   ├── scrape_lm_async.py
│   │   ├── download_pano.py
│   │   └── download_pano_parallel.py
│   └── data_processing/     # Scripts for data preprocessing
│       ├── preprocess_panorama.py
│       └── enrich_metas_with_coords.py
├── src/                     # Main dataset implementation
│   └── dataset.py
├── docs/                    # Documentation
│   ├── README_project.md    # Original project documentation
│   └── README_dataset.md    # Dataset documentation
├── data/                    # Data directory (created by scripts)
├── jobs/                    # SLURM job scripts
├── constants.py             # Configuration constants
├── environment.yml          # Conda environment
├── requirements*.txt        # Python requirements
└── README.md               # This file
```

## Quick Start

1. **Install dependencies:**
   ```bash
   conda env create -f environment.yml
   ```

2. **Set up constants:**
   Add your LearnableMeta API key to `constants.py`

3. **Collect data:**
   ```bash
   # Scrape meta information
   python scripts/data_collection/scrape_lm.py

   # Download panorama images
   python scripts/data_collection/download_pano.py
   ```

4. **Process data:**
   ```bash
   # Preprocess images
   python scripts/data_processing/preprocess_panorama.py --panorama-folder data/{geoguessrId}/panorama

   # Enrich meta files with coordinates
   python scripts/data_processing/enrich_metas_with_coords.py
   ```

5. **Use dataset:**
   ```python
   # Phase 1 concept dataset (cached patch tokens)
   from cbm.phase1.data import ConceptDataset

   dataset = ConceptDataset(
       csv_path="path/to/train.csv",
       cached_dir="path/to/cached_dir",
       concept_vocab_path="path/to/concept_vocab.json",
       s2_vocab_path="path/to/s2_cells.json",
       split="train",
   )
   ```

## Documentation

- [Project Documentation](docs/README_project.md) - Original setup and usage
- [Dataset Documentation](docs/README_dataset.md) - Detailed dataset guide

## License

See individual documentation files for licensing information.