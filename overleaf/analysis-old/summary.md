# Experiment Results Summary

This report summarizes the evaluation results for Stage 1 (concept classification) and Stage 2 (geolocation) models.

## Stage 1: Concept Classification Results

### Test Split Evaluation

| Variant | Child Concept (Top-1) | Parent Concept (Top-1) | Child Concept (Top-5) | Parent Concept (Top-5) |
|---------|----------------------|----------------------|---------------------|----------------------|
| Default | 0.4612 | 0.3857 | 0.7156 | 0.7133 |
| Finetuned | 0.4552 | 0.4796 | 0.6815 | 0.7254 |

## Stage 2: Geolocation Results

### Test Split Evaluation

| Variant | Ablation | Median Error (km) | Mean Error (km) | Cell Acc | City Acc | Region Acc | Country Acc |
|---------|----------|-------------------|----------------|----------|----------|------------|-------------|
| Default | image_only | 222.04 | 1070.54 | 0.3741 | 0.1750 | 0.4823 | 0.7533 |
| Finetuned | image_only | 153.98 | 790.45 | 0.4301 | 0.2017 | 0.5462 | 0.8058 |

### Hugging Face GeoGuessr Dataset Evaluation

| Variant | Ablation | Median Error (km) | Mean Error (km) | Cell Acc | City Acc | Region Acc | Country Acc |
|---------|----------|-------------------|----------------|----------|----------|------------|-------------|
| Default | image_only | 447.30 | 1851.63 | 0.2216 | 0.0301 | 0.3069 | 0.6298 |
| Finetuned | image_only | 375.11 | 1663.49 | 0.2427 | 0.0321 | 0.3392 | 0.6705 |

