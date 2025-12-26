#!/bin/bash
# Submit multiple cross-attention experiments with different hyperparameters

set -e

echo "Submitting cross-attention sweep experiments..."

# Experiment 1: Default (3-layer mixer, 8 heads, top-16 attention)
echo "Submitting: default config (depth=3, heads=8, topk=16)"
sbatch --export=ALL,MIX_DEPTH=3,XATTN_HEADS=8,XATTN_TOPK=16,CONCEPT_DIM=256 \
  jobs/train_cbm_cross_attention.job

# Experiment 2: Deeper mixer (4 layers)
echo "Submitting: deeper mixer (depth=4)"
sbatch --export=ALL,MIX_DEPTH=4,XATTN_HEADS=8,XATTN_TOPK=16,CONCEPT_DIM=256 \
  jobs/train_cbm_cross_attention.job

# Experiment 3: More heads (16 heads)
echo "Submitting: more heads (heads=16)"
sbatch --export=ALL,MIX_DEPTH=3,XATTN_HEADS=16,XATTN_TOPK=16,CONCEPT_DIM=512 \
  jobs/train_cbm_cross_attention.job

# Experiment 4: Full attention (no top-K sparsification)
echo "Submitting: full attention (topk=0)"
sbatch --export=ALL,MIX_DEPTH=3,XATTN_HEADS=8,XATTN_TOPK=0,CONCEPT_DIM=256 \
  jobs/train_cbm_cross_attention.job

# Experiment 5: Sharper attention (lower temperature)
echo "Submitting: sharper attention (temp=0.5)"
sbatch --export=ALL,MIX_DEPTH=3,XATTN_HEADS=8,XATTN_TOPK=16,XATTN_TEMP=0.5,CONCEPT_DIM=256 \
  jobs/train_cbm_cross_attention.job

# Experiment 6: Larger concept dim
echo "Submitting: larger concept dim (dim=512)"
sbatch --export=ALL,MIX_DEPTH=3,XATTN_HEADS=8,XATTN_TOPK=16,CONCEPT_DIM=512 \
  jobs/train_cbm_cross_attention.job

# Experiment 7: Top-32 attention (more patches)
echo "Submitting: top-32 attention"
sbatch --export=ALL,MIX_DEPTH=3,XATTN_HEADS=8,XATTN_TOPK=32,CONCEPT_DIM=256 \
  jobs/train_cbm_cross_attention.job

# Experiment 8: Combined improvements (depth=4, heads=8, dim=512)
echo "Submitting: combined (depth=4, dim=512)"
sbatch --export=ALL,MIX_DEPTH=4,XATTN_HEADS=8,XATTN_TOPK=16,CONCEPT_DIM=512 \
  jobs/train_cbm_cross_attention.job

echo ""
echo "All experiments submitted!"
echo "Monitor with: squeue -u \$USER"
echo "View outputs in: jobs/outputs/cbm_cross_attention_*.out"

