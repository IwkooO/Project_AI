#!/bin/bash
# Submit Deeper MIL sweep experiments
# Tier 1: Depth sweep (D2, D3, D4)

set -e

echo "============================================================"
echo "Submitting Deeper MIL Sweep - Tier 1 (Depth)"
echo "============================================================"
echo ""
echo "Baseline: mix_depth=1, mix_heads=4, concept_dim=256 -> 41.4% acc@1"
echo ""

# Experiment D2: 2-layer mixer
echo "Submitting D2: mix_depth=2, mix_heads=4, concept_dim=256"
sbatch --export=ALL,MIX_DEPTH=2,MIX_HEADS=4,CONCEPT_DIM=256 \
  jobs/train_cbm_deeper_mil.job

# Experiment D3: 3-layer mixer
echo "Submitting D3: mix_depth=3, mix_heads=4, concept_dim=256"
sbatch --export=ALL,MIX_DEPTH=3,MIX_HEADS=4,CONCEPT_DIM=256 \
  jobs/train_cbm_deeper_mil.job

# Experiment D4: 4-layer mixer
echo "Submitting D4: mix_depth=4, mix_heads=4, concept_dim=256"
sbatch --export=ALL,MIX_DEPTH=4,MIX_HEADS=4,CONCEPT_DIM=256 \
  jobs/train_cbm_deeper_mil.job

echo ""
echo "============================================================"
echo "Tier 1 experiments submitted!"
echo "============================================================"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "View outputs: jobs/outputs/cbm_deeper_mil_*.out"
echo "Checkpoints:  checkpoints/phase1_deeper_mil/"
echo ""
echo "After Tier 1 results, run Tier 2 if depth helps:"
echo "  - DW1: depth=3, heads=8, dim=256"
echo "  - DW2: depth=3, heads=4, dim=512"
echo "  - DW3: depth=3, heads=8, dim=512"
echo ""

