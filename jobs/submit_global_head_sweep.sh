#!/bin/bash
# Submit sweep of global head experiments (CLS-based global representation)

# Base configuration
export MIX_HEADS=4
export CONCEPT_DIM=256
export TAU=0.25
export TOPK=6
export STK_MASK_PROB=0.4
export STK_K_MASK=1
export MIX_LOCAL_KERNEL=0
export PROJ_TYPE=simple
export DROPOUT=0.3
export WEIGHT_DECAY=0.05
export EARLY_STOP_PATIENCE=10
export LR=5e-5
export EPOCHS=100
export BATCH_SIZE=64
export WARMUP_EPOCHS=8
export USE_POS_ENCODING=true
export MAX_PATCHES=576
export USE_GLOBAL_HEAD=true  # Enable global head
export POOLED_DIM=768        # StreetCLIP pooled embedding dimension
export USE_PER_CONCEPT_TAU=false  # Can combine with global head if desired

echo "============================================================"
echo "Submitting Global Head Sweep (CLS-based)"
echo "============================================================"
echo "Testing different mix depths with global head (local + CLS fusion)"
echo ""

# Tier 1: Test different depths with global head
for DEPTH in 2 3 4; do
  export MIX_DEPTH=$DEPTH
  export WANDB_RUN_NAME="global_head_d${DEPTH}_h${MIX_HEADS}_dim${CONCEPT_DIM}_pos"
  
  echo "Submitting: Depth=${DEPTH}, Heads=${MIX_HEADS}, Dim=${CONCEPT_DIM}, Pos=${USE_POS_ENCODING}, GlobalHead=${USE_GLOBAL_HEAD}"
  sbatch jobs/train_cbm_global_head.job
  sleep 1
done

echo ""
echo "============================================================"
echo "All jobs submitted!"
echo "============================================================"
echo "Monitor with: squeue -u $USER"
echo "Check outputs: jobs/outputs/cbm_global_head_*.out"
echo ""
echo "Note: These experiments use both local (patch-based) and global (CLS-based) heads"
echo "      with learnable per-concept fusion weights."

