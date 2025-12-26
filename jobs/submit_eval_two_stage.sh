#!/bin/bash
# Submit both two_stage evaluations in parallel

echo "Submitting two_stage evaluation jobs in parallel..."

# Submit Configuration 23
JOB_23=$(sbatch jobs/eval_two_stage_config_23.job | awk '{print $4}')
echo "Submitted Configuration 23 (job ID: $JOB_23)"

# Submit Configuration 24
JOB_24=$(sbatch jobs/eval_two_stage_config_24.job | awk '{print $4}')
echo "Submitted Configuration 24 (job ID: $JOB_24)"

echo ""
echo "Both jobs submitted! They will run in parallel."
echo "Monitor with: squeue -u $USER"

