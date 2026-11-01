#!/usr/bin/env bash
# Off-reservation debugging on donphan's shared A2 GPU:
#   module swap cluster/donphan
#   JOB_ID=$(sbatch --parsable jobs/interactive-donphan.sh)
#   squeue -j "$JOB_ID"                 # wait for RUNNING
#   srun --jobid="$JOB_ID" --overlap --pty bash -l
# Inside the shell:
#   module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
#   source /data/gent/courses/2026/ASIP_E092841/input/P2_venv/donphan/bin/activate
#   export DATA_ROOT=/data/gent/courses/2026/ASIP_E092841/input/P2_nnUnet_data/Task01_BrainTumour
# Leave with `exit`, then release the job with `scancel "$JOB_ID"`.
#SBATCH --job-name=brats-debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
echo "Debug allocation $SLURM_JOB_ID is ready on $(hostname)."
echo "Connect: srun --jobid=$SLURM_JOB_ID --overlap --pty bash -l"
sleep infinity
