#!/usr/bin/env bash
# Reserve one GPU for an interactive shell. From the login node:
#   module swap cluster/joltik
#   JOB_ID=$(sbatch --parsable jobs/interactive.sh)
#   squeue -j "$JOB_ID"                 # wait until the job is RUNNING
#   srun --jobid="$JOB_ID" --overlap --pty bash -l
# Inside the shell on joltik, from your group repository:
#   module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
#   source /data/gent/courses/2026/ASIP_E092841/input/P2_venv/joltik/bin/activate
#   export DATA_ROOT=/data/gent/courses/2026/ASIP_E092841/input/P2_nnUnet_data/Task01_BrainTumour
#   python main.py --smoke-test
# Leave the shell with `exit`; release the allocation with `scancel "$JOB_ID"`.
# One ASIP_1 reservation job per group: use it for this interactive job.
# Valid 27 Oct 2026, 13:00-18:00. Shorten --time if less than 2 hours remain.
#SBATCH --job-name=brats-interactive
#SBATCH --reservation=ASIP_1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
echo "GPU allocation $SLURM_JOB_ID is ready on $(hostname)."
echo "Connect from the login node: srun --jobid=$SLURM_JOB_ID --overlap --pty bash -l"
sleep infinity
