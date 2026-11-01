#!/usr/bin/env bash
# Submit a full run from your group repository through the regular queue:
#   module swap cluster/joltik
#   bash setup-wandb.sh             # once per account and repository
#   sbatch jobs/submission.sh
# The lab's one reservation job per group is jobs/interactive.sh.
#SBATCH --job-name=brats-baseline
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from your repository root}"

COURSE_INPUT=/data/gent/courses/2026/ASIP_E092841/input
DATA_ROOT="$COURSE_INPUT/P2_nnUnet_data/Task01_BrainTumour"
VENV_ROOT="$COURSE_INPUT/P2_venv/joltik"
PYTORCH_MODULE=PyTorch/2.7.1-foss-2024a-CUDA-12.6.0

# setup-wandb.sh stores the non-secret project name locally. An environment
# variable can override it for a particular run. The API key stays in W&B's
# per-user login, never in this script or repository.
if [[ -z "${WANDB_PROJECT:-}" && -s .wandb-project ]]; then
  WANDB_PROJECT=$(<.wandb-project)
fi
: "${WANDB_PROJECT:?Run bash setup-wandb.sh, or set WANDB_PROJECT before sbatch}"
test -d "$DATA_ROOT/imagesTr" && test -d "$DATA_ROOT/labelsTr"
test -f "$VENV_ROOT/bin/activate"

source /etc/profile.d/modules.sh
module load "$PYTORCH_MODULE"
source "$VENV_ROOT/bin/activate"

python main.py \
  --data-root "$DATA_ROOT" \
  --wandb-project "$WANDB_PROJECT" \
  --run-name "${RUN_NAME:-00-baseline}" \
  --model "${MODEL:-basic}" \
  --loss "${LOSS:-baseline}" \
  --optimizer "${OPTIMIZER:-adam}" \
  --epochs "${EPOCHS:-50}" \
  --val-interval 2 \
  --seed 42 \
  --batch-size "${BATCH_SIZE:-2}" \
  --patch-size "${PATCH_SIZE:-64,64,64}" \
  --patches-per-case 1 \
  --learning-rate "${LEARNING_RATE:-0.001}" \
  --num-workers "${SLURM_CPUS_PER_TASK:-4}"
