#!/bin/bash

#SBATCH --job-name=ram_s8_a8_dc
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/ram_s8_a8_dc_%A_%a.out
#SBATCH --error=logs/ram_s8_a8_dc_%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_ROOT=/home/students/studxuzho1/ram
PYTHON_BIN=/home/students/studxuzho1/envs/ram/bin/python
INPUT_H5=/mnt/qdata/rawdata/CINE/2D_h5_compressed/Sub0008.h5
MASK_TXT=/home/students/studxusiy1/mr_recon/masks/mask_VISTA_132x25_acc8_8.txt
RESULTS_ROOT=/home/students/studxuzho1/ram-results
DC_GAMMAS=(0.1 1 10)

dc_gamma="${DC_GAMMAS[$SLURM_ARRAY_TASK_ID]}"
experiment_id="sub0008-acc8-slice06-time10-p995-dc${dc_gamma}"

finish() {
    exit_status=$?
    echo "Exit status: $exit_status"
    echo "Finished: $(date --iso-8601=seconds)"
    exit "$exit_status"
}
trap finish EXIT

cd "$REPO_ROOT"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found or not executable: $PYTHON_BIN" >&2
    exit 2
fi
if [[ ! -f "$INPUT_H5" ]]; then
    echo "Input H5 not found: $INPUT_H5" >&2
    exit 2
fi
if [[ ! -f "$MASK_TXT" ]]; then
    echo "VISTA mask not found: $MASK_TXT" >&2
    exit 2
fi

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Repository: $REPO_ROOT"
echo "Git commit: $(git rev-parse HEAD)"
echo "Experiment: $experiment_id"
echo "Post-RAM data consistency gamma: $dc_gamma"

nvidia-smi
"$PYTHON_BIN" -c 'import torch; print("Torch:", torch.__version__); print("CUDA:", torch.version.cuda); print("CUDA available:", torch.cuda.is_available())'

bash scripts/run_cine_experiment.sh \
    --experiment-id "$experiment_id" \
    --input-h5 "$INPUT_H5" \
    --mask-txt "$MASK_TXT" \
    --results-root "$RESULTS_ROOT" \
    --slice-index 6 \
    --time-index 10 \
    --normalization p99.5 \
    --dc-gamma "$dc_gamma" \
    --batch-size 1 \
    --device cuda
