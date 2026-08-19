#!/bin/bash

# Evaluate job42720's adapter on held-out fastMRI brain volumes.

#SBATCH --job-name=ram_adapter_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/students/studxuzho1/ram-fastmri-brain-adapter/logs/adapter-eval-%j.out
#SBATCH --error=/home/students/studxuzho1/ram-fastmri-brain-adapter/logs/adapter-eval-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

REPO_ROOT=/home/students/studxuzho1/ram-fastmri-brain-adapter
PYTHON_BIN=/home/students/studxuzho1/envs/ram/bin/python
ESPIRIT_ROOT=/mnt/qdata/rawdata/fastMRI/brain/multicoil_val_espirit
BASE_CHECKPOINT=/mnt/ceph/vol_02_home_students/studxuzho1/.cache/huggingface/hub/models--mterris--ram/blobs/1292571adc3d15f9db5e7f5bd92265599030b6b816637367d0f5992d8dbdef7a
ADAPTER_CHECKPOINT=/home/students/studxuzho1/ram-results/fastmri-brain-finetune-r4-smoke-job42720/adapter-checkpoint.pt
RESULTS_ROOT=/home/students/studxuzho1/ram-results
EXPECTED_BRANCH=feature/fastmri-brain-adapter
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

INPUTS=(
    "$ESPIRIT_ROOT/file_brain_AXFLAIR_200_6002477.h5"
    "$ESPIRIT_ROOT/file_brain_AXFLAIR_200_6002483.h5"
    "$ESPIRIT_ROOT/file_brain_AXFLAIR_200_6002499.h5"
    "$ESPIRIT_ROOT/file_brain_AXFLAIR_200_6002536.h5"
    "$ESPIRIT_ROOT/file_brain_AXFLAIR_200_6002556.h5"
)
RUN_ID="fastmri-brain-adapter-eval-r4-job${SLURM_JOB_ID}"
OUTPUT_DIR="$RESULTS_ROOT/$RUN_ID"

cd "$REPO_ROOT"
if [[ "$(git branch --show-current)" != "$EXPECTED_BRANCH" ]]; then
    echo "Expected branch $EXPECTED_BRANCH" >&2
    exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Refusing to run with uncommitted repository changes" >&2
    git status --short >&2
    exit 2
fi
for required_file in "${INPUTS[@]}" "$BASE_CHECKPOINT" "$ADAPTER_CHECKPOINT"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Required input is missing: $required_file" >&2
        exit 2
    fi
done
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite $OUTPUT_DIR" >&2
    exit 2
fi

echo "Job ID: $SLURM_JOB_ID"
echo "Git commit: $(git rev-parse HEAD)"
echo "Adapter checkpoint: $ADAPTER_CHECKPOINT"
echo "Output: $OUTPUT_DIR"

"$PYTHON_BIN" scripts/evaluate_fastmri_brain_adapter.py \
    --input-h5 "${INPUTS[@]}" \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --adapter-checkpoint "$ADAPTER_CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --slices 7 8 9 \
    --seeds 0 1 2 \
    --acceleration 4 \
    --center-fraction 0.08 \
    --normalization-percentile 99.5 \
    --noise-sigma 0.001 \
    --device cuda

cp "$REPO_ROOT/logs/adapter-eval-${SLURM_JOB_ID}.out" "$OUTPUT_DIR/slurm.out"
if [[ -s "$REPO_ROOT/logs/adapter-eval-${SLURM_JOB_ID}.err" ]]; then
    cp "$REPO_ROOT/logs/adapter-eval-${SLURM_JOB_ID}.err" "$OUTPUT_DIR/slurm.err"
fi
