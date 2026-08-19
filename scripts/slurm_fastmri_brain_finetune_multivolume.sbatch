#!/bin/bash

# Second pilot: broaden volume and mask coverage before increasing epochs.

#SBATCH --job-name=ram_ft_multi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/students/studxuzho1/ram-fastmri-brain-adapter/logs/finetune-multivolume-%j.out
#SBATCH --error=/home/students/studxuzho1/ram-fastmri-brain-adapter/logs/finetune-multivolume-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

REPO_ROOT=/home/students/studxuzho1/ram-fastmri-brain-adapter
PYTHON_BIN=/home/students/studxuzho1/envs/ram/bin/python
ESPIRIT_ROOT=/mnt/qdata/rawdata/fastMRI/brain/multicoil_val_espirit
CHECKPOINT=/mnt/ceph/vol_02_home_students/studxuzho1/.cache/huggingface/hub/models--mterris--ram/blobs/1292571adc3d15f9db5e7f5bd92265599030b6b816637367d0f5992d8dbdef7a
RESULTS_ROOT=/home/students/studxuzho1/ram-results
EXPECTED_BRANCH=feature/fastmri-brain-adapter
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

TRAIN_INPUTS=(
    "$ESPIRIT_ROOT/file_brain_AXFLAIR_200_6002560.h5"
    "$ESPIRIT_ROOT/file_brain_AXFLAIR_200_6002562.h5"
    "$ESPIRIT_ROOT/file_brain_AXFLAIR_200_6002572.h5"
    "$ESPIRIT_ROOT/file_brain_AXFLAIR_200_6002574.h5"
    "$ESPIRIT_ROOT/file_brain_AXFLAIR_200_6002581.h5"
    "$ESPIRIT_ROOT/file_brain_AXFLAIR_200_6002584.h5"
    "$ESPIRIT_ROOT/file_brain_AXFLAIR_200_6002588.h5"
    "$ESPIRIT_ROOT/file_brain_AXFLAIR_200_6002603.h5"
    "$ESPIRIT_ROOT/file_brain_AXFLAIR_200_6002605.h5"
    "$ESPIRIT_ROOT/file_brain_AXFLAIR_200_6002634.h5"
)
VALIDATION_INPUTS=(
    "$ESPIRIT_ROOT/file_brain_AXFLAIR_200_6002447.h5"
    "$ESPIRIT_ROOT/file_brain_AXFLAIR_200_6002652.h5"
    "$ESPIRIT_ROOT/file_brain_AXFLAIR_200_6002656.h5"
)
RUN_ID="fastmri-brain-finetune-r4-multivolume-job${SLURM_JOB_ID}"
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
for required_file in \
    "${TRAIN_INPUTS[@]}" \
    "${VALIDATION_INPUTS[@]}" \
    "$CHECKPOINT"; do
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
echo "Training volumes: ${#TRAIN_INPUTS[@]}"
echo "Validation volumes: ${#VALIDATION_INPUTS[@]}"
echo "Training cases: 10 volumes x 3 slices x 3 masks = 90"
echo "Output: $OUTPUT_DIR"

"$PYTHON_BIN" scripts/finetune_fastmri_brain_smoke.py \
    --train-h5 "${TRAIN_INPUTS[@]}" \
    --validation-h5 "${VALIDATION_INPUTS[@]}" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --train-slices 7 8 9 \
    --validation-slices 8 \
    --train-mask-seeds 0 1 2 \
    --validation-mask-seeds 0 \
    --acceleration 4 \
    --center-fraction 0.08 \
    --normalization-percentile 99.5 \
    --noise-sigma 0.001 \
    --epochs 1 \
    --learning-rate 3e-6 \
    --device cuda

cp "$REPO_ROOT/logs/finetune-multivolume-${SLURM_JOB_ID}.out" "$OUTPUT_DIR/slurm.out"
if [[ -s "$REPO_ROOT/logs/finetune-multivolume-${SLURM_JOB_ID}.err" ]]; then
    cp "$REPO_ROOT/logs/finetune-multivolume-${SLURM_JOB_ID}.err" "$OUTPUT_DIR/slurm.err"
fi
