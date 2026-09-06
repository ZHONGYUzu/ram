#!/bin/bash
#SBATCH --job-name=ram-025-90cases
#SBATCH --array=0-29%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-025/logs/slurm-%A_%a.out
#SBATCH --error=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-025/logs/slurm-%A_%a.err

set -euo pipefail

REPO=/home/students/studxuzho1/ram-fastmri-brain-adapter
RUN="$REPO/runs/ram-025"
PYTHON=/home/students/studxuzho1/envs/ram/bin/python
CHECKPOINT=/mnt/ceph/vol_02_home_students/studxuzho1/.cache/huggingface/hub/models--mterris--ram/blobs/1292571adc3d15f9db5e7f5bd92265599030b6b816637367d0f5992d8dbdef7a
EXPECTED_SHA256=1292571adc3d15f9db5e7f5bd92265599030b6b816637367d0f5992d8dbdef7a
RAW_ROOT=/mnt/qdata/rawdata/fastMRI/brain/multicoil_val
ESPIRIT_ROOT=/mnt/qdata/rawdata/fastMRI/brain/multicoil_val_espirit
MANIFEST="$RUN/manifest.tsv"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export MPLBACKEND=Agg
export PYTHONPATH="$RUN/code:$RUN/code/scripts"

test -x "$PYTHON"
test -f "$CHECKPOINT"
test -f "$MANIFEST"
ACTUAL_SHA256="$(sha256sum "$CHECKPOINT" | cut -d ' ' -f 1)"
test "$ACTUAL_SHA256" = "$EXPECTED_SHA256"

LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 1))
IFS=$'\t' read -r ACQUISITION FILE_NAME SLICE_1 SLICE_2 SLICE_3 < <(
  sed -n "${LINE_NUMBER}p" "$MANIFEST"
)
test -n "$FILE_NAME"
RAW_H5="$RAW_ROOT/$FILE_NAME"
ESPIRIT_H5="$ESPIRIT_ROOT/$FILE_NAME"
test -f "$RAW_H5"
test -f "$ESPIRIT_H5"

VOLUME_ID="${FILE_NAME%.h5}"
VOLUME_OUTPUT="$RUN/output/$VOLUME_ID"
if test -e "$VOLUME_OUTPUT"; then
  echo "Refusing to overwrite existing volume output: $VOLUME_OUTPUT" >&2
  exit 2
fi
mkdir -p "$VOLUME_OUTPUT"

cd "$REPO"
for SLICE_INDEX in "$SLICE_1" "$SLICE_2" "$SLICE_3"; do
  CASE_ROOT="$VOLUME_OUTPUT/slice-$(printf '%03d' "$SLICE_INDEX")"
  CLEAN_OUTPUT="$CASE_ROOT/clean"
  NOISY_OUTPUT="$CASE_ROOT/noise-rms-0.05"
  NOISE_SEED=$((25000 + SLURM_ARRAY_TASK_ID * 100 + SLICE_INDEX))

  "$PYTHON" "$RUN/code/scripts/validate_fastmri_multicoil_ram.py" \
    --raw-h5 "$RAW_H5" \
    --espirit-h5 "$ESPIRIT_H5" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$CLEAN_OUTPUT" \
    --slice "$SLICE_INDEX" \
    --reference-key reference_acl15 \
    --smaps-key smaps_acl15 \
    --map-index 0 \
    --crop-size 320 \
    --acceleration 4 \
    --center-fraction 0.08 \
    --normalization-percentile 99.5 \
    --noise-sigma 0.001 \
    --synthetic-noise-relative-rms 0 \
    --synthetic-noise-seed "$NOISE_SEED" \
    --measurement-source measured \
    --post-data-consistency none \
    --seed 0 \
    --device cuda

  CLEAN_SCALE="$(
    sed -n 's/^[[:space:]]*"normalization_scale": \([^,]*\),/\1/p' \
      "$CLEAN_OUTPUT/experiment_setup.json" | head -n 1
  )"
  test -n "$CLEAN_SCALE"

  "$PYTHON" "$RUN/code/scripts/validate_fastmri_multicoil_ram.py" \
    --raw-h5 "$RAW_H5" \
    --espirit-h5 "$ESPIRIT_H5" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$NOISY_OUTPUT" \
    --slice "$SLICE_INDEX" \
    --reference-key reference_acl15 \
    --smaps-key smaps_acl15 \
    --map-index 0 \
    --crop-size 320 \
    --acceleration 4 \
    --center-fraction 0.08 \
    --normalization-percentile 99.5 \
    --normalization-scale "$CLEAN_SCALE" \
    --noise-sigma 0.001 \
    --synthetic-noise-relative-rms 0.05 \
    --synthetic-noise-seed "$NOISE_SEED" \
    --measurement-source measured \
    --post-data-consistency none \
    --seed 0 \
    --device cuda

  printf "%s\t%s\t%s\t%s\t%s\n" \
    "$ACQUISITION" "$FILE_NAME" "$SLICE_INDEX" "$NOISE_SEED" "$CLEAN_SCALE" \
    >> "$VOLUME_OUTPUT/completed_cases.tsv"
done
