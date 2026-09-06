#!/bin/bash
#SBATCH --job-name=ram-029-text
#SBATCH --array=0-4%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-029/logs/slurm-%A_%a.out
#SBATCH --error=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-029/logs/slurm-%A_%a.err

set -euo pipefail

REPO=/home/students/studxuzho1/ram-fastmri-brain-adapter
RUN="$REPO/runs/ram-029"
BASE="$REPO/runs/ram-024"
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
mkdir -p "$VOLUME_OUTPUT"

cd "$RUN"
for SLICE_INDEX in "$SLICE_1" "$SLICE_2" "$SLICE_3"; do
  CASE_ROOT="$VOLUME_OUTPUT/slice-$(printf '%03d' "$SLICE_INDEX")"
  CLEAN_DIR="$BASE/output/$VOLUME_ID/slice-$(printf '%03d' "$SLICE_INDEX")/clean"
  FINAL_OUTPUT="$CASE_ROOT/text-small"
  TEMP_OUTPUT="$CASE_ROOT/.text-small.tmp-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
  test -f "$CLEAN_DIR/experiment_setup.json"
  test -f "$CLEAN_DIR/metrics.json"
  test -f "$CLEAN_DIR/reconstructions.npz"

  if test -f "$FINAL_OUTPUT/COMPLETE"; then
    echo "Already complete, skipping: $FINAL_OUTPUT"
    continue
  fi
  if test -e "$FINAL_OUTPUT"; then
    echo "Refusing to overwrite incomplete final output: $FINAL_OUTPUT" >&2
    exit 2
  fi
  if test -e "$TEMP_OUTPUT"; then
    echo "Refusing to overwrite current-attempt temporary output: $TEMP_OUTPUT" >&2
    exit 2
  fi

  CLEAN_SCALE="$(
    sed -n 's/^[[:space:]]*"normalization_scale": \([^,]*\),/\1/p' \
      "$CLEAN_DIR/experiment_setup.json" | head -n 1
  )"
  test -n "$CLEAN_SCALE"

  "$PYTHON" "$RUN/code/scripts/validate_ram029_text.py" \
    --raw-h5 "$RAW_H5" \
    --espirit-h5 "$ESPIRIT_H5" \
    --checkpoint "$CHECKPOINT" \
    --clean-dir "$CLEAN_DIR" \
    --output-dir "$TEMP_OUTPUT" \
    --slice "$SLICE_INDEX" \
    --reference-key reference_acl15 \
    --smaps-key smaps_acl15 \
    --map-index 0 \
    --crop-size 320 \
    --acceleration 4 \
    --center-fraction 0.08 \
    --normalization-scale "$CLEAN_SCALE" \
    --noise-sigma 0.001 \
    --text RAM \
    --glyph-scale 2 \
    --text-top 270 \
    --text-left 246 \
    --text-amplitude-relative-reference-p995 0.10 \
    --seed 0 \
    --device cuda

  for REQUIRED in experiment_setup.json command.txt environment.json metrics.json metrics.csv preview.png reconstructions.npz; do
    test -s "$TEMP_OUTPUT/$REQUIRED"
  done
  printf "%s\t%s\t%s\t%s\n" "$ACQUISITION" "$FILE_NAME" "$SLICE_INDEX" "$CLEAN_SCALE" \
    > "$TEMP_OUTPUT/case_complete.tsv"
  touch "$TEMP_OUTPUT/COMPLETE"
  mkdir -p "$CASE_ROOT"
  mv "$TEMP_OUTPUT" "$FINAL_OUTPUT"
done
