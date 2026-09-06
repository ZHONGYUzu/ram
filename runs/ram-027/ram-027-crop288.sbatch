#!/bin/bash
#SBATCH --job-name=ram-027-crop288
#SBATCH --array=0-4%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-027/logs/slurm-%A_%a.out
#SBATCH --error=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-027/logs/slurm-%A_%a.err

set -euo pipefail
REPO=/home/students/studxuzho1/ram-fastmri-brain-adapter
RUN="$REPO/runs/ram-027"
BASELINE="$REPO/runs/ram-024"
PYTHON=/home/students/studxuzho1/envs/ram/bin/python
CHECKPOINT=/mnt/ceph/vol_02_home_students/studxuzho1/.cache/huggingface/hub/models--mterris--ram/blobs/1292571adc3d15f9db5e7f5bd92265599030b6b816637367d0f5992d8dbdef7a
EXPECTED_SHA256=1292571adc3d15f9db5e7f5bd92265599030b6b816637367d0f5992d8dbdef7a
RAW_ROOT=/mnt/qdata/rawdata/fastMRI/brain/multicoil_val
ESPIRIT_ROOT=/mnt/qdata/rawdata/fastMRI/brain/multicoil_val_espirit

export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export MPLBACKEND=Agg PYTHONPATH="$RUN/code"
test -x "$PYTHON"
test -f "$CHECKPOINT"
test "$(sha256sum "$CHECKPOINT" | cut -d ' ' -f 1)" = "$EXPECTED_SHA256"
LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 1))
IFS=$'\t' read -r ACQUISITION FILE_NAME SLICE_1 SLICE_2 SLICE_3 < <(sed -n "${LINE_NUMBER}p" "$RUN/manifest.tsv")
test -n "$FILE_NAME"
RAW_H5="$RAW_ROOT/$FILE_NAME"
ESPIRIT_H5="$ESPIRIT_ROOT/$FILE_NAME"
test -f "$RAW_H5"; test -f "$ESPIRIT_H5"
VOLUME_ID="${FILE_NAME%.h5}"
VOLUME_OUTPUT="$RUN/output/$VOLUME_ID"
if test -e "$VOLUME_OUTPUT"; then
  echo "Refusing to overwrite existing volume output: $VOLUME_OUTPUT" >&2
  exit 2
fi
mkdir -p "$VOLUME_OUTPUT"
cd "$REPO"
for SLICE_INDEX in "$SLICE_1" "$SLICE_2" "$SLICE_3"; do
  BASELINE_DIR="$BASELINE/output/$VOLUME_ID/slice-$(printf '%03d' "$SLICE_INDEX")/clean"
  OUTPUT_DIR="$VOLUME_OUTPUT/slice-$(printf '%03d' "$SLICE_INDEX")/center-crop-288"
  test -f "$BASELINE_DIR/experiment_setup.json"
  SCALE="$(sed -n 's/^[[:space:]]*"normalization_scale": \([^,]*\),/\1/p' "$BASELINE_DIR/experiment_setup.json" | head -n 1)"
  test -n "$SCALE"
  "$PYTHON" "$RUN/code/validate_ram027_crop.py" \
    --raw-h5 "$RAW_H5" --espirit-h5 "$ESPIRIT_H5" \
    --checkpoint "$CHECKPOINT" --baseline-dir "$BASELINE_DIR" \
    --output-dir "$OUTPUT_DIR" --slice "$SLICE_INDEX" \
    --normalization-scale "$SCALE" --crop-size 288 --noise-sigma 0.001 --device cuda
  printf "%s\t%s\t%s\t%s\n" "$ACQUISITION" "$FILE_NAME" "$SLICE_INDEX" "$SCALE" >> "$VOLUME_OUTPUT/completed_cases.tsv"
done
