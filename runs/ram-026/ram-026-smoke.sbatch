#!/bin/bash
#SBATCH --job-name=ram-026-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-026/logs/smoke-%j.out
#SBATCH --error=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-026/logs/smoke-%j.err

set -euo pipefail
REPO=/home/students/studxuzho1/ram-fastmri-brain-adapter
RUN="$REPO/runs/ram-026"
BASE="$REPO/runs/ram-025"
PYTHON=/home/students/studxuzho1/envs/ram/bin/python
CHECKPOINT=/mnt/ceph/vol_02_home_students/studxuzho1/.cache/huggingface/hub/models--mterris--ram/blobs/1292571adc3d15f9db5e7f5bd92265599030b6b816637367d0f5992d8dbdef7a
RAW_ROOT=/mnt/qdata/rawdata/fastMRI/brain/multicoil_val
ESPIRIT_ROOT=/mnt/qdata/rawdata/fastMRI/brain/multicoil_val_espirit
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 MPLBACKEND=Agg
export PYTHONPATH="$RUN/code:$RUN/code/scripts"

IFS=$'\t' read -r ACQUISITION FILE_NAME SLICE_1 SLICE_2 SLICE_3 < "$RUN/manifest.tsv"
VOLUME_ID="${FILE_NAME%.h5}"
SLICE_INDEX="$SLICE_1"
CLEAN_ROOT="$BASE/output/$VOLUME_ID/slice-$(printf '%03d' "$SLICE_INDEX")/clean"
OUTPUT="$RUN/smoke/output/$VOLUME_ID/slice-$(printf '%03d' "$SLICE_INDEX")/horizontal-flip"
test -f "$CLEAN_ROOT/experiment_setup.json"
test ! -e "$OUTPUT"
CLEAN_SCALE="$(sed -n 's/^[[:space:]]*"normalization_scale": \([^,]*\),/\1/p' "$CLEAN_ROOT/experiment_setup.json" | head -n 1)"
test -n "$CLEAN_SCALE"

cd "$REPO"
"$PYTHON" "$RUN/code/scripts/validate_fastmri_multicoil_ram.py" \
  --raw-h5 "$RAW_ROOT/$FILE_NAME" --espirit-h5 "$ESPIRIT_ROOT/$FILE_NAME" \
  --checkpoint "$CHECKPOINT" --output-dir "$OUTPUT" --slice "$SLICE_INDEX" \
  --reference-key reference_acl15 --smaps-key smaps_acl15 --map-index 0 \
  --crop-size 320 --acceleration 4 --center-fraction 0.08 \
  --normalization-percentile 99.5 --normalization-scale "$CLEAN_SCALE" \
  --noise-sigma 0.001 --synthetic-noise-relative-rms 0 \
  --measurement-source measured --spatial-transform horizontal_flip \
  --post-data-consistency none --seed 0 --device cuda

"$PYTHON" "$RUN/code/scripts/check_ram026_case.py" \
  --original "$CLEAN_ROOT" --flipped "$OUTPUT"
echo "ram-026 smoke test passed: $VOLUME_ID slice $SLICE_INDEX"
