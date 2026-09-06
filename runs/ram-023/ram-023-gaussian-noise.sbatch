#!/bin/bash
#SBATCH --job-name=ram-023-noise
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-023/logs/slurm-%j.out
#SBATCH --error=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-023/logs/slurm-%j.err

set -euo pipefail

REPO=/home/students/studxuzho1/ram-fastmri-brain-adapter
RUN="$REPO/runs/ram-023"
PYTHON=/home/students/studxuzho1/envs/ram/bin/python
CHECKPOINT=/mnt/ceph/vol_02_home_students/studxuzho1/.cache/huggingface/hub/models--mterris--ram/blobs/1292571adc3d15f9db5e7f5bd92265599030b6b816637367d0f5992d8dbdef7a
EXPECTED_SHA256=1292571adc3d15f9db5e7f5bd92265599030b6b816637367d0f5992d8dbdef7a
RAW_H5=/mnt/qdata/rawdata/fastMRI/brain/multicoil_val/file_brain_AXFLAIR_200_6002477.h5
ESPIRIT_H5=/mnt/qdata/rawdata/fastMRI/brain/multicoil_val_espirit/file_brain_AXFLAIR_200_6002477.h5

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export MPLBACKEND=Agg
export PYTHONPATH="$RUN/code:$RUN/code/scripts"

test -x "$PYTHON"
test -f "$CHECKPOINT"
test -f "$RAW_H5"
test -f "$ESPIRIT_H5"
ACTUAL_SHA256="$(sha256sum "$CHECKPOINT" | cut -d ' ' -f 1)"
test "$ACTUAL_SHA256" = "$EXPECTED_SHA256"
if test -d "$RUN/output" && test -n "$(find "$RUN/output" -mindepth 1 -print -quit)"; then
  echo "Refusing to overwrite non-empty output directory: $RUN/output" >&2
  exit 2
fi

mkdir -p "$RUN/output"
cd "$REPO"
"$PYTHON" "$RUN/code/scripts/validate_fastmri_multicoil_ram.py" \
  --raw-h5 "$RAW_H5" \
  --espirit-h5 "$ESPIRIT_H5" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$RUN/output" \
  --slice 8 \
  --reference-key reference_acl15 \
  --smaps-key smaps_acl15 \
  --map-index 0 \
  --crop-size 320 \
  --acceleration 4 \
  --center-fraction 0.08 \
  --normalization-percentile 99.5 \
  --normalization-scale 0.00015763349074404687 \
  --noise-sigma 0.001 \
  --synthetic-noise-relative-rms 0.01 \
  --synthetic-noise-seed 23001 \
  --measurement-source measured \
  --post-data-consistency none \
  --seed 0 \
  --device cuda
