#!/bin/bash

# ram-017: public-information reconstruction of RAM's simulated 15-coil R8 test.

#SBATCH --job-name=ram017_mc15
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-017/logs/slurm-%j.out
#SBATCH --error=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-017/logs/slurm-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

REPO_ROOT=/home/students/studxuzho1/ram-fastmri-brain-adapter
RUN_ROOT="$REPO_ROOT/runs/ram-017"
PYTHON_BIN=/home/students/studxuzho1/envs/ram/bin/python
INPUT_H5=/mnt/qdata/rawdata/fastMRI/brain/multicoil_val/file_brain_AXFLAIR_200_6002462.h5
CHECKPOINT=/mnt/ceph/vol_02_home_students/studxuzho1/.cache/huggingface/hub/models--mterris--ram/blobs/1292571adc3d15f9db5e7f5bd92265599030b6b816637367d0f5992d8dbdef7a
CHECKPOINT_SHA256=1292571adc3d15f9db5e7f5bd92265599030b6b816637367d0f5992d8dbdef7a
EXPECTED_BRANCH=feature/fastmri-brain-adapter
OUTPUT_DIR="$RUN_ROOT/output/job-${SLURM_JOB_ID}"
RUN_LOG="$RUN_ROOT/logs/run-${SLURM_JOB_ID}.log"
export PYTHONPATH="$REPO_ROOT/scripts:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_ROOT"
mkdir -p "$RUN_ROOT/output" "$RUN_ROOT/logs"
if [[ "$(git branch --show-current)" != "$EXPECTED_BRANCH" ]]; then
    echo "Expected branch $EXPECTED_BRANCH" >&2
    exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Refusing to run with uncommitted repository changes." >&2
    git status --short >&2
    exit 2
fi
if [[ ! -f "$CHECKPOINT" || ! -f "$INPUT_H5" ]]; then
    echo "Missing checkpoint or input H5; downloads are disabled." >&2
    exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite $OUTPUT_DIR" >&2
    exit 2
fi

cat > "$RUN_ROOT/config.yaml" <<EOF
experiment:
  id: ram-017
  purpose: public-information reconstruction of RAM paper simulated 15-coil R8 test
  exact_paper_reproduction: false
  reason_not_exact: paper does not disclose the coil-map generator, multicoil mask details, normalization implementation, or multicoil noise value
  slurm_job_id: ${SLURM_JOB_ID}
  git_branch: $(git branch --show-current)
  git_commit: $(git rev-parse HEAD)
data:
  source: fastMRI brain multicoil validation
  input_h5: $INPUT_H5
  source_volume: 6002462
  selected_slices: [2, 8, 13]
  clean_image_construction: volume-level ESC/VCC, then complex single-coil image
  crop_shape: [320, 320]
normalization:
  mode: noiseless multicoil zero-filled magnitude p99.5 per slice
physics:
  implementation: deepinv.physics.MultiCoilMRI
  simulated_coils: 15
  coil_map_generator: local deterministic implementation of the 2D sigpy.mri.birdcage_maps formula
  mask: fastMRI-style random Cartesian
  nominal_acceleration: 8
  center_fraction: 0.04
  seed: 0
  noise_model: Gaussian
  noise_sigma: 0.0005
arms:
  noisy: actual Gaussian measurement noise added
  conditioning_only: identical image, mask, and coil maps; sigma supplied to RAM but no noise added
model:
  implementation: deepinv.models.RAM
  checkpoint: $CHECKPOINT
  checkpoint_sha256: $CHECKPOINT_SHA256
  model_call: model(y, physics)
  post_ram_data_consistency: false
EOF

echo "Job ID: $SLURM_JOB_ID" | tee "$RUN_LOG"
echo "Git commit: $(git rev-parse HEAD)" | tee -a "$RUN_LOG"
nvidia-smi | tee -a "$RUN_LOG"
"$PYTHON_BIN" -c 'import torch, deepinv, ram.adapters; print("Torch:", torch.__version__); print("DeepInverse:", deepinv.__version__); print("RAM:", ram.__file__); print("CUDA:", torch.version.cuda)' | tee -a "$RUN_LOG"

"$PYTHON_BIN" scripts/benchmark_fastmri_brain_simulated_multicoil_ram.py \
    --input-h5 "$INPUT_H5" \
    --output-dir "$OUTPUT_DIR" \
    --checkpoint "$CHECKPOINT" \
    --checkpoint-sha256 "$CHECKPOINT_SHA256" \
    --acceleration 8 \
    --center-fraction 0.04 \
    --coils 15 \
    --normalization-percentile 99.5 \
    --noise-sigma 0.0005 \
    --slices 2 8 13 \
    --esc-max-iterations 60 \
    --seed 0 \
    --device cuda 2>&1 | tee -a "$RUN_LOG"

cp "$RUN_LOG" "$OUTPUT_DIR/run.log"
echo "Finished: $(date --iso-8601=seconds)" | tee -a "$RUN_LOG"
