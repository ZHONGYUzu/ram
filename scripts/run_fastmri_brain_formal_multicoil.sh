#!/bin/bash

# Shared runner for the acquisition-specific formal fastMRI brain benchmarks.

set -euo pipefail

if [[ "$#" -lt 5 ]]; then
    echo "Usage: $0 RUN_ID ACQUISITION CONDITION CASE.h5 [CASE.h5 ...]" >&2
    exit 2
fi

RUN_ID=$1
ACQUISITION=$2
CONDITION=$3
shift 3
CASES=("$@")

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

REPO_ROOT=/home/students/studxuzho1/ram-fastmri-brain-adapter
RUN_ROOT="$REPO_ROOT/runs/$RUN_ID"
DATA_ROOT=/mnt/qdata/rawdata/fastMRI/brain/multicoil_val
PYTHON_BIN=/home/students/studxuzho1/envs/ram/bin/python
CHECKPOINT=/mnt/ceph/vol_02_home_students/studxuzho1/.cache/huggingface/hub/models--mterris--ram/blobs/1292571adc3d15f9db5e7f5bd92265599030b6b816637367d0f5992d8dbdef7a
CHECKPOINT_SHA256=1292571adc3d15f9db5e7f5bd92265599030b6b816637367d0f5992d8dbdef7a
EXPECTED_BRANCH=feature/fastmri-brain-adapter
RUN_LOG="$RUN_ROOT/logs/run-${SLURM_JOB_ID}.log"
SUMMARY_DIR="$RUN_ROOT/summary/job-${SLURM_JOB_ID}"
export PYTHONPATH="$REPO_ROOT/scripts:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_ROOT"
mkdir -p "$RUN_ROOT/output" "$RUN_ROOT/logs" "$RUN_ROOT/summary"
if [[ "$(git branch --show-current)" != "$EXPECTED_BRANCH" ]]; then
    echo "Expected branch $EXPECTED_BRANCH" >&2
    exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Refusing to run with uncommitted repository changes." >&2
    git status --short >&2
    exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Cached checkpoint missing; downloads are disabled: $CHECKPOINT" >&2
    exit 2
fi
if [[ -e "$SUMMARY_DIR" ]]; then
    echo "Refusing to overwrite $SUMMARY_DIR" >&2
    exit 2
fi

printf '%s\n' "${CASES[@]}" > "$RUN_ROOT/cases.txt"
for case_name in "${CASES[@]}"; do
    if [[ ! -f "$DATA_ROOT/$case_name" ]]; then
        echo "Missing fastMRI volume: $DATA_ROOT/$case_name" >&2
        exit 2
    fi
done

cat > "$RUN_ROOT/config.yaml" <<EOF
experiment:
  id: $RUN_ID
  parent_run: ram-017
  purpose: formal $ACQUISITION simulated 15-coil R8 OOD benchmark
  exact_paper_reproduction: false
  slurm_job_id: ${SLURM_JOB_ID}
  git_branch: $(git branch --show-current)
  git_commit: $(git rev-parse HEAD)
  downloads_allowed: false
data:
  dataset: fastMRI brain multicoil validation
  acquisition: $ACQUISITION
  selection: first five $ACQUISITION validation filenames in lexical order
  data_root: $DATA_ROOT
  cases_file: $RUN_ROOT/cases.txt
  case_count: ${#CASES[@]}
  slices_per_case: 3
  slice_selection: equally spaced within the 15%-85% non-edge range
preprocessing:
  clean_image_construction: volume-level ESC/VCC, then complex single-coil image
  esc_max_iterations: 60
  image_shape: [320, 320]
  normalization: noiseless multicoil zero-filled magnitude p99.5 per slice
physics:
  implementation: deepinv.physics.MultiCoilMRI
  simulated_coils: 15
  coil_map_generator: local deterministic implementation of the 2D sigpy.mri.birdcage_maps formula
  mask: fastMRI-style random Cartesian
  nominal_acceleration: 8
  center_fraction: 0.04
  base_seeds: [0, 1, 2]
  per_volume_seed: sha256(filename + base_seed)
  noise_model: Gaussian
  noise_sigma: 0.0005
arms:
  noisy: primary arm with actual Gaussian measurement noise
  conditioning_only: diagnostic arm with sigma supplied to RAM but no added noise
model:
  implementation: deepinv.models.RAM
  checkpoint: $CHECKPOINT
  checkpoint_sha256: $CHECKPOINT_SHA256
  model_call: model(y, physics)
  post_ram_data_consistency: false
evaluation:
  unique_slice_mask_cases: 45
  formula: 5 volumes x 3 slices x 3 base seeds
  metric_rows: 90
  metrics: [PSNR, SSIM, NMSE]
output:
  root: $RUN_ROOT/output
  summary: $SUMMARY_DIR
logs:
  run: $RUN_LOG
  stdout: $RUN_ROOT/logs/slurm-${SLURM_JOB_ID}.out
  stderr: $RUN_ROOT/logs/slurm-${SLURM_JOB_ID}.err
EOF

echo "Job ID: $SLURM_JOB_ID" | tee "$RUN_LOG"
echo "Git commit: $(git rev-parse HEAD)" | tee -a "$RUN_LOG"
echo "Run root: $RUN_ROOT" | tee -a "$RUN_LOG"
nvidia-smi | tee -a "$RUN_LOG"
"$PYTHON_BIN" -c 'import torch, deepinv, ram.adapters; print("Torch:", torch.__version__); print("DeepInverse:", deepinv.__version__); print("RAM:", ram.__file__); print("CUDA:", torch.version.cuda)' | tee -a "$RUN_LOG"

SEEDS=(0 1 2)
INPUT_DIRS=()
for seed in "${SEEDS[@]}"; do
    for case_name in "${CASES[@]}"; do
        case_stem="${case_name%.h5}"
        output_dir="$RUN_ROOT/output/seed${seed}/${case_stem}-job${SLURM_JOB_ID}"
        if [[ -e "$output_dir" ]]; then
            echo "Refusing to overwrite $output_dir" >&2
            exit 2
        fi
        echo "Starting seed ${seed}: ${case_name}" | tee -a "$RUN_LOG"
        "$PYTHON_BIN" scripts/benchmark_fastmri_brain_simulated_multicoil_ram.py \
            --input-h5 "$DATA_ROOT/$case_name" \
            --output-dir "$output_dir" \
            --checkpoint "$CHECKPOINT" \
            --checkpoint-sha256 "$CHECKPOINT_SHA256" \
            --acceleration 8 \
            --center-fraction 0.04 \
            --coils 15 \
            --normalization-percentile 99.5 \
            --noise-sigma 0.0005 \
            --slices-per-volume 3 \
            --edge-fraction 0.15 \
            --esc-max-iterations 60 \
            --seed "$seed" \
            --device cuda 2>&1 | tee -a "$RUN_LOG"
        cp "$RUN_LOG" "$output_dir/run.log"
        INPUT_DIRS+=(--input-dir "$output_dir")
    done
done

"$PYTHON_BIN" scripts/summarize_fastmri_simulated_multicoil_run.py \
    --run-id "$RUN_ID" \
    --condition "$CONDITION" \
    --output-dir "$SUMMARY_DIR" \
    "${INPUT_DIRS[@]}" 2>&1 | tee -a "$RUN_LOG"
cp "$RUN_LOG" "$SUMMARY_DIR/run.log"
echo "Finished: $(date --iso-8601=seconds)" | tee -a "$RUN_LOG"
