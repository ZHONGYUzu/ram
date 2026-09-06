#!/bin/bash
#SBATCH --job-name=ram-027-summary
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-027/logs/summary-%j.out
#SBATCH --error=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-027/logs/summary-%j.err

set -euo pipefail
RUN=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-027
PYTHON=/home/students/studxuzho1/envs/ram/bin/python
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 MPLBACKEND=Agg
"$PYTHON" "$RUN/code/aggregate_ram027.py"
