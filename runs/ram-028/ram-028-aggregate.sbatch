#!/bin/bash
#SBATCH --job-name=ram-028-summary
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-028/logs/summary-%j.out
#SBATCH --error=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-028/logs/summary-%j.err

set -euo pipefail
RUN=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-028
BASE=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-024
PYTHON=/home/students/studxuzho1/envs/ram/bin/python
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 MPLBACKEND=Agg
test ! -e "$RUN/summary"
"$PYTHON" "$RUN/code/scripts/aggregate_ram028.py" --run-dir "$RUN" --baseline-run-dir "$BASE"
