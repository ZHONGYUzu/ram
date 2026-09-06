#!/bin/bash
#SBATCH --job-name=ram-029-summary
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-029/logs/summary-%j.out
#SBATCH --error=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-029/logs/summary-%j.err

set -euo pipefail

RUN=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-029
PYTHON=/home/students/studxuzho1/envs/ram/bin/python
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export MPLBACKEND=Agg
export PYTHONPATH="$RUN/code:$RUN/code/scripts"

test -x "$PYTHON"
test ! -e "$RUN/summary"
cd "$RUN"
"$PYTHON" "$RUN/code/scripts/aggregate_ram029.py" --run-dir "$RUN"
