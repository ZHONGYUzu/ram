#!/bin/bash

# ram-016: fastMRI random Cartesian R24 generalization, five cases, three seeds.

#SBATCH --job-name=ram016_r24
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-016/logs/slurm-%j.out
#SBATCH --error=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-016/logs/slurm-%j.err

exec /home/students/studxuzho1/ram-fastmri-brain-adapter/scripts/run_fastmri_paper_condition.sh ram-016 24 0.01
