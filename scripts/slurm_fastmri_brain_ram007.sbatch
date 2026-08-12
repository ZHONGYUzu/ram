#!/bin/bash
#SBATCH --job-name=ram007_vd_r16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-007/logs/slurm-%j.out
#SBATCH --error=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-007/logs/slurm-%j.err
exec /home/students/studxuzho1/ram-fastmri-brain-adapter/scripts/run_fastmri_mask_condition.sh ram-007 polynomial-vd-random 16 6 0.02
