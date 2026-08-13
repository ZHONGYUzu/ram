#!/bin/bash

# ram-014: paper-distribution fastMRI random Cartesian R8, five cases, three seeds.

#SBATCH --job-name=ram014_r8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-014/logs/slurm-%j.out
#SBATCH --error=/home/students/studxuzho1/ram-fastmri-brain-adapter/runs/ram-014/logs/slurm-%j.err

exec /home/students/studxuzho1/ram-fastmri-brain-adapter/scripts/run_fastmri_paper_condition.sh ram-014 8 0.04
