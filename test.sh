#!/bin/bash
#SBATCH --job-name=test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:02:00
#SBATCH --partition=gpucluster
#SBATCH --output=slurm-%j.out

echo "Hello SLURM"