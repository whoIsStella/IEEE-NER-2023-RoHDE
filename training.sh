#!/bin/bash
#
# ───────────── SBATCH OPTIONS ─────────────
#SBATCH --job-name=RoHDE              # job name
#SBATCH --partition=gpucluster
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --time=06:30:00              # time limit (hh:mm:ss)
#SBATCH --cpus-per-task=1  
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# ───────────── ENV SETUP ─────────────

cd $HOME/IEEE-NER-2023-RoHDE-1

# — Create conda env if missing —
export PATH="$HOME/miniconda/bin:$PATH"
source $HOME/miniconda/etc/profile.d/conda.sh
conda activate RoHDE
pip install scipy
pip install fastdtw 

echo "Conda env:" $(conda info --envs)
which python
python --version

# Ensure results directory exists
mkdir -p results

# ───────────── RUN SCRIPTS ─────────────

echo "[$(date +"%T")] Running WGAN-GP-train.py…"
python -u WGAN-GP-train.py \
  > results/WGAN-GP-train-${SLURM_JOB_ID}.out 2>&1

echo "[$(date +"%T")] Running EMG-Classifier.py…"
python -u EMG-Classifier.py \
  > results/EMG-Classifier-${SLURM_JOB_ID}.out 2>&1

echo "[$(date +"%T")] Running RoHDE.py…"
python -u RoHDE.py \
  > results/RoHDE-${SLURM_JOB_ID}.out 2>&1

echo "[$(date +"%T")] All done. Logs are in results/."

