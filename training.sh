#!/bin/bash
#
# ───────────── SBATCH OPTIONS ─────────────
#SBATCH --array=1-8
#SBATCH --job-name=RoHDE              # job name
#SBATCH --partition=gpucluster
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --time=8:00:00              # time limit (hh:mm:ss)
#SBATCH --cpus-per-task=4 
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

pip install --upgrade fastdtw

echo "Running on node $(hostname), GPU(s):"
nvidia-smi --query-gpu=name --format=csv

# Compute epoch ranges
EPOCHS_PER_JOB=500
START_EPOCH=$(( (SLURM_ARRAY_TASK_ID-1) * EPOCHS_PER_JOB + 1 ))
END_EPOCH=$(( SLURM_ARRAY_TASK_ID * EPOCHS_PER_JOB ))

echo "This is array task $SLURM_ARRAY_TASK_ID — epochs $START_EPOCH to $END_EPOCH"

mkdir -p results

# ───────────── RUN SCRIPTS ─────────────

echo "[$(date +"%T")] Task $SLURM_ARRAY_TASK_ID → epochs $START–$END"
python -u WGAN-GP-train.py --start-epoch $START --end-epoch $END
  > results/WGAN-GP-${SLURM_JOB_ID}.out 2>&1

echo "[$(date +"%T")] Running EMG-Classifier.py…"
python -u EMG-Classifier.py \
  > results/EMG-Classifier-${SLURM_JOB_ID}.out 2>&1

echo "[$(date +"%T")] Running RoHDE.py…"
python -u RoHDE.py \
  > results/RoHDE-${SLURM_JOB_ID}.out 2>&1

echo "[$(date +"%T")] All done. Logs are in results/."

