#!/bin/bash
#SBATCH --account=mth240012p
#SBATCH --job-name=autoencoder_job
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --mem=30G
#SBATCH -o output.log
#SBATCH -e error.log

module load anaconda3
conda activate env_214
python run_autoencoder.py configs/default.yaml