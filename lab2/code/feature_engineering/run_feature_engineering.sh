#!/bin/bash

# Run this code: sbatch job_full.sh

#SBATCH --account=mth240012p       # don't change this
#SBATCH --job-name=lr_final
#SBATCH --cpus-per-task=5          # GPU-shared allows max 5 cpus per GPU
#SBATCH --time 5:00:00
#SBATCH -o test_full.out          # write job console output to file test_full.out 
#SBATCH -e test_full.err          # write job console errors to file test_full.err
#SBATCH --partition=GPU-shared     # don't change this unless you need 8 GPUs
#SBATCH --gpus=v100-32:1           # don't increase this unless you need more than 1 GPU

module load anaconda3
conda activate env_214
echo "The python executable in this environment is:"
which python

python run_feature_engineering.py

echo "Codes successfully done"