#!/bin/bash
#SBATCH --job-name=cnn
#SBATCH --output=/home/habjan.e/SuperBIT_code/Redshift_ml/Sandbox_notebooks/cnn_redshift.log
#SBATCH --time=08:00:00
#SBATCH --mem=10G
#SBATCH --partition=short

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sb_metacal

python3 /home/habjan.e/SuperBIT_code/Redshift_ml/ml_redshifts/cnn_model_train.py