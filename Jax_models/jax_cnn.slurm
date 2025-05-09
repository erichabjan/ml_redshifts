#!/bin/bash
#SBATCH --job-name=jax_cnn
#SBATCH --output=/home/habjan.e/SuperBIT_code/Redshift_ml/Sandbox_notebooks/cnn_jax.log
#SBATCH --time=08:00:00
#SBATCH --mem=10G
#SBATCH --partition=short

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sb_metacal

python3 /home/habjan.e/SuperBIT_code/Redshift_ml/ml_redshifts/Jax_models/jax_cnn.py