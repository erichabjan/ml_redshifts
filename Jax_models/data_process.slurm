#!/bin/bash
#SBATCH --job-name=data_process
#SBATCH --output=/home/habjan.e/SuperBIT_code/Redshift_ml/Sandbox_notebooks/data_process_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=20G
#SBATCH --partition=short

source activate jax_gpu

python3 /home/habjan.e/SuperBIT_code/Redshift_ml/ml_redshifts/Jax_models/data_process.py