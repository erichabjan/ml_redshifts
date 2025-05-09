#!/bin/bash
#SBATCH --job-name=mlp_redshift
#SBATCH --output=/home/habjan.e/SuperBIT_code/Redshift_ml/Sandbox_notebooks/mlp_redshift.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=2:00:00
#SBATCH --mem=20G
#SBATCH --partition=short

source activate sb_metacal  

python3 /home/habjan.e/SuperBIT_code/Redshift_ml/mlp_redshifts/mlp_model_train.py