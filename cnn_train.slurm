#!/bin/bash
#SBATCH --job-name=cnn_redshift
#SBATCH --output=/home/habjan.e/SuperBIT_code/Redshift_ml/Sandbox_notebooks/cnn_redshift.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=2:00:00
#SBATCH --mem=20G
#SBATCH --partition=short

source activate sb_metacal  

python3 /home/habjan.e/SuperBIT_code/Redshift_ml/ml_redshifts/cnn_model_train.py