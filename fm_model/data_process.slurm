#!/bin/bash
#SBATCH --job-name=data_process
#SBATCH --output=/home/habjan.e/SuperBIT_code/Redshift_ml/Sandbox_notebooks/jax_slurm_logs/data_process_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=20G
#SBATCH --partition=short

set -euo pipefail

# Activate the conda environment (h5py + astropy live here, not in the system python)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jax_gpu

# Slurm inherits the submitting shell's modules, and the shared anaconda3 module
# sits ahead of the env on PATH -- so call the env's interpreter directly rather
# than relying on `python3` resolving to the right one.
PYTHON="$CONDA_PREFIX/bin/python3"

mkdir -p /projects/mccleary_group/habjan.e/SuperBIT/data/model_datasets

echo "python: $PYTHON"
"$PYTHON" -c "import h5py, astropy, numpy; print('h5py', h5py.__version__, '| astropy', astropy.__version__, '| numpy', numpy.__version__)"

"$PYTHON" /home/habjan.e/SuperBIT_code/Redshift_ml/ml_redshifts/fm_model/data_process.py
