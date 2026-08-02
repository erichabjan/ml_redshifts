#!/bin/bash
#SBATCH --job-name=redshift_flow
#SBATCH --output=/home/habjan.e/SuperBIT_code/Redshift_ml/Sandbox_notebooks/jax_slurm_logs/redshift_flow_%j.log
#SBATCH --time=00:30:00
#SBATCH --mem=24G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# Usage: sbatch jax_cnn.slurm [run_name] [extra args for train_redshift_flow.py]
#   sbatch jax_cnn.slurm baseline
#   sbatch jax_cnn.slurm no_mask --no-mask

set -euo pipefail

RUN_NAME="${1:-baseline}"
shift || true

source ~/miniconda3/etc/profile.d/conda.sh
conda activate jax_gpu

# Slurm inherits the submitting shell's modules, and the shared anaconda3
# module sits ahead of the conda env on PATH, so call the env's interpreter
# directly rather than relying on `python3` resolving to the right one.
# -u keeps stdout unbuffered, so the Slurm log still shows how far the job got
# if it dies rather than losing the whole buffer.
PYTHON=("$CONDA_PREFIX/bin/python3" -u)

# Do NOT load the system cuda module here. jax-cuda12-plugin ships its own
# CUDA 12.9 / cuDNN 9.10 wheels and is self-contained; pointing XLA at the
# cuda/12.3.0 module via LD_LIBRARY_PATH and --xla_gpu_cuda_data_dir makes it
# load mismatched cuBLAS/cuDNN and segfault on the first compile. Clear both in
# case they leaked in from the submitting shell.
unset LD_LIBRARY_PATH XLA_FLAGS

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

# Fail immediately rather than silently falling back to CPU, which would be
# far slower and waste the whole allocation
"${PYTHON[@]}" -c "import jax, sys; sys.exit(0 if any(d.platform == 'gpu' for d in jax.devices()) else 1)"

# --max-runtime-hours must stay under the #SBATCH --time above so the training
# loop can checkpoint gracefully before Slurm kills the job
"${PYTHON[@]}" /home/habjan.e/SuperBIT_code/Redshift_ml/ml_redshifts/fm_model/train_redshift_flow.py \
    --run-name "$RUN_NAME" \
    --max-runtime-hours 2.8 \
    "$@"
