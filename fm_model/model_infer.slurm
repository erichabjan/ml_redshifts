#!/bin/bash
#SBATCH --job-name=redshift_infer
#SBATCH --output=/home/habjan.e/SuperBIT_code/Redshift_ml/Sandbox_notebooks/jax_slurm_logs/redshift_infer_%j.log
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4

# Samples the trained model over train, validation and test and writes the
# draws plus the true redshifts to the model_datasets directory.
#
# Defaults to the "baseline" run and its best-validation checkpoint
# (<run>_best_state.msgpack), which is the one selected on sigma_NMAD rather
# than the last-step weights.
#
# Usage: sbatch model_infer.slurm [run_name] [extra args for model_infer.py]
#   sbatch model_infer.slurm
#   sbatch model_infer.slurm baseline --checkpoint final
#   sbatch model_infer.slurm shrunk --splits test --posterior-samples 1000

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

"${PYTHON[@]}" /home/habjan.e/SuperBIT_code/Redshift_ml/ml_redshifts/fm_model/model_infer.py \
    --run-name "$RUN_NAME" \
    "$@"
