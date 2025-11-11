#!/bin/bash
#SBATCH --job-name=jax_cnn
#SBATCH --output=/home/habjan.e/SuperBIT_code/Redshift_ml/Sandbox_notebooks/jax_slurm_logs/cnn_jax_%j.log
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1%1

H_VALUES=(32 64)
HIDDEN_SIZE=${H_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "Selected hidden_size: ${HIDDEN_SIZE} (task ${SLURM_ARRAY_TASK_ID})"

# Load CUDA module
module load cuda/12.3.0

# Activate the conda environment with JAX+CUDA support
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jax_gpu

# Unset any previously set CUDA-related environment variables
unset LD_LIBRARY_PATH
unset XLA_FLAGS

# Set correct CUDA environmental variables
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"

# To address CUBLAS issues, set memory limits
export TF_FORCE_GPU_ALLOW_GROWTH=true 
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 

# Print debug information
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
nvidia-smi  # Show GPU state before running

python -c "import jax; print('JAX version:', jax.__version__); print('Available devices:', jax.devices()); print('Using GPU:', any(d.platform == 'gpu' for d in jax.devices()))"

# Run your JAX script
python3 /home/habjan.e/SuperBIT_code/Redshift_ml/ml_redshifts/Jax_models/train_cnn.py "${HIDDEN_SIZE}"