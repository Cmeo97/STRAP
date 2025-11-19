#!/bin/bash
#SBATCH --job-name=dinov3_retrieval
#SBATCH --output=logs/dinov3_retrieval_%j.out
#SBATCH --error=logs/dinov3_retrieval_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Initialize conda
eval "$(/sw/arch/RHEL8/EB_production/2023/software/Miniconda3/23.5.2-0/bin/conda shell.bash hook)"

# Activate conda environment
conda activate strap

# Print Python and CUDA info
echo "Python: $(which python)"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")')"
echo "=========================================="

# Run the retrieval script with DINOv3
python strap/retrieval/retrieval.py --model dinov3

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="

