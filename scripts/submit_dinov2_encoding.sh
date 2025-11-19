#!/bin/bash
#SBATCH --job-name=dinov2_embed
#SBATCH --output=logs/dinov2_encoding_%j.out
#SBATCH --error=logs/dinov2_encoding_%j.err
#SBATCH --time=2:00:00
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

# Run the encoding script with DINOv2
python strap/embedding/encode_datasets.py --encoder dinov2 --batch-size 256

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
