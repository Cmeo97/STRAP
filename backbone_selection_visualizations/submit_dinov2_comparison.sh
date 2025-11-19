#!/bin/bash
#SBATCH --job-name=dinov2_compare
#SBATCH --output=logs/dinov2_compare_%j.out
#SBATCH --error=logs/dinov2_compare_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu

# Create logs directory
mkdir -p logs

# Activate environment
source activate strap

# Configuration (matches submit_patch_viz_static.sh)
DEMO_FILE="data/LIBERO/libero_90/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_demo.hdf5"
DEMO_IDX=0
TIMESTEPS="0 40 80 120"
OUTPUT_DIR="dinov2_comparison"
DEVICE="cuda"

# Model selection (leave empty for all models, or specify subset)
# Example: MODELS="vits14 vitb14 vitl14"
MODELS=""

echo "=========================================="
echo "DINOv2 Model Comparison (SLURM Job)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""
echo "Demo file: $DEMO_FILE"
echo "Demo index: $DEMO_IDX"
echo "Timesteps: $TIMESTEPS"
echo "Output directory: $OUTPUT_DIR"
echo "Device: $DEVICE"

if [ -n "$MODELS" ]; then
    echo "Models to compare: $MODELS"
else
    echo "Models to compare: ALL (4 models)"
fi

echo "=========================================="
echo ""

# Check if demo file exists
if [ ! -f "$DEMO_FILE" ]; then
    echo "Error: Demo file not found: $DEMO_FILE"
    echo ""
    echo "Available demos:"
    ls data/LIBERO/libero_90/*.hdf5 | grep -v embeds | head -10
    exit 1
fi

# Build command
CMD="python compare_dinov2_models.py \
    --demo-file \"$DEMO_FILE\" \
    --demo-idx $DEMO_IDX \
    --timesteps $TIMESTEPS \
    --output-dir \"$OUTPUT_DIR\" \
    --device $DEVICE"

# Add model selection if specified
if [ -n "$MODELS" ]; then
    CMD="$CMD --models $MODELS"
fi

# Run comparison
echo "Running model comparison..."
echo "Command: $CMD"
echo ""

eval $CMD

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job Complete!"
echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Model comparison successful!"
    echo ""
    echo "Results directory structure:"
    echo "  $OUTPUT_DIR/"

    # List generated directories
    if [ -d "$OUTPUT_DIR" ]; then
        for model_dir in "$OUTPUT_DIR"/*; do
            if [ -d "$model_dir" ]; then
                MODEL_NAME=$(basename "$model_dir")
                NUM_FILES=$(ls -1 "$model_dir"/*.png 2>/dev/null | wc -l)
                if [ $NUM_FILES -gt 0 ]; then
                    echo "    ├── $MODEL_NAME/ ($NUM_FILES files)"
                fi
            fi
        done

        if [ -f "$OUTPUT_DIR/model_comparison_summary.png" ]; then
            SIZE=$(du -h "$OUTPUT_DIR/model_comparison_summary.png" | cut -f1)
            echo "    └── model_comparison_summary.png ($SIZE)"
        fi
    fi

    echo ""
    echo "To view results:"
    echo "  ls $OUTPUT_DIR/*/"
    echo "  # View comparison summary:"
    echo "  open $OUTPUT_DIR/model_comparison_summary.png"
    echo ""
    echo "To compare specific models only, edit this script and set:"
    echo "  MODELS=\"vits14 vitb14 vitl14\""
else
    echo "✗ Model comparison failed with exit code: $EXIT_CODE"
    echo "Check the error log for details."
fi

echo ""
echo "Log files:"
echo "  Output: logs/dinov2_compare_${SLURM_JOB_ID}.out"
echo "  Errors: logs/dinov2_compare_${SLURM_JOB_ID}.err"
