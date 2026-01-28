#!/bin/bash
# Run all three baseline methods on a dataset using dataset name
# Usage: bash scripts/run_all_baselines.sh <dataset_name>
# Example: bash scripts/run_all_baselines.sh stanford_dogs

DATASET_NAME=${1:-"stanford_dogs"}  # Dataset name (default: stanford_dogs)

echo "=========================================="
echo "Running All CLIP Baseline Experiments"
echo "Dataset: $DATASET_NAME"
echo "=========================================="

# 1. Linear Probing (LP)
echo ""
echo "=========================================="
echo "1/3 Running Linear Probing (LP)..."
echo "=========================================="
python train_lp.py --dataset $DATASET_NAME

if [ $? -ne 0 ]; then
    echo "Error: LP training failed!"
    exit 1
fi

# 2. Fine-Tuning (FT)
echo ""
echo "=========================================="
echo "2/3 Running Fine-Tuning (FT)..."
echo "=========================================="
python train_ft.py --dataset $DATASET_NAME

if [ $? -ne 0 ]; then
    echo "Error: FT training failed!"
    exit 1
fi

# 3. LP-FT (Linear Probe then Fine-Tune)
echo ""
echo "=========================================="
echo "3/3 Running LP-FT..."
echo "=========================================="
python train_lp_ft.py --dataset $DATASET_NAME

if [ $? -ne 0 ]; then
    echo "Error: LP-FT training failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ All experiments completed successfully!"
echo "Check outputs/${DATASET_NAME}/ directory for results"
echo "=========================================="
