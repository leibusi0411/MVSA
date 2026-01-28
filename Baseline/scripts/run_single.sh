#!/bin/bash
# Run a single baseline method using dataset name
# Usage: bash scripts/run_single.sh <dataset_name> <method>
# Example: bash scripts/run_single.sh stanford_dogs ft

DATASET_NAME=${1:-"stanford_dogs"}  # Dataset name (e.g., stanford_dogs, cub, food101)
METHOD=${2:-"lp"}                    # Method: lp, ft, or lp_ft

echo "=========================================="
echo "Running CLIP Baseline Experiment"
echo "Dataset: $DATASET_NAME"
echo "Method: $METHOD"
echo "=========================================="

case $METHOD in
    lp)
        echo ""
        echo "Running Linear Probing (LP)..."
        python train_lp.py --dataset $DATASET_NAME
        ;;
    
    ft)
        echo ""
        echo "Running Fine-Tuning (FT)..."
        python train_ft.py --dataset $DATASET_NAME
        ;;
    
    lp_ft)
        echo ""
        echo "Running LP-FT..."
        python train_lp_ft.py --dataset $DATASET_NAME
        ;;
    
    *)
        echo "Unknown method: $METHOD"
        echo "Usage: bash scripts/run_single.sh <dataset_name> <method>"
        echo "  dataset_name: stanford_dogs, cub, food101, etc."
        echo "  method: lp, ft, lp_ft"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Done! Check outputs/ directory for results"
echo "=========================================="
