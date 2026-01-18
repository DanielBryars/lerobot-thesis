#!/bin/bash
# Train Pi0 on SO-101 pick-and-place dataset
#
# Usage:
#   ./scripts/pi0/train_so101.sh              # Full training (20k steps)
#   ./scripts/pi0/train_so101.sh --test       # Quick test (100 steps)
#   ./scripts/pi0/train_so101.sh --pi05       # Use Pi0.5 (larger model)
#
# Prerequisites:
#   1. Run from openpi directory: cd /app/openpi
#   2. Compute norm stats first (see below)
#
# Note: Uses pi0_libero as base config because:
#   - Libero uses action_dim=7 (matches SO-101: 6 joints + gripper)
#   - ALOHA uses action_dim=14 (bimanual, 2 arms) - wrong for SO-101

set -e

# Configuration
# Pi0-ready dataset with normalized gripper [0-1] and delta actions enabled
DATASET="danbhf/sim_pick_place_157ep_pi0"
PROMPT="Pick up the block and place it in the bowl"
EXP_NAME="so101_pick_place"
STEPS=5000  # 5k steps typically sufficient for Pi0 finetuning
BATCH_SIZE=16
SAVE_INTERVAL=1000

# Parse arguments
# Use our patched pi0_so101 config (6 action dims, delta actions enabled)
CONFIG="pi0_so101"
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            STEPS=100
            EXP_NAME="so101_test"
            shift
            ;;
        --pi05)
            # Pi0.5 uses more memory, smaller batch
            CONFIG="pi0_so101"  # Same config, will use pi05 base weights
            BATCH_SIZE=8
            EXP_NAME="so101_pi05"
            shift
            ;;
        --libero)
            # Fallback to libero config if pi0_so101 not available
            CONFIG="pi0_libero"
            EXP_NAME="so101_libero"
            shift
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --exp-name)
            EXP_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Pi0 Training for SO-101"
echo "=============================================="
echo "Config:     $CONFIG"
echo "Dataset:    $DATASET"
echo "Steps:      $STEPS"
echo "Batch size: $BATCH_SIZE"
echo "Exp name:   $EXP_NAME"
echo "=============================================="

# Check if we're in openpi directory
if [[ ! -f "scripts/train.py" ]]; then
    echo "Error: Run this script from the openpi directory"
    echo "  cd /app/openpi && bash /app/lerobot-thesis/scripts/pi0/train_so101.sh"
    exit 1
fi

# Check if norm stats exist
ASSETS_DIR="assets/${CONFIG}"
if [[ ! -d "$ASSETS_DIR" ]]; then
    echo ""
    echo "Normalization stats not found for $CONFIG"
    echo "Computing stats first (this may take a few minutes)..."
    echo ""
    # Dataset and prompt are baked into the pi0_so101 config via patch
    uv run scripts/compute_norm_stats.py --config-name="$CONFIG"
fi

# Run training
echo ""
echo "Starting training..."
echo ""

# Dataset and prompt are baked into the pi0_so101 config via patch
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py "$CONFIG" \
    --exp-name="$EXP_NAME" \
    --num-train-steps="$STEPS" \
    --batch-size="$BATCH_SIZE" \
    --save-interval="$SAVE_INTERVAL" \
    --overwrite

echo ""
echo "=============================================="
echo "Training complete!"
echo "Checkpoint: checkpoints/${CONFIG}/${EXP_NAME}"
echo "=============================================="
