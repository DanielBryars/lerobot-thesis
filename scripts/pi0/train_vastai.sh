#!/bin/bash
# LeRobot Pi0 Training on vast.ai
#
# Usage (on vast.ai instance):
#   # Set your HuggingFace token
#   huggingface-cli login
#
#   # Run training
#   DATASET=danbhf/sim_pick_place_157ep \
#   STEPS=5000 \
#   BATCH_SIZE=32 \
#   JOB_NAME=pi0_so101_5k \
#   REPO_ID=danbhf/pi0_so101_lerobot \
#   bash /app/train.sh

set -e

# Default values
DATASET=${DATASET:-"danbhf/sim_pick_place_157ep"}
STEPS=${STEPS:-5000}
BATCH_SIZE=${BATCH_SIZE:-32}
JOB_NAME=${JOB_NAME:-"pi0_training"}
REPO_ID=${REPO_ID:-""}
SAVE_FREQ=${SAVE_FREQ:-1000}

echo "================================================"
echo "LeRobot Pi0 Training"
echo "================================================"
echo "Dataset: $DATASET"
echo "Steps: $STEPS"
echo "Batch size: $BATCH_SIZE"
echo "Job name: $JOB_NAME"
echo "Save frequency: $SAVE_FREQ"
echo "Push to: $REPO_ID"
echo "================================================"

# Build command
CMD="lerobot-train \
    --dataset.repo_id=$DATASET \
    --policy.type=pi0 \
    --policy.pretrained_path=lerobot/pi0_base \
    --output_dir=/app/outputs/$JOB_NAME \
    --job_name=$JOB_NAME \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --steps=$STEPS \
    --batch_size=$BATCH_SIZE \
    --save_freq=$SAVE_FREQ \
    --wandb.enable=true"

if [ -n "$REPO_ID" ]; then
    CMD="$CMD --policy.repo_id=$REPO_ID"
fi

echo "Running: $CMD"
eval $CMD
