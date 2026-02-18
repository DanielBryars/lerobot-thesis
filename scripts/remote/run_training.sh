#!/bin/bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

cd /root/lerobot-thesis

# Add project to Python path
export PYTHONPATH="/root/lerobot-thesis:/root/lerobot-thesis/src:$PYTHONPATH"

echo "=== Starting training ==="
echo "Dataset: danbhf/sim_pick_place_220ep_pickup_tight"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python3 scripts/training/train_act_vit.py \
    danbhf/sim_pick_place_220ep_pickup_tight \
    --fix_state --chunk_size 50 \
    --output_dir outputs/train/act_vit_pickup_tight \
    --save_freq 5000 --steps 50000 \
    --eval_sweep --eval_episodes 10 \
    --eval_n_steps 5,10,20,40 \
    --batch_size 16

echo "=== Training complete ==="
