# Launch Octo training on RunPod - using ORIGINAL dataset (delta computed on-the-fly)
source /root/octo_env/bin/activate
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
cd /root/lerobot-thesis

# Quick verify the updated dataset module
python3 -c "from utils.octo_dataset import OctoDataset; print('OctoDataset import OK')" 2>&1; echo "IMPORT_CHECK"

# Configure wandb
export WANDB_SILENT=true

# Launch training - Variant A: overhead camera only, no proprio
# Uses ORIGINAL dataset (delta actions computed on-the-fly in OctoDataset)
nohup python3 scripts/training/train_octo.py danbhf/sim_pick_place_2pos_220ep_v2 --no-wrist-cam --no-proprio --run-name octo_A_overhead_only --batch-size 64 --steps 50000 --save-freq 5000 --log-freq 100 > /root/training_octo_A.log 2>&1 &
echo "TRAINING_PID=$!"
echo "TRAINING_LAUNCHED"
