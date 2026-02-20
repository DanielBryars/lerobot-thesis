# Install cv2 if needed, then launch training
source /root/octo_env/bin/activate
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
cd /root/lerobot-thesis

# Install opencv if not present
pip install opencv-python-headless 2>&1 | tail -3; echo "CV2_INSTALL_DONE"

# Verify imports
python3 -c "import cv2; from utils.octo_dataset import OctoDataset; print('All imports OK, cv2 version:', cv2.__version__)" 2>&1; echo "IMPORT_CHECK"

# Launch training - Variant A: overhead camera only, no proprio
export WANDB_SILENT=true
nohup python3 scripts/training/train_octo.py danbhf/sim_pick_place_2pos_220ep_v2 --no-wrist-cam --no-proprio --run-name octo_A_overhead_only --batch-size 64 --steps 50000 --save-freq 5000 --log-freq 100 > /root/training_octo_A.log 2>&1 &
echo "TRAINING_PID=$!"
echo "TRAINING_LAUNCHED"
