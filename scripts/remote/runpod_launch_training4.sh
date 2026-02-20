# Install imageio+pyav and launch training
source /root/octo_env/bin/activate
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
cd /root/lerobot-thesis

# Install imageio with pyav backend (handles AV1 video)
pip install imageio av 2>&1 | tail -3; echo "IMAGEIO_INSTALL_DONE"

# Quick verify
python3 -c "import imageio.v3 as iio; import av; print('imageio+pyav OK')" 2>&1; echo "VERIFY_DONE"

# Launch training
export WANDB_SILENT=true
nohup python3 scripts/training/train_octo.py danbhf/sim_pick_place_2pos_220ep_v2 --no-wrist-cam --no-proprio --run-name octo_A_overhead_only --batch-size 64 --steps 50000 --save-freq 5000 --log-freq 100 > /root/training_octo_A.log 2>&1 &
echo "TRAINING_PID=$!"
echo "TRAINING_LAUNCHED"
