# Kill any old training process and relaunch
source /root/octo_env/bin/activate
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
cd /root/lerobot-thesis

# Kill any running training
pkill -f train_octo 2>/dev/null; echo "KILLED_OLD"
sleep 2

# Launch training - Variant A: overhead camera only, no proprio
export WANDB_SILENT=true
nohup python3 scripts/training/train_octo.py danbhf/sim_pick_place_2pos_220ep_v2 --no-wrist-cam --no-proprio --run-name octo_A_overhead_only --batch-size 64 --steps 50000 --save-freq 5000 --log-freq 100 > /root/training_octo_A.log 2>&1 &
echo "TRAINING_PID=$!"
echo "TRAINING_LAUNCHED"
sleep 5
ps aux | grep train_octo | grep -v grep | head -1; echo "PS_DONE"
tail -30 /root/training_octo_A.log 2>&1; echo "LOG_DONE"
