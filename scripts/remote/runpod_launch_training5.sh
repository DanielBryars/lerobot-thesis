# Verify patched typing.py and launch training
source /root/octo_env/bin/activate
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
cd /root/lerobot-thesis

# Verify the typing.py patch works
python3 -c "from octo.utils.typing import Config; print('typing OK')" 2>&1; echo "TYPING_CHECK"

# Verify full Octo import
python3 -c "from octo.model.octo_model_pt import OctoModelPt; print('OctoModelPt OK')" 2>&1; echo "OCTO_IMPORT_CHECK"

# Launch training
export WANDB_SILENT=true
nohup python3 scripts/training/train_octo.py danbhf/sim_pick_place_2pos_220ep_v2 --no-wrist-cam --no-proprio --run-name octo_A_overhead_only --batch-size 64 --steps 50000 --save-freq 5000 --log-freq 100 > /root/training_octo_A.log 2>&1 &
echo "TRAINING_PID=$!"
echo "TRAINING_LAUNCHED"
