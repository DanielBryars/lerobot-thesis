# Launch Octo training on RunPod
source /root/octo_env/bin/activate
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
cd /root/lerobot-thesis

# Apply the text_processing patch
python3 scripts/remote/patch_text_processing.py 2>&1; echo "PATCH_DONE"

# Verify files exist
ls -la scripts/training/train_octo.py utils/octo_dataset.py scripts/inference/eval_octo.py 2>&1; echo "FILES_CHECK_DONE"

# Quick test: try importing our dataset
python3 -c "from utils.octo_dataset import OctoDataset; print('OctoDataset import OK')" 2>&1; echo "IMPORT_CHECK_DONE"

# Quick test: try loading Octo model
python3 -c "from octo.model.octo_model_pt import OctoModelPt; from octo.model.components.action_heads_pt import L1ActionHeadPt; from octo.utils.spec import ModuleSpec; print('All Octo imports OK')" 2>&1; echo "OCTO_IMPORT_DONE"

# Configure wandb (disable prompt)
export WANDB_SILENT=true

# Launch training - Variant A: overhead camera only, no proprio
nohup python3 scripts/training/train_octo.py danbhf/sim_pick_place_2pos_220ep_v2_delta --no-wrist-cam --no-proprio --run-name octo_A_overhead_only --batch-size 64 --steps 50000 --save-freq 5000 --log-freq 100 > /root/training_octo_A.log 2>&1 &
echo "TRAINING_PID=$!"
echo "TRAINING_LAUNCHED"
