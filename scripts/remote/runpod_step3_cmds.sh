# Step 3: Convert dataset to delta actions and push to HF
source /root/octo_env/bin/activate
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
cd /root/lerobot-thesis

# Convert dataset to delta actions and push to HF
python3 scripts/tools/convert_to_delta_actions.py danbhf/sim_pick_place_2pos_220ep_v2 danbhf/sim_pick_place_2pos_220ep_v2_delta 2>&1; echo "DELTA_CONVERSION_DONE"
