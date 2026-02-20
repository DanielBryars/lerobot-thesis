# Step 2: Check HF auth, convert dataset, download pretrained model, launch training
# All commands run inside the Python 3.10 venv on RunPod

source /root/octo_env/bin/activate
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Check HF auth (Python 3.10 safe - no backslash in f-string)
python3 -c "from huggingface_hub import HfApi; api = HfApi(); info = api.whoami(); print('Logged in as:', info['name'])" 2>&1; echo "HF_AUTH_CHECK_DONE"
