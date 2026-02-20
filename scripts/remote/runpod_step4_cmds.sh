# Step 4: Download pretrained Octo-Small and launch training
source /root/octo_env/bin/activate
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
cd /root/lerobot-thesis

# Pull latest code (in case we need updates)
git pull 2>&1; echo "GIT_PULL_DONE"

# Download pretrained Octo-Small (JAX checkpoint)
python3 -c "from octo.model.octo_model_pt import OctoModelPt; print('Loading Octo-Small from JAX...'); meta = OctoModelPt.load_config_and_meta_from_jax('hf://rail-berkeley/octo-small-1.5'); print('Pretrained Octo-Small downloaded and cached')" 2>&1; echo "OCTO_DOWNLOAD_DONE"
