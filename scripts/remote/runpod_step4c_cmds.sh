# Step 4c: Patch text_processing.py with sed and test
source /root/octo_env/bin/activate
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Patch text_processing.py - replace FlaxAutoModel import with try/except
sed -i 's/from transformers import AutoTokenizer, FlaxAutoModel  # lazy import/from transformers import AutoTokenizer  # lazy import/' /root/octo-pytorch/octo/data/utils/text_processing.py
echo "SED_PATCH_DONE"

# Verify the patch
grep -n "AutoTokenizer\|FlaxAutoModel" /root/octo-pytorch/octo/data/utils/text_processing.py
echo "GREP_DONE"

# Test loading Octo config
python3 -c "from octo.model.octo_model_pt import OctoModelPt; print('Loading...'); meta = OctoModelPt.load_config_and_meta_from_jax('hf://rail-berkeley/octo-small-1.5'); print('Octo-Small config loaded OK')" 2>&1; echo "OCTO_LOAD_DONE"
