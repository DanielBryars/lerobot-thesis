# Step 4b: Patch text_processing.py to skip FlaxAutoModel, then retry
source /root/octo_env/bin/activate
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
cd /root/lerobot-thesis

# Patch text_processing.py to handle missing FlaxAutoModel
python3 << 'PYEOF'
import os
fpath = "/root/octo-pytorch/octo/data/utils/text_processing.py"
with open(fpath, "r") as f:
    content = f.read()

# Replace the problematic import line
old = "from transformers import AutoTokenizer, FlaxAutoModel  # lazy import"
new = """from transformers import AutoTokenizer  # lazy import
        try:
            from transformers import FlaxAutoModel
        except ImportError:
            FlaxAutoModel = None  # Not available in PyTorch-only install"""

if old in content:
    content = content.replace(old, new)
    with open(fpath, "w") as f:
        f.write(content)
    print("Patched text_processing.py successfully")
else:
    print("Pattern not found - may already be patched")
    # Show the relevant lines
    for i, line in enumerate(content.split('\n')):
        if 'FlaxAutoModel' in line or 'AutoTokenizer' in line:
            print(f"  Line {i+1}: {line}")
PYEOF
echo "PATCH_DONE"

# Now retry downloading pretrained model
python3 -c "
from octo.model.octo_model_pt import OctoModelPt
print('Loading Octo-Small config from JAX...')
meta = OctoModelPt.load_config_and_meta_from_jax('hf://rail-berkeley/octo-small-1.5')
print('Config loaded successfully')
print('Config keys:', list(meta.keys()) if isinstance(meta, dict) else type(meta))
" 2>&1; echo "OCTO_CONFIG_DONE"
