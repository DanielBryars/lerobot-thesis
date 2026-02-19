#!/bin/bash
echo "=== Setting up RunPod for Octo-Small fine-tuning ==="

# Check GPU
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Install EGL dev libraries (for MuJoCo headless rendering)
echo "Installing EGL libraries..."
apt-get update -qq && apt-get install -y -qq libegl1-mesa-dev libgles2-mesa-dev 2>/dev/null

# Reinstall PyOpenGL cleanly
echo "Reinstalling PyOpenGL..."
pip uninstall -y PyOpenGL PyOpenGL-accelerate 2>/dev/null
pip install -q PyOpenGL PyOpenGL-accelerate

# Set rendering backends
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
echo "export MUJOCO_GL=egl" >> /root/.bashrc
echo "export PYOPENGL_PLATFORM=egl" >> /root/.bashrc

# Clone repos if not present
if [ ! -d /root/lerobot-thesis ]; then
    echo "Cloning lerobot-thesis..."
    cd /root
    git clone https://github.com/danbhf/lerobot-thesis.git
fi

if [ ! -d /root/octo-pytorch ]; then
    echo "Cloning octo-pytorch..."
    cd /root
    git clone https://github.com/emb-ai/octo-pytorch.git
    cd octo-pytorch
    git checkout pytorch
fi

# Install octo-pytorch
echo "Installing octo-pytorch..."
cd /root/octo-pytorch
pip install -q -e . 2>&1 | tail -5

# Install JAX CPU-only (for loading pretrained JAX weights)
echo "Installing JAX (CPU-only, for weight conversion)..."
pip install -q jax jaxlib flax orbax-checkpoint 2>&1 | tail -3

# Install other dependencies
echo "Installing dependencies..."
pip install -q transformers sentencepiece accelerate 2>&1 | tail -3
pip install -q wandb 2>&1 | tail -3
pip install -q mujoco 2>&1 | tail -3

# Install lerobot-thesis sim module
echo "Installing lerobot_robot_sim..."
cd /root/lerobot-thesis
pip install -q -e src/lerobot_robot_sim 2>&1 | tail -3

# Install lerobot
echo "Installing lerobot..."
pip install -q -e "src/lerobot[dynamixel]" 2>&1 | tail -5

# Test MuJoCo rendering
echo "Testing MuJoCo with EGL..."
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python3 << 'PYEOF'
import mujoco
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos="0 0 3"/><geom type="sphere" size=".1"/></worldbody></mujoco>')
d = mujoco.MjData(m)
mujoco.mj_step(m, d)
print("MuJoCo physics OK")
r = mujoco.Renderer(m, 64, 64)
mujoco.mj_forward(m, d)
r.update_scene(d)
img = r.render()
print(f"MuJoCo rendering OK: {img.shape}")
PYEOF

# Test sim module
echo "Testing sim module..."
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python3 << 'PYEOF'
import sys
sys.path.insert(0, '/root/lerobot-thesis')
sys.path.insert(0, '/root/lerobot-thesis/src')
from lerobot_robot_sim import SO100SimConfig, SO100Sim
print("Sim module OK")
PYEOF

# Test octo-pytorch import
echo "Testing octo-pytorch..."
python3 << 'PYEOF'
from octo.model.octo_model_pt import OctoModelPt
from octo.utils.spec import ModuleSpec
from octo.model.components.action_heads_pt import L1ActionHeadPt
print("Octo-PyTorch OK")
PYEOF

# Download pretrained Octo-Small
echo "Downloading pretrained Octo-Small..."
python3 << 'PYEOF'
from octo.model.octo_model_pt import OctoModelPt
print("Loading config and meta from JAX checkpoint...")
meta = OctoModelPt.load_config_and_meta_from_jax("hf://rail-berkeley/octo-small-1.5")
print("Pretrained Octo-Small downloaded and cached")
PYEOF

# Download delta dataset
echo "Downloading delta dataset..."
python3 << 'PYEOF'
from huggingface_hub import hf_hub_download, list_repo_files
repo_id = "danbhf/sim_pick_place_2pos_220ep_v2_delta"
try:
    files = list_repo_files(repo_id, repo_type="dataset")
    parquet_files = [f for f in files if f.endswith('.parquet')]
    for pf in parquet_files:
        path = hf_hub_download(repo_id, pf, repo_type="dataset")
        print(f"  Downloaded: {pf}")
    print(f"Dataset cached ({len(parquet_files)} parquet files)")
except Exception as e:
    print(f"WARNING: Delta dataset not found ({e})")
    print("You may need to create it first with: python scripts/tools/convert_to_delta_actions.py")
PYEOF

echo ""
echo "=== Setup complete ==="
echo ""
echo "To start training:"
echo "  cd /root/lerobot-thesis"
echo "  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python scripts/training/train_octo.py \\"
echo "    danbhf/sim_pick_place_2pos_220ep_v2_delta \\"
echo "    --run-name octo_A_overhead_only --no-wrist-cam --no-proprio"
