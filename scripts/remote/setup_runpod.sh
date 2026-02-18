#!/bin/bash
echo "=== Setting up RunPod training environment ==="

# Install EGL dev libraries
echo "Installing EGL libraries..."
apt-get update -qq && apt-get install -y -qq libegl1-mesa-dev libgles2-mesa-dev 2>/dev/null

# Reinstall PyOpenGL cleanly
echo "Reinstalling PyOpenGL..."
pip uninstall -y PyOpenGL PyOpenGL-accelerate 2>/dev/null
pip install -q PyOpenGL PyOpenGL-accelerate

# Set rendering backends - EGL for GPU-accelerated headless
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
echo "export MUJOCO_GL=egl" >> /root/.bashrc
echo "export PYOPENGL_PLATFORM=egl" >> /root/.bashrc

# Test MuJoCo rendering with EGL
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

# Install sim module
echo "Installing lerobot_robot_sim..."
cd /root/lerobot-thesis
pip install -q -e src/lerobot_robot_sim 2>&1 | tail -3

# Test sim import
echo "Testing sim module..."
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python3 << 'PYEOF'
import sys
sys.path.insert(0, '/root/lerobot-thesis')
sys.path.insert(0, '/root/lerobot-thesis/src')
from lerobot_robot_sim import SO100SimConfig, SO100Sim
print("Sim module OK")
PYEOF

echo "=== Setup complete ==="
