#!/usr/bin/env python
"""View workspace bounds for pickup coordinate normalization."""
import mujoco
import mujoco.viewer
import tempfile
import shutil
from pathlib import Path

# Load scene and add bounds visualization
scene_path = Path('scenes/so101_with_confuser.xml')
xml = scene_path.read_text()
bounds = '''
    <!-- Workspace bounds: X: 0.10-0.38, Y: -0.28-0.27 -->
    <geom name="bounds_x_min" type="box" size="0.002 0.275 0.002" pos="0.10 -0.005 0.002" rgba="0 1 0 0.5" contype="0" conaffinity="0"/>
    <geom name="bounds_x_max" type="box" size="0.002 0.275 0.002" pos="0.38 -0.005 0.002" rgba="0 1 0 0.5" contype="0" conaffinity="0"/>
    <geom name="bounds_y_min" type="box" size="0.14 0.002 0.002" pos="0.24 -0.28 0.002" rgba="0 1 0 0.5" contype="0" conaffinity="0"/>
    <geom name="bounds_y_max" type="box" size="0.14 0.002 0.002" pos="0.24 0.27 0.002" rgba="0 1 0 0.5" contype="0" conaffinity="0"/>
    <!-- Overhead camera'''
xml = xml.replace('<!-- Overhead camera', bounds)

# Write to temp file in same directory (so relative paths work)
temp_path = scene_path.parent / '_temp_bounds_view.xml'
temp_path.write_text(xml)

try:
    model = mujoco.MjModel.from_xml_path(str(temp_path))
    data = mujoco.MjData(model)
    mujoco.viewer.launch(model, data)
finally:
    temp_path.unlink()
