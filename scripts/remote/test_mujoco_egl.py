#!/usr/bin/env python
"""Test MuJoCo EGL rendering in Docker container."""
import os

print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL', 'not set')}")
print(f"PYOPENGL_PLATFORM={os.environ.get('PYOPENGL_PLATFORM', 'not set')}")

import mujoco
import numpy as np

# Create a simple model and render
xml = '''<mujoco>
  <worldbody>
    <light pos="0 0 1"/>
    <geom type="sphere" size="0.1"/>
  </worldbody>
</mujoco>'''

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, 480, 640)
renderer.update_scene(data)
img = renderer.render()

print(f"Rendered image shape: {img.shape}")
print(f"Image dtype: {img.dtype}")
print(f"Image range: [{img.min()}, {img.max()}]")
print("MuJoCo EGL rendering OK!")
