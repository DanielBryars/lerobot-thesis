#!/usr/bin/env python
"""
Interactive tool to adjust overhead camera angle in MuJoCo scene.

Controls:
    Arrow keys: Move camera X/Y
    W/S: Move camera up/down (Z)
    Q/E: Rotate camera (yaw)
    A/D: Tilt camera (pitch)
    Z/X: Zoom (change FOV)
    1/2: Adjust near clipping plane
    3/4: Adjust far clipping plane
    R: Reset to default
    P: Print current camera parameters
    SPACE: Toggle between overhead and free camera view
    ESC: Exit

Usage:
    python adjust_overhead_camera.py
    python adjust_overhead_camera.py --scene scenes/my_scene.xml
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
import time

# Default camera parameters (from so101_with_wrist_cam.xml)
DEFAULT_POS = np.array([0.3, -0.09, 0.6])
DEFAULT_AXISANGLE = np.array([0, 0, 1, -1.5708])  # -90 deg around Z
DEFAULT_FOVY = 52


def axisangle_to_quat(axis, angle):
    """Convert axis-angle to quaternion [w, x, y, z]."""
    axis = np.array(axis) / np.linalg.norm(axis)
    half_angle = angle / 2
    w = np.cos(half_angle)
    xyz = axis * np.sin(half_angle)
    return np.array([w, xyz[0], xyz[1], xyz[2]])


def quat_to_euler(q):
    """Convert quaternion [w, x, y, z] to euler angles [roll, pitch, yaw]."""
    w, x, y, z = q
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])


def euler_to_quat(roll, pitch, yaw):
    """Convert euler angles to quaternion [w, x, y, z]."""
    cr, cp, cy = np.cos(roll/2), np.cos(pitch/2), np.cos(yaw/2)
    sr, sp, sy = np.sin(roll/2), np.sin(pitch/2), np.sin(yaw/2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z])


class CameraAdjuster:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        # Find overhead camera
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_cam")
        if self.cam_id < 0:
            raise ValueError("overhead_cam not found in scene")

        # Get initial camera state
        self.pos = model.cam_pos[self.cam_id].copy()
        self.quat = model.cam_quat[self.cam_id].copy()
        self.fovy = model.cam_fovy[self.cam_id]

        # Convert to euler for easier manipulation
        self.euler = quat_to_euler(self.quat)

        # Movement speeds
        self.pos_speed = 0.02
        self.rot_speed = 0.05
        self.fov_speed = 2.0

        print("\n=== Overhead Camera Adjuster ===")
        print("Controls:")
        print("  Arrow keys: Move X/Y")
        print("  W/S: Move up/down (Z)")
        print("  Q/E: Rotate (yaw)")
        print("  A/D: Tilt (pitch)")
        print("  Z/X: Zoom (FOV)")
        print("  1/2: Near clip plane")
        print("  3/4: Far clip plane")
        print("  R: Reset to default")
        print("  P: Print parameters")
        print("  ESC: Exit")
        print("================================\n")
        self.print_params()

    def print_params(self):
        """Print current camera parameters in XML format."""
        # Convert euler back to axis-angle for XML
        yaw = self.euler[2]
        print(f"\nCurrent camera parameters:")
        print(f"  pos: ({self.pos[0]:.3f}, {self.pos[1]:.3f}, {self.pos[2]:.3f})")
        print(f"  euler (RPY): ({np.degrees(self.euler[0]):.1f}°, {np.degrees(self.euler[1]):.1f}°, {np.degrees(self.euler[2]):.1f}°)")
        print(f"  fovy: {self.fovy:.1f}")
        print(f"\nXML format:")
        print(f'  <camera name="overhead_cam" pos="{self.pos[0]:.3f} {self.pos[1]:.3f} {self.pos[2]:.3f}" euler="{np.degrees(self.euler[0]):.1f} {np.degrees(self.euler[1]):.1f} {np.degrees(self.euler[2]):.1f}" fovy="{self.fovy:.0f}"/>')
        print()

    def reset(self):
        """Reset to default parameters."""
        self.pos = DEFAULT_POS.copy()
        self.quat = axisangle_to_quat(DEFAULT_AXISANGLE[:3], DEFAULT_AXISANGLE[3])
        self.euler = quat_to_euler(self.quat)
        self.fovy = DEFAULT_FOVY
        self.update_camera()
        print("Reset to default")
        self.print_params()

    def update_camera(self):
        """Apply current parameters to MuJoCo camera."""
        self.model.cam_pos[self.cam_id] = self.pos
        self.quat = euler_to_quat(*self.euler)
        self.model.cam_quat[self.cam_id] = self.quat
        self.model.cam_fovy[self.cam_id] = self.fovy

    def move(self, dx=0, dy=0, dz=0):
        self.pos += np.array([dx, dy, dz]) * self.pos_speed
        self.update_camera()

    def rotate(self, dyaw=0, dpitch=0):
        self.euler[2] += dyaw * self.rot_speed  # yaw
        self.euler[1] += dpitch * self.rot_speed  # pitch
        self.update_camera()

    def zoom(self, dfov):
        self.fovy = np.clip(self.fovy + dfov * self.fov_speed, 10, 120)
        self.update_camera()


def key_callback(keycode, adjuster, model):
    """Handle keyboard input."""
    # Arrow keys
    if keycode == 265:  # UP
        adjuster.move(dx=1)
    elif keycode == 264:  # DOWN
        adjuster.move(dx=-1)
    elif keycode == 263:  # LEFT
        adjuster.move(dy=1)
    elif keycode == 262:  # RIGHT
        adjuster.move(dy=-1)
    # W/S for Z
    elif keycode == ord('W'):
        adjuster.move(dz=1)
    elif keycode == ord('S'):
        adjuster.move(dz=-1)
    # Q/E for yaw
    elif keycode == ord('Q'):
        adjuster.rotate(dyaw=1)
    elif keycode == ord('E'):
        adjuster.rotate(dyaw=-1)
    # A/D for pitch
    elif keycode == ord('A'):
        adjuster.rotate(dpitch=1)
    elif keycode == ord('D'):
        adjuster.rotate(dpitch=-1)
    # Z/X for FOV
    elif keycode == ord('Z'):
        adjuster.zoom(1)
    elif keycode == ord('X'):
        adjuster.zoom(-1)
    # 1/2 for near clipping
    elif keycode == ord('1'):
        model.vis.map.znear = max(0.001, model.vis.map.znear * 0.8)
        print(f"Near clip: {model.vis.map.znear:.4f}")
    elif keycode == ord('2'):
        model.vis.map.znear = min(1.0, model.vis.map.znear * 1.25)
        print(f"Near clip: {model.vis.map.znear:.4f}")
    # 3/4 for far clipping
    elif keycode == ord('3'):
        model.vis.map.zfar = max(1.0, model.vis.map.zfar * 0.8)
        print(f"Far clip: {model.vis.map.zfar:.2f}")
    elif keycode == ord('4'):
        model.vis.map.zfar = min(100.0, model.vis.map.zfar * 1.25)
        print(f"Far clip: {model.vis.map.zfar:.2f}")
    # R to reset
    elif keycode == ord('R'):
        adjuster.reset()
    # P to print
    elif keycode == ord('P'):
        adjuster.print_params()


def main():
    parser = argparse.ArgumentParser(description="Adjust overhead camera interactively")
    parser.add_argument("--scene", type=str, default="scenes/so101_with_wrist_cam.xml",
                        help="Path to MuJoCo scene XML")
    args = parser.parse_args()

    # Find scene file
    scene_path = Path(args.scene)
    if not scene_path.is_absolute():
        # Try relative to script location
        repo_root = Path(__file__).parent.parent.parent
        scene_path = repo_root / args.scene

    if not scene_path.exists():
        print(f"Scene not found: {scene_path}")
        sys.exit(1)

    print(f"Loading scene: {scene_path}")

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    # Increase clipping distances to avoid losing geometry
    model.vis.map.znear = 0.01  # Near clip plane (1cm)
    model.vis.map.zfar = 10.0   # Far clip plane (10m)

    # Reset to initial state
    mujoco.mj_resetData(model, data)

    # Move duplo to a visible position
    data.qpos[0] = 0.27  # X
    data.qpos[1] = 0.12  # Y
    mujoco.mj_forward(model, data)

    # Create camera adjuster
    adjuster = CameraAdjuster(model, data)

    # Launch viewer
    with mujoco.viewer.launch_passive(model, data, key_callback=lambda k: key_callback(k, adjuster, model)) as viewer:
        # Set initial view to overhead camera
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = adjuster.cam_id

        print("Viewer launched. Press P to print current parameters.")
        print("Use arrow keys, WASD, QE, ZX to adjust camera.")

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
