"""
Wrist camera preview with arm teleoperation.

View the wrist camera (as defined in XML) while moving the arm around.
Edit the XML directly to adjust camera position/orientation.

Usage:
    python adjust_wrist_camera_teleop.py

Controls:
    Arm: Joint sliders (J1-J6) or keyboard controls

    Keyboard:
        Q/A - Shoulder pan +/-
        W/S - Shoulder lift +/-
        E/D - Elbow flex +/-
        R/F - Wrist flex +/-
        T/G - Wrist roll +/-
        Y/H - Gripper open/close

        1-4 - Preset arm poses
        ESC - Quit
"""

import cv2
import numpy as np
import mujoco
import time
import json
from pathlib import Path


SCENE_XML = Path(__file__).parent.parent / "scenes" / "so101_with_wrist_cam.xml"
SETTINGS_FILE = Path(__file__).parent.parent / ".wrist_camera_settings.json"
AUTO_RELOAD_INTERVAL = 1.0  # seconds - check interval, only reloads if file changed

# Joint limits (radians)
JOINT_LIMITS = [
    (-1.92, 1.92),   # shoulder_pan
    (-1.75, 1.75),   # shoulder_lift
    (-1.69, 1.69),   # elbow_flex
    (-1.66, 1.66),   # wrist_flex
    (-2.74, 2.84),   # wrist_roll
    (-0.17, 1.75),   # gripper
]

JOINT_NAMES = ["Shoulder Pan", "Shoulder Lift", "Elbow Flex", "Wrist Flex", "Wrist Roll", "Gripper"]


class WristCameraPreview:
    def __init__(self):
        print(f"Loading {SCENE_XML}...")
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)

        self.wrist_cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam"
        )
        self.overhead_cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_cam"
        )
        print(f"Wrist camera ID: {self.wrist_cam_id}, Overhead camera ID: {self.overhead_cam_id}")

        # Track file modification time
        self.last_mtime = SCENE_XML.stat().st_mtime

        # External camera settings (defaults)
        self.ext_distance = 0.8
        self.ext_azimuth = 120
        self.ext_elevation = -25
        self.ext_lookat_x = 0.0
        self.ext_lookat_y = 0.0
        self.ext_lookat_z = 0.1

        # Joint positions (defaults)
        self.joint_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])

        # Load saved settings
        self._load_settings()
        self.joint_step = 0.05  # radians per keypress

        # Apply initial joint positions
        self._apply_joints()

        # Renderers (wrist and overhead cameras match real 1080p camera aspect ratio)
        self.external_renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.wrist_renderer = mujoco.Renderer(self.model, height=1080, width=1920)
        self.overhead_renderer = mujoco.Renderer(self.model, height=1080, width=1920)

        # Window (AUTOSIZE maintains aspect ratio)
        self.window_name = "Wrist Camera Preview"
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        self._create_trackbars()

    def _load_settings(self):
        """Load settings from file if it exists."""
        try:
            if SETTINGS_FILE.exists():
                with open(SETTINGS_FILE) as f:
                    settings = json.load(f)
                self.ext_distance = settings.get("ext_distance", self.ext_distance)
                self.ext_azimuth = settings.get("ext_azimuth", self.ext_azimuth)
                self.ext_elevation = settings.get("ext_elevation", self.ext_elevation)
                self.ext_lookat_x = settings.get("ext_lookat_x", self.ext_lookat_x)
                self.ext_lookat_y = settings.get("ext_lookat_y", self.ext_lookat_y)
                self.ext_lookat_z = settings.get("ext_lookat_z", self.ext_lookat_z)
                self.joint_pos = np.array(settings.get("joint_pos", self.joint_pos.tolist()))
                print("Loaded saved settings.")
        except Exception:
            pass

    def _save_settings(self):
        """Save current settings to file."""
        try:
            settings = {
                "ext_distance": self.ext_distance,
                "ext_azimuth": self.ext_azimuth,
                "ext_elevation": self.ext_elevation,
                "ext_lookat_x": self.ext_lookat_x,
                "ext_lookat_y": self.ext_lookat_y,
                "ext_lookat_z": self.ext_lookat_z,
                "joint_pos": self.joint_pos.tolist(),
            }
            with open(SETTINGS_FILE, "w") as f:
                json.dump(settings, f, indent=2)
        except Exception:
            pass

    def _apply_joints(self):
        """Apply current joint positions to simulation."""
        self.data.ctrl[:] = self.joint_pos
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

    def _create_trackbars(self):
        """Create trackbars for joint and external camera control."""
        def joint_to_slider(val, idx):
            lo, hi = JOINT_LIMITS[idx]
            return int((val - lo) / (hi - lo) * 1000)

        # Joint trackbars
        for i, name in enumerate(JOINT_NAMES):
            cv2.createTrackbar(f"J{i+1}", self.window_name, joint_to_slider(self.joint_pos[i], i), 1000, lambda x: None)

        # External camera trackbars
        cv2.createTrackbar("Dist", self.window_name, int(self.ext_distance * 100), 300, lambda x: None)  # 0-3m
        cv2.createTrackbar("Azi", self.window_name, int(self.ext_azimuth + 180), 360, lambda x: None)  # -180 to 180
        cv2.createTrackbar("Elev", self.window_name, int(self.ext_elevation + 90), 180, lambda x: None)  # -90 to 90
        cv2.createTrackbar("LookX", self.window_name, int((self.ext_lookat_x + 1) * 500), 1000, lambda x: None)  # -1 to 1
        cv2.createTrackbar("LookY", self.window_name, int((self.ext_lookat_y + 1) * 500), 1000, lambda x: None)  # -1 to 1
        cv2.createTrackbar("LookZ", self.window_name, int((self.ext_lookat_z + 1) * 500), 1000, lambda x: None)  # -1 to 1

    def _read_trackbars(self):
        """Read trackbar values and save if changed."""
        def slider_to_joint(val, idx):
            lo, hi = JOINT_LIMITS[idx]
            return lo + (val / 1000) * (hi - lo)

        # Store old values to detect changes
        old_values = (
            self.ext_distance, self.ext_azimuth, self.ext_elevation,
            self.ext_lookat_x, self.ext_lookat_y, self.ext_lookat_z,
            tuple(self.joint_pos)
        )

        # Joints
        for i in range(6):
            self.joint_pos[i] = slider_to_joint(cv2.getTrackbarPos(f"J{i+1}", self.window_name), i)

        # External camera
        self.ext_distance = cv2.getTrackbarPos("Dist", self.window_name) / 100.0
        self.ext_azimuth = cv2.getTrackbarPos("Azi", self.window_name) - 180
        self.ext_elevation = cv2.getTrackbarPos("Elev", self.window_name) - 90
        self.ext_lookat_x = cv2.getTrackbarPos("LookX", self.window_name) / 500.0 - 1.0
        self.ext_lookat_y = cv2.getTrackbarPos("LookY", self.window_name) / 500.0 - 1.0
        self.ext_lookat_z = cv2.getTrackbarPos("LookZ", self.window_name) / 500.0 - 1.0

        # Check if changed and save
        new_values = (
            self.ext_distance, self.ext_azimuth, self.ext_elevation,
            self.ext_lookat_x, self.ext_lookat_y, self.ext_lookat_z,
            tuple(self.joint_pos)
        )
        if new_values != old_values:
            self._save_settings()

    def _update_joint_sliders(self):
        """Update joint sliders to match current joint_pos (after keyboard input)."""
        for i in range(6):
            lo, hi = JOINT_LIMITS[i]
            slider_val = int((self.joint_pos[i] - lo) / (hi - lo) * 1000)
            cv2.setTrackbarPos(f"J{i+1}", self.window_name, slider_val)
        self._save_settings()

    def _adjust_joint(self, idx, delta):
        """Adjust a joint by delta, respecting limits."""
        lo, hi = JOINT_LIMITS[idx]
        self.joint_pos[idx] = np.clip(self.joint_pos[idx] + delta, lo, hi)
        self._update_joint_sliders()

    def render_wrist(self):
        """Render from wrist camera exactly as defined in XML."""
        self.wrist_renderer.update_scene(self.data, camera="wrist_cam")
        return self.wrist_renderer.render()

    def render_overhead(self):
        """Render from overhead camera exactly as defined in XML."""
        self.overhead_renderer.update_scene(self.data, camera="overhead_cam")
        return self.overhead_renderer.render()

    def render_external(self):
        """Render external view."""
        cam = mujoco.MjvCamera()
        cam.distance = self.ext_distance
        cam.azimuth = self.ext_azimuth
        cam.elevation = self.ext_elevation
        cam.lookat[0] = self.ext_lookat_x
        cam.lookat[1] = self.ext_lookat_y
        cam.lookat[2] = self.ext_lookat_z
        self.external_renderer.update_scene(self.data, camera=cam)
        return self.external_renderer.render()

    def reload_model_if_changed(self):
        """Reload model from XML if file has changed. Returns True if reloaded."""
        try:
            # Check if file modified
            current_mtime = SCENE_XML.stat().st_mtime
            if current_mtime == self.last_mtime:
                return False  # No change

            saved_joints = self.joint_pos.copy()

            # Try to reload model
            new_model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
            new_data = mujoco.MjData(new_model)

            # Re-find cameras
            new_wrist_cam_id = mujoco.mj_name2id(
                new_model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam"
            )
            new_overhead_cam_id = mujoco.mj_name2id(
                new_model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_cam"
            )

            # Close old renderers explicitly
            del self.external_renderer
            del self.wrist_renderer
            del self.overhead_renderer

            # Success - swap in new model
            self.model = new_model
            self.data = new_data
            self.wrist_cam_id = new_wrist_cam_id
            self.overhead_cam_id = new_overhead_cam_id
            self.last_mtime = current_mtime

            # Restore joint positions and step simulation
            self.joint_pos = saved_joints
            self.data.ctrl[:] = self.joint_pos
            for _ in range(50):  # More steps to stabilize
                mujoco.mj_step(self.model, self.data)

            # Create new renderers
            self.external_renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.wrist_renderer = mujoco.Renderer(self.model, height=1080, width=1920)
            self.overhead_renderer = mujoco.Renderer(self.model, height=1080, width=1920)

            print("XML reloaded!")
            return True

        except Exception as e:
            # XML parse error or other issue - keep using old model
            return False

    def run(self):
        print("\n" + "="*70)
        print("WRIST CAMERA PREVIEW + TELEOP")
        print("="*70)
        print("Camera renders exactly as defined in XML.")
        print("Edit scenes/so101_with_wrist_cam.xml - auto-reloads every 1s.")
        print("")
        print("Arm:    Sliders J1-J6 or keyboard:")
        print("        Q/A=J1  W/S=J2  E/D=J3  R/F=J4  T/G=J5  Y/H=J6")
        print("        1-4=Preset poses  ESC=Quit")
        print("="*70 + "\n")

        last_reload_time = time.time()

        presets = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
            [0.3, -0.5, 0.8, 0.5, 0.0, 0.5],
            [0.5, -0.7, 1.0, 0.8, 0.3, 0.2],
            [-0.5, -0.3, 0.5, 0.3, -0.5, 0.8],
        ]

        while True:
            # Check for XML changes every 1 second
            now = time.time()
            if now - last_reload_time >= AUTO_RELOAD_INTERVAL:
                self.reload_model_if_changed()
                last_reload_time = now

            self._read_trackbars()
            self._apply_joints()

            # Render all cameras
            external = cv2.cvtColor(self.render_external(), cv2.COLOR_RGB2BGR)
            wrist_full = cv2.cvtColor(self.render_wrist(), cv2.COLOR_RGB2BGR)
            overhead_full = cv2.cvtColor(self.render_overhead(), cv2.COLOR_RGB2BGR)

            # Scale cameras to fit display (keep 16:9 for wrist/overhead)
            wrist = cv2.resize(wrist_full, (640, 360))
            overhead = cv2.resize(overhead_full, (640, 360))

            # Labels
            cv2.putText(external, "External", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(wrist, "Wrist Cam", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(overhead, "Overhead Cam", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Joint info on external view
            joint_info = [f"J{i+1}: {np.degrees(self.joint_pos[i]):.1f}deg" for i in range(6)]
            for i, line in enumerate(joint_info):
                cv2.putText(external, line, (10, 60 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Layout: External on left, wrist+overhead stacked on right
            right_col = np.vstack([wrist, overhead])  # 640x720
            # Pad external to match height (480 -> 720)
            external_padded = np.zeros((720, 640, 3), dtype=np.uint8)
            external_padded[:480, :, :] = external

            display = np.hstack([external_padded, right_col])
            cv2.putText(display, "Auto-reloads XML | 1-4=Presets | Q/A W/S E/D R/F T/G Y/H=Joints | ESC=Quit",
                       (10, display.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # ESC
                break
            # Preset poses
            elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                self.joint_pos = np.array(presets[key - ord('1')])
                self._update_joint_sliders()
            # Joint controls
            elif key == ord('q'):
                self._adjust_joint(0, self.joint_step)
            elif key == ord('a'):
                self._adjust_joint(0, -self.joint_step)
            elif key == ord('w'):
                self._adjust_joint(1, self.joint_step)
            elif key == ord('s'):
                self._adjust_joint(1, -self.joint_step)
            elif key == ord('e'):
                self._adjust_joint(2, self.joint_step)
            elif key == ord('d'):
                self._adjust_joint(2, -self.joint_step)
            elif key == ord('r'):
                self._adjust_joint(3, self.joint_step)
            elif key == ord('f'):
                self._adjust_joint(3, -self.joint_step)
            elif key == ord('t'):
                self._adjust_joint(4, self.joint_step)
            elif key == ord('g'):
                self._adjust_joint(4, -self.joint_step)
            elif key == ord('y'):
                self._adjust_joint(5, self.joint_step)
            elif key == ord('h'):
                self._adjust_joint(5, -self.joint_step)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    WristCameraPreview().run()
