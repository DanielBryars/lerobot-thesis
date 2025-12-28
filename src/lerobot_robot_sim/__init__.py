"""
LeRobot plugin for SO100/SO101 Simulation with VR support.

This package registers a simulated robot that can be used as a drop-in
replacement for real hardware. Uses MuJoCo for physics and optionally
renders to VR headset.

Install with: pip install -e . --no-deps
"""

import logging
import time
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

logger = logging.getLogger(__name__)

# Motor names in order
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Sim action space bounds (radians)
SIM_ACTION_LOW = np.array([-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.17453])
SIM_ACTION_HIGH = np.array([1.91986, 1.74533, 1.69, 1.65806, 2.84121, 1.74533])

# Scene XML path (relative to this package in src/)
REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_SCENE_XML = REPO_ROOT / "scenes" / "so101_with_wrist_cam.xml"


def radians_to_normalized(radians: np.ndarray, use_degrees: bool = False) -> dict[str, float]:
    """Convert sim radians to lerobot normalized values."""
    result = {}
    for i, name in enumerate(MOTOR_NAMES):
        if use_degrees:
            result[name] = float(np.degrees(radians[i]))
        else:
            if name == "gripper":
                t = (radians[i] - SIM_ACTION_LOW[i]) / (SIM_ACTION_HIGH[i] - SIM_ACTION_LOW[i])
                result[name] = float(t * 100)
            else:
                t = (radians[i] - SIM_ACTION_LOW[i]) / (SIM_ACTION_HIGH[i] - SIM_ACTION_LOW[i])
                result[name] = float(t * 200 - 100)
    return result


def normalized_to_radians(normalized: dict[str, float], use_degrees: bool = False) -> np.ndarray:
    """Convert lerobot normalized values to sim radians."""
    radians = np.zeros(6, dtype=np.float32)
    for i, name in enumerate(MOTOR_NAMES):
        val = normalized.get(name, 0.0)
        if use_degrees:
            radians[i] = np.radians(val)
        else:
            if name == "gripper":
                t = val / 100.0
            else:
                t = (val + 100) / 200.0
            radians[i] = SIM_ACTION_LOW[i] + t * (SIM_ACTION_HIGH[i] - SIM_ACTION_LOW[i])
    return radians


@RobotConfig.register_subclass("so100_sim")
@dataclass
class SO100SimConfig(RobotConfig):
    """Configuration for the simulated SO100 follower."""

    # Path to MuJoCo scene XML
    scene_xml: str | None = None

    # Number of physics steps per action
    n_sim_steps: int = 10

    # Simulated camera names (must exist in MuJoCo model)
    # These are MuJoCo camera names, not real cameras
    sim_cameras: list[str] = field(default_factory=lambda: ["wrist_cam"])

    # Camera dimensions for MuJoCo rendering
    camera_width: int = 640
    camera_height: int = 480

    # Use degrees instead of normalized [-100, 100]
    use_degrees: bool = False

    # Enable VR output
    enable_vr: bool = False

    # Real cameras config (not used for sim, but needed for interface compat)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


class SO100Sim(Robot):
    """
    Simulated SO100 Follower using MuJoCo.

    Drop-in replacement for real SO100Follower that uses MuJoCo for physics.
    Optionally renders to VR headset.
    """

    config_class = SO100SimConfig
    name = "so100_sim"

    def __init__(self, config: SO100SimConfig):
        super().__init__(config)
        self.config = config

        # MuJoCo state
        self.mj_model = None
        self.mj_data = None
        self.mj_renderer = None
        self._connected = False

        # VR renderer
        self.vr_renderer = None

        # Scene XML
        if config.scene_xml:
            self.scene_xml = Path(config.scene_xml)
        else:
            self.scene_xml = DEFAULT_SCENE_XML

        # Camera IDs (MuJoCo camera name -> ID)
        self._camera_ids = {}

        # Cameras dict (lerobot expects this - we use sim camera names as keys)
        # Values are None since we render from MuJoCo, not real cameras
        self.cameras = {cam: None for cam in config.sim_cameras}

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in MOTOR_NAMES}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.camera_height, self.config.camera_width, 3)
            for cam in self.config.sim_cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True  # Sim doesn't need calibration

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        logger.info(f"Loading MuJoCo scene: {self.scene_xml}")
        if not self.scene_xml.exists():
            raise FileNotFoundError(f"Scene XML not found: {self.scene_xml}")

        self.mj_model = mujoco.MjModel.from_xml_path(str(self.scene_xml))
        self.mj_data = mujoco.MjData(self.mj_model)

        # Init VR FIRST if enabled (creates OpenGL context that MuJoCo will share)
        if self.config.enable_vr:
            self._init_vr()

        # Create MuJoCo renderer AFTER VR (so it uses VR's OpenGL context)
        self.mj_renderer = mujoco.Renderer(
            self.mj_model,
            height=self.config.camera_height,
            width=self.config.camera_width
        )

        # Find cameras in model
        for cam_name in self.config.sim_cameras:
            try:
                cam_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                self._camera_ids[cam_name] = cam_id
                logger.info(f"Found camera '{cam_name}' (ID {cam_id})")
            except Exception:
                logger.warning(f"Camera '{cam_name}' not in model, using default view")
                self._camera_ids[cam_name] = None

        # Initialize sim
        for _ in range(100):
            mujoco.mj_step(self.mj_model, self.mj_data)

        self._connected = True
        logger.info(f"{self} connected (simulated).")

    def _init_vr(self):
        try:
            # Import VR renderer from scripts directory
            import sys
            scripts_path = REPO_ROOT / "scripts"
            sys.path.insert(0, str(scripts_path))
            from teleop_sim_vr import VRRenderer

            self.vr_renderer = VRRenderer(self.mj_model, self.mj_data)
            self.vr_renderer.init_all()
            logger.info("VR renderer initialized")
        except Exception as e:
            logger.warning(f"Failed to init VR: {e}")
            self.vr_renderer = None

    def calibrate(self) -> None:
        pass  # No calibration needed

    def configure(self) -> None:
        pass  # No configuration needed

    def reset_scene(self, randomize: bool = True, pos_range: float = 0.02, rot_range: float = np.pi) -> None:
        """Reset the simulation to initial state with optional randomization.

        Args:
            randomize: If True, randomize duplo position and orientation
            pos_range: Max random offset in meters (default ±2cm)
            rot_range: Max random rotation in radians (default ±180°)
        """
        if not self.is_connected:
            return

        mujoco.mj_resetData(self.mj_model, self.mj_data)

        if randomize:
            # Duplo free joint is at qpos[0:7]: pos(3) + quat(4)
            # Add random XY offset (keep Z the same)
            self.mj_data.qpos[0] += np.random.uniform(-pos_range, pos_range)  # X
            self.mj_data.qpos[1] += np.random.uniform(-pos_range, pos_range)  # Y

            # Random Z rotation (yaw)
            angle = np.random.uniform(-rot_range, rot_range)
            # Quaternion for Z rotation: [cos(θ/2), 0, 0, sin(θ/2)]
            self.mj_data.qpos[3] = np.cos(angle / 2)  # w
            self.mj_data.qpos[4] = 0  # x
            self.mj_data.qpos[5] = 0  # y
            self.mj_data.qpos[6] = np.sin(angle / 2)  # z

            logger.info(f"Duplo randomized: pos=({self.mj_data.qpos[0]:.3f}, {self.mj_data.qpos[1]:.3f}), "
                       f"rot={np.degrees(angle):.1f}°")

        # Step a few times to settle
        for _ in range(50):
            mujoco.mj_step(self.mj_model, self.mj_data)

        logger.info("Scene reset")

    def set_duplo_position(self, x: float, y: float, z: float = None, quat: list = None) -> None:
        """Set the duplo block position and orientation.

        Args:
            x: X position in meters
            y: Y position in meters
            z: Z position in meters (optional, keeps current if None)
            quat: Quaternion [w, x, y, z] (optional, resets to upright if None)
        """
        if not self.is_connected:
            return

        # Duplo free joint is at qpos[0:7]: pos(3) + quat(4)
        self.mj_data.qpos[0] = x
        self.mj_data.qpos[1] = y
        if z is not None:
            self.mj_data.qpos[2] = z

        # Set quaternion (or reset to upright)
        if quat is not None and len(quat) == 4:
            self.mj_data.qpos[3] = quat[0]  # w
            self.mj_data.qpos[4] = quat[1]  # x
            self.mj_data.qpos[5] = quat[2]  # y
            self.mj_data.qpos[6] = quat[3]  # z
        else:
            # Default to upright
            self.mj_data.qpos[3] = 1.0  # w
            self.mj_data.qpos[4] = 0.0  # x
            self.mj_data.qpos[5] = 0.0  # y
            self.mj_data.qpos[6] = 0.0  # z

        # Step to settle
        for _ in range(10):
            mujoco.mj_step(self.mj_model, self.mj_data)

        logger.info(f"Duplo set to pos=({x:.3f}, {y:.3f}), quat={quat if quat else 'upright'}")

    def get_scene_info(self) -> dict:
        """Get initial positions and orientations of scene objects for metadata."""
        if not self.is_connected:
            return {}

        scene_info = {
            "scene_xml": self.scene_xml.as_posix(),  # Use forward slashes
            "objects": {}
        }

        # Get positions of key bodies
        for body_name in ["duplo", "bowl"]:
            try:
                body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                pos = self.mj_data.xpos[body_id].tolist()
                obj_info = {
                    "position": {"x": pos[0], "y": pos[1], "z": pos[2]}
                }
                # For duplo, also save orientation (from qpos of free joint)
                if body_name == "duplo":
                    # Duplo free joint quat is at qpos[3:7]
                    quat = self.mj_data.qpos[3:7].tolist()
                    obj_info["quaternion"] = {"w": quat[0], "x": quat[1], "y": quat[2], "z": quat[3]}
                scene_info["objects"][body_name] = obj_info
            except Exception:
                pass

        return scene_info

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read joint positions
        joint_radians = self.mj_data.qpos[:6].copy()
        obs_dict = radians_to_normalized(joint_radians, self.config.use_degrees)
        obs_dict = {f"{motor}.pos": obs_dict[motor] for motor in MOTOR_NAMES}

        # Render cameras
        for cam_name in self.config.sim_cameras:
            cam_id = self._camera_ids.get(cam_name)
            if cam_id is not None:
                self.mj_renderer.update_scene(self.mj_data, camera=cam_id)
            else:
                self.mj_renderer.update_scene(self.mj_data)
            obs_dict[cam_name] = self.mj_renderer.render().copy()

        return obs_dict

    def is_task_complete(self) -> bool:
        """Check if the duplo block is inside the bowl.

        Returns True if the duplo center is within the bowl bounds and resting.
        """
        if not self.is_connected:
            return False

        try:
            # Get duplo body position using MuJoCo body xpos (world position)
            duplo_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "duplo")
            duplo_pos = self.mj_data.xpos[duplo_body_id].copy()

            # Bowl position (static body at 0.217, -0.225, 0)
            bowl_x, bowl_y = 0.217, -0.225
            bowl_half_size = 0.06  # 12cm x 12cm bowl

            # Check if duplo is within bowl XY bounds
            in_x = abs(duplo_pos[0] - bowl_x) < bowl_half_size
            in_y = abs(duplo_pos[1] - bowl_y) < bowl_half_size

            # Check if duplo is at bowl height (resting inside, not above)
            # Bowl base is at z=0.002, duplo half-height is 0.0096
            # So duplo should be around z = 0.002 + 0.0096 = ~0.012 when resting in bowl
            # Use 5cm threshold to account for settling
            in_z = duplo_pos[2] < 0.05  # Below 5cm means it's resting, not held up

            result = in_x and in_y and in_z

            # Debug logging every 100 calls
            if not hasattr(self, '_task_check_count'):
                self._task_check_count = 0
            self._task_check_count += 1
            if self._task_check_count % 100 == 0:
                logger.info(f"Task check: duplo=({duplo_pos[0]:.3f}, {duplo_pos[1]:.3f}, {duplo_pos[2]:.3f}) "
                           f"in_x={in_x} in_y={in_y} in_z={in_z} → {result}")

            return result
        except Exception as e:
            logger.warning(f"Task completion check failed: {e}")
            return False

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Extract goal positions
        goal_pos = {}
        for key, val in action.items():
            if key.endswith(".pos"):
                motor = key.removesuffix(".pos")
                goal_pos[motor] = val

        # Convert and apply
        joint_radians = normalized_to_radians(goal_pos, self.config.use_degrees)
        joint_radians = np.clip(joint_radians, SIM_ACTION_LOW, SIM_ACTION_HIGH)
        self.mj_data.ctrl[:6] = joint_radians

        # Debug logging (every 100th call)
        if not hasattr(self, '_action_count'):
            self._action_count = 0
        self._action_count += 1
        if self._action_count % 100 == 1:
            actual_qpos = self.mj_data.qpos[:6]
            norm_vals = [goal_pos.get(m, 0.0) for m in MOTOR_NAMES]
            logger.info(f"Action #{self._action_count}:")
            logger.info(f"  Normalized: [{norm_vals[0]:6.1f}, {norm_vals[1]:6.1f}, {norm_vals[2]:6.1f}, {norm_vals[3]:6.1f}, {norm_vals[4]:6.1f}, {norm_vals[5]:5.1f}]")
            logger.info(f"  Target rad: [{joint_radians[0]:6.3f}, {joint_radians[1]:6.3f}, {joint_radians[2]:6.3f}, {joint_radians[3]:6.3f}, {joint_radians[4]:6.3f}, {joint_radians[5]:6.3f}]")
            logger.info(f"  Actual pos: [{actual_qpos[0]:6.3f}, {actual_qpos[1]:6.3f}, {actual_qpos[2]:6.3f}, {actual_qpos[3]:6.3f}, {actual_qpos[4]:6.3f}, {actual_qpos[5]:6.3f}]")

        # Step physics
        for _ in range(self.config.n_sim_steps):
            mujoco.mj_step(self.mj_model, self.mj_data)

        # Update VR
        if self.vr_renderer is not None:
            if not self.vr_renderer.render_frame():
                logger.warning("VR session ended")
                self.vr_renderer = None

        return {f"{motor}.pos": goal_pos.get(motor, 0.0) for motor in MOTOR_NAMES}

    def render_vr(self) -> bool:
        """Render a VR frame without changing robot state. Returns False if VR ended."""
        if self.vr_renderer is not None:
            if not self.vr_renderer.render_frame():
                logger.warning("VR session ended")
                self.vr_renderer = None
                return False
        return True

    def render(self) -> bool:
        """Render to screen using MuJoCo passive viewer. Returns False if window closed."""
        if not hasattr(self, '_viewer') or self._viewer is None:
            # Lazy init viewer
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
            logger.info("Launched MuJoCo viewer")

        if self._viewer.is_running():
            self._viewer.sync()
            return True
        else:
            self._viewer = None
            return False

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.vr_renderer:
            self.vr_renderer.cleanup()
            self.vr_renderer = None

        if hasattr(self, '_viewer') and self._viewer is not None:
            self._viewer.close()
            self._viewer = None

        del self.mj_renderer, self.mj_data, self.mj_model
        self.mj_renderer = self.mj_data = self.mj_model = None

        self._connected = False
        logger.info(f"{self} disconnected.")


# Also register as so101_sim for convenience
@RobotConfig.register_subclass("so101_sim")
@dataclass
class SO101SimConfig(SO100SimConfig):
    pass


class SO101Sim(SO100Sim):
    config_class = SO101SimConfig
    name = "so101_sim"


print("Registered: so100_sim, so101_sim")
