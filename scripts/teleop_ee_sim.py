#!/usr/bin/env python
"""
End-Effector Teleop: Leader arm -> FK -> End-effector pose -> IK -> MuJoCo Sim

This script demonstrates the FK/IK pipeline for end-effector control:
1. Read joint angles from leader arm
2. Convert to end-effector pose using Forward Kinematics
3. Display the end-effector pose (position + orientation)
4. Convert back to joint angles using Inverse Kinematics
5. Apply to MuJoCo simulation

This is a stepping stone to training ACT in end-effector action space.

Usage:
    python scripts/teleop_ee_sim.py                    # With leader arm
    python scripts/teleop_ee_sim.py --test             # Test mode (no arm)
    python scripts/teleop_ee_sim.py --test --keyboard  # Keyboard control
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import mujoco

# Add project paths
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(Path(__file__).parent))

# Import shared utilities
from utils.constants import SIM_ACTION_LOW, SIM_ACTION_HIGH
from utils.conversions import normalized_to_radians

# Import FK/IK from our test module
from test_fk_ik import MuJoCoFK, MuJoCoIK, ARM_JOINT_NAMES


def load_config():
    """Load config.json for COM port settings."""
    repo_root = Path(__file__).parent.parent
    config_paths = [
        repo_root / "configs" / "config.json",
        Path("config.json"),
    ]
    for path in config_paths:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


def create_leader_bus(port: str):
    """Create motor bus for leader arm with STS3250 motors."""
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    bus = FeetechMotorsBus(
        port=port,
        motors={
            "shoulder_pan": Motor(1, "sts3250", MotorNormMode.RANGE_M100_100),
            "shoulder_lift": Motor(2, "sts3250", MotorNormMode.RANGE_M100_100),
            "elbow_flex": Motor(3, "sts3250", MotorNormMode.RANGE_M100_100),
            "wrist_flex": Motor(4, "sts3250", MotorNormMode.RANGE_M100_100),
            "wrist_roll": Motor(5, "sts3250", MotorNormMode.RANGE_M100_100),
            "gripper": Motor(6, "sts3250", MotorNormMode.RANGE_0_100),
        },
    )
    return bus


def load_calibration(arm_id: str = "leader_so100", is_follower: bool = False):
    """Load calibration from JSON file."""
    import draccus
    from lerobot.motors import MotorCalibration
    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION

    if is_follower:
        calib_path = HF_LEROBOT_CALIBRATION / "robots" / "so100_follower_sts3250" / f"{arm_id}.json"
    else:
        calib_path = HF_LEROBOT_CALIBRATION / "teleoperators" / "so100_leader_sts3250" / f"{arm_id}.json"

    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    with open(calib_path) as f, draccus.config_type("json"):
        return draccus.load(dict[str, MotorCalibration], f)


def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to euler angles (roll, pitch, yaw) in degrees."""
    # Extract euler angles from rotation matrix (ZYX convention)
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0

    return np.degrees(np.array([roll, pitch, yaw]))


class MuJoCoViewer:
    """Simple MuJoCo viewer using passive viewer."""

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.viewer = None

    def launch(self):
        """Launch the passive viewer."""
        try:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            return True
        except Exception as e:
            print(f"Warning: Could not launch viewer: {e}")
            return False

    def sync(self):
        """Sync the viewer with current simulation state."""
        if self.viewer is not None:
            self.viewer.sync()

    def is_running(self):
        """Check if viewer is still running."""
        if self.viewer is None:
            return True  # No viewer, continue running
        return self.viewer.is_running()

    def close(self):
        """Close the viewer."""
        if self.viewer is not None:
            self.viewer.close()


def run_teleop_ee(port: str, fps: int = 30, is_follower: bool = False):
    """Run end-effector teleoperation with arm controlling sim via FK/IK."""

    arm_type = "follower" if is_follower else "leader"
    arm_id = "follower_so100" if is_follower else "leader_so100"

    print(f"Connecting to {arm_type} arm on {port}...")
    bus = create_leader_bus(port)  # Same motor config for both
    bus.connect()

    print("Loading calibration...")
    bus.calibration = load_calibration(arm_id, is_follower=is_follower)
    bus.disable_torque()
    print(f"{arm_type.capitalize()} arm connected!")

    # Initialize FK/IK (separate instance for calculations)
    print("Initializing FK/IK...")
    scene_xml = str(Path(__file__).parent.parent / "scenes" / "so101_with_wrist_cam.xml")
    fk = MuJoCoFK(scene_xml)
    ik = MuJoCoIK(fk)

    # Create SEPARATE model/data for simulation (FK has its own for calculations)
    mj_model = mujoco.MjModel.from_xml_path(scene_xml)
    mj_data = mujoco.MjData(mj_model)

    # Initialize simulation
    for _ in range(100):
        mujoco.mj_step(mj_model, mj_data)

    # Launch viewer
    print("Launching viewer...")
    viewer = MuJoCoViewer(mj_model, mj_data)
    viewer.launch()

    print("\n" + "="*60)
    print("End-Effector Teleop Started!")
    print("="*60)
    print("\nPipeline: Leader joints -> FK -> EE pose -> IK -> Sim joints")
    print("\nMove the leader arm to control the simulation.")
    print("Press Ctrl+C to exit")
    print("="*60 + "\n")

    frame_time = 1.0 / fps
    step_count = 0
    n_sim_steps = 10
    last_normalized = np.zeros(6, dtype=np.float32)
    consecutive_errors = 0
    max_consecutive_errors = 30

    # Statistics
    ik_successes = 0
    ik_attempts = 0

    try:
        while viewer.is_running():
            loop_start = time.time()

            # Read leader arm
            try:
                positions = bus.sync_read("Present_Position")
                normalized = np.array([
                    positions["shoulder_pan"],
                    positions["shoulder_lift"],
                    positions["elbow_flex"],
                    positions["wrist_flex"],
                    positions["wrist_roll"],
                    positions["gripper"],
                ], dtype=np.float32)
                last_normalized = normalized.copy()
                consecutive_errors = 0
            except ConnectionError:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    print(f"\n Too many read errors, exiting.")
                    break
                normalized = last_normalized

            # Convert to radians
            joint_radians = normalized_to_radians(normalized)
            arm_joints = joint_radians[:5]
            gripper_joint = joint_radians[5]

            # Forward Kinematics: joints -> end-effector pose
            ee_pos, ee_rot = fk.forward(arm_joints)
            ee_euler = rotation_matrix_to_euler(ee_rot)

            # Inverse Kinematics: end-effector pose -> joints
            ik_attempts += 1
            ik_joints, ik_success, ik_error = ik.solve(
                target_pos=ee_pos,
                target_rot=ee_rot,
                initial_angles=arm_joints,  # Use current as initial guess
                max_iterations=100,
                pos_tolerance=1e-3,
            )

            if ik_success:
                ik_successes += 1
            else:
                # IK failed, fall back to direct joint control
                ik_joints = arm_joints

            # Combine IK result with gripper
            sim_joints = np.concatenate([ik_joints, [gripper_joint]])
            sim_joints = np.clip(sim_joints, SIM_ACTION_LOW, SIM_ACTION_HIGH)

            # Apply to simulation
            mj_data.ctrl[:] = sim_joints
            for _ in range(n_sim_steps):
                mujoco.mj_step(mj_model, mj_data)

            # Sync viewer
            viewer.sync()
            step_count += 1

            # Print status every 50 steps
            if step_count % 50 == 0:
                ik_rate = 100.0 * ik_successes / ik_attempts if ik_attempts > 0 else 0
                print(f"\nStep {step_count} | IK success: {ik_rate:.1f}%")
                print(f"  EE Position: [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}] m")
                print(f"  EE Euler:    [{ee_euler[0]:.1f}, {ee_euler[1]:.1f}, {ee_euler[2]:.1f}] deg")
                print(f"  Leader:      [{arm_joints[0]:.3f}, {arm_joints[1]:.3f}, {arm_joints[2]:.3f}, {arm_joints[3]:.3f}, {arm_joints[4]:.3f}] rad")
                print(f"  IK result:   [{ik_joints[0]:.3f}, {ik_joints[1]:.3f}, {ik_joints[2]:.3f}, {ik_joints[3]:.3f}, {ik_joints[4]:.3f}] rad")
                print(f"  Gripper:     {gripper_joint:.3f} rad")

            # Maintain frame rate
            elapsed = time.time() - loop_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        viewer.close()
        bus.disconnect()
        print(f"\nFinal IK success rate: {100.0 * ik_successes / ik_attempts:.1f}%")
        print("Done.")


def run_test_mode(fps: int = 30, keyboard: bool = False):
    """Test mode without physical arm - use preset poses or keyboard control."""

    print("Initializing FK/IK...")
    scene_xml = str(Path(__file__).parent.parent / "scenes" / "so101_with_wrist_cam.xml")
    fk = MuJoCoFK(scene_xml)
    ik = MuJoCoIK(fk)

    mj_model = fk.model
    mj_data = fk.data

    # Initialize simulation
    for _ in range(100):
        mujoco.mj_step(mj_model, mj_data)

    # Launch viewer
    print("Launching viewer...")
    viewer = MuJoCoViewer(mj_model, mj_data)
    viewer.launch()

    print("\n" + "="*60)
    print("End-Effector Teleop - TEST MODE")
    print("="*60)

    if keyboard:
        print("\nKeyboard Controls:")
        print("  W/S: Move EE forward/backward (X)")
        print("  A/D: Move EE left/right (Y)")
        print("  Q/E: Move EE up/down (Z)")
        print("  R:   Reset to home position")
        print("\nPress Ctrl+C to exit")
    else:
        print("\nCycling through preset end-effector positions...")
        print("Press Ctrl+C to exit")

    print("="*60 + "\n")

    # Home position (arm joints)
    home_joints = np.array([0.0, 0.5, -1.0, -0.5, 0.0])
    current_joints = home_joints.copy()

    # Get initial EE position
    ee_pos, ee_rot = fk.forward(current_joints)
    target_pos = ee_pos.copy()

    # Preset positions for cycling (non-keyboard mode)
    presets = [
        np.array([0.15, 0.0, 0.20]),   # Forward center
        np.array([0.15, 0.10, 0.20]),  # Forward left
        np.array([0.15, -0.10, 0.20]), # Forward right
        np.array([0.20, 0.0, 0.15]),   # Forward low
        np.array([0.10, 0.0, 0.25]),   # Back high
    ]
    preset_idx = 0
    last_preset_change = time.time()
    preset_interval = 3.0  # seconds

    frame_time = 1.0 / fps
    step_count = 0
    n_sim_steps = 10
    move_step = 0.005  # 5mm per key press

    try:
        while viewer.is_running():
            loop_start = time.time()

            if keyboard:
                # Keyboard control - check for key presses
                # Note: This requires a separate input method in practice
                # For now, we'll just hold position
                pass
            else:
                # Cycle through presets
                if time.time() - last_preset_change > preset_interval:
                    preset_idx = (preset_idx + 1) % len(presets)
                    target_pos = presets[preset_idx].copy()
                    last_preset_change = time.time()
                    print(f"\nMoving to preset {preset_idx}: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")

            # Inverse Kinematics
            ik_joints, ik_success, ik_error = ik.solve(
                target_pos=target_pos,
                target_rot=None,  # Don't constrain orientation
                initial_angles=current_joints,
                max_iterations=100,
                pos_tolerance=1e-3,
            )

            if ik_success:
                current_joints = ik_joints

            # Apply to simulation (with zero gripper)
            sim_joints = np.concatenate([current_joints, [0.5]])  # Half-open gripper
            sim_joints = np.clip(sim_joints, SIM_ACTION_LOW, SIM_ACTION_HIGH)

            mj_data.ctrl[:] = sim_joints
            for _ in range(n_sim_steps):
                mujoco.mj_step(mj_model, mj_data)

            # Get actual EE position after physics
            actual_pos, actual_rot = fk.forward(current_joints)

            viewer.sync()
            step_count += 1

            # Print status
            if step_count % 100 == 0:
                error = np.linalg.norm(actual_pos - target_pos)
                print(f"Step {step_count} | Target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}] | "
                      f"Actual: [{actual_pos[0]:.3f}, {actual_pos[1]:.3f}, {actual_pos[2]:.3f}] | "
                      f"Error: {error*1000:.1f}mm")

            elapsed = time.time() - loop_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        viewer.close()
        print("Done.")


def main():
    parser = argparse.ArgumentParser(description="End-Effector Teleop for SO101 sim using FK/IK")
    parser.add_argument("--port", "-p", type=str, default=None,
                        help="Serial port for arm (default: from config.json)")
    parser.add_argument("--fps", "-f", type=int, default=30,
                        help="Target frame rate (default: 30)")
    parser.add_argument("--test", "-t", action="store_true",
                        help="Test mode: no arm required, cycle through preset positions")
    parser.add_argument("--keyboard", "-k", action="store_true",
                        help="Enable keyboard control in test mode")
    parser.add_argument("--follower", action="store_true",
                        help="Use follower arm instead of leader arm")

    args = parser.parse_args()
    is_follower = args.follower

    if args.test:
        run_test_mode(args.fps, args.keyboard)
        return

    port = args.port
    if port is None:
        config = load_config()
        arm_key = "follower" if is_follower else "leader"
        if config and arm_key in config:
            port = config[arm_key]["port"]
            print(f"Using {arm_key} port from config: {port}")
        else:
            port = "COM8"
            print(f"Using default port: {port}")

    run_teleop_ee(port, args.fps, is_follower=is_follower)


if __name__ == "__main__":
    main()
