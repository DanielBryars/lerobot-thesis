#!/usr/bin/env python3
"""
Visualize policy action predictions as "whiskers" in MuJoCo simulation.

Shows predicted future trajectories as faint lines emanating from the robot,
allowing visual inspection of what the policy is predicting at each timestep.

Usage:
    python scripts/tools/visualize_whiskers.py --checkpoint outputs/train/act_xxx/checkpoint_045000
    python scripts/tools/visualize_whiskers.py --model danbhf/act_so101_157ep/checkpoint_045000
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer
import torch

# Add project paths
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot_robot_sim import SO100Sim, SO100SimConfig

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class WhiskerVisualizer:
    """Visualizes action chunk predictions as whiskers in MuJoCo."""

    def __init__(
        self,
        policy,
        sim: SO100Sim,
        device: str = "cuda",
        whisker_color: tuple = (0.2, 0.8, 0.2, 0.3),  # RGBA - green, semi-transparent
        whisker_radius: float = 0.003,
        show_mean: bool = True,
        mean_color: tuple = (1.0, 0.3, 0.3, 0.8),  # Red, more opaque for mean
    ):
        self.policy = policy
        self.sim = sim
        self.device = device
        self.whisker_color = whisker_color
        self.whisker_radius = whisker_radius
        self.show_mean = show_mean
        self.mean_color = mean_color

        # Get end-effector site ID
        self.ee_site_id = mujoco.mj_name2id(
            sim.mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripper_site"
        )
        if self.ee_site_id == -1:
            # Try alternate names
            for name in ["ee_site", "end_effector", "gripper"]:
                self.ee_site_id = mujoco.mj_name2id(
                    sim.mj_model, mujoco.mjtObj.mjOBJ_SITE, name
                )
                if self.ee_site_id != -1:
                    break

        # Create rollout data (copy of sim state for forward simulation)
        self.data_rollout = mujoco.MjData(sim.mj_model)

        # Storage for current whiskers
        self.whisker_points = None  # Shape: (horizon, 3)
        self.current_ee_pos = None

    def get_action_chunk(self, obs: dict) -> np.ndarray:
        """Get full action chunk from policy."""
        # Prepare observation for policy
        batch = self._prepare_obs(obs)

        with torch.no_grad():
            # For ACT, we want the full chunk, not just temporal ensembled action
            # Access the internal action queue or get raw output
            if hasattr(self.policy, '_action_queue') and len(self.policy._action_queue) > 0:
                # Already have queued actions
                actions = list(self.policy._action_queue)
            else:
                # Get fresh prediction - need to access internal method
                # ACT stores chunk_size actions
                action = self.policy.select_action(batch)

                # After select_action, the queue should be populated
                if hasattr(self.policy, '_action_queue'):
                    actions = list(self.policy._action_queue)
                    # Add the returned action at the start
                    actions = [action.cpu().numpy().flatten()] + [a.cpu().numpy().flatten() for a in actions]
                else:
                    # Fallback - just use single action repeated
                    actions = [action.cpu().numpy().flatten()] * 10

        return np.array(actions)

    def _prepare_obs(self, obs: dict) -> dict:
        """Convert sim observation to policy input format."""
        batch = {}

        # State
        state = np.array([obs[m + ".pos"] for m in MOTOR_NAMES], dtype=np.float32)
        batch["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(self.device)

        # Images
        for key, value in obs.items():
            if isinstance(value, np.ndarray) and value.ndim == 3:
                img = torch.from_numpy(value).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                batch[f"observation.images.{key}"] = img.to(self.device)

        return batch

    def forward_simulate_chunk(self, actions: np.ndarray) -> np.ndarray:
        """Forward simulate action chunk and return EE positions."""
        # Copy current sim state to rollout data
        self.data_rollout.qpos[:] = self.sim.mj_data.qpos[:]
        self.data_rollout.qvel[:] = self.sim.mj_data.qvel[:]
        self.data_rollout.ctrl[:] = self.sim.mj_data.ctrl[:]
        mujoco.mj_forward(self.sim.mj_model, self.data_rollout)

        # Record starting position
        positions = [self.data_rollout.site_xpos[self.ee_site_id].copy()]

        # Step through each action in the chunk
        for action in actions:
            # Set control (joint positions)
            for i, motor in enumerate(MOTOR_NAMES):
                actuator_id = mujoco.mj_name2id(
                    self.sim.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, motor
                )
                if actuator_id != -1 and i < len(action):
                    self.data_rollout.ctrl[actuator_id] = action[i]

            # Step simulation
            mujoco.mj_step(self.sim.mj_model, self.data_rollout)

            # Record EE position
            positions.append(self.data_rollout.site_xpos[self.ee_site_id].copy())

        return np.array(positions)

    def update_whiskers(self, obs: dict):
        """Update whisker visualization based on current observation."""
        # Get current EE position
        self.current_ee_pos = self.sim.mj_data.site_xpos[self.ee_site_id].copy()

        # Get action chunk from policy
        actions = self.get_action_chunk(obs)

        # Forward simulate to get predicted positions
        self.whisker_points = self.forward_simulate_chunk(actions)

    def add_whiskers_to_scene(self, scene: mujoco.MjvScene):
        """Add whisker geometry to the MuJoCo scene for rendering."""
        if self.whisker_points is None or len(self.whisker_points) < 2:
            return

        pts = self.whisker_points
        radius = self.whisker_radius

        # Add line segments as capsules
        for i in range(len(pts) - 1):
            if scene.ngeom >= scene.maxgeom:
                print(f"Warning: max geoms reached ({scene.maxgeom})")
                return

            # Calculate alpha fade based on distance into future
            alpha = self.whisker_color[3] * (1.0 - i / len(pts))
            color = np.array([
                self.whisker_color[0],
                self.whisker_color[1],
                self.whisker_color[2],
                alpha
            ], dtype=np.float64)

            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                g,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                np.eye(3, dtype=np.float64).flatten(),
                color,
            )
            mujoco.mjv_makeConnector(
                g,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                radius,
                pts[i][0], pts[i][1], pts[i][2],
                pts[i+1][0], pts[i+1][1], pts[i+1][2],
            )
            scene.ngeom += 1


def load_act_policy(checkpoint_path: str, device: str):
    """Load ACT policy from checkpoint."""
    from lerobot.policies.act.modeling_act import ACTPolicy

    policy = ACTPolicy.from_pretrained(checkpoint_path)
    policy.to(device)
    policy.eval()
    return policy


def main():
    parser = argparse.ArgumentParser(description="Visualize policy whiskers")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint (local or HuggingFace)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=300,
                        help="Max steps per episode")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load policy
    print(f"Loading policy from {args.checkpoint}...")
    policy = load_act_policy(args.checkpoint, device)
    print("Policy loaded")

    # Create simulation
    print("Creating simulation...")
    scene_path = REPO_ROOT / "scenes" / "so101_with_wrist_cam.xml"
    sim_config = SO100SimConfig(
        scene_xml=str(scene_path),
        sim_cameras=["overhead_cam", "wrist_cam"],
        camera_width=640,
        camera_height=480,
    )
    sim = SO100Sim(sim_config)
    sim.connect()

    # Create visualizer
    visualizer = WhiskerVisualizer(policy, sim, device=device)

    # Create MuJoCo viewer with custom render callback
    print("Starting visualization...")
    print("Controls: Click and drag to rotate, scroll to zoom, R to reset")

    with mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data) as viewer:
        for ep in range(args.episodes):
            print(f"\nEpisode {ep + 1}/{args.episodes}")
            sim.reset_scene(randomize=True, pos_range=0.04, rot_range=np.pi)
            policy.reset()

            for step in range(args.max_steps):
                if not viewer.is_running():
                    print("Viewer closed")
                    sim.disconnect()
                    return

                # Get observation
                obs = sim.get_observation()

                # Update whiskers based on current observation
                visualizer.update_whiskers(obs)

                # Get action and step simulation
                batch = visualizer._prepare_obs(obs)
                with torch.no_grad():
                    action = policy.select_action(batch)
                action = action.cpu().numpy().flatten()

                # Apply action
                action_dict = {m + ".pos": float(action[i]) for i, m in enumerate(MOTOR_NAMES)}
                sim.send_action(action_dict)

                # Add whiskers to scene and sync viewer
                with viewer.lock():
                    visualizer.add_whiskers_to_scene(viewer.user_scn)

                viewer.sync()

                # Check task completion
                if sim.is_task_complete():
                    print(f"  SUCCESS at step {step + 1}")
                    time.sleep(1)  # Pause to see success
                    break

                # Small delay for visualization
                time.sleep(0.02)
            else:
                print(f"  TIMEOUT after {args.max_steps} steps")

    sim.disconnect()
    print("Done")


if __name__ == "__main__":
    main()
