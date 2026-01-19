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
        preprocessor=None,
        postprocessor=None,
        whisker_color: tuple = (0.2, 0.8, 0.2, 0.3),  # RGBA - green, semi-transparent
        whisker_radius: float = 0.003,
        show_mean: bool = True,
        mean_color: tuple = (1.0, 0.3, 0.3, 0.8),  # Red, more opaque for mean
    ):
        self.policy = policy
        self.sim = sim
        self.device = device
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
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

        # Apply preprocessor (normalizes observations)
        if self.preprocessor is not None:
            batch = self.preprocessor(batch)

        with torch.no_grad():
            # Get action from policy
            action = self.policy.select_action(batch)

            # Apply postprocessor (denormalizes action)
            if self.postprocessor is not None:
                action = self.postprocessor(action)

            action_np = action.cpu().numpy().flatten()

            # Try to get full action chunk from queue
            actions = [action_np]

            if hasattr(self.policy, '_action_queue') and len(self.policy._action_queue) > 0:
                for a in self.policy._action_queue:
                    if isinstance(a, torch.Tensor):
                        a_processed = self.postprocessor(a) if self.postprocessor else a
                        actions.append(a_processed.cpu().numpy().flatten())
                    else:
                        actions.append(np.array(a).flatten())

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

    def forward_simulate_chunk(self, actions: np.ndarray, max_steps: int = 20) -> np.ndarray:
        """Forward simulate action chunk and return EE positions."""
        # Copy current sim state to rollout data
        self.data_rollout.qpos[:] = self.sim.mj_data.qpos[:]
        self.data_rollout.qvel[:] = self.sim.mj_data.qvel[:]
        self.data_rollout.ctrl[:] = self.sim.mj_data.ctrl[:]
        mujoco.mj_forward(self.sim.mj_model, self.data_rollout)

        # Record starting position
        positions = [self.data_rollout.site_xpos[self.ee_site_id].copy()]

        # Step through each action in the chunk (limited to max_steps for performance)
        for action in actions[:max_steps]:
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
            mujoco.mjv_connector(
                g,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                radius,
                pts[i].astype(np.float64),
                pts[i+1].astype(np.float64),
            )
            scene.ngeom += 1


def load_act_policy(checkpoint_path: str, device: str):
    """Load ACT policy and processors from checkpoint."""
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = ACTPolicy.from_pretrained(checkpoint_path)
    policy.to(device)
    policy.eval()

    # Load preprocessor/postprocessor for normalization
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=checkpoint_path
    )

    return policy, preprocessor, postprocessor


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
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Allow CPU (will be slow)")
    args = parser.parse_args()

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        if args.allow_cpu:
            print("WARNING: CUDA not available, using CPU (will be slow)")
            device = "cpu"
        else:
            raise RuntimeError(
                "CUDA is not available. PyTorch may be CPU-only.\n"
                "Fix with: pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall\n"
                "Or use --allow-cpu to run on CPU (will be very slow)"
            )
    else:
        device = args.device
    print(f"Using device: {device}")

    # Load policy and processors
    print(f"Loading policy from {args.checkpoint}...")
    policy, preprocessor, postprocessor = load_act_policy(args.checkpoint, device)
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
    visualizer = WhiskerVisualizer(policy, sim, device=device,
                                   preprocessor=preprocessor, postprocessor=postprocessor)

    # Create MuJoCo viewer with custom render callback
    print("Starting visualization...")
    print("Controls:")
    print("  Mouse: Click and drag to rotate, scroll to zoom")
    print("  SPACE: Pause/unpause")
    print("  RIGHT ARROW: Step forward (when paused)")
    print("  LEFT ARROW: Step back through history (when paused)")
    print("  R: Reset episode")
    print("  Q/ESC: Quit")

    # State for interactive control
    paused = False
    step_forward = False
    history_index = -1  # -1 means "live", >= 0 means viewing history
    whisker_history = []  # List of (whisker_points, ee_pos, obs) tuples
    max_history = 500

    # Keyboard callback
    def key_callback(key):
        nonlocal paused, step_forward, history_index
        if key == 32:  # SPACE
            paused = not paused
            if paused:
                print("\n  [PAUSED] - RIGHT to step, LEFT for history, SPACE to resume")
                history_index = -1  # Reset to live view
            else:
                print("\n  [RESUMED]")
                history_index = -1
        elif key == 262 and paused:  # RIGHT ARROW
            step_forward = True
            history_index = -1  # Back to live
        elif key == 263 and paused:  # LEFT ARROW
            if len(whisker_history) > 0:
                if history_index == -1:
                    history_index = len(whisker_history) - 1
                else:
                    history_index = max(0, history_index - 1)
                print(f"\r  [HISTORY {history_index + 1}/{len(whisker_history)}]", end="", flush=True)

    with mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data, key_callback=key_callback) as viewer:
        for ep in range(args.episodes):
            print(f"\nEpisode {ep + 1}/{args.episodes}")
            sim.reset_scene(randomize=True, pos_range=0.04, rot_range=np.pi)
            policy.reset()
            whisker_history.clear()  # Clear history for new episode
            history_index = -1

            for step in range(args.max_steps):
                if not viewer.is_running():
                    print("Viewer closed")
                    sim.disconnect()
                    return

                # Handle pause state
                while paused and not step_forward and viewer.is_running():
                    # Show history if navigating
                    if history_index >= 0 and history_index < len(whisker_history):
                        hist_whiskers, hist_ee, _ = whisker_history[history_index]
                        with viewer.lock():
                            viewer.user_scn.ngeom = 0
                            visualizer.whisker_points = hist_whiskers
                            visualizer.add_whiskers_to_scene(viewer.user_scn)
                    viewer.sync()
                    time.sleep(0.05)

                if step_forward:
                    step_forward = False

                # Get observation
                obs = sim.get_observation()

                # Update whiskers every frame when paused stepping, otherwise every 5
                update_interval = 1 if paused else 5
                if step % update_interval == 0:
                    visualizer.update_whiskers(obs)

                    # Record to history
                    if visualizer.whisker_points is not None:
                        whisker_history.append((
                            visualizer.whisker_points.copy(),
                            visualizer.current_ee_pos.copy() if visualizer.current_ee_pos is not None else None,
                            {k: v.copy() if hasattr(v, 'copy') else v for k, v in obs.items()}
                        ))
                        # Trim history if too long
                        if len(whisker_history) > max_history:
                            whisker_history.pop(0)

                # Get action and step simulation
                batch = visualizer._prepare_obs(obs)
                if preprocessor is not None:
                    batch = preprocessor(batch)
                with torch.no_grad():
                    action = policy.select_action(batch)
                if postprocessor is not None:
                    action = postprocessor(action)
                action = action.cpu().numpy().flatten()

                # Apply action
                action_dict = {m + ".pos": float(action[i]) for i, m in enumerate(MOTOR_NAMES)}
                sim.send_action(action_dict)

                # Add whiskers to scene and sync viewer
                with viewer.lock():
                    # Clear previous custom geometry
                    viewer.user_scn.ngeom = 0
                    # Add new whiskers
                    visualizer.add_whiskers_to_scene(viewer.user_scn)

                viewer.sync()

                # Check task completion
                if sim.is_task_complete():
                    print(f"  SUCCESS at step {step + 1}")
                    paused = True
                    print("  [PAUSED] - Use arrows to review, SPACE to continue")
                    while paused and viewer.is_running():
                        if history_index >= 0 and history_index < len(whisker_history):
                            hist_whiskers, _, _ = whisker_history[history_index]
                            with viewer.lock():
                                viewer.user_scn.ngeom = 0
                                visualizer.whisker_points = hist_whiskers
                                visualizer.add_whiskers_to_scene(viewer.user_scn)
                        viewer.sync()
                        time.sleep(0.05)
                    break

                # Small delay for visualization (reduce if too slow)
                time.sleep(0.01)
            else:
                print(f"  TIMEOUT after {args.max_steps} steps")

            # Pause at end of episode
            print("    Pausing 2 seconds...")
            pause_start = time.time()
            while viewer.is_running() and (time.time() - pause_start) < 2:
                viewer.sync()
                time.sleep(0.05)

        # Final pause before exit - stay open until user closes window
        print("\nAll episodes complete. Close viewer window to exit...")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.1)

    sim.disconnect()
    print("Done")


if __name__ == "__main__":
    main()
