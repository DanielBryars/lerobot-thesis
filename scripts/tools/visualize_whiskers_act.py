#!/usr/bin/env python3
"""
Visualize ACT policy action predictions as "whiskers" in MuJoCo simulation.

Shows predicted future trajectories as faint lines emanating from the robot,
allowing visual inspection of what the policy is predicting at each timestep.

Usage:
    python scripts/tools/visualize_whiskers_act.py --checkpoint outputs/train/act_xxx/checkpoint_045000
    python scripts/tools/visualize_whiskers_act.py --model danbhf/act_so101_157ep/checkpoint_045000
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add project paths
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot_robot_sim import SO100Sim, SO100SimConfig

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class JointPlotter:
    """Real-time matplotlib plot showing predicted joint trajectories."""

    def __init__(self, chunk_size: int = 100):
        self.chunk_size = chunk_size

        # Set up interactive mode
        plt.ion()

        # Create figure with subplots for each joint
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title('Joint Predictions')
        gs = GridSpec(3, 2, figure=self.fig, hspace=0.3, wspace=0.25)

        self.axes = []
        self.lines = []
        self.current_step_lines = []  # Vertical lines showing current step

        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']

        for i, (motor, color) in enumerate(zip(MOTOR_NAMES, colors)):
            row, col = i // 2, i % 2
            ax = self.fig.add_subplot(gs[row, col])
            ax.set_title(motor, fontsize=10, fontweight='bold')
            ax.set_xlabel('Step', fontsize=8)
            ax.set_ylabel('Value', fontsize=8)
            ax.set_xlim(0, chunk_size)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

            # Create line for predictions
            line, = ax.plot([], [], color=color, linewidth=2, label='Predicted')
            # Create vertical line for current execution point
            vline = ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7, label='Executed')

            self.axes.append(ax)
            self.lines.append(line)
            self.current_step_lines.append(vline)

            # Set appropriate y-limits based on joint type
            if motor == 'gripper':
                ax.set_ylim(-5, 55)
            else:
                ax.set_ylim(-3.5, 3.5)

        self.fig.tight_layout()
        plt.show(block=False)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, action_chunk: np.ndarray, executed_steps: int = 0):
        """Update the plot with new action chunk predictions.

        Args:
            action_chunk: Shape (chunk_size, 6) - predicted actions
            executed_steps: How many steps have been executed (for vertical line)
        """
        if action_chunk is None or len(action_chunk) == 0:
            return

        x = np.arange(len(action_chunk))

        for i, (line, vline, ax) in enumerate(zip(self.lines, self.current_step_lines, self.axes)):
            if i < action_chunk.shape[1]:
                y = action_chunk[:, i]
                line.set_data(x, y)

                # Auto-adjust y-limits if needed (except gripper which is fixed)
                if MOTOR_NAMES[i] != 'gripper':
                    ymin, ymax = y.min(), y.max()
                    margin = (ymax - ymin) * 0.1 + 0.1
                    ax.set_ylim(ymin - margin, ymax + margin)

                # Update vertical line position
                vline.set_xdata([executed_steps, executed_steps])

        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        """Close the plot window."""
        plt.close(self.fig)


class WhiskerVisualizer:
    """Visualizes action chunk predictions as whiskers in MuJoCo."""

    def __init__(
        self,
        policy,
        sim: SO100Sim,
        device: str = "cuda",
        preprocessor=None,
        postprocessor=None,
        whisker_color: tuple = (0.2, 0.8, 0.2, 0.7),  # RGBA - green, semi-transparent
        ghost_color: tuple = (0.5, 0.5, 0.9, 0.3),  # RGBA - light blue
        actual_path_color: tuple = (1.0, 0.3, 0.1, 0.5),  # RGBA - orange for actual path taken
        whisker_radius: float = 0.003,
        ghost_radius: float = 0.002,
        max_ghost_trails: int = 12,  # Recent ghost trails
        ghost_trail_interval: int = 1,  # Every prediction
    ):
        self.policy = policy
        self.sim = sim
        self.device = device
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.whisker_color = whisker_color
        self.ghost_color = ghost_color
        self.actual_path_color = actual_path_color
        self.whisker_radius = whisker_radius
        self.ghost_radius = ghost_radius
        self.max_ghost_trails = max_ghost_trails
        self.ghost_trail_interval = ghost_trail_interval
        self._ghost_counter = 0  # Counter for spacing out ghost trails

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
        self.whisker_points = None  # Shape: (horizon, 3) - EE positions
        self.whisker_gripper = None  # Shape: (horizon,) - gripper values
        self.whisker_joints = None  # Dict of joint_name -> positions (unused, kept for compatibility)
        self.whisker_moving_jaw = None  # Shape: (horizon, 3) - moving jaw positions
        self.current_ee_pos = None
        self.current_action_chunk = None  # Shape: (chunk_size, 6) - for joint plotting

        # Storage for ghost trails (past predictions)
        self.ghost_trails = []  # List of past whisker data dicts

        # Storage for actual path taken
        self.actual_path = []  # List of actual EE positions

    def get_action_chunk(self, obs: dict) -> np.ndarray:
        """Get full action chunk from policy."""
        # Prepare observation for policy
        batch = self._prepare_obs(obs)

        # Apply preprocessor (normalizes observations)
        if self.preprocessor is not None:
            batch = self.preprocessor(batch)

        with torch.no_grad():
            # Get full action chunk directly from model (not through select_action queue)
            if hasattr(self.policy, 'predict_action_chunk'):
                # ACT policy - get full chunk prediction
                actions = self.policy.predict_action_chunk(batch)  # Shape: [1, chunk_size, action_dim]
                actions = actions.squeeze(0)  # Shape: [chunk_size, action_dim]

                # Apply postprocessor (denormalizes actions) to each action
                if self.postprocessor is not None:
                    # Postprocessor expects single actions, so process each
                    actions_list = []
                    for i in range(actions.shape[0]):
                        a = self.postprocessor(actions[i])
                        actions_list.append(a.cpu().numpy())
                    return np.array(actions_list)
                else:
                    return actions.cpu().numpy()
            else:
                # Fallback for other policies - just get single action
                action = self.policy.select_action(batch)
                if self.postprocessor is not None:
                    action = self.postprocessor(action)
                return action.cpu().numpy().reshape(1, -1)

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

    def forward_simulate_chunk(self, actions: np.ndarray, max_steps: int = None) -> dict:
        """Forward simulate action chunk and return positions and gripper state.

        Args:
            actions: Action chunk to simulate [chunk_size, 6]
            max_steps: Max steps to simulate. If None, simulates full chunk.

        Returns:
            dict with:
                'ee_positions': EE trajectory [N, 3]
                'gripper_values': Gripper action values [N]
                'joint_positions': Dict of joint_name -> positions [N, 3]
        """
        if max_steps is None:
            max_steps = len(actions)  # Use full chunk by default

        # Copy current sim state to rollout data
        self.data_rollout.qpos[:] = self.sim.mj_data.qpos[:]
        self.data_rollout.qvel[:] = self.sim.mj_data.qvel[:]
        self.data_rollout.ctrl[:] = self.sim.mj_data.ctrl[:]
        mujoco.mj_forward(self.sim.mj_model, self.data_rollout)

        # Find body IDs for joint visualization (use actual MuJoCo body names)
        # Note: shoulder body doesn't translate (only rotates in place), so skip it
        joint_body_ids = {}
        for name in ["upper_arm", "lower_arm", "wrist"]:
            body_id = mujoco.mj_name2id(self.sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id != -1:
                joint_body_ids[name] = body_id

        # Find moving jaw body for gripper visualization
        moving_jaw_id = mujoco.mj_name2id(self.sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")
        if moving_jaw_id == -1:
            moving_jaw_id = mujoco.mj_name2id(self.sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw")

        # Record starting positions
        ee_positions = [self.data_rollout.site_xpos[self.ee_site_id].copy()]
        gripper_values = [actions[0][5] if len(actions) > 0 and len(actions[0]) > 5 else 0]
        joint_positions = {name: [self.data_rollout.xpos[bid].copy()]
                          for name, bid in joint_body_ids.items()}
        # Track moving jaw position for gripper whisker
        moving_jaw_positions = []
        if moving_jaw_id != -1:
            moving_jaw_positions.append(self.data_rollout.xpos[moving_jaw_id].copy())

        # Step through each action in the chunk
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
            ee_positions.append(self.data_rollout.site_xpos[self.ee_site_id].copy())

            # Record gripper value (index 5 = gripper)
            gripper_values.append(action[5] if len(action) > 5 else 0)

            # Record joint body positions
            for name, bid in joint_body_ids.items():
                joint_positions[name].append(self.data_rollout.xpos[bid].copy())

            # Record moving jaw position
            if moving_jaw_id != -1:
                moving_jaw_positions.append(self.data_rollout.xpos[moving_jaw_id].copy())

        return {
            'ee_positions': np.array(ee_positions),
            'gripper_values': np.array(gripper_values),
            'joint_positions': {k: np.array(v) for k, v in joint_positions.items()},
            'moving_jaw_positions': np.array(moving_jaw_positions) if moving_jaw_positions else None,
        }

    def record_actual_position(self):
        """Record current EE position to actual path. Call every simulation step."""
        pos = self.sim.mj_data.site_xpos[self.ee_site_id].copy()
        self.actual_path.append(pos)
        # Keep actual path trimmed
        max_actual_path = 500  # More points for smoother path
        if len(self.actual_path) > max_actual_path:
            self.actual_path.pop(0)

    def update_whiskers(self, obs: dict):
        """Update whisker visualization based on current observation."""
        # Get current EE position
        self.current_ee_pos = self.sim.mj_data.site_xpos[self.ee_site_id].copy()

        # Save old whiskers as ghost trail (only every N predictions to avoid clutter)
        self._ghost_counter += 1
        if self.whisker_points is not None and len(self.whisker_points) > 1:
            if self._ghost_counter >= self.ghost_trail_interval:
                self._ghost_counter = 0
                self.ghost_trails.append({
                    'ee_positions': self.whisker_points.copy(),
                    'gripper_values': self.whisker_gripper.copy() if self.whisker_gripper is not None else None,
                    'joint_positions': {k: v.copy() for k, v in self.whisker_joints.items()} if self.whisker_joints else None,
                })
                # Keep only recent ghost trails
                if len(self.ghost_trails) > self.max_ghost_trails:
                    self.ghost_trails.pop(0)

        # Get action chunk from policy
        actions = self.get_action_chunk(obs)
        self.current_action_chunk = actions  # Store for joint plotting

        # Forward simulate to get predicted positions (full chunk)
        result = self.forward_simulate_chunk(actions)
        self.whisker_points = result['ee_positions']
        self.whisker_gripper = result['gripper_values']
        self.whisker_joints = result['joint_positions']
        self.whisker_moving_jaw = result.get('moving_jaw_positions')

    def clear_trails(self):
        """Clear ghost trails and actual path (call on episode reset)."""
        self.ghost_trails.clear()
        self.actual_path.clear()
        self.whisker_points = None

    def _add_line_segment(self, scene: mujoco.MjvScene, p1: np.ndarray, p2: np.ndarray,
                           color: np.ndarray, radius: float) -> bool:
        """Add a single line segment to scene. Returns False if scene is full."""
        if scene.ngeom >= scene.maxgeom:
            return False

        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64).flatten(),
            color.astype(np.float64),
        )
        mujoco.mjv_connector(
            g,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            radius,
            p1.astype(np.float64),
            p2.astype(np.float64),
        )
        scene.ngeom += 1
        return True

    def add_whiskers_to_scene(self, scene: mujoco.MjvScene, show_ghosts: bool = True,
                               show_actual_path: bool = True, actual_path_limit: int = None,
                               ghost_trails_limit: int = None):
        """Add whisker geometry to the MuJoCo scene for rendering.

        Args:
            scene: MuJoCo scene to add geometry to
            show_ghosts: Whether to show ghost trails of past predictions
            show_actual_path: Whether to show the actual path taken
            actual_path_limit: Only show first N points of actual path (for history viewing)
            ghost_trails_limit: Only show first N ghost trails (for history viewing)
        """
        # First render ghost trails (oldest first, so newer ones are on top)
        ghost_trails_to_show = self.ghost_trails[:ghost_trails_limit] if ghost_trails_limit else self.ghost_trails
        if show_ghosts and ghost_trails_to_show:
            for trail_idx, ghost_data in enumerate(ghost_trails_to_show):
                # Handle both old format (array) and new format (dict)
                if isinstance(ghost_data, dict):
                    ghost_pts = ghost_data.get('ee_positions')
                else:
                    ghost_pts = ghost_data  # Old format compatibility

                if ghost_pts is None or len(ghost_pts) < 2:
                    continue

                # Fade older ghosts very aggressively - only recent ones visible
                age_factor = (trail_idx + 1) / len(ghost_trails_to_show)  # 0->1 as newer
                # Very aggressive fade - older trails nearly invisible
                base_alpha = self.ghost_color[3] * (age_factor ** 2.5)

                # Shift color slightly for older trails (more purple/faded)
                color_fade = 1.0 - (1.0 - age_factor) * 0.3
                r = self.ghost_color[0] * color_fade
                g = self.ghost_color[1] * color_fade
                b = self.ghost_color[2]

                # Sample every few points to reduce geometry count
                step = max(1, len(ghost_pts) // 20)  # More segments for smoother trails
                for i in range(0, len(ghost_pts) - step, step):
                    # Fade along the trail too
                    dist_factor = 1.0 - (i / len(ghost_pts)) * 0.4
                    alpha = base_alpha * dist_factor
                    color = np.array([r, g, b, alpha])

                    if not self._add_line_segment(scene, ghost_pts[i], ghost_pts[min(i+step, len(ghost_pts)-1)],
                                                   color, self.ghost_radius):
                        break

        # Render actual path taken (orange line showing where robot actually went)
        actual_path_to_show = self.actual_path[:actual_path_limit] if actual_path_limit else self.actual_path
        if show_actual_path and len(actual_path_to_show) >= 2:
            # Sample to reduce geometry count but keep it smooth (60 segments max)
            step = max(1, len(actual_path_to_show) // 60)
            for i in range(0, len(actual_path_to_show) - step, step):
                # Fade older positions
                age_factor = i / len(actual_path_to_show)
                alpha = self.actual_path_color[3] * (0.3 + 0.7 * age_factor)
                color = np.array([
                    self.actual_path_color[0],
                    self.actual_path_color[1],
                    self.actual_path_color[2],
                    alpha
                ])

                if not self._add_line_segment(scene, actual_path_to_show[i], actual_path_to_show[min(i+step, len(actual_path_to_show)-1)],
                                               color, self.ghost_radius):
                    break

        # Joint arcs removed - now shown in separate matplotlib graph

        # Render current whisker (EE trajectory) - single bright green color
        if self.whisker_points is None or len(self.whisker_points) < 2:
            return

        pts = self.whisker_points
        radius = self.whisker_radius

        # Sample every few points for very long chunks
        step = max(1, len(pts) // 30)
        for i in range(0, len(pts) - step, step):
            # Calculate alpha fade based on distance into future
            alpha = self.whisker_color[3] * (1.0 - (i / len(pts)) * 0.5)
            color = np.array([
                self.whisker_color[0],
                self.whisker_color[1],
                self.whisker_color[2],
                alpha
            ])

            if not self._add_line_segment(scene, pts[i], pts[min(i+step, len(pts)-1)],
                                           color, radius):
                print(f"Warning: max geoms reached ({scene.maxgeom})")
                break


def load_act_policy(checkpoint_path: str, device: str):
    """Load ACT policy and processors from checkpoint."""
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = ACTPolicy.from_pretrained(checkpoint_path)
    policy.to(device)
    policy.eval()

    # Load preprocessor/postprocessor for normalization
    # Override device in processor config (needs to be passed via preprocessor_overrides kwarg)
    device_override = {"device_processor": {"device": device}}
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=checkpoint_path,
        preprocessor_overrides=device_override,
        postprocessor_overrides=device_override,
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
    parser.add_argument("--show-joint-graph", action="store_true",
                        help="Show real-time matplotlib graph of joint predictions")
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

    # Create joint plotter for real-time graphs (optional)
    joint_plotter = None
    if args.show_joint_graph:
        chunk_size = policy.config.chunk_size if hasattr(policy.config, 'chunk_size') else 100
        joint_plotter = JointPlotter(chunk_size=chunk_size)

    # Create MuJoCo viewer with custom render callback
    print("Starting visualization...")
    print("\nColor Legend:")
    print("  EE Whisker: GREEN (predicted trajectory)")
    print("  Ghost trails: BLUE (past predictions)")
    print("  Actual path: ORANGE")
    if args.show_joint_graph:
        print("  Joint Graph: Separate window showing all 6 joint predictions")
    print("\nControls:")
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
            if history_index >= 0:
                # Move forward in history
                history_index = min(len(whisker_history) - 1, history_index + 1)
                if history_index == len(whisker_history) - 1:
                    print(f"\r  [HISTORY {history_index + 1}/{len(whisker_history)}] (latest)", end="", flush=True)
                else:
                    print(f"\r  [HISTORY {history_index + 1}/{len(whisker_history)}]          ", end="", flush=True)
            else:
                # Step forward in simulation
                step_forward = True
        elif key == 263 and paused:  # LEFT ARROW
            if len(whisker_history) > 0:
                if history_index == -1:
                    history_index = len(whisker_history) - 1
                else:
                    history_index = max(0, history_index - 1)
                print(f"\r  [HISTORY {history_index + 1}/{len(whisker_history)}]          ", end="", flush=True)

    with mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data, key_callback=key_callback) as viewer:
        for ep in range(args.episodes):
            print(f"\nEpisode {ep + 1}/{args.episodes}")
            sim.reset_scene(randomize=True, pos_range=0.04, rot_range=np.pi)
            policy.reset()
            whisker_history.clear()  # Clear history for new episode
            visualizer.clear_trails()  # Clear ghost trails and actual path
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
                        hist = whisker_history[history_index]
                        # Restore robot state
                        sim.mj_data.qpos[:] = hist['qpos']
                        sim.mj_data.qvel[:] = hist['qvel']
                        mujoco.mj_forward(sim.mj_model, sim.mj_data)
                        # Update joint plotter with historical action chunk
                        if joint_plotter and hist.get('action_chunk') is not None:
                            joint_plotter.update(hist['action_chunk'], executed_steps=0)
                        # Show whiskers from that moment using saved snapshots
                        with viewer.lock():
                            viewer.user_scn.ngeom = 0
                            # Temporarily swap in the historical state
                            saved_whiskers = visualizer.whisker_points
                            saved_gripper = visualizer.whisker_gripper
                            saved_joints = visualizer.whisker_joints
                            saved_path = visualizer.actual_path
                            saved_ghosts = visualizer.ghost_trails
                            visualizer.whisker_points = hist['whiskers']
                            visualizer.whisker_gripper = hist.get('gripper', saved_gripper)
                            visualizer.whisker_joints = hist.get('joints', saved_joints)
                            visualizer.actual_path = hist.get('actual_path', saved_path)
                            visualizer.ghost_trails = hist.get('ghost_trails', saved_ghosts)
                            visualizer.add_whiskers_to_scene(viewer.user_scn)
                            # Restore current state
                            visualizer.whisker_points = saved_whiskers
                            visualizer.whisker_gripper = saved_gripper
                            visualizer.whisker_joints = saved_joints
                            visualizer.actual_path = saved_path
                            visualizer.ghost_trails = saved_ghosts
                    viewer.sync()
                    time.sleep(0.05)

                if step_forward:
                    step_forward = False

                # Get observation
                obs = sim.get_observation()

                # Update whiskers every frame for continuous predictions
                if step % 1 == 0:  # Every step now
                    visualizer.update_whiskers(obs)
                    # Update joint plotter with new action chunk
                    if joint_plotter and visualizer.current_action_chunk is not None:
                        joint_plotter.update(visualizer.current_action_chunk, executed_steps=0)

                    # Record to history (including robot state for playback)
                    if visualizer.whisker_points is not None:
                        # Store snapshots of current state for perfect history playback
                        whisker_history.append({
                            'whiskers': visualizer.whisker_points.copy(),
                            'gripper': visualizer.whisker_gripper.copy() if visualizer.whisker_gripper is not None else None,
                            'joints': {k: v.copy() for k, v in visualizer.whisker_joints.items()} if visualizer.whisker_joints else None,
                            'action_chunk': visualizer.current_action_chunk.copy() if visualizer.current_action_chunk is not None else None,
                            'ee_pos': visualizer.current_ee_pos.copy() if visualizer.current_ee_pos is not None else None,
                            'qpos': sim.mj_data.qpos.copy(),
                            'qvel': sim.mj_data.qvel.copy(),
                            'step': step,
                            'actual_path': [p.copy() for p in visualizer.actual_path],  # Snapshot of path
                            'ghost_trails': [g.copy() for g in visualizer.ghost_trails],  # Snapshot of ghosts
                        })
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

                # Record actual EE position every step for smooth path
                visualizer.record_actual_position()

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
                    print("  [PAUSED] - Use LEFT/RIGHT arrows to review, SPACE to continue")
                    while paused and viewer.is_running():
                        if history_index >= 0 and history_index < len(whisker_history):
                            hist = whisker_history[history_index]
                            # Restore robot state
                            sim.mj_data.qpos[:] = hist['qpos']
                            sim.mj_data.qvel[:] = hist['qvel']
                            mujoco.mj_forward(sim.mj_model, sim.mj_data)
                            # Update joint plotter with historical action chunk
                            if hist.get('action_chunk') is not None:
                                joint_plotter.update(hist['action_chunk'], executed_steps=0)
                            # Show whiskers from that moment using saved snapshots
                            with viewer.lock():
                                viewer.user_scn.ngeom = 0
                                saved_whiskers = visualizer.whisker_points
                                saved_gripper = visualizer.whisker_gripper
                                saved_joints = visualizer.whisker_joints
                                saved_path = visualizer.actual_path
                                saved_ghosts = visualizer.ghost_trails
                                visualizer.whisker_points = hist['whiskers']
                                visualizer.whisker_gripper = hist.get('gripper', saved_gripper)
                                visualizer.whisker_joints = hist.get('joints', saved_joints)
                                visualizer.actual_path = hist.get('actual_path', saved_path)
                                visualizer.ghost_trails = hist.get('ghost_trails', saved_ghosts)
                                visualizer.add_whiskers_to_scene(viewer.user_scn)
                                visualizer.whisker_points = saved_whiskers
                                visualizer.whisker_gripper = saved_gripper
                                visualizer.whisker_joints = saved_joints
                                visualizer.actual_path = saved_path
                                visualizer.ghost_trails = saved_ghosts
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

    if joint_plotter:
        joint_plotter.close()
    sim.disconnect()
    print("Done")


if __name__ == "__main__":
    main()
