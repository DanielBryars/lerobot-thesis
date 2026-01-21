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
    """Real-time matplotlib plot showing predicted vs actual joint trajectories."""

    def __init__(self, chunk_size: int = 100, history_size: int = 200):
        self.chunk_size = chunk_size
        self.history_size = history_size

        # History of actual joint positions and desired positions
        self.actual_history = []  # List of (step, joint_values) tuples
        self.desired_history = []  # List of (step, joint_values) tuples
        self.global_step = 0

        # Set up interactive mode
        plt.ion()

        # Create figure with subplots for each joint
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.canvas.manager.set_window_title('Joint Predictions vs Actual')
        gs = GridSpec(3, 2, figure=self.fig, hspace=0.35, wspace=0.25)

        self.axes = []
        self.predicted_lines = []  # Current chunk prediction
        self.actual_lines = []  # Actual joint position history
        self.desired_lines = []  # Desired (commanded) position history
        self.current_step_lines = []  # Vertical lines showing current step

        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']

        for i, (motor, color) in enumerate(zip(MOTOR_NAMES, colors)):
            row, col = i // 2, i % 2
            ax = self.fig.add_subplot(gs[row, col])
            ax.set_title(motor, fontsize=10, fontweight='bold')
            ax.set_xlabel('Step', fontsize=8)
            ax.set_ylabel('Value (model-norm)', fontsize=8)
            ax.set_xlim(0, chunk_size)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

            # Create line for current chunk prediction (dashed, shows future)
            pred_line, = ax.plot([], [], color=color, linewidth=2, linestyle='--',
                                  alpha=0.7, label='Predicted chunk')
            # Create line for actual joint positions (solid, historical)
            actual_line, = ax.plot([], [], color='black', linewidth=1.5,
                                    alpha=0.8, label='Actual')
            # Create line for desired/commanded positions (dotted, historical)
            desired_line, = ax.plot([], [], color=color, linewidth=1.5, linestyle=':',
                                     alpha=0.6, label='Commanded')
            # Create vertical line for current execution point
            vline = ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)

            # Add legend only on first subplot
            if i == 0:
                ax.legend(loc='upper right', fontsize=7)

            self.axes.append(ax)
            self.predicted_lines.append(pred_line)
            self.actual_lines.append(actual_line)
            self.desired_lines.append(desired_line)
            self.current_step_lines.append(vline)

            # Set appropriate y-limits for model-normalized space (roughly -3 to 3)
            ax.set_ylim(-4, 4)

        self.fig.tight_layout()
        plt.show(block=False)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def record_actual(self, joint_positions: np.ndarray):
        """Record actual joint position at current step.

        Args:
            joint_positions: Array of 6 joint positions (normalized)
        """
        self.actual_history.append((self.global_step, joint_positions.copy()))
        # Trim history
        if len(self.actual_history) > self.history_size:
            self.actual_history.pop(0)

    def record_desired(self, joint_positions: np.ndarray):
        """Record desired/commanded joint position at current step.

        Args:
            joint_positions: Array of 6 joint positions (normalized, from action)
        """
        self.desired_history.append((self.global_step, joint_positions.copy()))
        # Trim history
        if len(self.desired_history) > self.history_size:
            self.desired_history.pop(0)
        self.global_step += 1

    def update(self, action_chunk: np.ndarray, executed_steps: int = 0):
        """Update the plot with new action chunk predictions and history.

        Args:
            action_chunk: Shape (chunk_size, 6) - predicted actions for current chunk
            executed_steps: How many steps have been executed in current chunk
        """
        if action_chunk is None or len(action_chunk) == 0:
            return

        # Calculate x-axis range to show: history leading up to now + future prediction
        # The current step is at global_step, chunk extends into future
        history_steps = min(len(self.actual_history), self.history_size // 2)
        future_steps = len(action_chunk) - executed_steps

        # X-axis: [global_step - history_steps, global_step + future_steps]
        x_min = max(0, self.global_step - history_steps)
        x_max = self.global_step + future_steps + 10

        # Prepare chunk prediction x-coords (starting from current global step - executed)
        chunk_start_step = self.global_step - executed_steps
        chunk_x = np.arange(len(action_chunk)) + chunk_start_step

        for i, (pred_line, actual_line, desired_line, vline, ax) in enumerate(
            zip(self.predicted_lines, self.actual_lines, self.desired_lines,
                self.current_step_lines, self.axes)):

            # Update predicted chunk line
            if i < action_chunk.shape[1]:
                pred_line.set_data(chunk_x, action_chunk[:, i])

            # Update actual history line
            if self.actual_history:
                actual_x = [h[0] for h in self.actual_history]
                actual_y = [h[1][i] for h in self.actual_history if i < len(h[1])]
                actual_x = actual_x[:len(actual_y)]
                actual_line.set_data(actual_x, actual_y)

            # Update desired history line
            if self.desired_history:
                desired_x = [h[0] for h in self.desired_history]
                desired_y = [h[1][i] for h in self.desired_history if i < len(h[1])]
                desired_x = desired_x[:len(desired_y)]
                desired_line.set_data(desired_x, desired_y)

            # Update x-limits to follow the action
            ax.set_xlim(x_min, x_max)

            # Auto-adjust y-limits based on visible data
            if MOTOR_NAMES[i] != 'gripper':
                all_y = list(action_chunk[:, i])
                if self.actual_history:
                    all_y.extend([h[1][i] for h in self.actual_history if i < len(h[1])])
                if all_y:
                    ymin, ymax = min(all_y), max(all_y)
                    margin = (ymax - ymin) * 0.15 + 1
                    ax.set_ylim(ymin - margin, ymax + margin)

            # Update vertical line to show current position
            vline.set_xdata([self.global_step, self.global_step])

        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def reset(self):
        """Reset history for new episode."""
        self.actual_history.clear()
        self.desired_history.clear()
        self.global_step = 0

    def close(self):
        """Close the plot window."""
        plt.close(self.fig)


class WhiskerVisualizer:
    """Visualizes action chunk predictions as whiskers in MuJoCo.

    Uses utility classes from utils.mujoco_viz for clean separation of concerns.
    """

    def __init__(
        self,
        sim: SO100Sim,
        whisker_color: tuple = (0.2, 0.8, 0.2, 0.7),  # RGBA - green
        ghost_color: tuple = (0.5, 0.5, 0.9, 0.3),  # RGBA - light blue
        actual_path_color: tuple = (1.0, 0.3, 0.1, 0.5),  # RGBA - orange
        whisker_radius: float = 0.003,
        ghost_radius: float = 0.002,
        max_ghost_trails: int = 12,
    ):
        self.sim = sim
        self.whisker_color = whisker_color
        self.ghost_color = ghost_color
        self.actual_path_color = actual_path_color
        self.whisker_radius = whisker_radius
        self.ghost_radius = ghost_radius
        self.max_ghost_trails = max_ghost_trails

        # Import utilities
        from utils.mujoco_viz import MujocoPathRenderer, TrajectoryTracker, FKSolver

        # Initialize components
        self.fk_solver = FKSolver(sim.mj_model, sim.mj_data, ee_site_name="gripperframe")
        self.actual_path_tracker = TrajectoryTracker(max_length=500)
        self.renderer = MujocoPathRenderer()

        # Get end-effector site ID (for recording actual position)
        self.ee_site_id = self.fk_solver.ee_site_id

        # Current whisker data
        self.whisker_points = None  # Shape: (N, 3) - predicted EE positions
        self.ghost_trails = []  # List of past whisker points arrays

        # Action chunk management
        self.current_action_chunk_normalized = None
        self.current_action_chunk_denorm = None
        self.chunk_step = 0

    def _prepare_obs(self, obs: dict, device: str = "cuda") -> dict:
        """Convert sim observation to policy input format."""
        batch = {}

        # State
        state = np.array([obs[m + ".pos"] for m in MOTOR_NAMES], dtype=np.float32)
        batch["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(device)

        # Images
        for key, value in obs.items():
            if isinstance(value, np.ndarray) and value.ndim == 3:
                img = torch.from_numpy(value).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                batch[f"observation.images.{key}"] = img.to(device)

        return batch

    def update_whiskers_from_actions(self, action_chunk_denorm: np.ndarray):
        """Update whisker visualization from action chunk.

        Args:
            action_chunk_denorm: Denormalized joint targets, shape (N, 6)
        """
        # Save old whiskers as ghost trail
        if self.whisker_points is not None and len(self.whisker_points) > 1:
            self.ghost_trails.append(self.whisker_points.copy())
            if len(self.ghost_trails) > self.max_ghost_trails:
                self.ghost_trails.pop(0)

        # Compute predicted EE positions using FK
        self.whisker_points = self.fk_solver.compute_ee_positions(action_chunk_denorm)

    def record_actual_position(self):
        """Record current EE position. Call every simulation step."""
        pos = self.sim.mj_data.site_xpos[self.ee_site_id].copy()
        self.actual_path_tracker.record(pos)

    def clear_trails(self):
        """Clear all trails and whiskers (call on episode reset)."""
        self.ghost_trails.clear()
        self.actual_path_tracker.clear()
        self.whisker_points = None

    def draw_actual_path(self, scene: mujoco.MjvScene):
        """Draw the actual path taken (updates every step)."""       
        actual_pts = self.actual_path_tracker.get_points()
        if len(actual_pts) >= 2:
            self.renderer.draw_path(scene, actual_pts, self.actual_path_color, self.ghost_radius, max_segments=60)

    def draw_whisker(self, scene: mujoco.MjvScene):
        """Draw current whisker prediction and ghost trails."""
        # Ghost trails (older = more faded)
        for i, ghost_pts in enumerate(self.ghost_trails):
            if ghost_pts is None or len(ghost_pts) < 2:
                continue
            age_factor = (i + 1) / max(len(self.ghost_trails), 1)
            alpha = self.ghost_color[3] * (age_factor ** 2)
            color = (self.ghost_color[0], self.ghost_color[1], self.ghost_color[2], alpha)
            self.renderer.draw_path(scene, ghost_pts, color, self.ghost_radius, max_segments=20)

        # Current whisker
        if self.whisker_points is not None and len(self.whisker_points) >= 2:
            self.renderer.draw_path(scene, self.whisker_points, self.whisker_color, self.whisker_radius, max_segments=30)

    def add_whiskers_to_scene(self, scene: mujoco.MjvScene, show_ghosts: bool = True, show_actual_path: bool = True,
                               actual_path_limit: int = None, ghost_trails_limit: int = None):
        """Draw all visualization elements to the MuJoCo scene (legacy method)."""
        if show_ghosts:
            self.draw_whisker(scene)
        if show_actual_path:
            self.draw_actual_path(scene)


def load_act_policy(checkpoint_path: str, device: str):
    """Load ACT policy and processors from checkpoint."""
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from pathlib import Path

    # Convert to absolute path if it's a local path
    if Path(checkpoint_path).exists():
        checkpoint_path = str(Path(checkpoint_path).resolve())

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
    parser.add_argument("--rollout-length", type=int, default=None,
                        help="Override n_action_steps (rollout length before re-prediction)")
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

    # Override rollout length if specified
    if args.rollout_length is not None:
        from collections import deque
        original_n_action_steps = policy.config.n_action_steps
        policy.config.n_action_steps = args.rollout_length
        policy._action_queue = deque([], maxlen=args.rollout_length)
        print(f"Rollout length overridden: {original_n_action_steps} -> {args.rollout_length}")

    print(f"Policy loaded (n_action_steps={policy.config.n_action_steps}, chunk_size={policy.config.chunk_size})")

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
    visualizer = WhiskerVisualizer(sim)

    # Get chunk size for display
    chunk_size = policy.config.n_action_steps if hasattr(policy.config, 'n_action_steps') else 100

    # Create joint plotter for real-time graphs (optional)
    joint_plotter = None
    if args.show_joint_graph:
        joint_plotter = JointPlotter(chunk_size=chunk_size)

    # Create MuJoCo viewer with custom render callback
    print("Starting visualization...")
    print("\nColor Legend:")
    print("  EE Whisker: GREEN (predicted trajectory)")
    print("  Ghost trails: BLUE (past predictions)")
    print("  Actual path: ORANGE")
    if args.show_joint_graph:
        print("  Joint Graph: Separate window showing all 6 joints (in MODEL-NORMALIZED space)")
        print("    - Black solid: ACTUAL position (observation after preprocessing)")
        print("    - Colored dotted: COMMANDED position (model output before postprocessing)")
        print("    - Colored dashed: PREDICTED chunk (future trajectory)")
        print("    (Values near 0 = at dataset mean, +/-3 = ~3 std devs from mean)")
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
                if history_index >= len(whisker_history) - 1:
                    # At end of history - exit history mode to live view
                    history_index = -1
                    print(f"\r  [LIVE - at step {whisker_history[-1]['step'] if whisker_history else 0}]          ", end="", flush=True)
                else:
                    history_index += 1
                    print(f"\r  [HISTORY {history_index + 1}/{len(whisker_history)}] step {whisker_history[history_index]['step']}          ", end="", flush=True)
            else:
                # Step forward in simulation
                step_forward = True
        elif key == 263 and paused:  # LEFT ARROW
            if len(whisker_history) > 0:
                if history_index == -1:
                    history_index = len(whisker_history) - 1
                else:
                    history_index = max(0, history_index - 1)
                hist = whisker_history[history_index]
                print(f"\r  [HISTORY {history_index + 1}/{len(whisker_history)}] step {hist['step']} [{hist.get('executed_in_chunk', 0)}/{chunk_size}]          ", end="", flush=True)

    with mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data, key_callback=key_callback) as viewer:
        for ep in range(args.episodes):
            print(f"\nEpisode {ep + 1}/{args.episodes}")
            sim.reset_scene(randomize=True, pos_range=0.04, rot_range=np.pi)
            policy.reset()
            whisker_history.clear()  # Clear history for new episode
            visualizer.clear_trails()  # Clear ghost trails and actual path
            visualizer.chunk_step = 0  # Reset chunk step counter
            visualizer.current_action_chunk_normalized = None
            visualizer.current_action_chunk_denorm = None
            if joint_plotter:
                joint_plotter.reset()  # Reset joint position history
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
                        # Update joint plotter
                        if joint_plotter and hist.get('action_chunk') is not None:
                            joint_plotter.update(hist['action_chunk'], executed_steps=hist.get('executed_in_chunk', 0))
                        # Draw historical state
                        with viewer.lock():
                            viewer.user_scn.ngeom = 0
                            saved_whiskers = visualizer.whisker_points
                            saved_ghosts = visualizer.ghost_trails
                            visualizer.whisker_points = hist['whiskers']
                            visualizer.ghost_trails = hist.get('ghost_trails', [])
                            visualizer.draw_whisker(viewer.user_scn)
                            if hist.get('actual_path'):
                                from utils.mujoco_viz import MujocoPathRenderer
                                MujocoPathRenderer.draw_path(viewer.user_scn, np.array(hist['actual_path']),
                                                              visualizer.actual_path_color, visualizer.ghost_radius)
                            visualizer.whisker_points = saved_whiskers
                            visualizer.ghost_trails = saved_ghosts
                    viewer.sync()
                    time.sleep(0.05)

                if step_forward:
                    step_forward = False

                # Timing debug
                t_loop_start = time.perf_counter()

                # Get observation
                t0 = time.perf_counter()
                obs = sim.get_observation()
                t_obs = time.perf_counter() - t0

                # Get action from our managed chunk (not the policy's queue)
                # This ensures whiskers match exactly what we execute
                t0 = time.perf_counter()
                n_action_steps = policy.config.n_action_steps

                batch = visualizer._prepare_obs(obs, device)
                if preprocessor is not None:
                    batch = preprocessor(batch)

                # Record ACTUAL joint positions in MODEL-NORMALIZED space
                if joint_plotter:
                    actual_normalized = batch["observation.state"].cpu().numpy().flatten()
                    joint_plotter.record_actual(actual_normalized)

                # Check if we need a new prediction
                need_new_chunk = (visualizer.current_action_chunk_denorm is None or
                                  visualizer.chunk_step >= n_action_steps)

                # ===== GET CHUNK (only when needed) =====
                with torch.no_grad():
                    if need_new_chunk:
                        print(f"  [GetChunk] step={step}")
                        full_chunk_tensor = policy.predict_action_chunk(batch)
                        full_chunk_tensor = full_chunk_tensor.squeeze(0)

                        full_chunk_normalized = full_chunk_tensor.cpu().numpy()
                        visualizer.current_action_chunk_normalized = full_chunk_normalized

                        if postprocessor is not None:
                            full_chunk_denorm = np.array([
                                postprocessor(full_chunk_tensor[i]).cpu().numpy().flatten()
                                for i in range(full_chunk_tensor.shape[0])
                            ])
                        else:
                            full_chunk_denorm = full_chunk_normalized.copy()

                        visualizer.current_action_chunk_denorm = full_chunk_denorm
                        visualizer.chunk_step = 0

                        # ===== CALCULATE FK (only when new chunk) =====
                        print(f"  [CalculateFK] {len(full_chunk_denorm)} positions")
                        visualizer.update_whiskers_from_actions(full_chunk_denorm)

                        # ===== DRAW WHISKER (only when new chunk) =====
                        print(f"  [DrawWhisker] {len(visualizer.whisker_points)} pts, first={visualizer.whisker_points[0]}, last={visualizer.whisker_points[-1]}")

                    # Get action from stored chunk
                    action_normalized = visualizer.current_action_chunk_normalized[visualizer.chunk_step]
                    action = visualizer.current_action_chunk_denorm[visualizer.chunk_step]
                    visualizer.chunk_step += 1

                executed_in_chunk = visualizer.chunk_step
                t_policy = time.perf_counter() - t0

                # Record for joint plotter
                if joint_plotter:
                    joint_plotter.record_desired(action_normalized)
                    joint_plotter.update(visualizer.current_action_chunk_normalized, executed_steps=executed_in_chunk)

                # Record history for playback
                whisker_history.append({
                    'whiskers': visualizer.whisker_points.copy() if visualizer.whisker_points is not None else None,
                    'action_chunk': visualizer.current_action_chunk_normalized.copy() if visualizer.current_action_chunk_normalized is not None else None,
                    'qpos': sim.mj_data.qpos.copy(),
                    'qvel': sim.mj_data.qvel.copy(),
                    'step': step,
                    'executed_in_chunk': executed_in_chunk,
                    'actual_path': [p.copy() for p in visualizer.actual_path_tracker.points],
                    'ghost_trails': [g.copy() for g in visualizer.ghost_trails],
                })
                if len(whisker_history) > max_history:
                    whisker_history.pop(0)

                # ===== APPLY ACTION =====
                t0 = time.perf_counter()
                action_dict = {m + ".pos": float(action[i]) for i, m in enumerate(MOTOR_NAMES)}
                sim.send_action(action_dict)
                t_sim = time.perf_counter() - t0

                # ===== RECORD PATH (every step) =====
                visualizer.record_actual_position()

                # ===== RENDER SCENE (every step - MuJoCo requires redraw) =====
                t0 = time.perf_counter()
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    visualizer.draw_whisker(viewer.user_scn)
                    visualizer.draw_actual_path(viewer.user_scn)
                viewer.sync()
                t_render = time.perf_counter() - t0

                t_loop = time.perf_counter() - t_loop_start

                # Status every 10 steps
                if step % 10 == 0:
                    print(f"  Step {step} [{executed_in_chunk}/{n_action_steps}]: {t_loop*1000:.1f}ms")

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
                            # Update joint plotter
                            if joint_plotter and hist.get('action_chunk') is not None:
                                joint_plotter.update(hist['action_chunk'], executed_steps=hist.get('executed_in_chunk', 0))
                            # Draw historical state
                            with viewer.lock():
                                viewer.user_scn.ngeom = 0
                                # Temporarily swap state for drawing
                                saved_whiskers = visualizer.whisker_points
                                saved_ghosts = visualizer.ghost_trails
                                visualizer.whisker_points = hist['whiskers']
                                visualizer.ghost_trails = hist.get('ghost_trails', [])
                                visualizer.draw_whisker(viewer.user_scn)
                                # Draw historical actual path
                                if hist.get('actual_path'):
                                    from utils.mujoco_viz import MujocoPathRenderer
                                    MujocoPathRenderer.draw_path(viewer.user_scn, np.array(hist['actual_path']),
                                                                  visualizer.actual_path_color, visualizer.ghost_radius)
                                visualizer.whisker_points = saved_whiskers
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
