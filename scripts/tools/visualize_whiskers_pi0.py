#!/usr/bin/env python3
"""
Visualize Pi0 policy action predictions as "whiskers" in MuJoCo simulation.

Shows predicted future trajectories as faint lines emanating from the robot,
allowing visual inspection of what the policy is predicting at each timestep.
Useful for debugging why Pi0 might be failing.

Usage:
    python scripts/tools/visualize_whiskers_pi0.py --checkpoint danbhf/pi0_so101_lerobot
    python scripts/tools/visualize_whiskers_pi0.py --checkpoint danbhf/pi0_so101_lerobot_20k --instruction "pick up the block"
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
        language_tokens=None,  # Pre-tokenized language instruction
        language_mask=None,    # Attention mask for language
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
        self.language_tokens = language_tokens
        self.language_mask = language_mask
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
        self.whisker_points = None  # Shape: (horizon, 3)
        self.current_ee_pos = None

        # Storage for ghost trails (past predictions)
        self.ghost_trails = []  # List of past whisker_points arrays

        # Storage for actual path taken
        self.actual_path = []  # List of actual EE positions

    def get_action_chunk(self, obs: dict) -> np.ndarray:
        """Get full action chunk from policy."""
        # Prepare observation for policy (Pi0 doesn't use preprocessor)
        batch = self._prepare_obs(obs)

        with torch.no_grad():
            # Get full action chunk directly from model
            if hasattr(self.policy, 'predict_action_chunk'):
                actions = self.policy.predict_action_chunk(batch)  # Shape: [1, chunk_size, action_dim]
                actions = actions.squeeze(0)  # Shape: [chunk_size, action_dim]
                return actions.cpu().numpy()
            else:
                # Fallback for other policies - just get single action
                action = self.policy.select_action(batch)
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

        # Language tokens (required for Pi0)
        if self.language_tokens is not None:
            batch["observation.language.tokens"] = self.language_tokens
            batch["observation.language.attention_mask"] = self.language_mask

        return batch

    def forward_simulate_chunk(self, actions: np.ndarray, max_steps: int = None) -> np.ndarray:
        """Forward simulate action chunk and return EE positions.

        Args:
            actions: Action chunk to simulate
            max_steps: Max steps to simulate. If None, simulates full chunk.
        """
        if max_steps is None:
            max_steps = len(actions)  # Use full chunk by default
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
                self.ghost_trails.append(self.whisker_points.copy())
                # Keep only recent ghost trails
                if len(self.ghost_trails) > self.max_ghost_trails:
                    self.ghost_trails.pop(0)

        # Get action chunk from policy
        actions = self.get_action_chunk(obs)

        # Forward simulate to get predicted positions (full chunk)
        self.whisker_points = self.forward_simulate_chunk(actions)

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
            for trail_idx, ghost_pts in enumerate(ghost_trails_to_show):
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

        # Render current whiskers (green, main prediction) last so they're on top
        if self.whisker_points is None or len(self.whisker_points) < 2:
            return

        pts = self.whisker_points
        radius = self.whisker_radius

        # Sample every few points for very long chunks
        step = max(1, len(pts) // 30)
        for i in range(0, len(pts) - step, step):
            # Calculate alpha fade based on distance into future
            alpha = self.whisker_color[3] * (1.0 - (i / len(pts)) * 0.7)
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


def load_pi0_policy(checkpoint_path: str, device: str):
    """Load Pi0 policy from checkpoint."""
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy

    print(f"Loading Pi0 from {checkpoint_path}...")
    policy = PI0Policy.from_pretrained(checkpoint_path)
    policy.to(device)
    policy.eval()
    print(f"Pi0 config: chunk_size={policy.config.chunk_size}, n_action_steps={policy.config.n_action_steps}")

    return policy


def tokenize_instruction(policy, instruction: str, device: str):
    """Tokenize language instruction for Pi0."""
    from transformers import AutoTokenizer

    # Try to get tokenizer from policy
    try:
        tokenizer = policy.processor.tokenizer
    except AttributeError:
        try:
            tokenizer = policy.model.processor.tokenizer
        except AttributeError:
            # Fall back to loading tokenizer directly
            print("Loading tokenizer from paligemma...")
            tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

    encoding = tokenizer(
        instruction,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=policy.config.tokenizer_max_length,
    )
    tokens = encoding["input_ids"].to(device)
    mask = encoding["attention_mask"].bool().to(device)

    return tokens, mask


def main():
    parser = argparse.ArgumentParser(description="Visualize Pi0 policy whiskers")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint (local or HuggingFace)")
    parser.add_argument("--instruction", type=str,
                        default="Pick up the block and place it in the bowl",
                        help="Language instruction for Pi0")
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

    # Load policy (Pi0 doesn't use preprocessor/postprocessor)
    print(f"Loading Pi0 policy from {args.checkpoint}...")
    policy = load_pi0_policy(args.checkpoint, device)

    # Tokenize language instruction
    print(f"Language instruction: '{args.instruction}'")
    lang_tokens, lang_mask = tokenize_instruction(policy, args.instruction, device)
    print("Policy loaded")

    # Create simulation (Pi0 uses 224x224 images)
    print("Creating simulation...")
    scene_path = REPO_ROOT / "scenes" / "so101_with_wrist_cam.xml"
    sim_config = SO100SimConfig(
        scene_xml=str(scene_path),
        sim_cameras=["overhead_cam", "wrist_cam"],
        camera_width=224,
        camera_height=224,
    )
    sim = SO100Sim(sim_config)
    sim.connect()

    # Create visualizer with language tokens (Pi0 doesn't use preprocessor/postprocessor)
    visualizer = WhiskerVisualizer(policy, sim, device=device,
                                   language_tokens=lang_tokens, language_mask=lang_mask)

    # Create MuJoCo viewer with custom render callback
    print("Starting visualization...")
    print("\nColor Legend:")
    print("  GREEN:  Current prediction (where policy thinks robot will go)")
    print("  BLUE:   Ghost trails (past predictions)")
    print("  ORANGE: Actual path taken")
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
                        # Show whiskers from that moment using saved snapshots
                        with viewer.lock():
                            viewer.user_scn.ngeom = 0
                            # Temporarily swap in the historical state
                            saved_whiskers = visualizer.whisker_points
                            saved_path = visualizer.actual_path
                            saved_ghosts = visualizer.ghost_trails
                            visualizer.whisker_points = hist['whiskers']
                            visualizer.actual_path = hist.get('actual_path', saved_path)
                            visualizer.ghost_trails = hist.get('ghost_trails', saved_ghosts)
                            visualizer.add_whiskers_to_scene(viewer.user_scn)
                            # Restore current state
                            visualizer.whisker_points = saved_whiskers
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

                    # Record to history (including robot state for playback)
                    if visualizer.whisker_points is not None:
                        # Store snapshots of current state for perfect history playback
                        whisker_history.append({
                            'whiskers': visualizer.whisker_points.copy(),
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

                # Get action and step simulation (Pi0 doesn't use preprocessor)
                batch = visualizer._prepare_obs(obs)
                with torch.no_grad():
                    action = policy.select_action(batch)
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
                            # Show whiskers from that moment using saved snapshots
                            with viewer.lock():
                                viewer.user_scn.ngeom = 0
                                saved_whiskers = visualizer.whisker_points
                                saved_path = visualizer.actual_path
                                saved_ghosts = visualizer.ghost_trails
                                visualizer.whisker_points = hist['whiskers']
                                visualizer.actual_path = hist.get('actual_path', saved_path)
                                visualizer.ghost_trails = hist.get('ghost_trails', saved_ghosts)
                                visualizer.add_whiskers_to_scene(viewer.user_scn)
                                visualizer.whisker_points = saved_whiskers
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

    sim.disconnect()
    print("Done")


if __name__ == "__main__":
    main()
