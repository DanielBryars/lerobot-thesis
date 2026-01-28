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

# Add project paths
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot_robot_sim import SO100Sim, SO100SimConfig
from utils.joint_plotter import JointPlotter
from utils.whisker_visualizer import WhiskerVisualizer

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def set_arm_start_position(sim, start_near: int):
    """Set the robot arm to start near a specific block position.

    Args:
        sim: The SO100Sim instance
        start_near: 1 for block at pos1 (positive Y), 2 for block at pos2 (negative Y)
    """
    # In two-block scene, qpos layout:
    # [0:7] = duplo freejoint (pos xyz + quat wxyz)
    # [7:14] = duplo2 freejoint
    # [14:20] = robot joints (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper)
    # Single block scene: [0:7] = duplo, [7:13] = robot joints

    # Try to find robot joint indices
    try:
        shoulder_pan_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_pan")
        shoulder_pan_qpos_idx = sim.mj_model.jnt_qposadr[shoulder_pan_id]
    except:
        # Fall back to common layouts
        shoulder_pan_qpos_idx = 14 if sim.mj_model.nq > 14 else 7

    if start_near == 1:
        # Position arm toward block 1 (positive Y, x=0.22, y=0.225)
        sim.mj_data.qpos[shoulder_pan_qpos_idx] = 0.8  # ~45 degrees toward pos1
        sim.mj_data.qpos[shoulder_pan_qpos_idx + 1] = -0.3  # shoulder_lift
        sim.mj_data.qpos[shoulder_pan_qpos_idx + 2] = 0.5   # elbow_flex
    elif start_near == 2:
        # Position arm toward block 2 (negative Y, x=0.32, y=-0.03)
        sim.mj_data.qpos[shoulder_pan_qpos_idx] = -0.5  # ~-30 degrees toward pos2
        sim.mj_data.qpos[shoulder_pan_qpos_idx + 1] = -0.3  # shoulder_lift
        sim.mj_data.qpos[shoulder_pan_qpos_idx + 2] = 0.5   # elbow_flex

    # Step to apply the changes
    for _ in range(10):
        mujoco.mj_step(sim.mj_model, sim.mj_data)


def load_act_policy(checkpoint_path: str, device: str):
    """Load ACT policy and processors from checkpoint."""
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    # Convert to absolute path if it's a local path
    if Path(checkpoint_path).exists():
        checkpoint_path = str(Path(checkpoint_path).resolve())

    policy = ACTPolicy.from_pretrained(checkpoint_path)
    policy.to(device)
    policy.eval()

    # Load preprocessor/postprocessor for normalization
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
    parser.add_argument("--num-inferences", type=int, default=1,
                        help="Number of inferences per chunk (shows variance, follows last one)")
    parser.add_argument("--stochastic", action="store_true",
                        help="Enable stochastic sampling from VAE latent (required for variance visualization)")
    parser.add_argument("--scene", type=str, default="so101_with_wrist_cam.xml",
                        help="Scene XML file (in scenes/ directory)")
    parser.add_argument("--no-randomize", action="store_true",
                        help="Disable block position randomization (for fixed scenes)")
    parser.add_argument("--start-near", type=int, choices=[1, 2], default=None,
                        help="Start arm near block 1 or 2 (for two-block scenes)")
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
    print(f"Creating simulation with scene: {args.scene}")
    scene_path = REPO_ROOT / "scenes" / args.scene
    if not scene_path.exists():
        print(f"ERROR: Scene not found: {scene_path}")
        sys.exit(1)
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
    # Warn if using multiple inferences without stochastic mode
    if args.num_inferences > 1 and not args.stochastic:
        print("\nWARNING: --num-inferences > 1 but --stochastic not set.")
        print("  ACT is deterministic in eval mode - all inferences will be identical!")
        print("  Add --stochastic to enable VAE latent sampling for variance visualization.\n")

    print("\nColor Legend:")
    print("  EE Whisker: GREEN (predicted trajectory - the one we follow)")
    if args.num_inferences > 1:
        print(f"  Variance whiskers: GREY ({args.num_inferences - 1} alternative predictions)")
        if args.stochastic:
            print("    (stochastic VAE sampling enabled)")
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
    whisker_history = []  # List of state snapshots
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
            randomize = not args.no_randomize
            sim.reset_scene(randomize=randomize, pos_range=0.04, rot_range=np.pi)
            if args.start_near is not None:
                set_arm_start_position(sim, args.start_near)
            policy.reset()
            whisker_history.clear()
            visualizer.clear_trails()
            visualizer.chunk_step = 0
            visualizer.current_action_chunk_normalized = None
            visualizer.current_action_chunk_denorm = None
            if joint_plotter:
                joint_plotter.reset()
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

                # Timing
                t_loop_start = time.perf_counter()

                # Get observation
                t0 = time.perf_counter()
                obs = sim.get_observation()
                t_obs = time.perf_counter() - t0

                # Get action from our managed chunk (not the policy's queue)
                t0 = time.perf_counter()
                n_action_steps = policy.config.n_action_steps

                batch = visualizer.prepare_obs(obs, device)
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

                        # Run multiple inferences for variance visualization
                        all_chunks_denorm = []
                        last_chunk_normalized = None

                        for inf_idx in range(args.num_inferences):
                            if args.stochastic:
                                # Stay in eval mode but patch torch.zeros to sample latent from N(0,1)
                                # instead of using zeros (which makes ACT deterministic)
                                latent_dim = policy.config.latent_dim
                                _orig_zeros = torch.zeros

                                def _randn_for_latent(size, *args, **kwargs):
                                    # Only replace zeros for latent-shaped tensors [batch, latent_dim]
                                    if isinstance(size, (list, tuple)) and len(size) == 2 and size[1] == latent_dim:
                                        return torch.randn(size, *args, **kwargs)
                                    return _orig_zeros(size, *args, **kwargs)

                                torch.zeros = _randn_for_latent
                                try:
                                    chunk_tensor = policy.predict_action_chunk(batch).squeeze(0)
                                finally:
                                    torch.zeros = _orig_zeros
                            else:
                                chunk_tensor = policy.predict_action_chunk(batch).squeeze(0)

                            if postprocessor is not None:
                                chunk_denorm = np.array([
                                    postprocessor(chunk_tensor[i]).cpu().numpy().flatten()
                                    for i in range(chunk_tensor.shape[0])
                                ])
                            else:
                                chunk_denorm = chunk_tensor.cpu().numpy()

                            all_chunks_denorm.append(chunk_denorm)
                            last_chunk_normalized = chunk_tensor.cpu().numpy()

                        # Use the LAST inference as the one to follow
                        full_chunk_denorm = all_chunks_denorm[-1]
                        full_chunk_normalized = last_chunk_normalized

                        visualizer.current_action_chunk_normalized = full_chunk_normalized
                        visualizer.current_action_chunk_denorm = full_chunk_denorm
                        visualizer.chunk_step = 0

                        # Alternative chunks (all except the last one)
                        alt_chunks = all_chunks_denorm[:-1] if len(all_chunks_denorm) > 1 else []

                        # ===== CALCULATE FK (only when new chunk) =====
                        visualizer.update_whiskers_from_actions(full_chunk_denorm, alt_chunks=alt_chunks)

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
                            sim.mj_data.qpos[:] = hist['qpos']
                            sim.mj_data.qvel[:] = hist['qvel']
                            mujoco.mj_forward(sim.mj_model, sim.mj_data)
                            if joint_plotter and hist.get('action_chunk') is not None:
                                joint_plotter.update(hist['action_chunk'], executed_steps=hist.get('executed_in_chunk', 0))
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
                    break

                # Small delay for visualization
                time.sleep(0.01)
            else:
                print(f"  TIMEOUT after {args.max_steps} steps")

            # Pause at end of episode
            print("    Pausing 2 seconds...")
            pause_start = time.time()
            while viewer.is_running() and (time.time() - pause_start) < 2:
                viewer.sync()
                time.sleep(0.05)

        # Final pause before exit
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
