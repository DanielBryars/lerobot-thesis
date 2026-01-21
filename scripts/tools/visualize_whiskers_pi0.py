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

import os
import sys
# Fix Windows encoding issue for model loading
os.environ['PYTHONIOENCODING'] = 'utf-8'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import argparse
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
from utils.mujoco_viz import MujocoPathRenderer

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def load_pi0_policy(checkpoint_path: str, device: str, dataset: str = None):
    """Load Pi0 policy from checkpoint with processors for normalization."""
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.policies.factory import make_pre_post_processors

    print(f"Loading Pi0 from {checkpoint_path}...")
    policy = PI0Policy.from_pretrained(checkpoint_path)
    policy.to(device)
    policy.eval()
    print(f"Pi0 config: chunk_size={policy.config.chunk_size}, n_action_steps={policy.config.n_action_steps}")

    # Try to load pre/post processors for normalization
    preprocessor, postprocessor = None, None
    try:
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            pretrained_path=checkpoint_path
        )
        print("Loaded pre/post processors from checkpoint")
    except Exception as e:
        print(f"Warning: Could not load processors from checkpoint: {e}")
        if dataset:
            try:
                from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
                dataset_metadata = LeRobotDatasetMetadata(dataset)
                preprocessor, postprocessor = make_pre_post_processors(
                    policy.config,
                    dataset_stats=dataset_metadata.stats
                )
                print(f"Loaded pre/post processors from dataset {dataset}")
            except Exception as e2:
                print(f"Warning: Could not load processors from dataset: {e2}")

    return policy, preprocessor, postprocessor


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
    parser.add_argument("--show-joint-graph", action="store_true",
                        help="Show real-time matplotlib graph of joint predictions")
    parser.add_argument("--blind-camera", type=str, choices=["none", "overhead", "wrist", "both"],
                        default="none", help="Blind camera(s) with black images for debugging")
    parser.add_argument("--dataset", type=str, default="danbhf/so101_pick_place_1k",
                        help="Dataset for loading normalization stats (if not in checkpoint)")
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

    # Load policy with processors for normalization
    print(f"Loading Pi0 policy from {args.checkpoint}...")
    policy, preprocessor, postprocessor = load_pi0_policy(args.checkpoint, device, args.dataset)

    # Override rollout length if specified
    if args.rollout_length is not None:
        from collections import deque
        original_n_action_steps = policy.config.n_action_steps
        policy.config.n_action_steps = args.rollout_length
        policy._action_queue = deque([], maxlen=args.rollout_length)
        print(f"Rollout length overridden: {original_n_action_steps} -> {args.rollout_length}")

    # Tokenize language instruction
    print(f"Language instruction: '{args.instruction}'")
    lang_tokens, lang_mask = tokenize_instruction(policy, args.instruction, device)
    print(f"Policy loaded (n_action_steps={policy.config.n_action_steps}, chunk_size={policy.config.chunk_size})")

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

    # Create visualizer (uses FK for fast whisker computation)
    visualizer = WhiskerVisualizer(sim)

    # Get chunk size for display
    chunk_size = policy.config.n_action_steps if hasattr(policy.config, 'n_action_steps') else 50
    n_action_steps = policy.config.n_action_steps

    # Create joint plotter for real-time graphs (optional)
    joint_plotter = None
    if args.show_joint_graph:
        joint_plotter = JointPlotter(chunk_size=chunk_size)

    # Create MuJoCo viewer with custom render callback
    if args.blind_camera != "none":
        print(f"\n*** CAMERA BLINDING: {args.blind_camera} camera(s) replaced with BLACK images ***\n")

    print("Starting visualization...")
    print("\nColor Legend:")
    print("  EE Whisker: GREEN (predicted trajectory - the one we follow)")
    print("  Ghost trails: BLUE (past predictions)")
    print("  Actual path: ORANGE")
    if args.show_joint_graph:
        print("  Joint Graph: Separate window showing all 6 joints (in MODEL-NORMALIZED space)")
        print("    - Black solid: ACTUAL position (observation after preprocessing)")
        print("    - Colored dotted: COMMANDED position (model output before postprocessing)")
        print("    - Colored dashed: PREDICTED chunk (future trajectory)")
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
                print(f"\r  [HISTORY {history_index + 1}/{len(whisker_history)}] step {hist['step']} [{hist.get('executed_in_chunk', 0)}/{n_action_steps}]          ", end="", flush=True)

    with mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data, key_callback=key_callback) as viewer:
        for ep in range(args.episodes):
            print(f"\nEpisode {ep + 1}/{args.episodes}")
            sim.reset_scene(randomize=True, pos_range=0.04, rot_range=np.pi)
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

                # Prepare batch with language tokens
                batch = visualizer.prepare_obs(obs, device)

                # Apply blind camera if requested
                if args.blind_camera != "none":
                    for key in list(batch.keys()):
                        if "images" in key:
                            should_blind = False
                            if args.blind_camera == "both":
                                should_blind = True
                            elif args.blind_camera == "overhead" and "overhead" in key:
                                should_blind = True
                            elif args.blind_camera == "wrist" and "wrist" in key:
                                should_blind = True
                            if should_blind:
                                batch[key] = torch.zeros_like(batch[key])

                # Add task string for preprocessor to tokenize (Pi0 requirement)
                batch["task"] = args.instruction

                if preprocessor is not None:
                    batch = preprocessor(batch)
                else:
                    # If no preprocessor, add language tokens manually
                    batch["observation.language.tokens"] = lang_tokens
                    batch["observation.language.attention_mask"] = lang_mask

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

                        # Get full action chunk from policy
                        chunk_tensor = policy.predict_action_chunk(batch).squeeze(0)

                        # Save normalized version
                        full_chunk_normalized = chunk_tensor.cpu().numpy()

                        # Denormalize
                        if postprocessor is not None:
                            chunk_denorm = np.array([
                                postprocessor(chunk_tensor[i]).cpu().numpy().flatten()
                                for i in range(chunk_tensor.shape[0])
                            ])
                        else:
                            chunk_denorm = chunk_tensor.cpu().numpy()

                        full_chunk_denorm = chunk_denorm

                        visualizer.current_action_chunk_normalized = full_chunk_normalized
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
