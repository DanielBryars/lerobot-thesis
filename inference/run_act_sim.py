#!/usr/bin/env python
"""
Run a trained ACT policy in the MuJoCo VR simulation.

Usage:
    python inference/run_act_sim.py outputs/train/act_20251229_120000/final
    python inference/run_act_sim.py outputs/train/act_20251229_120000/checkpoint_005000
    python inference/run_act_sim.py outputs/train/act_20251229_120000/final --episodes 10
"""

# Suppress noisy warnings before importing anything else
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io")
warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")
warnings.filterwarnings("ignore", message=".*video decoding.*deprecated.*")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import wandb

# Add src to path for simulation plugin
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

# Motor names in order
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Sim action space bounds (radians) for EE action conversion
SIM_ACTION_LOW = np.array([-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.17453])
SIM_ACTION_HIGH = np.array([1.91986, 1.74533, 1.69, 1.65806, 2.84121, 1.74533])

# Global IK solver (initialized lazily)
_ik_solver = None
_fk_solver = None

# Track IK failures for diagnostics
_ik_failure_count = 0
_ik_total_count = 0


def get_ik_solver():
    """Lazily initialize IK solver for EE action space."""
    global _ik_solver, _fk_solver
    if _ik_solver is None:
        from test_fk_ik import MuJoCoFK, MuJoCoIK
        scene_xml = str(REPO_ROOT / "scenes" / "so101_with_wrist_cam.xml")
        _fk_solver = MuJoCoFK(scene_xml)
        _ik_solver = MuJoCoIK(_fk_solver)
        print("Initialized IK solver for EE action space")
    return _ik_solver


def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [qw, qx, qy, qz] to rotation matrix."""
    w, x, y, z = quat
    n = np.sqrt(w*w + x*x + y*y + z*z)
    if n > 0:
        w, x, y, z = w/n, x/n, y/n, z/n
    R = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    return R


def ee_action_to_joint_action(ee_action: np.ndarray, last_joints: np.ndarray = None) -> tuple[np.ndarray, bool]:
    """Convert EE action [xyz, quat, gripper] to joint action [6 joints].

    Args:
        ee_action: 8-dim array [x, y, z, qw, qx, qy, qz, gripper]
        last_joints: Previous joint angles for IK initial guess (5-dim, no gripper)

    Returns:
        Tuple of (6-dim joint action in radians, IK success flag)
    """
    global _ik_failure_count, _ik_total_count
    ik = get_ik_solver()

    # Extract EE pose
    ee_pos = ee_action[:3]
    ee_quat = ee_action[3:7]
    gripper = ee_action[7]

    # Convert quaternion to rotation matrix
    ee_rot = quaternion_to_rotation_matrix(ee_quat)

    # Initial guess
    if last_joints is None:
        last_joints = np.zeros(5)

    # Solve IK
    ik_joints, success, error = ik.solve(
        target_pos=ee_pos,
        target_rot=ee_rot,
        initial_angles=last_joints,
        max_iterations=100,
        pos_tolerance=1e-3,
    )

    _ik_total_count += 1

    if not success:
        _ik_failure_count += 1
        # Log warning (but don't spam - only every 10th failure)
        if _ik_failure_count <= 3 or _ik_failure_count % 10 == 0:
            print(f"  [WARNING] IK failed ({_ik_failure_count}/{_ik_total_count}): "
                  f"target_pos={ee_pos}, error={error:.4f}mm, using last joints")
        # Fall back to last joints - for safety during inference
        # (hard failure could cause jerky robot motion)
        ik_joints = last_joints

    # Combine with gripper and clip to bounds
    joint_action = np.concatenate([ik_joints, [gripper]])
    joint_action = np.clip(joint_action, SIM_ACTION_LOW, SIM_ACTION_HIGH)

    return joint_action, success


def load_policy(checkpoint_path: Path, device: torch.device):
    """Load a trained ACT policy from checkpoint."""
    print(f"Loading policy from: {checkpoint_path}")

    # Load ACT policy directly
    policy = ACTPolicy.from_pretrained(str(checkpoint_path))
    policy.eval()
    policy.to(device)

    # Load preprocessor (normalizes observations) and postprocessor (unnormalizes actions)
    print("Loading preprocessor/postprocessor...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint_path),
    )

    print(f"Policy loaded: {type(policy).__name__}")
    print(f"  Chunk size: {policy.config.chunk_size}")
    print(f"  Input features: {list(policy.config.input_features.keys())}")
    return policy, preprocessor, postprocessor


def prepare_observation(obs: dict, device: torch.device) -> dict:
    """Convert simulation observation to policy input format.

    The policy expects:
    - observation.state: [batch, state_dim] tensor of normalized joint positions
    - observation.images.{camera_name}: [batch, C, H, W] tensor
    """
    batch = {}

    # Extract state (joint positions)
    state = []
    for motor in MOTOR_NAMES:
        key = f"{motor}.pos"
        state.append(obs.get(key, 0.0))

    # State tensor: [1, 6]
    batch["observation.state"] = torch.tensor([state], dtype=torch.float32, device=device)

    # Camera images
    for key, value in obs.items():
        if isinstance(value, np.ndarray) and value.ndim == 3:
            # Image: [H, W, C] -> [1, C, H, W]
            img = torch.from_numpy(value).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            batch[f"observation.images.{key}"] = img.to(device)

    return batch


# Track last joint state for IK initial guess (global state for continuity)
_last_ik_joints = None


def actions_to_dict(actions: torch.Tensor, is_ee_action_space: bool = False) -> list[dict]:
    """Convert policy action tensor to list of action dicts.

    Args:
        actions: Tensor of shape [action_dim] or [chunk_size, action_dim] or [batch, chunk_size, action_dim]
        is_ee_action_space: If True, actions are 8-dim EE and need IK conversion

    Returns:
        List of action dicts with motor names
    """
    global _last_ik_joints

    actions = actions.cpu().numpy()

    # Flatten to 2D: [num_actions, action_dim]
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    elif actions.ndim == 3:
        actions = actions[0]  # Take first batch

    action_dim = actions.shape[-1]
    action_dicts = []

    for t in range(actions.shape[0]):
        action = actions[t]

        # Convert EE actions to joint actions if needed
        if is_ee_action_space or action_dim == 8:
            joint_action, ik_success = ee_action_to_joint_action(action, _last_ik_joints)
            _last_ik_joints = joint_action[:5].copy()  # Save for next IK guess
        else:
            joint_action = action

        # Build action dict
        action_dict = {}
        for i, motor in enumerate(MOTOR_NAMES):
            action_dict[f"{motor}.pos"] = float(joint_action[i])
        action_dicts.append(action_dict)

    return action_dicts


def reset_ik_state():
    """Reset IK state at the start of each episode."""
    global _last_ik_joints, _ik_failure_count, _ik_total_count
    _last_ik_joints = None
    _ik_failure_count = 0
    _ik_total_count = 0


def get_ik_stats() -> tuple[int, int, float]:
    """Get IK statistics for current episode.

    Returns:
        Tuple of (failures, total_calls, failure_rate)
    """
    rate = _ik_failure_count / max(1, _ik_total_count)
    return _ik_failure_count, _ik_total_count, rate


def _old_actions_to_dict(actions: torch.Tensor) -> list[dict]:
    """DEPRECATED: Original joint-only version."""
    actions = actions.cpu().numpy()

    # Handle different output shapes from select_action
    if actions.ndim == 1:
        # Single action: [action_dim] -> list of 1 dict
        action_dict = {}
        for i, motor in enumerate(MOTOR_NAMES):
            action_dict[f"{motor}.pos"] = float(actions[i])
        return [action_dict]
    elif actions.ndim == 2:
        # Chunk of actions: [chunk_size, action_dim]
        action_dicts = []
        for t in range(actions.shape[0]):
            action_dict = {}
            for i, motor in enumerate(MOTOR_NAMES):
                action_dict[f"{motor}.pos"] = float(actions[t, i])
            action_dicts.append(action_dict)
        return action_dicts
    elif actions.ndim == 3:
        # Batched chunk: [batch, chunk_size, action_dim]
        actions = actions[0]  # Take first batch
        action_dicts = []
        for t in range(actions.shape[0]):
            action_dict = {}
            for i, motor in enumerate(MOTOR_NAMES):
                action_dict[f"{motor}.pos"] = float(actions[t, i])
            action_dicts.append(action_dict)
        return action_dicts
    else:
        raise ValueError(f"Unexpected action shape: {actions.shape}")


def run_episode(sim_robot, policy, preprocessor, postprocessor, device, fps: int = 30, max_steps: int = 300, use_vr: bool = True):
    """Run one episode with the policy.

    Returns:
        success: True if task completed
        steps: Number of steps taken
        elapsed_time: Time taken in seconds
        ik_failures: Number of IK failures (0 for joint action space)
        ik_total: Total IK calls (0 for joint action space)
    """
    frame_time = 1.0 / fps

    # Reset IK state for new episode (for EE action space)
    reset_ik_state()

    # Reset the policy's internal action queue
    policy.reset()

    print(f"  Running episode (chunk_size={policy.config.chunk_size}, max_steps={max_steps})...")

    episode_start = time.time()

    for step in range(max_steps):
        step_start = time.time()

        # Get observation
        obs = sim_robot.get_observation()

        # Prepare batch for policy
        batch = prepare_observation(obs, device)

        # Apply preprocessor (normalizes observations using training stats)
        batch = preprocessor(batch)

        # Run inference - select_action handles chunking internally
        with torch.no_grad():
            action = policy.select_action(batch)

        # Apply postprocessor (unnormalizes actions back to real values)
        action = postprocessor(action)

        # Convert to action dict
        action_dicts = actions_to_dict(action)
        action_dict = action_dicts[0]  # Take first (or only) action

        # Execute action
        sim_robot.send_action(action_dict)

        # Render (VR is handled in send_action, but MuJoCo viewer needs explicit call)
        if not use_vr:
            if not sim_robot.render():
                print("  Viewer closed")
                elapsed = time.time() - episode_start
                ik_failures, ik_total, _ = get_ik_stats()
                return False, step + 1, elapsed, ik_failures, ik_total

        # Check task completion
        if sim_robot.is_task_complete():
            elapsed = time.time() - episode_start
            ik_failures, ik_total, _ = get_ik_stats()
            print(f"  Task completed at step {step + 1} ({elapsed:.2f}s)")
            if ik_total > 0:
                print(f"  IK stats: {ik_failures}/{ik_total} failures ({100*ik_failures/ik_total:.1f}%)")
            return True, step + 1, elapsed, ik_failures, ik_total

        # Maintain frame rate
        step_elapsed = time.time() - step_start
        if step_elapsed < frame_time:
            time.sleep(frame_time - step_elapsed)

    elapsed = time.time() - episode_start
    ik_failures, ik_total, _ = get_ik_stats()
    print(f"  Episode timed out after {max_steps} steps ({elapsed:.2f}s)")
    if ik_total > 0:
        print(f"  IK stats: {ik_failures}/{ik_total} failures ({100*ik_failures/ik_total:.1f}%)")
    return False, max_steps, elapsed, ik_failures, ik_total


def main():
    parser = argparse.ArgumentParser(description="Run ACT policy in VR simulation")
    parser.add_argument("checkpoint", type=str, help="Path to trained model checkpoint")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--fps", type=int, default=30, help="Simulation FPS")
    parser.add_argument("--max_steps", type=int, default=300, help="Max steps per episode")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--no_vr", action="store_true", help="Disable VR (use MuJoCo viewer)")
    parser.add_argument("--no_randomize", action="store_true", help="Disable object randomization")
    parser.add_argument("--pos_range", type=float, default=4.0, help="Position randomization in cm")
    parser.add_argument("--rot_range", type=float, default=180.0, help="Rotation randomization in degrees")
    parser.add_argument("--wandb_project", type=str, default="lerobot-thesis", help="WandB project name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Select device
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load policy with preprocessor/postprocessor
    policy, preprocessor, postprocessor = load_policy(checkpoint_path, device)

    # Initialize simulation
    print("Initializing simulation...")
    from lerobot_robot_sim import SO100SimConfig, SO100Sim

    config = SO100SimConfig(
        sim_cameras=["wrist_cam", "overhead_cam"],
        enable_vr=not args.no_vr,
        camera_width=640,
        camera_height=480,
    )

    sim_robot = SO100Sim(config)
    sim_robot.connect()

    # Randomization settings
    pos_range_m = args.pos_range / 100.0  # cm to m
    rot_range_rad = np.radians(args.rot_range)

    print()
    print("=" * 60)
    print("ACT Policy Inference")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Episodes: {args.episodes}")
    print(f"FPS: {args.fps}")
    print(f"Max steps: {args.max_steps}")
    print(f"VR: {'disabled' if args.no_vr else 'enabled'}")
    print(f"Randomization: {'disabled' if args.no_randomize else f'±{args.pos_range}cm, ±{args.rot_range}°'}")
    print(f"WandB: {'disabled' if args.no_wandb else args.wandb_project}")
    print("=" * 60)
    print()

    # Initialize WandB
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"eval_{checkpoint_path.parent.name}",
            config={
                "checkpoint": str(checkpoint_path),
                "episodes": args.episodes,
                "fps": args.fps,
                "max_steps": args.max_steps,
                "randomize": not args.no_randomize,
                "pos_range_cm": args.pos_range,
                "rot_range_deg": args.rot_range,
            },
            tags=["inference", "evaluation"],
        )

    print("Controls:")
    print("  SPACEBAR - Recenter VR view")
    print("  Q - Quit")
    print()

    # Run episodes
    successes = 0
    total_steps = 0
    total_time = 0.0
    total_ik_failures = 0
    total_ik_calls = 0
    episode_results = []

    try:
        for ep in range(args.episodes):
            print(f"Episode {ep + 1}/{args.episodes}")

            # Reset scene
            sim_robot.reset_scene(
                randomize=not args.no_randomize,
                pos_range=pos_range_m,
                rot_range=rot_range_rad
            )

            # Small delay to see initial state
            time.sleep(0.5)

            # Run episode
            success, steps, elapsed, ik_failures, ik_calls = run_episode(
                sim_robot, policy, preprocessor, postprocessor, device,
                fps=args.fps,
                max_steps=args.max_steps,
                use_vr=not args.no_vr
            )

            if success:
                successes += 1
            total_steps += steps
            total_time += elapsed
            total_ik_failures += ik_failures
            total_ik_calls += ik_calls

            episode_results.append({
                "episode": ep + 1,
                "success": success,
                "steps": steps,
                "time": elapsed,
                "ik_failures": ik_failures,
                "ik_calls": ik_calls,
            })

            # Log to WandB
            if not args.no_wandb:
                log_data = {
                    "episode/success": 1 if success else 0,
                    "episode/steps": steps,
                    "episode/time": elapsed,
                    "episode/cumulative_success_rate": successes / (ep + 1),
                }
                if ik_calls > 0:
                    log_data["episode/ik_failure_rate"] = ik_failures / ik_calls
                    log_data["episode/ik_failures"] = ik_failures
                wandb.log(log_data, step=ep + 1)

            # Brief pause between episodes
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        sim_robot.disconnect()

    # Summary
    num_episodes = len(episode_results)
    success_rate = successes / max(1, num_episodes)
    avg_steps = total_steps / max(1, num_episodes)
    avg_time = total_time / max(1, num_episodes)
    ik_failure_rate = total_ik_failures / max(1, total_ik_calls)

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Success rate: {successes}/{num_episodes} ({100*success_rate:.1f}%)")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Average time: {avg_time:.2f}s")
    if total_ik_calls > 0:
        print(f"IK failures: {total_ik_failures}/{total_ik_calls} ({100*ik_failure_rate:.2f}%)")
    print("=" * 60)

    # Log final summary to WandB
    if not args.no_wandb:
        summary_data = {
            "summary/success_rate": success_rate,
            "summary/avg_steps": avg_steps,
            "summary/avg_time": avg_time,
            "summary/total_episodes": num_episodes,
        }
        if total_ik_calls > 0:
            summary_data["summary/ik_failure_rate"] = ik_failure_rate
            summary_data["summary/total_ik_failures"] = total_ik_failures
            summary_data["summary/total_ik_calls"] = total_ik_calls
        wandb.log(summary_data)
        wandb.finish()


if __name__ == "__main__":
    main()
