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

# Add src to path for simulation plugin
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

# Motor names in order
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


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


def actions_to_dict(actions: torch.Tensor) -> list[dict]:
    """Convert policy action tensor to list of action dicts.

    Args:
        actions: Tensor of shape [action_dim] or [chunk_size, action_dim] or [batch, chunk_size, action_dim]

    Returns:
        List of action dicts with motor names
    """
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
    """
    frame_time = 1.0 / fps

    # Reset the policy's internal action queue
    policy.reset()

    print(f"  Running episode (chunk_size={policy.config.chunk_size}, max_steps={max_steps})...")

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
                return False, step + 1

        # Check task completion
        if sim_robot.is_task_complete():
            print(f"  Task completed at step {step + 1}!")
            return True, step + 1

        # Maintain frame rate
        elapsed = time.time() - step_start
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)

    print(f"  Episode timed out after {max_steps} steps")
    return False, max_steps


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
    print("=" * 60)
    print()

    print("Controls:")
    print("  SPACEBAR - Recenter VR view")
    print("  Q - Quit")
    print()

    # Run episodes
    successes = 0
    total_steps = 0

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
            success, steps = run_episode(
                sim_robot, policy, preprocessor, postprocessor, device,
                fps=args.fps,
                max_steps=args.max_steps,
                use_vr=not args.no_vr
            )

            if success:
                successes += 1
            total_steps += steps

            # Brief pause between episodes
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        sim_robot.disconnect()

    # Summary
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Success rate: {successes}/{args.episodes} ({100*successes/max(1,args.episodes):.1f}%)")
    print(f"Average steps: {total_steps/max(1,args.episodes):.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
