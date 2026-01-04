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

# Add project paths
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

# Import shared utilities
from utils.constants import MOTOR_NAMES
from utils.ik_solver import IKSolver


# Global IK solver instance
_ik_solver: IKSolver | None = None


def get_ik_solver() -> IKSolver:
    """Get or create the global IK solver."""
    global _ik_solver
    if _ik_solver is None:
        _ik_solver = IKSolver()
    return _ik_solver


def load_policy(checkpoint_path: Path, device: torch.device):
    """Load a trained ACT policy from checkpoint."""
    print(f"Loading policy from: {checkpoint_path}")

    # Load ACT policy directly
    policy = ACTPolicy.from_pretrained(str(checkpoint_path))
    policy.eval()
    policy.to(device)

    # Load preprocessor (normalizes observations) and postprocessor (unnormalizes actions)
    print("Loading preprocessor/postprocessor...")

    # Override device in processor configs (needed when CUDA isn't available)
    device_str = str(device).replace("cuda:", "cuda")
    processor_overrides = {"device_processor": {"device": device_str}}

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint_path),
        preprocessor_overrides=processor_overrides,
        postprocessor_overrides=processor_overrides,
    )

    print(f"Policy loaded: {type(policy).__name__}")
    print(f"  Chunk size: {policy.config.chunk_size}")
    print(f"  Input features: {list(policy.config.input_features.keys())}")
    return policy, preprocessor, postprocessor


def prepare_observation(obs: dict, device: torch.device) -> dict:
    """Convert simulation observation to policy input format."""
    batch = {}

    # Extract state (joint positions)
    state = [obs.get(f"{motor}.pos", 0.0) for motor in MOTOR_NAMES]
    batch["observation.state"] = torch.tensor([state], dtype=torch.float32, device=device)

    # Camera images
    for key, value in obs.items():
        if isinstance(value, np.ndarray) and value.ndim == 3:
            img = torch.from_numpy(value).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            batch[f"observation.images.{key}"] = img.to(device)

    return batch


def actions_to_dict(actions: torch.Tensor, ik_solver: IKSolver) -> list[dict]:
    """Convert policy action tensor to list of action dicts."""
    actions = actions.cpu().numpy()

    # Flatten to 2D: [num_actions, action_dim]
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    elif actions.ndim == 3:
        actions = actions[0]

    action_dim = actions.shape[-1]
    is_ee = action_dim == 8
    action_dicts = []

    for t in range(actions.shape[0]):
        action = actions[t]

        if is_ee:
            # Convert EE action to normalized joint action using IK solver
            joint_action, _, _ = ik_solver.ee_to_joint_action(action, return_normalized=True)
        else:
            joint_action = action

        action_dict = {f"{MOTOR_NAMES[i]}.pos": float(joint_action[i]) for i in range(6)}
        action_dicts.append(action_dict)

    return action_dicts


def run_episode(
    sim_robot,
    policy,
    preprocessor,
    postprocessor,
    device,
    fps: int = 30,
    max_steps: int = 300,
    use_vr: bool = True,
    debug: bool = True,
):
    """Run one episode with the policy."""
    frame_time = 1.0 / fps
    ik_solver = get_ik_solver()
    ik_solver.reset_stats()
    policy.reset()

    action_dim = policy.config.output_features["action"].shape[0]
    is_ee = action_dim == 8
    print(f"  Running episode (chunk_size={policy.config.chunk_size}, max_steps={max_steps})...")
    print(f"  Action space: {'EE (8-dim)' if is_ee else f'Joint ({action_dim}-dim)'}")

    episode_start = time.time()

    for step in range(max_steps):
        step_start = time.time()

        # Get observation
        obs = sim_robot.get_observation()
        batch = prepare_observation(obs, device)

        # Debug output
        if debug and step < 3:
            raw_state = batch["observation.state"].cpu().numpy()[0]
            print(f"  Step {step}: raw_obs.state (before norm) = {raw_state}")

        # Apply preprocessor and run inference
        batch = preprocessor(batch)
        with torch.no_grad():
            action = policy.select_action(batch)
        action = postprocessor(action)

        # Debug output
        if debug and step < 5:
            raw_action = action.cpu().numpy()
            if raw_action.ndim == 3:
                raw_action = raw_action[0, 0]
            elif raw_action.ndim == 2:
                raw_action = raw_action[0]
            print(f"  Step {step}: raw_action = {raw_action}")
            print(f"           obs.state = {batch['observation.state'].cpu().numpy()[0]}")

        # Convert to action dict
        action_dicts = actions_to_dict(action, ik_solver)
        action_dict = action_dicts[0]

        if debug and step < 5:
            joints = [action_dict[f"{m}.pos"] for m in MOTOR_NAMES]
            print(f"           joint_cmd (normalized) = {[f'{j:.1f}' for j in joints]}")

        # Execute action
        sim_robot.send_action(action_dict)

        # Render
        if not use_vr and not sim_robot.render():
            print("  Viewer closed")
            stats = ik_solver.get_stats()
            return False, step + 1, time.time() - episode_start, stats

        # Check task completion
        if sim_robot.is_task_complete():
            elapsed = time.time() - episode_start
            stats = ik_solver.get_stats()
            print(f"  Task completed at step {step + 1} ({elapsed:.2f}s)")
            if stats["total_count"] > 0:
                print(f"  IK stats: {stats['failure_count']}/{stats['total_count']} failures "
                      f"({100*stats['failure_rate']:.1f}%)")
            return True, step + 1, elapsed, stats

        # Maintain frame rate
        step_elapsed = time.time() - step_start
        if step_elapsed < frame_time:
            time.sleep(frame_time - step_elapsed)

    elapsed = time.time() - episode_start
    stats = ik_solver.get_stats()
    print(f"  Episode timed out after {max_steps} steps ({elapsed:.2f}s)")
    if stats["total_count"] > 0:
        print(f"  IK stats: {stats['failure_count']}/{stats['total_count']} failures "
              f"({100*stats['failure_rate']:.1f}%)")
    return False, max_steps, elapsed, stats


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

    # Load policy
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
    pos_range_m = args.pos_range / 100.0
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

            sim_robot.reset_scene(
                randomize=not args.no_randomize,
                pos_range=pos_range_m,
                rot_range=rot_range_rad
            )
            time.sleep(0.5)

            success, steps, elapsed, ik_stats = run_episode(
                sim_robot, policy, preprocessor, postprocessor, device,
                fps=args.fps,
                max_steps=args.max_steps,
                use_vr=not args.no_vr
            )

            if success:
                successes += 1
            total_steps += steps
            total_time += elapsed
            total_ik_failures += ik_stats["failure_count"]
            total_ik_calls += ik_stats["total_count"]

            episode_results.append({
                "episode": ep + 1,
                "success": success,
                "steps": steps,
                "time": elapsed,
                "ik_stats": ik_stats,
            })

            if not args.no_wandb:
                log_data = {
                    "episode/success": 1 if success else 0,
                    "episode/steps": steps,
                    "episode/time": elapsed,
                    "episode/cumulative_success_rate": successes / (ep + 1),
                }
                if ik_stats["total_count"] > 0:
                    log_data["episode/ik_failure_rate"] = ik_stats["failure_rate"]
                    log_data["episode/ik_failures"] = ik_stats["failure_count"]
                wandb.log(log_data, step=ep + 1)

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
