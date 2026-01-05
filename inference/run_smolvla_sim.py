#!/usr/bin/env python
"""
Run a trained SmolVLA policy in MuJoCo simulation.

SmolVLA is a Vision-Language-Action model that can follow natural language instructions.

Usage:
    # Run with default language instruction
    python run_smolvla_sim.py outputs/train/smolvla_20260105_120000/final

    # Run with custom language instruction
    python run_smolvla_sim.py outputs/train/smolvla/final --language "Pick up the red block"

    # Run multiple episodes with randomization
    python run_smolvla_sim.py outputs/train/smolvla/final --num_episodes 30 --randomize

    # Run from HuggingFace pretrained model
    python run_smolvla_sim.py lerobot/smolvla_base --language "Pick up the block"
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

from utils.constants import MOTOR_NAMES, NUM_JOINTS
from utils.training import prepare_obs_for_policy
from utils.ik_solver import IKSolver


def run_episode(
    sim_robot,
    policy,
    preprocessor,
    postprocessor,
    device,
    language_instruction: str,
    is_ee_action_space: bool,
    ik_solver: IKSolver = None,
    depth_cameras: list = None,
    max_steps: int = 300,
    verbose: bool = True,
):
    """Run a single episode with the SmolVLA policy.

    Returns:
        Tuple of (success, steps, elapsed_time)
    """
    sim_robot.reset()
    start_time = time.time()

    for step in range(max_steps):
        # Get observation
        obs = sim_robot.get_observation()
        batch = prepare_obs_for_policy(obs, device, depth_cameras)

        # Add language instruction
        batch["observation.language"] = language_instruction

        # Preprocess
        batch = preprocessor(batch)

        # Get action from policy
        with torch.no_grad():
            action = policy.select_action(batch)

        # Postprocess
        action = postprocessor(action)

        # Convert to numpy
        action_np = action.cpu().numpy()
        if action_np.ndim > 1:
            action_np = action_np.flatten()

        # Convert EE actions to joint actions if needed
        if is_ee_action_space:
            action_np = action_np[:8]
            joint_action, _, _ = ik_solver.ee_to_joint_action(action_np, return_normalized=True)
        else:
            joint_action = action_np[:NUM_JOINTS]

        # Apply action
        action_dict = {f"{MOTOR_NAMES[i]}.pos": float(joint_action[i]) for i in range(NUM_JOINTS)}
        sim_robot.send_action(action_dict)

        # Check task completion
        if sim_robot.is_task_complete():
            elapsed = time.time() - start_time
            if verbose:
                print(f"  Task completed at step {step + 1} ({elapsed:.2f}s)")
            return True, step + 1, elapsed

    elapsed = time.time() - start_time
    if verbose:
        print(f"  Timed out after {max_steps} steps ({elapsed:.2f}s)")
    return False, max_steps, elapsed


def main():
    parser = argparse.ArgumentParser(description="Run SmolVLA policy in simulation")
    parser.add_argument("model_path", type=str, help="Path to trained model or HuggingFace model ID")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes (default: 10)")
    parser.add_argument("--randomize", action="store_true", help="Randomize object positions")
    parser.add_argument("--language", type=str, default="Pick up the block and place it in the bowl",
                        help="Language instruction")
    parser.add_argument("--max_steps", type=int, default=300, help="Max steps per episode (default: 300)")
    parser.add_argument("--fps", type=int, default=30, help="Simulation FPS (default: 30)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (default: cuda)")
    parser.add_argument("--scene", type=str, default=None,
                        help="Scene XML file (default: auto-detect from policy)")
    parser.add_argument("--verbose", action="store_true", help="Print per-episode results")
    parser.add_argument("--wandb_project", type=str, default=None, help="Log to WandB project")
    parser.add_argument("--wandb_run", type=str, default=None, help="WandB run name")

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load policy
    print(f"Loading SmolVLA model from: {args.model_path}")
    model_path = Path(args.model_path)

    if model_path.exists():
        # Load from local path
        policy = SmolVLAPolicy.from_pretrained(str(model_path))
    else:
        # Try loading from HuggingFace
        policy = SmolVLAPolicy.from_pretrained(args.model_path)

    policy.eval()
    policy.to(device)

    # Get action dimension to determine action space
    action_dim = 6  # Default to joint space
    try:
        action_shape = policy.config.output_features['action'].shape
        action_dim = action_shape[0] if action_shape else 6
    except:
        pass

    is_ee_action_space = action_dim == 8
    print(f"Action space: {'EE (8-dim)' if is_ee_action_space else f'Joint ({action_dim}-dim)'}")

    # Detect depth cameras
    depth_cameras = []
    try:
        for key in policy.config.input_features.keys():
            if "_depth" in key:
                cam_name = key.replace("observation.images.", "")
                depth_cameras.append(cam_name)
    except:
        pass

    if depth_cameras:
        print(f"Depth cameras: {depth_cameras}")

    # Create preprocessor/postprocessor
    # Try to load stats from model directory
    stats = None
    stats_path = model_path / "stats.pt" if model_path.exists() else None
    if stats_path and stats_path.exists():
        stats = torch.load(stats_path)

    preprocessor, postprocessor = make_pre_post_processors(policy.config, dataset_stats=stats)

    # Initialize IK solver if using EE actions
    ik_solver = None
    if is_ee_action_space:
        print("Initializing IK solver...")
        ik_solver = IKSolver(verbose=True)

    # Determine scene
    scene_xml = args.scene
    if scene_xml is None:
        scene_xml = "so101_rgbd.xml" if depth_cameras else "so101_sim.xml"
    print(f"Using scene: {scene_xml}")

    # Import simulation
    from lerobot_robot_sim import SO100SimConfig, SO100Sim

    # Create simulation
    config = SO100SimConfig(
        fps=args.fps,
        enable_vr=False,
        randomize_position=args.randomize,
        randomize_rotation=args.randomize,
        scene_xml=scene_xml,
    )
    sim_robot = SO100Sim(config)
    sim_robot.connect()

    # Initialize WandB if requested
    if args.wandb_project:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run or f"smolvla_eval_{Path(args.model_path).name}",
            config={
                "model_path": args.model_path,
                "num_episodes": args.num_episodes,
                "randomize": args.randomize,
                "language": args.language,
                "max_steps": args.max_steps,
                "fps": args.fps,
                "action_space": "EE" if is_ee_action_space else "Joint",
            },
        )

    # Run episodes
    print(f"\nRunning {args.num_episodes} episodes...")
    print(f"Language instruction: \"{args.language}\"")
    print(f"Randomization: {'enabled' if args.randomize else 'disabled'}")
    print()

    successes = 0
    total_steps = 0
    total_time = 0
    results = []

    for ep in range(args.num_episodes):
        if args.verbose:
            print(f"Episode {ep + 1}/{args.num_episodes}:")

        success, steps, elapsed = run_episode(
            sim_robot=sim_robot,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
            language_instruction=args.language,
            is_ee_action_space=is_ee_action_space,
            ik_solver=ik_solver,
            depth_cameras=depth_cameras,
            max_steps=args.max_steps,
            verbose=args.verbose,
        )

        results.append({
            "episode": ep + 1,
            "success": success,
            "steps": steps,
            "time": elapsed,
        })

        if success:
            successes += 1
        total_steps += steps
        total_time += elapsed

        if args.wandb_project:
            import wandb
            wandb.log({
                "episode": ep + 1,
                "success": int(success),
                "steps": steps,
                "time": elapsed,
            })

    sim_robot.disconnect()

    # Print summary
    success_rate = successes / args.num_episodes
    avg_steps = total_steps / args.num_episodes
    avg_time = total_time / args.num_episodes

    print()
    print("=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Language: \"{args.language}\"")
    print(f"Episodes: {args.num_episodes}")
    print(f"Randomization: {'enabled' if args.randomize else 'disabled'}")
    print()
    print(f"Success rate: {successes}/{args.num_episodes} ({100*success_rate:.1f}%)")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Average time: {avg_time:.2f}s")
    print("=" * 50)

    # Report IK stats if applicable
    if ik_solver:
        stats = ik_solver.get_stats()
        if stats["total_count"] > 0:
            print(f"\nIK Statistics:")
            print(f"  Failures: {stats['failure_count']}/{stats['total_count']} ({100*stats['failure_rate']:.2f}%)")
            print(f"  Avg error: {stats['avg_error_mm']:.2f}mm")

    if args.wandb_project:
        import wandb
        wandb.log({
            "summary/success_rate": success_rate,
            "summary/avg_steps": avg_steps,
            "summary/avg_time": avg_time,
        })
        wandb.finish()

    return success_rate


if __name__ == "__main__":
    main()
