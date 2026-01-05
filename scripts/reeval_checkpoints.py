#!/usr/bin/env python
"""
Re-evaluate all checkpoints from a training run and upload results to WandB.

Usage:
    python scripts/reeval_checkpoints.py outputs/train/act_20260104_165201 --episodes 30
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import wandb

# Add project paths
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from lerobot_robot_sim import SO100Sim, SO100SimConfig, MOTOR_NAMES
from utils.ik_solver import IKSolver


def prepare_obs_for_policy(obs: dict, device: torch.device) -> dict:
    """Convert simulation observation to policy input format."""
    batch = {}

    # State (joint positions)
    state = [obs[f"{name}.pos"] for name in MOTOR_NAMES]
    batch["observation.state"] = torch.tensor([state], dtype=torch.float32, device=device)

    # Camera images
    for key, value in obs.items():
        if isinstance(value, np.ndarray) and value.ndim == 3:
            # Check if this is a depth image (grayscale stored as 3-channel)
            if "_depth" in key:
                # Depth: take first channel, normalize to 0-1, repeat to 3 channels for CNN
                # Stored as 0-255 = 0-2m range
                depth_uint8 = value[:, :, 0]  # Take first channel
                img = torch.from_numpy(depth_uint8).unsqueeze(0).unsqueeze(0).float() / 255.0
                img = img.repeat(1, 3, 1, 1)  # [1, 1, H, W] -> [1, 3, H, W] for CNN compatibility
            else:
                # RGB image: normalize to [0, 1]
                img = torch.from_numpy(value).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            batch[f"observation.images.{key}"] = img.to(device)

    return batch


def run_evaluation(policy, preprocessor, postprocessor, device, num_episodes: int,
                   randomize: bool = True, fps: int = 30, depth_cameras: list = None):
    """Run evaluation episodes in simulation."""

    # Detect action space from policy config
    action_dim = None
    for key, ft in policy.config.output_features.items():
        if key == "action":
            action_dim = ft.shape[0]
            break

    use_ee_actions = (action_dim == 8)

    if use_ee_actions:
        print("  Using EE action space (8-dim) with IK conversion")
    else:
        print("  Using joint action space (6-dim)")

    # Setup depth cameras
    if depth_cameras is None:
        depth_cameras = []
    if depth_cameras:
        print(f"  Using depth cameras: {depth_cameras}")

    # Initialize simulation
    sim_config = SO100SimConfig(
        id="evaluator",
        sim_cameras=["wrist_cam", "overhead_cam"],
        depth_cameras=depth_cameras,
        camera_width=640,
        camera_height=480,
        enable_vr=False,
        n_sim_steps=10,
    )
    sim_robot = SO100Sim(sim_config)
    sim_robot.connect()

    # Initialize IK if using EE actions
    ik_solver = None
    if use_ee_actions:
        ik_solver = IKSolver()

    results = []
    ik_failures = 0
    ik_errors = []
    total_steps = 0

    for ep in range(num_episodes):
        sim_robot.reset_scene(randomize=randomize)

        episode_start = time.time()
        success = False
        steps = 0
        max_steps = 300

        while steps < max_steps:
            # Get observation
            obs = sim_robot.get_observation()

            # Prepare for policy
            batch = prepare_obs_for_policy(obs, device)
            batch = preprocessor(batch)

            # Fix tensor dimensions
            if "observation.state" in batch and batch["observation.state"].dim() == 3:
                batch["observation.state"] = batch["observation.state"].squeeze(1)

            # Get action from policy
            with torch.no_grad():
                action = policy.select_action(batch)

            # Denormalize action
            action = postprocessor(action)

            # Convert action to numpy
            action_np = action.cpu().numpy()
            if action_np.ndim > 1:
                action_np = action_np.flatten()

            # Convert to action dict
            if use_ee_actions:
                # EE action: [x, y, z, qw, qx, qy, qz, gripper]
                action_np = action_np[:8]  # Take first 8 values

                # Use IKSolver which returns normalized values for sim
                joint_action, ik_error, ik_failed = ik_solver.ee_to_joint_action(action_np, return_normalized=True)

                if ik_failed:
                    ik_failures += 1
                    steps += 1
                    total_steps += 1
                    continue

                if ik_error is not None:
                    ik_errors.append(ik_error * 1000)  # Convert to mm

                action_dict = {f"{MOTOR_NAMES[i]}.pos": float(joint_action[i]) for i in range(6)}
            else:
                # Joint action: direct normalized values
                joint_action = action_np[:6]  # Take first 6 values
                action_dict = {f"{MOTOR_NAMES[i]}.pos": float(joint_action[i]) for i in range(6)}

            # Send to sim
            sim_robot.send_action(action_dict)

            # Check success
            if sim_robot.is_task_complete():
                success = True
                break

            steps += 1
            total_steps += 1

        episode_time = time.time() - episode_start
        results.append({
            "success": success,
            "steps": steps,
            "time": episode_time,
        })

    # Cleanup
    sim_robot.disconnect()

    # Compute stats
    success_rate = sum(r["success"] for r in results) / len(results) * 100
    avg_steps = sum(r["steps"] for r in results) / len(results)
    avg_time = sum(r["time"] for r in results) / len(results)
    ik_failure_rate = ik_failures / total_steps * 100 if total_steps > 0 else 0
    avg_ik_error = np.mean(ik_errors) if ik_errors else 0

    return {
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "avg_time": avg_time,
        "ik_failure_rate": ik_failure_rate,
        "avg_ik_error_mm": avg_ik_error,
    }


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate checkpoints and upload to WandB")
    parser.add_argument("train_dir", type=str, help="Training output directory")
    parser.add_argument("--episodes", type=int, default=30, help="Episodes per checkpoint")
    parser.add_argument("--randomize", action="store_true", default=True, help="Randomize object positions")
    parser.add_argument("--no_randomize", action="store_true", help="Disable randomization")
    parser.add_argument("--wandb_project", type=str, default="lerobot-thesis", help="WandB project name")
    parser.add_argument("--run_name", type=str, default=None, help="WandB run name (default: reeval_<dir_name>)")

    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    if not train_dir.exists():
        print(f"ERROR: Training directory not found: {train_dir}")
        sys.exit(1)

    randomize = not args.no_randomize

    # Find checkpoints
    checkpoints = sorted(train_dir.glob("checkpoint_*"))
    if (train_dir / "final").exists():
        checkpoints.append(train_dir / "final")

    if not checkpoints:
        print(f"ERROR: No checkpoints found in {train_dir}")
        sys.exit(1)

    print(f"Found {len(checkpoints)} checkpoints")
    for cp in checkpoints:
        print(f"  - {cp.name}")

    # Load first checkpoint to get config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    first_policy = ACTPolicy.from_pretrained(str(checkpoints[0]))

    # Get dataset info for normalization stats
    # Try to find dataset from policy config or infer from directory name
    dataset_name = None

    # Check if there's a config file with dataset info
    config_file = train_dir / "config.json"
    if config_file.exists():
        import json
        with open(config_file) as f:
            config = json.load(f)
            dataset_name = config.get("dataset")

    if not dataset_name:
        # Determine action space from policy to pick correct dataset
        action_dim = None
        for key, ft in first_policy.config.output_features.items():
            if key == "action":
                action_dim = ft.shape[0]
                break

        # Check for depth cameras to determine if RGBD dataset
        has_depth = any(key.endswith("_depth") for key in first_policy.config.input_features)

        if action_dim == 8:  # EE action space
            if has_depth:
                dataset_name = "danbhf/sim_pick_place_40ep_rgbd_ee"
            else:
                dataset_name = "danbhf/sim_pick_place_merged_40ep_ee_2"
        else:  # Joint action space
            dataset_name = "danbhf/sim_pick_place_merged_40ep"

    print(f"Using dataset for normalization: {dataset_name}")
    dataset_metadata = LeRobotDatasetMetadata(dataset_name)

    # Detect depth cameras from policy config
    depth_cameras = []
    for key in first_policy.config.input_features:
        if key.startswith("observation.images.") and key.endswith("_depth"):
            cam_name = key.replace("observation.images.", "").replace("_depth", "")
            if cam_name not in depth_cameras:
                depth_cameras.append(cam_name)

    # Extract camera names
    camera_names = [key.replace("observation.images.", "") for key in first_policy.config.input_features
                    if key.startswith("observation.images.")]

    # Determine action space
    action_dim = None
    for key, ft in first_policy.config.output_features.items():
        if key == "action":
            action_dim = ft.shape[0]
            break

    if action_dim == 8:
        action_space = "end-effector (8-dim)"
    elif action_dim == 6:
        action_space = "joint (6-dim)"
    else:
        action_space = f"unknown ({action_dim}-dim)"

    # Initialize WandB
    run_name = args.run_name or f"reeval_{train_dir.name}"
    print(f"\nInitializing WandB run: {run_name}")

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "train_dir": str(train_dir),
            "num_checkpoints": len(checkpoints),
            "episodes_per_checkpoint": args.episodes,
            "randomize": randomize,
            "dataset": dataset_name,
            "cameras": camera_names,
            "depth_cameras": depth_cameras,
            "action_space": action_space,
            "reeval": True,  # Mark this as a re-evaluation run
        },
    )

    print(f"\n{'='*60}")
    print("Re-evaluation Configuration")
    print(f"{'='*60}")
    print(f"Training dir: {train_dir}")
    print(f"Checkpoints: {len(checkpoints)}")
    print(f"Episodes per checkpoint: {args.episodes}")
    print(f"Randomize: {randomize}")
    print(f"Cameras: {', '.join(camera_names)}")
    print(f"Depth cameras: {', '.join(depth_cameras) if depth_cameras else 'none'}")
    print(f"Action space: {action_space}")
    print(f"{'='*60}\n")

    # Evaluate each checkpoint
    all_results = []

    for i, checkpoint_path in enumerate(checkpoints):
        checkpoint_name = checkpoint_path.name

        # Extract step number
        if checkpoint_name.startswith("checkpoint_"):
            step = int(checkpoint_name.split("_")[1])
        else:
            step = 50000  # Assume final is at 50k

        print(f"\n[{i+1}/{len(checkpoints)}] Evaluating {checkpoint_name} (step {step})...")

        # Load policy
        policy = ACTPolicy.from_pretrained(str(checkpoint_path))
        policy.eval()
        policy.to(device)

        # Create preprocessor/postprocessor
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config, dataset_stats=dataset_metadata.stats
        )

        # Run evaluation
        results = run_evaluation(
            policy, preprocessor, postprocessor, device,
            num_episodes=args.episodes,
            randomize=randomize,
            depth_cameras=depth_cameras,
        )

        # Log to WandB
        wandb.log({
            "checkpoint/step": step,
            "eval/success_rate": results["success_rate"],
            "eval/avg_steps": results["avg_steps"],
            "eval/avg_time": results["avg_time"],
            "eval/ik_failure_rate": results["ik_failure_rate"],
            "eval/avg_ik_error_mm": results["avg_ik_error_mm"],
        })

        print(f"  Success rate: {results['success_rate']:.1f}%")
        print(f"  Avg steps: {results['avg_steps']:.1f}")
        print(f"  Avg time: {results['avg_time']:.2f}s")
        if results["ik_failure_rate"] > 0:
            print(f"  IK failures: {results['ik_failure_rate']:.2f}%")

        all_results.append({
            "checkpoint": checkpoint_name,
            "step": step,
            **results,
        })

    # Summary
    print(f"\n{'='*60}")
    print("Re-evaluation Complete")
    print(f"{'='*60}")

    # Find best checkpoint
    best = max(all_results, key=lambda x: x["success_rate"])
    print(f"Best checkpoint: {best['checkpoint']} ({best['success_rate']:.1f}% success)")

    # Log summary to WandB
    wandb.summary["best_checkpoint"] = best["checkpoint"]
    wandb.summary["best_success_rate"] = best["success_rate"]
    wandb.summary["best_step"] = best["step"]

    # Create results table
    table = wandb.Table(columns=["checkpoint", "step", "success_rate", "avg_steps", "avg_time", "ik_failure_rate", "ik_error_mm"])
    for r in all_results:
        table.add_data(r["checkpoint"], r["step"], r["success_rate"], r["avg_steps"], r["avg_time"], r["ik_failure_rate"], r["avg_ik_error_mm"])
    wandb.log({"results_table": table})

    wandb.finish()
    print("\nResults uploaded to WandB!")


if __name__ == "__main__":
    main()
