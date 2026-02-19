#!/usr/bin/env python
"""
Train an ACT (Action Chunking Transformer) policy on a LeRobot dataset.

Based on LeRobot examples and lerobot-scratch training scripts.

Usage:
    python train_act.py danbhf/sim_pick_place_20251229_101340
    python train_act.py danbhf/sim_pick_place_20251229_101340 --steps 50000
    python train_act.py danbhf/sim_pick_place_20251229_101340 --batch_size 4 --steps 5000
"""

# Suppress noisy warnings before importing anything else
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io")
warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")
warnings.filterwarnings("ignore", message=".*video decoding.*deprecated.*")
warnings.filterwarnings("ignore", message=".*NumPy version.*SciPy.*")

import argparse
from pathlib import Path
from datetime import datetime
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

# Add project root to path for shared utilities
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

# Import shared utilities
from utils.constants import MOTOR_NAMES
from utils.ik_solver import IKSolver
from utils.training import (
    CachedDataset,
    DiskCachedDataset,
    EpisodeFilterDataset,
    PickupCoordinateDataset,
    cycle,
    prepare_obs_for_policy,
    run_evaluation,
    save_checkpoint,
    load_checkpoint,
    get_scene_metadata,
)


def main():
    parser = argparse.ArgumentParser(description="Train ACT policy on LeRobot dataset")
    parser.add_argument("dataset", type=str, help="HuggingFace dataset repo ID")
    parser.add_argument("--steps", type=int, default=50000, help="Training steps (default: 50000)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (default: 1e-5)")
    parser.add_argument("--chunk_size", type=int, default=100, help="ACT chunk size (default: 100)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--log_freq", type=int, default=100, help="Log frequency (default: 100)")
    parser.add_argument("--save_freq", type=int, default=5000, help="Checkpoint save frequency (default: 5000)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (default: cuda)")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers (default: 4)")
    parser.add_argument("--wandb_project", type=str, default="lerobot-thesis", help="WandB project name")
    parser.add_argument("--run_name", type=str, default=None, help="WandB run name (default: auto-generated)")
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--eval_episodes", type=int, default=0, help="Evaluation episodes per checkpoint (0=disabled)")
    parser.add_argument("--eval_randomize", action="store_true", help="Randomize object position during eval")
    parser.add_argument("--cache_dataset", action="store_true", help="Pre-cache dataset in memory (eliminates GPU idle time, high RAM usage)")
    parser.add_argument("--disk_cache", action="store_true", help="Cache decoded frames to disk (fast training, low RAM, persists across runs)")
    parser.add_argument("--cache_image_size", type=int, default=None, help="Resize images during caching (e.g., 224)")
    parser.add_argument("--use_joint_actions", action="store_true", help="Use action_joints instead of action (for EE datasets with preserved joint actions)")
    parser.add_argument("--cameras", type=str, default=None,
                        help="Comma-separated camera names to use (e.g., 'wrist_cam,overhead_cam,overhead_cam_depth'). Default: all available")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint directory to resume training from")
    parser.add_argument("--pickup_coords", action="store_true",
                        help="Add pickup location (x,y) conditioning from episode_scenes.json")
    parser.add_argument("--pos1_only", action="store_true",
                        help="Filter dataset to only include position 1 episodes (Y > 0.1)")
    parser.add_argument("--episode_filter", type=str, default=None,
                        help="Comma-separated episode ranges to include, e.g. '0-19,100-119,200-219'")

    args = parser.parse_args()

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"outputs/train/act_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load dataset metadata to get feature shapes
    print(f"Loading dataset metadata: {args.dataset}")
    dataset_metadata = LeRobotDatasetMetadata(args.dataset)

    # Get features for policy
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}

    # Handle action field selection for EE datasets with preserved joint actions
    if args.use_joint_actions:
        if 'action_joints' not in output_features:
            print("ERROR: --use_joint_actions specified but dataset has no 'action_joints' field")
            sys.exit(1)
        # Use action_joints as the action, remove original action
        print("Using 'action_joints' (6-dim joint space) instead of 'action'")
        if 'action' in output_features:
            del output_features['action']
        # Rename action_joints to action for the policy
        output_features['action'] = output_features.pop('action_joints')
    else:
        # Default: use 'action', exclude 'action_joints'
        if 'action_joints' in output_features:
            del output_features['action_joints']

    # Exclude action_joints from input features (it's only for comparison during playback)
    input_features = {key: ft for key, ft in features.items()
                      if key not in output_features and key != 'action_joints' and key != 'action'}

    # Filter cameras if --cameras specified
    if args.cameras:
        selected_cams = [c.strip() for c in args.cameras.split(",")]
        print(f"Selected cameras: {selected_cams}")
        # Keep only observation.images.* keys that exactly match selected cameras
        input_features = {
            key: ft for key, ft in input_features.items()
            if not key.startswith("observation.images.")
            or key.replace("observation.images.", "") in selected_cams
        }

    # Load pickup coordinate conditioning if enabled
    episode_scenes = None
    pickup_coord_stats = None
    pos1_episode_indices = None  # For --pos1_only filtering

    if args.pickup_coords or args.pos1_only:
        print("Loading episode_scenes.json...")
        episode_scenes = PickupCoordinateDataset.load_episode_scenes(args.dataset)

        if episode_scenes:
            # Filter to position 1 only if requested
            if args.pos1_only:
                pos1_episode_indices = set()
                for ep_idx_str, scene_info in episode_scenes.items():
                    try:
                        y = scene_info['objects']['duplo']['position']['y']
                        if y > 0.1:  # Position 1
                            pos1_episode_indices.add(int(ep_idx_str))
                    except (KeyError, TypeError):
                        pass
                print(f"  Filtering to position 1 only: {len(pos1_episode_indices)} episodes")
                # Filter episode_scenes too
                episode_scenes = {k: v for k, v in episode_scenes.items() if int(k) in pos1_episode_indices}

            if args.pickup_coords:
                # Add observation.environment_state to input features for pickup coordinates
                from lerobot.configs.types import PolicyFeature
                input_features['observation.environment_state'] = PolicyFeature(
                    type=FeatureType.STATE,
                    shape=(2,),  # (pickup_x, pickup_y)
                )
                # Compute normalization stats
                pickup_coord_stats = PickupCoordinateDataset.compute_stats(episode_scenes)
                print(f"  Loaded coordinates for {len(episode_scenes)} episodes")
        else:
            print("  WARNING: No episode_scenes found")
            if args.pickup_coords:
                print("  Disabling pickup_coords")
                args.pickup_coords = False
            if args.pos1_only:
                print("  Cannot filter by position without episode_scenes!")
                args.pos1_only = False

    print(f"Input features: {list(input_features.keys())}")
    print(f"Output features: {list(output_features.keys())}")

    # Create ACT policy config
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
    )

    # Create policy
    print("Creating ACT policy...")
    policy = ACTPolicy(cfg)
    policy.train()
    policy.to(device)

    # Create pre/post processors for normalization
    # If using joint actions, rename stats key from action_joints to action
    import copy
    stats = copy.deepcopy(dict(dataset_metadata.stats))  # Deep copy to avoid modifying original
    if args.use_joint_actions:
        if 'action_joints' in stats:
            stats['action'] = stats.pop('action_joints')
        elif 'action' in stats:
            # Remove EE action stats if present (wrong shape)
            del stats['action']
    # Add pickup coordinate stats if enabled
    if pickup_coord_stats:
        stats.update(pickup_coord_stats)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=stats)

    # Count parameters
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Set up delta timestamps for ACT
    delta_timestamps = {
        "action": [i / dataset_metadata.fps for i in range(args.chunk_size)],
    }
    # If using joint actions, also need to load action_joints with chunk timestamps
    if args.use_joint_actions:
        delta_timestamps["action_joints"] = [i / dataset_metadata.fps for i in range(args.chunk_size)]
    for key in input_features:
        # Skip observation.environment_state - it's added by PickupCoordinateDataset wrapper, not in raw dataset
        if key == 'observation.environment_state':
            continue
        delta_timestamps[key] = [0.0]

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    raw_dataset = LeRobotDataset(args.dataset, delta_timestamps=delta_timestamps)
    print(f"Dataset size: {len(raw_dataset)} frames")

    # Optionally cache dataset to eliminate video decoding bottleneck
    if args.cache_dataset and args.disk_cache:
        print("Warning: Both --cache_dataset and --disk_cache specified, using disk cache")
        args.cache_dataset = False

    if args.disk_cache:
        # Disk cache: persists across runs, supports num_workers > 0
        dataset = DiskCachedDataset(raw_dataset, resize_images_to=args.cache_image_size)
        num_workers = args.num_workers
        pin_memory = device.type != "cpu"
    elif args.cache_dataset:
        # RAM cache: fast but high memory usage, no workers
        dataset = CachedDataset(raw_dataset, resize_images_to=args.cache_image_size)
        num_workers = 0
        pin_memory = False
    else:
        dataset = raw_dataset
        num_workers = args.num_workers
        pin_memory = device.type != "cpu"

    # Filter to position 1 episodes only if requested
    if args.pos1_only and pos1_episode_indices:
        dataset = EpisodeFilterDataset(dataset, pos1_episode_indices)

    # Filter to specific episode ranges if requested
    if args.episode_filter:
        filter_indices = set()
        for part in args.episode_filter.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-", 1)
                filter_indices.update(range(int(start), int(end) + 1))
            else:
                filter_indices.add(int(part))
        print(f"Episode filter: {len(filter_indices)} episodes from ranges '{args.episode_filter}'")
        dataset = EpisodeFilterDataset(dataset, filter_indices)

    # Wrap dataset with pickup coordinate conditioning if enabled
    if args.pickup_coords and episode_scenes:
        dataset = PickupCoordinateDataset(dataset, episode_scenes)

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        drop_last=True,
    )

    # Create optimizer with weight decay
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=1e-7
    )

    # Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"Resuming from checkpoint: {resume_path}")
            start_step = load_checkpoint(resume_path, policy, optimizer, scheduler)
            print(f"  Resumed at step {start_step}")
        else:
            print(f"WARNING: Checkpoint not found: {resume_path}, starting from scratch")

    # Extract camera names from input features
    camera_names = [key.replace("observation.images.", "") for key in input_features.keys()
                    if key.startswith("observation.images.")]

    # Determine action space type
    action_feature = output_features.get('action')
    if action_feature:
        action_dim = action_feature.shape[0] if hasattr(action_feature, 'shape') else len(action_feature.shape)
        # Check the actual shape from the feature
        action_shape = list(action_feature.shape) if hasattr(action_feature, 'shape') else action_feature.shape
        action_dim = action_shape[0] if action_shape else 0
        if action_dim == 8:
            action_space = "end-effector (8-dim: xyz + quat + gripper)"
        elif action_dim == 6:
            action_space = "joint (6-dim: normalized joints)"
        else:
            action_space = f"unknown ({action_dim}-dim)"
    else:
        action_space = "unknown"

    # Extract camera resolutions from input features
    camera_resolutions = {}
    for key, feat in input_features.items():
        if key.startswith("observation.images."):
            cam_name = key.replace("observation.images.", "")
            if hasattr(feat, 'shape') and len(feat.shape) == 3:
                camera_resolutions[cam_name] = f"{feat.shape[2]}x{feat.shape[1]}"

    # Get scene metadata (camera FOVs, positions, etc.)
    scene_meta = get_scene_metadata()

    # Create training metadata to save with checkpoints
    training_metadata = {
        "dataset_repo_id": args.dataset,
        "scene": scene_meta.get("scene_xml", "unknown"),
        "scene_cameras": scene_meta.get("cameras", {}),
        "cameras": camera_names,
        "camera_resolutions": camera_resolutions,
        "action_space": action_space,
        "action_dim": action_dim,
        "chunk_size": args.chunk_size,
        "fps": dataset_metadata.fps,
        "total_frames": len(dataset),
        "pickup_coords": args.pickup_coords,
        "pos1_only": args.pos1_only,
        "episode_filter": args.episode_filter,
    }

    # Initialize WandB
    if not args.no_wandb:
        print("Initializing WandB...")
        wandb.init(
            project=args.wandb_project,
            name=args.run_name or f"act_{args.dataset.split('/')[-1]}",
            config={
                "dataset": args.dataset,
                "training_steps": args.steps,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "chunk_size": args.chunk_size,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "dataset_frames": len(dataset),
                "dataset_fps": dataset_metadata.fps,
                "device": str(device),
                "cache_dataset": args.cache_dataset,
                "disk_cache": args.disk_cache,
                "cache_image_size": args.cache_image_size,
                "cameras": camera_names,
                "action_space": action_space,
                "pickup_coords": args.pickup_coords,
                "pos1_only": args.pos1_only,
                "episode_filter": args.episode_filter,
            },
        )

    # Training info
    print()
    print("=" * 60)
    print("ACT Training Configuration")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {output_dir}")
    print(f"Training steps: {args.steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Device: {device}")
    cache_mode = "disk" if args.disk_cache else ("memory" if args.cache_dataset else "disabled")
    print(f"Dataset caching: {cache_mode}")
    print(f"Cameras: {', '.join(camera_names)}")
    print(f"Action space: {action_space}")
    print(f"Pickup coords: {'enabled' if args.pickup_coords else 'disabled'}")
    print(f"Position 1 only: {'enabled' if args.pos1_only else 'disabled'}")
    print(f"Episode filter: {args.episode_filter or 'disabled'}")
    print(f"WandB: {'disabled' if args.no_wandb else args.wandb_project}")
    print("=" * 60)
    print()

    # Training loop
    print("Starting training...")
    step = start_step
    best_loss = float('inf')
    running_loss = 0.0
    running_kl_loss = 0.0
    start_time = time.time()

    data_iter = cycle(dataloader)
    pbar = tqdm(total=args.steps, initial=start_step, desc="Training")

    while step < args.steps:
        batch = next(data_iter)

        # Move batch to device and preprocess
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # If using joint actions from EE dataset, rename action_joints to action
        if args.use_joint_actions and 'action_joints' in batch:
            batch['action'] = batch.pop('action_joints')

        batch = preprocessor(batch)

        # Fix tensor dimensions for ACT model expectations:
        # State: ACT expects [batch, state_dim], not [batch, n_obs_steps, state_dim]
        if "observation.state" in batch and isinstance(batch["observation.state"], torch.Tensor):
            if batch["observation.state"].dim() == 3:
                batch["observation.state"] = batch["observation.state"].squeeze(1)

        # Same for environment_state (pickup coordinates)
        if "observation.environment_state" in batch and isinstance(batch["observation.environment_state"], torch.Tensor):
            if batch["observation.environment_state"].dim() == 3:
                batch["observation.environment_state"] = batch["observation.environment_state"].squeeze(1)

        # Forward pass
        loss, output_dict = policy.forward(batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)

        optimizer.step()
        scheduler.step()

        # Update metrics
        running_loss += loss.item()
        if "kld_loss" in output_dict:
            running_kl_loss += output_dict["kld_loss"]

        step += 1
        pbar.update(1)

        # Log to WandB every step
        if not args.no_wandb:
            log_dict = {
                "train/loss": loss.item(),
                "train/grad_norm": grad_norm.item(),
                "train/lr": scheduler.get_last_lr()[0],
            }
            if "kld_loss" in output_dict:
                log_dict["train/kl_loss"] = output_dict["kld_loss"]
            wandb.log(log_dict, step=step)

        # Logging
        if step % args.log_freq == 0:
            avg_loss = running_loss / args.log_freq
            avg_kl_loss = running_kl_loss / args.log_freq if running_kl_loss > 0 else 0
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            eta_minutes = (args.steps - step) / steps_per_sec / 60

            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'kl': f'{avg_kl_loss:.4f}',
                'lr': f'{current_lr:.2e}',
                'eta': f'{eta_minutes:.1f}m'
            })

            # Log averaged metrics to WandB
            if not args.no_wandb:
                wandb.log({
                    "train/avg_loss": avg_loss,
                    "train/avg_kl_loss": avg_kl_loss,
                    "train/steps_per_sec": steps_per_sec,
                    "train/eta_minutes": eta_minutes,
                }, step=step)

            running_loss = 0.0
            running_kl_loss = 0.0

            # Track best loss
            if avg_loss < best_loss:
                best_loss = avg_loss
                if not args.no_wandb:
                    wandb.log({"train/best_loss": best_loss}, step=step)

        # Save checkpoint
        if step % args.save_freq == 0 or step == args.steps:
            print(f"\n  Saving checkpoint_{step:06d}")
            save_checkpoint(
                policy, optimizer, scheduler, step, output_dir,
                training_metadata=training_metadata,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                best_loss=best_loss,
            )

            if not args.no_wandb:
                wandb.log({"checkpoint/step": step}, step=step)

            # Run evaluation if enabled
            if args.eval_episodes > 0:
                print(f"\n  Running {args.eval_episodes} evaluation episodes...")
                policy.eval()
                success_rate, avg_steps, avg_time, ik_failure_rate, avg_ik_error, failure_summary = run_evaluation(
                    policy, preprocessor, postprocessor, device,
                    num_episodes=args.eval_episodes,
                    randomize=args.eval_randomize
                )
                policy.train()
                print(f"  Eval: {success_rate*100:.1f}% success, {avg_steps:.1f} avg steps, {avg_time:.2f}s avg time")

                if not args.no_wandb:
                    log_data = {
                        "eval/success_rate": success_rate,
                        "eval/avg_steps": avg_steps,
                        "eval/avg_time": avg_time,
                    }
                    if ik_failure_rate is not None:
                        log_data["eval/ik_failure_rate"] = ik_failure_rate
                    if avg_ik_error is not None:
                        log_data["eval/avg_ik_error_mm"] = avg_ik_error
                    # Log failure analysis metrics
                    if failure_summary:
                        log_data["eval/pick_rate"] = failure_summary.get("pick_rate", 0)
                        log_data["eval/drop_rate"] = failure_summary.get("drop_rate", 0)
                        outcome_counts = failure_summary.get("outcome_counts", {})
                        from utils.failure_analysis import Outcome
                        for outcome in Outcome:
                            count = outcome_counts.get(outcome, 0)
                            log_data[f"eval/outcome_{outcome.value}"] = count
                    wandb.log(log_data, step=step)

    pbar.close()

    # Save final model
    print()
    print("Saving final model...")
    save_checkpoint(
        policy, optimizer, scheduler, args.steps, output_dir,
        training_metadata=training_metadata,
        checkpoint_name="final",
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        best_loss=best_loss,
    )

    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final model: {output_dir / 'final'}")

    # Final evaluation
    if args.eval_episodes > 0:
        print()
        print(f"Running final evaluation ({args.eval_episodes} episodes)...")
        policy.eval()
        success_rate, avg_steps, avg_time, ik_failure_rate, avg_ik_error, failure_summary = run_evaluation(
            policy, preprocessor, postprocessor, device,
            num_episodes=args.eval_episodes,
            randomize=args.eval_randomize
        )
        print(f"Final success rate: {success_rate*100:.1f}%")
        print(f"Final avg steps: {avg_steps:.1f}")
        print(f"Final avg time: {avg_time:.2f}s")
        if ik_failure_rate is not None:
            print(f"Final IK failure rate: {ik_failure_rate*100:.2f}%")
        if avg_ik_error is not None:
            print(f"Final avg IK error: {avg_ik_error:.2f}mm")
        if failure_summary:
            print(f"Final pick rate: {failure_summary.get('pick_rate', 0)*100:.1f}%")
            print(f"Final drop rate: {failure_summary.get('drop_rate', 0)*100:.1f}%")

        if not args.no_wandb:
            log_data = {
                "final/success_rate": success_rate,
                "final/avg_steps": avg_steps,
                "final/avg_time": avg_time,
            }
            if ik_failure_rate is not None:
                log_data["final/ik_failure_rate"] = ik_failure_rate
            if avg_ik_error is not None:
                log_data["final/avg_ik_error_mm"] = avg_ik_error
            if failure_summary:
                log_data["final/pick_rate"] = failure_summary.get("pick_rate", 0)
                log_data["final/drop_rate"] = failure_summary.get("drop_rate", 0)
            wandb.log(log_data)

    # Log final metrics and finish WandB
    if not args.no_wandb:
        wandb.log({
            "final/total_time_minutes": elapsed / 60,
            "final/best_loss": best_loss,
        })
        wandb.finish()


if __name__ == "__main__":
    main()
