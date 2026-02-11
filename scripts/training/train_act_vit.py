#!/usr/bin/env python
"""
Train an ACT policy with Vision Transformer backbone on a LeRobot dataset.

This is a variant of train_act.py that uses ViT instead of ResNet for image encoding.

Usage:
    python train_act_vit.py danbhf/sim_pick_place_157ep
    python train_act_vit.py danbhf/sim_pick_place_157ep --steps 50000
"""

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

# Add project root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.factory import make_pre_post_processors

# Import our custom ViT ACT model
from models.act_vit import ACTViTPolicy

# Import shared utilities
from utils.constants import MOTOR_NAMES
from utils.training import (
    CachedDataset,
    DiskCachedDataset,
    PickupCoordinateDataset,
    SubtaskDataset,
    DeltaActionDataset,
    EpisodeFilterDataset,
    FixedStateDataset,
    SubtaskChunkDataset,
    cycle,
    prepare_obs_for_policy,
    run_evaluation,
    save_checkpoint,
    load_checkpoint,
    get_scene_metadata,
)


def main():
    parser = argparse.ArgumentParser(description="Train ACT-ViT policy on LeRobot dataset")
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
    parser.add_argument("--cache_dataset", action="store_true", help="Pre-cache dataset in memory")
    parser.add_argument("--disk_cache", action="store_true", help="Cache decoded frames to disk")
    parser.add_argument("--cache_image_size", type=int, default=None, help="Resize images during caching")
    parser.add_argument("--cameras", type=str, default=None, help="Comma-separated camera names to use")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint directory to resume from")
    parser.add_argument("--pickup_coords", action="store_true", help="Add pickup location conditioning")
    parser.add_argument("--subtask", action="store_true", help="Add subtask phase conditioning (requires subtask_annotations.json)")
    parser.add_argument("--pos1_only", action="store_true", help="Filter to position 1 episodes only")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze ViT backbone (only train projection layers)")
    parser.add_argument("--vit_model", type=str, default="vit_b_16",
                        choices=["vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32"],
                        help="ViT model variant (default: vit_b_16). vit_b_32 has fewer patches (49 vs 196)")
    parser.add_argument("--delta_actions", action="store_true",
                        help="Use delta/relative actions instead of absolute (predicts frame-to-frame changes)")
    parser.add_argument("--blinkering", action="store_true",
                        help="Enable blinkering: mask overhead camera during PICK_UP/DROP subtasks (requires --subtask)")
    parser.add_argument("--fix_state", action="store_true",
                        help="Fix observation.state bug: replace duplo position with commanded joint positions")
    parser.add_argument("--subtask_chunks", action="store_true",
                        help="Enable completion head for subtask progress prediction (requires --subtask). "
                             "By default also masks action loss at subtask boundaries; use --no_mask_actions to disable masking.")
    parser.add_argument("--no_mask_actions", action="store_true",
                        help="With --subtask_chunks: disable action masking, only use completion head as auxiliary task. "
                             "Actions are supervised across subtask boundaries (like normal training).")
    parser.add_argument("--completion_weight", type=float, default=0.1,
                        help="Weight for completion loss (default: 0.1)")

    args = parser.parse_args()

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"outputs/train/act_vit_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load dataset metadata
    print(f"Loading dataset metadata: {args.dataset}")
    dataset_metadata = LeRobotDatasetMetadata(args.dataset)

    # Get features
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}

    # Default: use 'action', exclude 'action_joints'
    if 'action_joints' in output_features:
        del output_features['action_joints']

    input_features = {key: ft for key, ft in features.items()
                      if key not in output_features and key != 'action_joints' and key != 'action'}

    # Filter cameras if specified
    if args.cameras:
        selected_cams = [c.strip() for c in args.cameras.split(",")]
        print(f"Selected cameras: {selected_cams}")
        input_features = {
            key: ft for key, ft in input_features.items()
            if not key.startswith("observation.images.")
            or key.replace("observation.images.", "") in selected_cams
        }

    # Load pickup coordinate conditioning
    episode_scenes = None
    pickup_coord_stats = None
    pos1_episode_indices = None

    if args.pickup_coords or args.pos1_only:
        print("Loading episode_scenes.json...")
        episode_scenes = PickupCoordinateDataset.load_episode_scenes(args.dataset)

        if episode_scenes:
            if args.pos1_only:
                pos1_episode_indices = set()
                for ep_idx_str, scene_info in episode_scenes.items():
                    try:
                        y = scene_info['objects']['duplo']['position']['y']
                        if y > 0.1:
                            pos1_episode_indices.add(int(ep_idx_str))
                    except (KeyError, TypeError):
                        pass
                print(f"  Filtering to position 1 only: {len(pos1_episode_indices)} episodes")
                episode_scenes = {k: v for k, v in episode_scenes.items() if int(k) in pos1_episode_indices}

            if args.pickup_coords:
                from lerobot.configs.types import PolicyFeature
                input_features['observation.environment_state'] = PolicyFeature(
                    type=FeatureType.STATE,
                    shape=(2,),
                )
                pickup_coord_stats = PickupCoordinateDataset.compute_stats(episode_scenes)
                print(f"  Loaded coordinates for {len(episode_scenes)} episodes")
        else:
            print("  WARNING: No episode_scenes found")
            if args.pickup_coords:
                args.pickup_coords = False
            if args.pos1_only:
                args.pos1_only = False

    # Load subtask conditioning
    subtask_annotations = None
    subtask_stats = None

    if args.subtask:
        print("Loading subtask_annotations.json...")
        # Try local path first, then HuggingFace
        local_path = Path(f"datasets/{args.dataset.split('/')[-1]}")
        if local_path.exists():
            subtask_annotations = SubtaskDataset.load_annotations_local(str(local_path))
        if not subtask_annotations:
            subtask_annotations = SubtaskDataset.load_annotations(args.dataset)

        if subtask_annotations:
            # Subtask is added to observation.environment_state (concatenated with pickup_coords if present)
            # Determine combined feature size:
            # - pickup_coords only: 2 dims
            # - subtask only: 4 dims (one-hot)
            # - both: 6 dims
            env_state_dim = SubtaskDataset.NUM_SUBTASKS  # 4 for subtask one-hot
            if args.pickup_coords and episode_scenes:
                env_state_dim += 2  # Add pickup coord dims

            from lerobot.configs.types import PolicyFeature
            # Override/set the environment_state feature with combined size
            input_features['observation.environment_state'] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(env_state_dim,),
            )
            subtask_stats = SubtaskDataset.compute_stats(use_one_hot=True)
            # Merge subtask stats with pickup_coord stats if needed
            if pickup_coord_stats:
                # Combine stats: pickup_coords (2) + subtask (4) = 6 dims
                combined_mean = torch.cat([
                    pickup_coord_stats['observation.environment_state']['mean'],
                    subtask_stats['observation.environment_state']['mean']
                ])
                combined_std = torch.cat([
                    pickup_coord_stats['observation.environment_state']['std'],
                    subtask_stats['observation.environment_state']['std']
                ])
                pickup_coord_stats['observation.environment_state']['mean'] = combined_mean
                pickup_coord_stats['observation.environment_state']['std'] = combined_std
            else:
                # Use subtask stats alone
                pickup_coord_stats = subtask_stats
            print(f"  Loaded subtask annotations for {len(subtask_annotations)} episodes")
            print(f"  Combined environment_state dim: {env_state_dim}")
        else:
            print("  WARNING: No subtask_annotations found - disabling subtask conditioning")
            args.subtask = False

    print(f"Input features: {list(input_features.keys())}")
    print(f"Output features: {list(output_features.keys())}")

    # Compute delta action stats if enabled
    delta_action_stats = None
    if args.delta_actions:
        print("Computing delta action stats...")
        # Use fast path: compute directly from parquet files
        delta_action_stats = DeltaActionDataset.compute_stats(
            args.dataset,  # Pass repo_id string for fast path
            action_dim=output_features['action'].shape[0],
            num_samples=10000
        )

    # Create ACT config
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
    )
    # Store ViT model name as a custom attribute (ACTConfig validates vision_backbone as ResNet)
    cfg.vit_model = args.vit_model
    # Store blinkering flag
    cfg.blinkering = args.blinkering
    # Completion head for subtask chunk truncation
    cfg.use_completion_head = args.subtask_chunks
    cfg.completion_loss_weight = args.completion_weight

    # Create ViT-based ACT policy
    print("Creating ACT-ViT policy...")
    policy = ACTViTPolicy(cfg)

    # Optionally freeze the ViT backbone
    if args.freeze_backbone:
        print("Freezing ViT backbone - only projection layers will be trained")
        for param in policy.model.backbone.parameters():
            param.requires_grad = False

    policy.train()
    policy.to(device)

    # Create processors
    import copy
    stats = copy.deepcopy(dict(dataset_metadata.stats))
    # pickup_coord_stats contains merged stats if both pickup_coords and subtask are enabled
    if pickup_coord_stats:
        stats.update(pickup_coord_stats)
    # Update action stats for delta actions
    if delta_action_stats:
        stats.update(delta_action_stats)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=stats)

    # Count parameters
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    backbone_params = sum(p.numel() for n, p in policy.named_parameters() if "backbone" in n)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Backbone (ViT) parameters: {backbone_params:,}")

    # Set up delta timestamps
    delta_timestamps = {
        "action": [i / dataset_metadata.fps for i in range(args.chunk_size)],
    }
    for key in input_features:
        if key == 'observation.environment_state':
            continue  # Added by dataset wrappers (pickup coords and/or subtask)
        delta_timestamps[key] = [0.0]

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    raw_dataset = LeRobotDataset(args.dataset, delta_timestamps=delta_timestamps)
    print(f"Dataset size: {len(raw_dataset)} frames")

    # Cache handling
    if args.cache_dataset and args.disk_cache:
        print("Warning: Both --cache_dataset and --disk_cache specified, using disk cache")
        args.cache_dataset = False

    if args.disk_cache:
        dataset = DiskCachedDataset(raw_dataset, resize_images_to=args.cache_image_size)
        num_workers = args.num_workers
        pin_memory = device.type != "cpu"
    elif args.cache_dataset:
        dataset = CachedDataset(raw_dataset, resize_images_to=args.cache_image_size)
        num_workers = 0
        pin_memory = False
    else:
        dataset = raw_dataset
        num_workers = args.num_workers
        pin_memory = device.type != "cpu"

    # Filter episodes if needed
    if args.pos1_only and pos1_episode_indices:
        dataset = EpisodeFilterDataset(dataset, pos1_episode_indices)

    # Add pickup coordinates
    if args.pickup_coords and episode_scenes:
        dataset = PickupCoordinateDataset(dataset, episode_scenes)

    # Add subtask conditioning
    if args.subtask and subtask_annotations:
        dataset = SubtaskDataset(dataset, subtask_annotations, use_one_hot=True)

    # Fix observation.state bug (replace duplo position with commanded joints)
    if args.fix_state:
        dataset = FixedStateDataset(dataset)

    # Add completion head labels (and optionally mask actions at subtask boundaries)
    if args.subtask_chunks:
        if not args.subtask or not subtask_annotations:
            print("WARNING: --subtask_chunks requires --subtask with valid annotations. Skipping.")
        else:
            mask_actions = not args.no_mask_actions
            dataset = SubtaskChunkDataset(dataset, subtask_annotations,
                                          chunk_size=args.chunk_size, mask_actions=mask_actions)

    # Add delta action transformation
    if args.delta_actions:
        action_dim = output_features['action'].shape[0]
        dataset = DeltaActionDataset(dataset, action_dim=action_dim)

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        drop_last=True,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-7)

    # Resume if specified
    start_step = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"Resuming from checkpoint: {resume_path}")
            start_step = load_checkpoint(resume_path, policy, optimizer, scheduler)
            print(f"  Resumed at step {start_step}")
        else:
            print(f"WARNING: Checkpoint not found: {resume_path}")

    # Extract camera names
    camera_names = [key.replace("observation.images.", "") for key in input_features.keys()
                    if key.startswith("observation.images.")]

    # Action space info
    action_feature = output_features.get('action')
    if action_feature:
        action_shape = list(action_feature.shape) if hasattr(action_feature, 'shape') else action_feature.shape
        action_dim = action_shape[0] if action_shape else 0
        action_space = f"joint ({action_dim}-dim)" if action_dim == 6 else f"unknown ({action_dim}-dim)"
    else:
        action_space = "unknown"
        action_dim = 0

    # Training metadata
    training_metadata = {
        "dataset_repo_id": args.dataset,
        "model_type": "act_vit",
        "vision_backbone": args.vit_model,
        "cameras": camera_names,
        "action_space": action_space,
        "action_dim": action_dim,
        "chunk_size": args.chunk_size,
        "fps": dataset_metadata.fps,
        "total_frames": len(dataset),
        "pickup_coords": args.pickup_coords,
        "delta_actions": args.delta_actions,
        "pos1_only": args.pos1_only,
        "blinkering": args.blinkering,
        "state_bug_fixed": args.fix_state,
        "subtask_chunks": args.subtask_chunks,
        "mask_actions": args.subtask_chunks and not args.no_mask_actions,
        "completion_head": args.subtask_chunks,
        "completion_weight": args.completion_weight if args.subtask_chunks else None,
    }

    # Initialize WandB
    if not args.no_wandb:
        print("Initializing WandB...")
        wandb.init(
            project=args.wandb_project,
            name=args.run_name or f"act_vit_{args.dataset.split('/')[-1]}",
            config={
                "model_type": "act_vit",
                "vision_backbone": args.vit_model,
                "dataset": args.dataset,
                "training_steps": args.steps,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "chunk_size": args.chunk_size,
                "total_params": total_params,
                "backbone_params": backbone_params,
                "trainable_params": trainable_params,
                "cameras": camera_names,
                "pickup_coords": args.pickup_coords,
            },
        )

    # Print training info
    print()
    print("=" * 60)
    print("ACT-ViT Training Configuration")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {output_dir}")
    print(f"Training steps: {args.steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Vision backbone: {args.vit_model}")
    print(f"Cameras: {', '.join(camera_names)}")
    print(f"Pickup coords: {'enabled' if args.pickup_coords else 'disabled'}")
    print(f"Delta actions: {'enabled' if args.delta_actions else 'disabled'}")
    print(f"Blinkering: {'enabled' if args.blinkering else 'disabled'}")
    print(f"State fix: {'enabled' if args.fix_state else 'disabled (buggy duplo state)'}")
    if args.subtask_chunks:
        mask_str = "action masking + " if not args.no_mask_actions else "auxiliary "
        print(f"Subtask chunks: enabled ({mask_str}completion head, weight={args.completion_weight})")
    else:
        print(f"Subtask chunks: disabled")
    print(f"Frozen backbone: {'yes' if args.freeze_backbone else 'no'}")
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

        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Preserve custom keys that the preprocessor strips
        action_mask = batch.get("action_mask")
        completion_progress = batch.get("completion_progress")

        batch = preprocessor(batch)

        # Restore custom keys after preprocessing
        if action_mask is not None:
            batch["action_mask"] = action_mask
        if completion_progress is not None:
            batch["completion_progress"] = completion_progress

        # Fix tensor dimensions
        if "observation.state" in batch and isinstance(batch["observation.state"], torch.Tensor):
            if batch["observation.state"].dim() == 3:
                batch["observation.state"] = batch["observation.state"].squeeze(1)

        if "observation.environment_state" in batch and isinstance(batch["observation.environment_state"], torch.Tensor):
            if batch["observation.environment_state"].dim() == 3:
                batch["observation.environment_state"] = batch["observation.environment_state"].squeeze(1)

        # Forward pass
        loss, output_dict = policy.forward(batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()

        # Update metrics
        running_loss += loss.item()
        if "kld_loss" in output_dict:
            running_kl_loss += output_dict["kld_loss"]

        step += 1
        pbar.update(1)

        # WandB logging
        if not args.no_wandb:
            log_dict = {
                "train/loss": loss.item(),
                "train/grad_norm": grad_norm.item(),
                "train/lr": scheduler.get_last_lr()[0],
            }
            if "kld_loss" in output_dict:
                log_dict["train/kl_loss"] = output_dict["kld_loss"]
            if "completion_loss" in output_dict:
                log_dict["train/completion_loss"] = output_dict["completion_loss"]
            wandb.log(log_dict, step=step)

        # Console logging
        if step % args.log_freq == 0:
            avg_loss = running_loss / args.log_freq
            avg_kl_loss = running_kl_loss / args.log_freq if running_kl_loss > 0 else 0
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            eta_minutes = (args.steps - step) / steps_per_sec / 60

            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'kl': f'{avg_kl_loss:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                'eta': f'{eta_minutes:.1f}m'
            })

            if not args.no_wandb:
                wandb.log({
                    "train/avg_loss": avg_loss,
                    "train/avg_kl_loss": avg_kl_loss,
                }, step=step)

            running_loss = 0.0
            running_kl_loss = 0.0

            if avg_loss < best_loss:
                best_loss = avg_loss

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

    if not args.no_wandb:
        wandb.log({
            "final/total_time_minutes": elapsed / 60,
            "final/best_loss": best_loss,
        })
        wandb.finish()


if __name__ == "__main__":
    main()
