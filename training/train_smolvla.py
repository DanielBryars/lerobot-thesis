#!/usr/bin/env python
"""
Train a SmolVLA (Vision-Language-Action) policy on a LeRobot dataset.

SmolVLA is a compact (450M) VLA model that combines:
- SmolVLM2-500M as the vision-language backbone
- A 100M parameter action expert using flow matching

Usage:
    # Finetune from pretrained SmolVLA
    python train_smolvla.py danbhf/sim_pick_place_40ep_rgbd_ee --from_pretrained lerobot/smolvla_base

    # Train action expert from scratch (with pretrained VLM)
    python train_smolvla.py danbhf/sim_pick_place_40ep_rgbd_ee

    # With language instruction
    python train_smolvla.py danbhf/sim_pick_place_40ep_rgbd_ee --language "Pick up the block and place it in the bowl"

References:
    - Paper: https://huggingface.co/papers/2506.01844
    - Blog: https://huggingface.co/blog/smolvla
    - Model: https://huggingface.co/lerobot/smolvla_base
"""

# Suppress noisy warnings before importing anything else
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io")
warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")
warnings.filterwarnings("ignore", message=".*video decoding.*deprecated.*")

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
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

# Import shared utilities
from utils.constants import MOTOR_NAMES
from utils.training import (
    CachedDataset,
    cycle,
    prepare_obs_for_policy,
    run_evaluation,
    get_action_space_info,
    get_camera_names,
)


def main():
    parser = argparse.ArgumentParser(description="Train SmolVLA policy on LeRobot dataset")
    parser.add_argument("dataset", type=str, help="HuggingFace dataset repo ID")
    parser.add_argument("--from_pretrained", type=str, default=None,
                        help="Pretrained model to finetune (e.g., 'lerobot/smolvla_base')")
    parser.add_argument("--steps", type=int, default=50000, help="Training steps (default: 50000)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--chunk_size", type=int, default=50, help="Action chunk size (default: 50)")
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
    parser.add_argument("--cache_dataset", action="store_true", help="Pre-cache dataset in memory")
    parser.add_argument("--cameras", type=str, default=None,
                        help="Comma-separated camera names (default: all available)")
    parser.add_argument("--language", type=str, default="Pick up the block and place it in the bowl",
                        help="Language instruction for the task")
    parser.add_argument("--use_joint_actions", action="store_true",
                        help="Use action_joints instead of action (for EE datasets)")

    # SmolVLA-specific arguments
    parser.add_argument("--freeze_vision", action="store_true", default=True,
                        help="Freeze vision encoder (default: True)")
    parser.add_argument("--train_expert_only", action="store_true", default=True,
                        help="Only train action expert (default: True)")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps (default: 1000)")

    args = parser.parse_args()

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"outputs/train/smolvla_{timestamp}")
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
        print("Using 'action_joints' (6-dim joint space) instead of 'action'")
        if 'action' in output_features:
            del output_features['action']
        output_features['action'] = output_features.pop('action_joints')
    else:
        if 'action_joints' in output_features:
            del output_features['action_joints']

    # Exclude action fields from input features
    input_features = {key: ft for key, ft in features.items()
                      if key not in output_features and key != 'action_joints' and key != 'action'}

    # Filter cameras if --cameras specified
    if args.cameras:
        selected_cams = [c.strip() for c in args.cameras.split(",")]
        print(f"Selected cameras: {selected_cams}")
        input_features = {
            key: ft for key, ft in input_features.items()
            if not key.startswith("observation.images.")
            or key.replace("observation.images.", "") in selected_cams
        }

    print(f"Input features: {list(input_features.keys())}")
    print(f"Output features: {list(output_features.keys())}")

    # Get action space info
    action_dim, action_space = get_action_space_info(output_features)
    camera_names = get_camera_names(input_features)

    # Create or load SmolVLA policy
    if args.from_pretrained:
        print(f"Loading pretrained SmolVLA from: {args.from_pretrained}")
        policy = SmolVLAPolicy.from_pretrained(args.from_pretrained)
        # Update config with our features
        policy.config.input_features = input_features
        policy.config.output_features = output_features
    else:
        print("Creating SmolVLA policy from scratch...")
        cfg = SmolVLAConfig(
            input_features=input_features,
            output_features=output_features,
            chunk_size=args.chunk_size,
            n_action_steps=args.chunk_size,
            freeze_vision_encoder=args.freeze_vision,
            train_expert_only=args.train_expert_only,
            optimizer_lr=args.lr,
            scheduler_warmup_steps=args.warmup_steps,
            scheduler_decay_steps=args.steps,
        )
        policy = SmolVLAPolicy(cfg)

    policy.train()
    policy.to(device)

    # Create pre/post processors for normalization
    stats = dataset_metadata.stats
    if args.use_joint_actions:
        import copy
        stats = copy.deepcopy(dict(stats))
        if 'action_joints' in stats:
            stats['action'] = stats.pop('action_joints')
        elif 'action' in stats:
            del stats['action']
    preprocessor, postprocessor = make_pre_post_processors(policy.config, dataset_stats=stats)

    # Count parameters
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Set up delta timestamps for SmolVLA
    delta_timestamps = {
        "action": [i / dataset_metadata.fps for i in range(args.chunk_size)],
    }
    if args.use_joint_actions:
        delta_timestamps["action_joints"] = [i / dataset_metadata.fps for i in range(args.chunk_size)]
    for key in input_features:
        delta_timestamps[key] = [0.0]

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    raw_dataset = LeRobotDataset(args.dataset, delta_timestamps=delta_timestamps)
    print(f"Dataset size: {len(raw_dataset)} frames")

    # Optionally cache dataset
    if args.cache_dataset:
        dataset = CachedDataset(raw_dataset)
        num_workers = 0
        pin_memory = False
    else:
        dataset = raw_dataset
        num_workers = args.num_workers
        pin_memory = device.type != "cpu"

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        drop_last=True,
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
        betas=(0.9, 0.95),
    )

    # Create scheduler with warmup
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        else:
            progress = (step - args.warmup_steps) / (args.steps - args.warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Initialize WandB
    if not args.no_wandb:
        print("Initializing WandB...")
        wandb.init(
            project=args.wandb_project,
            name=args.run_name or f"smolvla_{args.dataset.split('/')[-1]}",
            config={
                "model": "SmolVLA",
                "dataset": args.dataset,
                "from_pretrained": args.from_pretrained,
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
                "cameras": camera_names,
                "action_space": action_space,
                "language_instruction": args.language,
                "freeze_vision": args.freeze_vision,
                "train_expert_only": args.train_expert_only,
            },
        )

    # Training info
    print()
    print("=" * 60)
    print("SmolVLA Training Configuration")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"From pretrained: {args.from_pretrained or 'None (training from scratch)'}")
    print(f"Output directory: {output_dir}")
    print(f"Training steps: {args.steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Device: {device}")
    print(f"Dataset caching: {'enabled' if args.cache_dataset else 'disabled'}")
    print(f"Cameras: {', '.join(camera_names)}")
    print(f"Action space: {action_space}")
    print(f"Language: {args.language}")
    print(f"Freeze vision: {args.freeze_vision}")
    print(f"Train expert only: {args.train_expert_only}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"WandB: {args.wandb_project if not args.no_wandb else 'disabled'}")
    print("=" * 60)
    print()

    # Training loop
    print("Starting training...")
    data_iter = cycle(dataloader)
    step = 0
    pbar = tqdm(total=args.steps, desc="Training")

    while step < args.steps:
        batch = next(data_iter)

        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass
        loss_dict = policy.forward(batch)
        loss = loss_dict["loss"]

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        optimizer.step()
        scheduler.step()

        step += 1

        # Update progress bar
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr": f"{current_lr:.2e}",
            "eta": f"{(args.steps - step) / max(step, 1) * pbar.format_dict['elapsed'] / 60:.1f}m"
        })
        pbar.update(1)

        # Log to WandB
        if not args.no_wandb and step % args.log_freq == 0:
            log_dict = {
                "train/loss": loss.item(),
                "train/lr": current_lr,
                "train/step": step,
            }
            # Add any additional losses from loss_dict
            for k, v in loss_dict.items():
                if k != "loss" and isinstance(v, (int, float, torch.Tensor)):
                    val = v.item() if isinstance(v, torch.Tensor) else v
                    log_dict[f"train/{k}"] = val
            wandb.log(log_dict)

        # Save checkpoint and evaluate
        if step % args.save_freq == 0 or step == args.steps:
            checkpoint_dir = output_dir / f"checkpoint_{step:06d}"
            print(f"\n  Saving checkpoint to {checkpoint_dir}")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(str(checkpoint_dir))

            # Save optimizer state
            torch.save({
                'step': step,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_dir / "training_state.pt")

            # Run evaluation
            if args.eval_episodes > 0:
                print(f"\n  Running {args.eval_episodes} evaluation episodes...")
                policy.eval()

                # Detect depth cameras
                depth_cameras = [cam for cam in camera_names if "_depth" in cam]

                success_rate, avg_steps, avg_time, ik_failure_rate, avg_ik_error, failure_summary = run_evaluation(
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    device=device,
                    num_episodes=args.eval_episodes,
                    randomize=args.eval_randomize,
                    fps=dataset_metadata.fps,
                    action_dim=action_dim,
                    depth_cameras=depth_cameras,
                    language_instruction=args.language,
                )

                print(f"  Eval: {100*success_rate:.1f}% success, {avg_steps:.1f} avg steps, {avg_time:.2f}s avg time")

                if not args.no_wandb:
                    eval_log = {
                        "eval/success_rate": success_rate,
                        "eval/avg_steps": avg_steps,
                        "eval/avg_time": avg_time,
                        "eval/step": step,
                    }
                    if ik_failure_rate is not None:
                        eval_log["eval/ik_failure_rate"] = ik_failure_rate
                        eval_log["eval/avg_ik_error_mm"] = avg_ik_error
                    # Log failure analysis metrics
                    if failure_summary:
                        eval_log["eval/pick_rate"] = failure_summary.get("pick_rate", 0)
                        eval_log["eval/drop_rate"] = failure_summary.get("drop_rate", 0)
                        from utils.failure_analysis import Outcome
                        outcome_counts = failure_summary.get("outcome_counts", {})
                        for outcome in Outcome:
                            count = outcome_counts.get(outcome, 0)
                            eval_log[f"eval/outcome_{outcome.value}"] = count
                    wandb.log(eval_log)

                policy.train()

    pbar.close()

    # Save final model
    final_dir = output_dir / "final"
    print(f"\nSaving final model to {final_dir}")
    final_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(final_dir))

    # Final evaluation
    if args.eval_episodes > 0:
        print(f"\nRunning final evaluation ({args.eval_episodes} episodes)...")
        policy.eval()
        depth_cameras = [cam for cam in camera_names if "_depth" in cam]

        success_rate, avg_steps, avg_time, ik_failure_rate, avg_ik_error, failure_summary = run_evaluation(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
            num_episodes=args.eval_episodes,
            randomize=args.eval_randomize,
            fps=dataset_metadata.fps,
            action_dim=action_dim,
            depth_cameras=depth_cameras,
            language_instruction=args.language,
        )

        print(f"Final: {100*success_rate:.1f}% success, {avg_steps:.1f} avg steps, {avg_time:.2f}s avg time")
        if failure_summary:
            print(f"Final pick rate: {failure_summary.get('pick_rate', 0)*100:.1f}%")
            print(f"Final drop rate: {failure_summary.get('drop_rate', 0)*100:.1f}%")

        if not args.no_wandb:
            final_log = {
                "eval/final_success_rate": success_rate,
                "eval/final_avg_steps": avg_steps,
                "eval/final_avg_time": avg_time,
            }
            if failure_summary:
                final_log["eval/final_pick_rate"] = failure_summary.get("pick_rate", 0)
                final_log["eval/final_drop_rate"] = failure_summary.get("drop_rate", 0)
            wandb.log(final_log)

    if not args.no_wandb:
        wandb.finish()

    print(f"\nTraining complete! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
