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

import argparse
from pathlib import Path
from datetime import datetime
import time

import torch
from tqdm import tqdm
import wandb

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def cycle(dataloader):
    """Infinite dataloader iterator."""
    while True:
        for batch in dataloader:
            yield batch


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
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")

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
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

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
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    # Count parameters
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Set up delta timestamps for ACT
    delta_timestamps = {
        "action": [i / dataset_metadata.fps for i in range(args.chunk_size)],
    }
    for key in input_features:
        delta_timestamps[key] = [0.0]

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = LeRobotDataset(args.dataset, delta_timestamps=delta_timestamps)
    print(f"Dataset size: {len(dataset)} frames")

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Create optimizer with weight decay
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=1e-7
    )

    # Initialize WandB
    if not args.no_wandb:
        print("Initializing WandB...")
        wandb.init(
            project=args.wandb_project,
            name=f"act_{args.dataset.split('/')[-1]}",
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
    print(f"WandB: {'disabled' if args.no_wandb else args.wandb_project}")
    print("=" * 60)
    print()

    # Training loop
    print("Starting training...")
    step = 0
    best_loss = float('inf')
    running_loss = 0.0
    running_kl_loss = 0.0
    start_time = time.time()

    data_iter = cycle(dataloader)
    pbar = tqdm(total=args.steps, desc="Training")

    while step < args.steps:
        batch = next(data_iter)

        # Move batch to device and preprocess
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch = preprocessor(batch)

        # Fix tensor dimensions for ACT model expectations:
        # State: ACT expects [batch, state_dim], not [batch, n_obs_steps, state_dim]
        if "observation.state" in batch and isinstance(batch["observation.state"], torch.Tensor):
            if batch["observation.state"].dim() == 3:
                batch["observation.state"] = batch["observation.state"].squeeze(1)

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
        if "kl_loss" in output_dict:
            running_kl_loss += output_dict["kl_loss"].item()

        step += 1
        pbar.update(1)

        # Log to WandB every step
        if not args.no_wandb:
            log_dict = {
                "train/loss": loss.item(),
                "train/grad_norm": grad_norm.item(),
                "train/lr": scheduler.get_last_lr()[0],
            }
            if "kl_loss" in output_dict:
                log_dict["train/kl_loss"] = output_dict["kl_loss"].item()
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
            checkpoint_dir = output_dir / f"checkpoint_{step:06d}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n  Saving checkpoint to {checkpoint_dir}")
            policy.save_pretrained(checkpoint_dir)
            preprocessor.save_pretrained(checkpoint_dir)
            postprocessor.save_pretrained(checkpoint_dir)

            # Save training state for resuming
            torch.save({
                'step': step,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
            }, checkpoint_dir / "training_state.pt")

            if not args.no_wandb:
                wandb.log({"checkpoint/step": step}, step=step)

    pbar.close()

    # Save final model
    print()
    print("Saving final model...")
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(final_dir)
    preprocessor.save_pretrained(final_dir)
    postprocessor.save_pretrained(final_dir)

    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final model: {final_dir}")

    # Log final metrics and finish WandB
    if not args.no_wandb:
        wandb.log({
            "final/total_time_minutes": elapsed / 60,
            "final/best_loss": best_loss,
        })
        wandb.finish()


if __name__ == "__main__":
    main()
