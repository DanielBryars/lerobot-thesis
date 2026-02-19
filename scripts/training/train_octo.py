#!/usr/bin/env python
"""
Fine-tune Octo-Small on a LeRobot dataset with delta joint actions.

Uses octo-pytorch (https://github.com/emb-ai/octo-pytorch) for the model,
with a custom PyTorch dataloader (no RLDS/TensorFlow dependency).

Usage:
    # Overhead camera only (variant A)
    python scripts/training/train_octo.py danbhf/sim_pick_place_2pos_220ep_v2_delta \
        --no-wrist-cam --no-proprio --run-name octo_A_overhead_only

    # Overhead + wrist (variant B)
    python scripts/training/train_octo.py danbhf/sim_pick_place_2pos_220ep_v2_delta \
        --no-proprio --run-name octo_B_overhead_wrist

    # With proprio (variant C)
    python scripts/training/train_octo.py danbhf/sim_pick_place_2pos_220ep_v2_delta \
        --no-wrist-cam --run-name octo_C_overhead_proprio

    # Full setup (variant D)
    python scripts/training/train_octo.py danbhf/sim_pick_place_2pos_220ep_v2_delta \
        --run-name octo_D_full

    # Resume from checkpoint
    python scripts/training/train_octo.py danbhf/sim_pick_place_2pos_220ep_v2_delta \
        --resume outputs/train/octo_XXXX/checkpoint_010000
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import wandb

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from utils.constants import MOTOR_NAMES, NUM_JOINTS
from utils.octo_dataset import OctoDataset


def inverse_sqrt_schedule(step: int, warmup_steps: int = 2000) -> float:
    """Inverse square root LR schedule with linear warmup (matches Octo paper)."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    return (warmup_steps ** 0.5) * (step ** -0.5)


def freeze_weights(model, frozen_keys: list):
    """Freeze parameters matching any of the given key patterns."""
    for name, param in model.named_parameters():
        for key in frozen_keys:
            if key in name:
                param.requires_grad = False
                break


def save_octo_checkpoint(
    model,
    optimizer,
    scheduler,
    step: int,
    output_dir: Path,
    training_metadata: dict,
    action_stats: dict,
    proprio_stats: dict = None,
    checkpoint_name: str = None,
    best_loss: float = None,
):
    """Save an Octo training checkpoint."""
    if checkpoint_name is None:
        checkpoint_name = f"checkpoint_{step:06d}"

    checkpoint_dir = output_dir / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), checkpoint_dir / "model.pt")

    # Save optimizer + scheduler state
    state = {
        "step": step,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    }
    if best_loss is not None:
        state["best_loss"] = best_loss
    torch.save(state, checkpoint_dir / "training_state.pt")

    # Save training metadata
    with open(checkpoint_dir / "training_metadata.json", "w") as f:
        json.dump(training_metadata, f, indent=2)

    # Save action stats (needed for inference denormalization)
    stats = {"action": {k: v.tolist() for k, v in action_stats.items()}}
    if proprio_stats:
        stats["proprio"] = {k: v.tolist() for k, v in proprio_stats.items()}
    with open(checkpoint_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


def load_octo_checkpoint(checkpoint_dir: Path, model, optimizer=None, scheduler=None) -> int:
    """Load an Octo training checkpoint. Returns the step number."""
    checkpoint_dir = Path(checkpoint_dir)

    # Find latest checkpoint if parent dir given
    state_path = checkpoint_dir / "training_state.pt"
    if not state_path.exists():
        checkpoints = sorted(
            [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint_")],
            key=lambda x: int(x.name.split("_")[1])
        )
        if checkpoints:
            checkpoint_dir = checkpoints[-1]
            state_path = checkpoint_dir / "training_state.pt"
            print(f"  Found latest checkpoint: {checkpoint_dir.name}")

    # Load model weights
    model_path = checkpoint_dir / "model.pt"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"  Loaded model weights from {model_path}")

    # Load training state
    if state_path.exists():
        state = torch.load(state_path, weights_only=True)
        if optimizer and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler and state.get("scheduler_state_dict"):
            scheduler.load_state_dict(state["scheduler_state_dict"])
        return state["step"]

    return 0


def cycle(dataloader):
    """Infinite dataloader iterator."""
    while True:
        for batch in dataloader:
            yield batch


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Octo-Small on LeRobot dataset")
    parser.add_argument("dataset", type=str, help="HuggingFace dataset repo ID (delta actions)")
    parser.add_argument("--pretrained", type=str, default="hf://rail-berkeley/octo-small-1.5",
                        help="Pretrained Octo checkpoint path")
    parser.add_argument("--steps", type=int, default=50000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=2000, help="LR warmup steps")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clip norm")
    parser.add_argument("--action-horizon", type=int, default=4, help="Action prediction horizon")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--log-freq", type=int, default=100, help="Log every N steps")
    parser.add_argument("--save-freq", type=int, default=5000, help="Save checkpoint every N steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--wandb-project", type=str, default="lerobot-thesis", help="WandB project")
    parser.add_argument("--run-name", type=str, default=None, help="WandB run name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB")
    parser.add_argument("--no-wrist-cam", action="store_true", help="Disable wrist camera")
    parser.add_argument("--no-proprio", action="store_true", help="Disable proprioception input")
    parser.add_argument("--no-augment", action="store_true", help="Disable image augmentation")
    parser.add_argument("--fix-state", action="store_true", default=True,
                        help="Fix state bug (default: True)")
    parser.add_argument("--no-fix-state", dest="fix_state", action="store_false")
    parser.add_argument("--freeze-transformer", action="store_true",
                        help="Freeze transformer backbone (only train heads)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--instruction", type=str,
                        default="Pick up the block and place it in the bowl",
                        help="Language instruction")

    args = parser.parse_args()

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"outputs/train/octo_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # ---------------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------------
    print(f"\nLoading dataset: {args.dataset}")
    dataset = OctoDataset(
        dataset_repo_id=args.dataset,
        action_horizon=args.action_horizon,
        primary_image_size=256,
        wrist_image_size=128,
        use_wrist_cam=not args.no_wrist_cam,
        use_proprio=not args.no_proprio,
        augment=not args.no_augment,
        instruction=args.instruction,
        fix_state=args.fix_state,
    )

    action_stats = dataset.compute_action_stats()
    proprio_stats = dataset.compute_proprio_stats() if not args.no_proprio else None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # ---------------------------------------------------------------
    # Load pretrained Octo model
    # ---------------------------------------------------------------
    print(f"\nLoading pretrained Octo-Small from: {args.pretrained}")

    from octo.model.octo_model_pt import OctoModelPt
    from octo.utils.spec import ModuleSpec

    meta = OctoModelPt.load_config_and_meta_from_jax(args.pretrained)

    # Modify observation tokenizers
    if args.no_wrist_cam and "wrist" in meta["config"]["model"]["observation_tokenizers"]:
        del meta["config"]["model"]["observation_tokenizers"]["wrist"]
        print("  Removed wrist camera tokenizer")

    if not args.no_proprio:
        from octo.model.components.tokenizers_pt import LowdimObsTokenizerPt
        meta["config"]["model"]["observation_tokenizers"]["proprio"] = ModuleSpec.create(
            LowdimObsTokenizerPt,
            n_bins=256,
            bin_type="normal",
            low=-2.0,
            high=2.0,
            obs_keys=["proprio"],
        )
        print("  Added proprio tokenizer (256 bins, range [-2, 2])")

    # Update token counts
    num_tokens = {"primary": 256, "language": 16, "action": 1}
    if not args.no_wrist_cam:
        num_tokens["wrist"] = 64  # 128x128 / 16^2 = 64 patches
    if not args.no_proprio:
        num_tokens["proprio"] = NUM_JOINTS  # 6 tokens for 6 joints
    meta["config"]["model"]["num_tokens_dict"] = num_tokens

    # Re-initialize action head for our action space
    # Use L1 (regression) head for delta joint actions
    from octo.model.components.action_heads_pt import L1ActionHeadPt
    meta["config"]["model"]["heads"]["action"] = ModuleSpec.create(
        L1ActionHeadPt,
        input_dim=384,  # Octo-Small transformer dim
        action_horizon=args.action_horizon,
        action_dim=NUM_JOINTS,  # 6 (5 arm + 1 gripper)
        readout_key="readout_action",
    )
    print(f"  Action head: L1, horizon={args.action_horizon}, dim={NUM_JOINTS}")

    # Build model from config
    model = OctoModelPt.from_config(**meta, verbose=True)

    # Load pretrained weights (skip mismatched heads and new tokenizers)
    skip_regex = ".*hf_model"  # Skip language model weights if not needed
    if not args.no_proprio:
        skip_regex += "|.*proprio.*"  # Skip proprio tokenizer (randomly initialized)
    model.load_weights_from_jax(args.pretrained, skip_keys_regex=skip_regex)
    print("  Pretrained weights loaded (action head + new tokenizers randomly initialized)")

    model.to(device)

    # Freeze transformer if requested
    if args.freeze_transformer:
        frozen_keys = meta["config"].get("optimizer", {}).get("frozen_keys", [])
        frozen_keys.append("BlockTransformer_0")
        freeze_weights(model.module, frozen_keys)
        print("  Transformer backbone FROZEN")

    model.train()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # ---------------------------------------------------------------
    # Optimizer and scheduler
    # ---------------------------------------------------------------
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params_list, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: inverse_sqrt_schedule(step, args.warmup_steps))

    # Resume if specified
    start_step = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"\nResuming from: {resume_path}")
            start_step = load_octo_checkpoint(resume_path, model, optimizer, scheduler)
            print(f"  Resumed at step {start_step}")

    # ---------------------------------------------------------------
    # Training metadata
    # ---------------------------------------------------------------
    cameras = ["overhead_cam"]
    if not args.no_wrist_cam:
        cameras.append("wrist_cam")

    training_metadata = {
        "dataset_repo_id": args.dataset,
        "model_type": "octo_small",
        "pretrained_path": args.pretrained,
        "cameras": cameras,
        "action_space": f"delta_joint ({NUM_JOINTS}-dim)",
        "action_dim": NUM_JOINTS,
        "action_horizon": args.action_horizon,
        "fps": 30,
        "total_frames": len(dataset),
        "use_wrist_cam": not args.no_wrist_cam,
        "use_proprio": not args.no_proprio,
        "fix_state": args.fix_state,
        "instruction": args.instruction,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "batch_size": args.batch_size,
        "freeze_transformer": args.freeze_transformer,
    }

    # ---------------------------------------------------------------
    # WandB
    # ---------------------------------------------------------------
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name or f"octo_{args.dataset.split('/')[-1]}",
            config=training_metadata,
        )

    # ---------------------------------------------------------------
    # Print config
    # ---------------------------------------------------------------
    print()
    print("=" * 60)
    print("Octo-Small Fine-tuning Configuration")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Output: {output_dir}")
    print(f"Steps: {args.steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR: {args.lr} (warmup={args.warmup_steps}, inv sqrt decay)")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Grad clip: {args.grad_clip}")
    print(f"Action horizon: {args.action_horizon}")
    print(f"Cameras: {', '.join(cameras)}")
    print(f"Proprio: {'enabled' if not args.no_proprio else 'disabled'}")
    print(f"Augmentation: {'enabled' if not args.no_augment else 'disabled'}")
    print(f"Instruction: {args.instruction}")
    print(f"Parameters: {trainable_params:,} trainable / {total_params:,} total")
    print("=" * 60)
    print()

    # ---------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------
    print("Starting training...")
    step = start_step
    best_loss = float("inf")
    running_loss = 0.0
    start_time = time.time()

    data_iter = cycle(dataloader)
    pbar = tqdm(total=args.steps, initial=start_step, desc="Training")

    while step < args.steps:
        batch = next(data_iter)

        # Build Octo observation dict
        observations = {
            "image_primary": batch["image_primary"].to(device),  # (B, T=2, 3, H, W)
            "timestep_pad_mask": batch["obs_pad_mask"].to(device).bool(),  # (B, T=2)
        }
        if not args.no_wrist_cam and "image_wrist" in batch:
            observations["image_wrist"] = batch["image_wrist"].to(device)
        if not args.no_proprio and "proprio" in batch:
            observations["proprio"] = batch["proprio"].to(device)

        # Build task dict (language instruction)
        task = model.create_tasks(texts=[args.instruction] * batch["action"].shape[0], device=device)

        # Actions and masks
        gt_actions = batch["action"].to(device)  # (B, horizon, D)
        action_pad_mask = batch["action_pad_mask"].to(device)  # (B, horizon)

        # Forward pass
        _, head_outputs = model(
            observations=observations,
            tasks=task,
            timestep_pad_mask=observations["timestep_pad_mask"],
            action_pad_mask=action_pad_mask,
            gt_actions=gt_actions,
            train=True,
            verbose=False,
        )

        loss = head_outputs["action"][0]
        info = head_outputs["action"][1]

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=args.grad_clip,
        )
        optimizer.step()
        scheduler.step()

        # Metrics
        running_loss += loss.item()
        step += 1
        pbar.update(1)

        # WandB logging
        if use_wandb:
            log_dict = {
                "train/loss": loss.item(),
                "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                "train/lr": scheduler.get_last_lr()[0],
            }
            # Log any extra info from the action head
            if isinstance(info, dict):
                for k, v in info.items():
                    if isinstance(v, (int, float)):
                        log_dict[f"train/{k}"] = v
                    elif isinstance(v, torch.Tensor) and v.numel() == 1:
                        log_dict[f"train/{k}"] = v.item()
            wandb.log(log_dict, step=step)

        # Console logging
        if step % args.log_freq == 0:
            avg_loss = running_loss / args.log_freq
            elapsed = time.time() - start_time
            steps_per_sec = (step - start_step) / elapsed
            eta_minutes = (args.steps - step) / max(steps_per_sec, 1e-6) / 60

            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                "eta": f"{eta_minutes:.1f}m",
            })

            if use_wandb:
                wandb.log({"train/avg_loss": avg_loss}, step=step)

            running_loss = 0.0
            if avg_loss < best_loss:
                best_loss = avg_loss

        # Save checkpoint
        if step % args.save_freq == 0 or step == args.steps:
            print(f"\n  Saving checkpoint_{step:06d}")
            save_octo_checkpoint(
                model, optimizer, scheduler, step, output_dir,
                training_metadata=training_metadata,
                action_stats=action_stats,
                proprio_stats=proprio_stats,
                best_loss=best_loss,
            )

    pbar.close()

    # Save final
    print("\nSaving final model...")
    save_octo_checkpoint(
        model, optimizer, scheduler, args.steps, output_dir,
        training_metadata=training_metadata,
        action_stats=action_stats,
        proprio_stats=proprio_stats,
        checkpoint_name="final",
        best_loss=best_loss,
    )

    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Total time: {elapsed / 60:.1f} minutes")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final model: {output_dir / 'final'}")

    if use_wandb:
        wandb.log({
            "final/total_time_minutes": elapsed / 60,
            "final/best_loss": best_loss,
        })
        wandb.finish()


if __name__ == "__main__":
    main()
