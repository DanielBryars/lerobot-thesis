#!/usr/bin/env python
"""
Training script for Pi0/Pi0.5 using openpi framework.

This script trains Pi0 or Pi0.5 on a converted LeRobot dataset using
Physical Intelligence's openpi implementation.

Prerequisites:
    # Clone openpi repository
    git clone https://github.com/Physical-Intelligence/openpi.git
    cd openpi

    # Install dependencies (requires JAX with GPU support)
    pip install -e .

    # Or using conda/mamba
    GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

Usage:
    # Train Pi0 on converted dataset
    python scripts/openpi/train_pi0.py --dataset danbhf/openpi_sim_pick_place \
        --model pi0 --steps 50000 --batch_size 16

    # Train Pi0.5 (larger model)
    python scripts/openpi/train_pi0.py --dataset danbhf/openpi_sim_pick_place \
        --model pi0.5 --steps 50000 --batch_size 8

    # Resume from checkpoint
    python scripts/openpi/train_pi0.py --dataset danbhf/openpi_sim_pick_place \
        --resume checkpoints/pi0_step_10000

Notes:
    - Requires JAX with GPU/TPU support
    - Memory requirements: Pi0 ~16GB, Pi0.5 ~40GB VRAM
    - Training typically requires H100 or A100 for Pi0.5

References:
    - Paper: https://arxiv.org/abs/2410.24164
    - Repo: https://github.com/Physical-Intelligence/openpi
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

# Add project root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def check_openpi_installed():
    """Check if openpi is installed."""
    try:
        import openpi
        return True
    except ImportError:
        return False


def download_dataset(repo_id: str, cache_dir: str = "./datasets") -> Path:
    """Download dataset from HuggingFace.

    Args:
        repo_id: HuggingFace dataset repo ID
        cache_dir: Local cache directory

    Returns:
        Path to downloaded dataset
    """
    from huggingface_hub import snapshot_download

    local_dir = Path(cache_dir) / repo_id.replace("/", "_")

    if local_dir.exists():
        print(f"Using cached dataset: {local_dir}")
        return local_dir

    print(f"Downloading dataset from {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
    )

    return local_dir


def load_episode(npz_path: Path) -> Dict[str, np.ndarray]:
    """Load a single episode from NPZ file.

    Args:
        npz_path: Path to episode NPZ file

    Returns:
        Dict with episode data
    """
    data = np.load(str(npz_path), allow_pickle=True)

    episode = {
        "observation": {
            "state": data["observation/state"],
            "image": data["observation/image"],
        },
        "action": data["action"],
        "is_first": data["is_first"],
        "is_last": data["is_last"],
        "is_terminal": data["is_terminal"],
        "language_instruction": str(data["language_instruction"]),
    }

    # Add wrist image if available
    if "observation/wrist_image" in data.files:
        episode["observation"]["wrist_image"] = data["observation/wrist_image"]

    return episode


def create_dataloader(
    dataset_dir: Path,
    batch_size: int,
    shuffle: bool = True,
) -> Iterator[Dict[str, np.ndarray]]:
    """Create a simple dataloader for training.

    Args:
        dataset_dir: Path to dataset directory
        batch_size: Batch size
        shuffle: Whether to shuffle episodes

    Yields:
        Batches of training data
    """
    # Find all episode files
    episode_files = sorted(dataset_dir.glob("episode_*.npz"))
    print(f"Found {len(episode_files)} episodes")

    # Load all episodes into memory
    episodes = []
    for ep_file in episode_files:
        ep = load_episode(ep_file)
        episodes.append(ep)

    # Flatten into individual steps
    all_steps = []
    for ep in episodes:
        n_steps = len(ep["action"])
        for i in range(n_steps):
            step = {
                "observation": {
                    "state": ep["observation"]["state"][i],
                    "image": ep["observation"]["image"][i],
                },
                "action": ep["action"][i],
                "language_instruction": ep["language_instruction"],
            }
            if "wrist_image" in ep["observation"]:
                step["observation"]["wrist_image"] = ep["observation"]["wrist_image"][i]
            all_steps.append(step)

    print(f"Total steps: {len(all_steps)}")

    # Create batches
    indices = np.arange(len(all_steps))

    while True:
        if shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            if len(batch_indices) < batch_size:
                continue  # Skip incomplete batches

            batch = {
                "observation": {
                    "state": np.stack([all_steps[j]["observation"]["state"] for j in batch_indices]),
                    "image": np.stack([all_steps[j]["observation"]["image"] for j in batch_indices]),
                },
                "action": np.stack([all_steps[j]["action"] for j in batch_indices]),
                "language_instruction": [all_steps[j]["language_instruction"] for j in batch_indices],
            }

            if "wrist_image" in all_steps[0]["observation"]:
                batch["observation"]["wrist_image"] = np.stack(
                    [all_steps[j]["observation"]["wrist_image"] for j in batch_indices]
                )

            yield batch


def train_with_openpi(
    dataset_dir: Path,
    model_type: str,
    batch_size: int,
    steps: int,
    learning_rate: float,
    checkpoint_dir: Path,
    resume_from: Optional[str],
    wandb_project: Optional[str],
):
    """Train using openpi framework.

    This function uses openpi's native training infrastructure.

    Args:
        dataset_dir: Path to dataset
        model_type: "pi0" or "pi0.5"
        batch_size: Training batch size
        steps: Number of training steps
        learning_rate: Learning rate
        checkpoint_dir: Directory for checkpoints
        resume_from: Optional checkpoint to resume from
        wandb_project: Optional WandB project name
    """
    try:
        # Import openpi modules
        from openpi.models import load_model
        from openpi.training import Trainer
        from openpi.data import make_dataset
    except ImportError as e:
        print(f"Error: openpi not installed or import failed: {e}")
        print("\nTo install openpi:")
        print("  git clone https://github.com/Physical-Intelligence/openpi.git")
        print("  cd openpi && pip install -e .")
        sys.exit(1)

    # Load model
    print(f"Loading {model_type} model...")
    model = load_model(model_type)

    # Create dataset
    print("Creating dataset...")
    dataset = make_dataset(str(dataset_dir))

    # Create trainer
    trainer = Trainer(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        learning_rate=learning_rate,
        checkpoint_dir=str(checkpoint_dir),
        wandb_project=wandb_project,
    )

    # Resume if specified
    if resume_from:
        print(f"Resuming from {resume_from}")
        trainer.load_checkpoint(resume_from)

    # Train
    print(f"Starting training for {steps} steps...")
    trainer.train(steps)

    print("Training complete!")


def train_standalone(
    dataset_dir: Path,
    model_type: str,
    batch_size: int,
    steps: int,
    learning_rate: float,
    checkpoint_dir: Path,
    log_freq: int,
    save_freq: int,
):
    """Standalone training loop without full openpi installation.

    This is a simplified training loop for testing and development.
    For production training, use train_with_openpi() with the full
    openpi installation.

    Args:
        dataset_dir: Path to dataset
        model_type: "pi0" or "pi0.5"
        batch_size: Training batch size
        steps: Number of training steps
        learning_rate: Learning rate
        checkpoint_dir: Directory for checkpoints
        log_freq: Logging frequency
        save_freq: Checkpoint save frequency
    """
    try:
        import jax
        import jax.numpy as jnp
        import optax
    except ImportError:
        print("Error: JAX not installed. Install with: pip install jax jaxlib")
        print("For GPU support: pip install jax[cuda12]")
        sys.exit(1)

    print(f"JAX devices: {jax.devices()}")

    # Create dataloader
    dataloader = create_dataloader(dataset_dir, batch_size)

    # Placeholder for actual model loading
    # In practice, this would load the Pi0/Pi0.5 weights
    print(f"[PLACEHOLDER] Would load {model_type} model here")
    print("This standalone mode is for testing only.")
    print("For actual training, install openpi and use --use_openpi flag.")

    # Training loop placeholder
    print(f"\nStarting training loop for {steps} steps...")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for step, batch in enumerate(dataloader):
        if step >= steps:
            break

        # Placeholder loss computation
        loss = np.random.random()  # Fake loss for demonstration

        if step % log_freq == 0:
            print(f"Step {step}/{steps} | Loss: {loss:.4f}")

        if step > 0 and step % save_freq == 0:
            ckpt_path = checkpoint_dir / f"checkpoint_{step:06d}"
            print(f"[PLACEHOLDER] Would save checkpoint to {ckpt_path}")

    print("\nTraining complete (placeholder mode)")


def main():
    parser = argparse.ArgumentParser(
        description="Train Pi0/Pi0.5 on converted LeRobot dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="danbhf/openpi_sim_pick_place",
                        help="HuggingFace dataset repo ID or local path")
    parser.add_argument("--local", action="store_true",
                        help="Treat dataset path as local directory")

    # Model arguments
    parser.add_argument("--model", type=str, default="pi0", choices=["pi0", "pi0.5"],
                        help="Model type (default: pi0)")

    # Training arguments
    parser.add_argument("--steps", type=int, default=50000,
                        help="Training steps (default: 50000)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for checkpoints")
    parser.add_argument("--log_freq", type=int, default=100,
                        help="Logging frequency (default: 100)")
    parser.add_argument("--save_freq", type=int, default=5000,
                        help="Checkpoint save frequency (default: 5000)")

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")

    # Logging
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="WandB project name (optional)")

    # Mode
    parser.add_argument("--use_openpi", action="store_true",
                        help="Use full openpi training infrastructure")

    args = parser.parse_args()

    # Set output directory
    if args.output_dir:
        checkpoint_dir = Path(args.output_dir)
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = Path(f"outputs/train/{args.model}_{timestamp}")

    print("=" * 60)
    print("Pi0/Pi0.5 Training")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Steps: {args.steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Output: {checkpoint_dir}")
    print("=" * 60)

    # Get dataset path
    if args.local:
        dataset_dir = Path(args.dataset)
        if not dataset_dir.exists():
            print(f"Error: Local dataset not found: {dataset_dir}")
            sys.exit(1)
    else:
        dataset_dir = download_dataset(args.dataset)

    # Check for openpi installation
    openpi_available = check_openpi_installed()

    if args.use_openpi:
        if not openpi_available:
            print("Error: --use_openpi specified but openpi not installed")
            print("\nTo install openpi:")
            print("  git clone https://github.com/Physical-Intelligence/openpi.git")
            print("  cd openpi && pip install -e .")
            sys.exit(1)

        train_with_openpi(
            dataset_dir=dataset_dir,
            model_type=args.model,
            batch_size=args.batch_size,
            steps=args.steps,
            learning_rate=args.lr,
            checkpoint_dir=checkpoint_dir,
            resume_from=args.resume,
            wandb_project=args.wandb_project,
        )
    else:
        if openpi_available:
            print("Note: openpi is installed. Use --use_openpi for full training.")

        train_standalone(
            dataset_dir=dataset_dir,
            model_type=args.model,
            batch_size=args.batch_size,
            steps=args.steps,
            learning_rate=args.lr,
            checkpoint_dir=checkpoint_dir,
            log_freq=args.log_freq,
            save_freq=args.save_freq,
        )


if __name__ == "__main__":
    main()
