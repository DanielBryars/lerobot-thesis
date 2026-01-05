#!/usr/bin/env python
"""
Remote training script for vast.ai / RunPod / etc.

This script:
1. Downloads the dataset from HuggingFace
2. Runs training with specified config (ACT or SmolVLA)
3. Uploads checkpoints to HuggingFace
4. Logs everything to WandB

Usage:
    # ACT training (default)
    python remote/train_remote.py --dataset danbhf/sim_pick_place_40ep_rgbd_ee \
        --cameras wrist_cam,overhead_cam,overhead_cam_depth \
        --steps 50000 --batch_size 16 --eval_episodes 30

    # SmolVLA finetuning from pretrained
    python remote/train_remote.py --policy smolvla --dataset danbhf/sim_pick_place_40ep_rgbd_ee \
        --from_pretrained lerobot/smolvla_base \
        --language "Pick up the block and place it in the bowl" \
        --steps 20000 --batch_size 16 --eval_episodes 30

Environment variables required:
    HF_TOKEN: HuggingFace token for dataset access and model upload
    WANDB_API_KEY: WandB API key for logging
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def setup_environment():
    """Setup environment variables and authenticate services."""
    # Check for required environment variables
    hf_token = os.environ.get("HF_TOKEN")
    wandb_key = os.environ.get("WANDB_API_KEY")

    if not hf_token:
        print("WARNING: HF_TOKEN not set. Dataset download may fail for private datasets.")
    else:
        # Login to HuggingFace
        print("Logging in to HuggingFace...")
        subprocess.run(["huggingface-cli", "login", "--token", hf_token], check=False)

    if not wandb_key:
        print("WARNING: WANDB_API_KEY not set. WandB logging will be disabled.")
        return False

    # Login to WandB
    print("Logging in to WandB...")
    subprocess.run(["wandb", "login", wandb_key], check=False)
    return True


def run_training(args):
    """Run the training script with specified arguments."""
    # Determine which training script to use
    if args.policy == "smolvla":
        script = "training/train_smolvla.py"
    else:
        script = "training/train_act.py"

    # Build the training command
    cmd = [
        "python", script,
        args.dataset,
        "--steps", str(args.steps),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--save_freq", str(args.save_freq),
    ]

    if args.cameras:
        cmd.extend(["--cameras", args.cameras])

    if args.eval_episodes > 0:
        cmd.extend(["--eval_episodes", str(args.eval_episodes)])

    if args.eval_randomize:
        cmd.append("--eval_randomize")

    if args.use_joint_actions:
        cmd.append("--use_joint_actions")

    if args.cache_dataset:
        cmd.append("--cache_dataset")

    if args.no_wandb:
        cmd.append("--no_wandb")

    if args.wandb_project:
        cmd.extend(["--wandb_project", args.wandb_project])

    if args.run_name:
        cmd.extend(["--run_name", args.run_name])

    # SmolVLA-specific arguments
    if args.policy == "smolvla":
        if args.from_pretrained:
            cmd.extend(["--from_pretrained", args.from_pretrained])
        if args.language:
            cmd.extend(["--language", args.language])

    print(f"\nRunning training command:")
    print(f"  {' '.join(cmd)}\n")

    # Run training
    result = subprocess.run(cmd)
    return result.returncode


def upload_results(args, output_dir: Path):
    """Upload trained model to HuggingFace."""
    if not args.upload_repo:
        print("No upload repo specified, skipping upload.")
        return

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("WARNING: HF_TOKEN not set, cannot upload results.")
        return

    # Find the latest training output (ACT or SmolVLA)
    pattern = "smolvla_*" if args.policy == "smolvla" else "act_*"
    train_dirs = sorted(Path("outputs/train").glob(pattern))
    if not train_dirs:
        print(f"No training outputs found matching {pattern}.")
        return

    latest_dir = train_dirs[-1]
    final_dir = latest_dir / "final"

    if not final_dir.exists():
        print(f"Final model not found at {final_dir}")
        return

    print(f"\nUploading {final_dir} to {args.upload_repo}...")

    cmd = [
        "huggingface-cli", "upload",
        args.upload_repo,
        str(final_dir),
        "--repo-type", "model",
    ]

    subprocess.run(cmd, check=False)
    print("Upload complete!")


def main():
    parser = argparse.ArgumentParser(description="Remote training script")

    # Policy type
    parser.add_argument("--policy", type=str, default="act", choices=["act", "smolvla"],
                        help="Policy type to train (default: act)")

    # Dataset arguments
    parser.add_argument("--dataset", type=str, required=True,
                        help="HuggingFace dataset ID")

    # Training arguments
    parser.add_argument("--steps", type=int, default=50000,
                        help="Training steps")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--save_freq", type=int, default=5000,
                        help="Checkpoint save frequency")

    # Camera arguments
    parser.add_argument("--cameras", type=str, default=None,
                        help="Comma-separated camera names")

    # Evaluation arguments
    parser.add_argument("--eval_episodes", type=int, default=30,
                        help="Evaluation episodes per checkpoint")
    parser.add_argument("--eval_randomize", action="store_true", default=True,
                        help="Randomize object positions during eval")

    # Action space
    parser.add_argument("--use_joint_actions", action="store_true",
                        help="Use joint action space instead of EE")

    # Caching
    parser.add_argument("--cache_dataset", action="store_true",
                        help="Cache dataset in memory")

    # WandB arguments
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="lerobot-thesis",
                        help="WandB project name")
    parser.add_argument("--run_name", type=str, default=None,
                        help="WandB run name")

    # Upload arguments
    parser.add_argument("--upload_repo", type=str, default=None,
                        help="HuggingFace repo to upload final model")

    # SmolVLA-specific arguments
    parser.add_argument("--from_pretrained", type=str, default=None,
                        help="SmolVLA: pretrained model to finetune from")
    parser.add_argument("--language", type=str, default="Pick up the block and place it in the bowl",
                        help="SmolVLA: language instruction for the task")

    args = parser.parse_args()

    print("=" * 60)
    print("LeRobot Remote Training")
    print("=" * 60)
    print(f"Policy: {args.policy.upper()}")
    print(f"Dataset: {args.dataset}")
    print(f"Steps: {args.steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Cameras: {args.cameras or 'all'}")
    print(f"Eval episodes: {args.eval_episodes}")
    print(f"Action space: {'joint' if args.use_joint_actions else 'EE'}")
    if args.policy == "smolvla":
        print(f"Pretrained: {args.from_pretrained or 'scratch'}")
        print(f"Language: {args.language}")
    print("=" * 60)

    # Setup environment
    wandb_enabled = setup_environment()
    if not wandb_enabled:
        args.no_wandb = True

    # Run training
    print("\nStarting training...")
    return_code = run_training(args)

    if return_code != 0:
        print(f"\nTraining failed with return code {return_code}")
        sys.exit(return_code)

    # Upload results
    if args.upload_repo:
        upload_results(args, Path("outputs"))

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
