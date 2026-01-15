#!/usr/bin/env python
"""
Launch training jobs on vast.ai.

This script helps you:
1. Find suitable GPU instances
2. Launch training with your configuration
3. Monitor running instances

Prerequisites:
    pip install vastai
    vastai set api-key YOUR_API_KEY

Usage:
    # Search for available instances
    python remote/launch_vast.py search --gpu RTX4090

    # Launch a training job
    python remote/launch_vast.py launch --gpu RTX4090 \\
        --dataset danbhf/sim_pick_place_40ep_rgbd_ee \\
        --cameras wrist_cam,overhead_cam,overhead_cam_depth \\
        --steps 100000

    # List running instances
    python remote/launch_vast.py list

    # Stop an instance
    python remote/launch_vast.py stop --instance_id 12345
"""

import argparse
import json
import os
import subprocess
import sys


# Docker image on Docker Hub
DOCKER_IMAGE = "aerdanielbryars101/lerobot-training:latest"


def run_vastai_cmd(args: list, capture_output: bool = True):
    """Run a vastai CLI command."""
    cmd = ["vastai"] + args
    if capture_output:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout, result.stderr, result.returncode
    else:
        return subprocess.run(cmd)


def search_instances(args):
    """Search for available GPU instances."""
    # Minimum disk space needed:
    # - Pi0 checkpoints are ~13GB each (train_state + params + assets)
    # - 40k steps with save_interval=5000 → 8 checkpoints → ~100GB
    # - Plus model weights, datasets, etc. → need 150GB minimum
    min_disk = args.min_disk if hasattr(args, 'min_disk') and args.min_disk else 150

    # Build search query
    query_parts = [
        "reliability > 0.95",      # High reliability
        "inet_down > 100",         # Fast internet
        f"disk_space > {min_disk}", # Enough disk for Pi0 checkpoints
        "cuda_vers >= 12.0",       # CUDA 12+
    ]

    if args.gpu:
        # GPU names use underscores: RTX_4090, RTX_5090, etc.
        gpu_name = args.gpu.replace(" ", "_")
        query_parts.append(f"gpu_name={gpu_name}")

    if args.max_price:
        query_parts.append(f"dph_total <= {args.max_price}")

    query = " ".join(query_parts)

    print(f"Searching for instances with:")
    print(f"  Min disk: {min_disk}GB (for Pi0 checkpoints)")
    print(f"  GPU: {args.gpu or 'any'}")
    print(f"  Query: {query}")
    print("-" * 60)

    stdout, stderr, code = run_vastai_cmd([
        "search", "offers",
        query,
        "--order", "dph_total",  # Sort by price
        "--limit", str(args.limit),
    ])

    if code != 0:
        print(f"Error: {stderr}")
        return

    print(stdout)


def launch_instance(args):
    """Launch a training instance."""
    # Check for required environment variables
    hf_token = os.environ.get("HF_TOKEN")
    wandb_key = os.environ.get("WANDB_API_KEY")

    if not hf_token:
        print("ERROR: HF_TOKEN environment variable not set")
        sys.exit(1)

    if not wandb_key:
        print("WARNING: WANDB_API_KEY not set, WandB logging will be disabled")

    # Build the training command
    train_cmd = [
        "--dataset", args.dataset,
        "--steps", str(args.steps),
        "--batch_size", str(args.batch_size),
        "--save_freq", str(args.save_freq),
        "--eval_episodes", str(args.eval_episodes),
    ]

    if args.cameras:
        train_cmd.extend(["--cameras", args.cameras])

    if args.use_joint_actions:
        train_cmd.append("--use_joint_actions")

    if args.run_name:
        train_cmd.extend(["--run_name", args.run_name])

    if args.upload_repo:
        train_cmd.extend(["--upload_repo", args.upload_repo])

    # Environment variables to pass
    env_vars = f"-e HF_TOKEN={hf_token}"
    if wandb_key:
        env_vars += f" -e WANDB_API_KEY={wandb_key}"

    # Build onstart script (runs when instance starts)
    onstart_script = f'''
cd /app
python remote/train_remote.py {" ".join(train_cmd)}
'''

    # Search for a suitable instance first
    # Need 150GB+ for Pi0 checkpoints (see search_instances for details)
    min_disk = args.min_disk if hasattr(args, 'min_disk') and args.min_disk else 150
    query_parts = [
        "reliability > 0.95",
        "inet_down > 100",
        f"disk_space > {min_disk}",
        "cuda_vers >= 12.0",
    ]

    if args.gpu:
        query_parts.append(f"gpu_name={args.gpu}")

    if args.max_price:
        query_parts.append(f"dph_total <= {args.max_price}")

    query = " ".join(query_parts)

    # Disk space to request
    disk_gb = args.disk if hasattr(args, 'disk') and args.disk else 150

    print(f"Launching training job:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Steps: {args.steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Cameras: {args.cameras or 'all'}")
    print(f"  GPU filter: {args.gpu or 'any'}")
    print(f"  Min disk: {min_disk}GB (filter)")
    print(f"  Disk request: {disk_gb}GB")
    print(f"  Max price: ${args.max_price}/hr" if args.max_price else "  Max price: unlimited")
    print("-" * 60)

    # Create the instance
    create_cmd = [
        "create", "instance",
        query,
        "--image", DOCKER_IMAGE,
        "--disk", str(disk_gb),  # Need 150GB+ for Pi0 checkpoints
        "--onstart-cmd", onstart_script,
        "--env", env_vars.replace("-e ", ""),
    ]

    print(f"Running: vastai {' '.join(create_cmd[:5])}...")

    stdout, stderr, code = run_vastai_cmd(create_cmd)

    if code != 0:
        print(f"Error creating instance: {stderr}")
        return

    print(stdout)
    print("\nInstance launched! Use 'python remote/launch_vast.py list' to monitor.")


def list_instances(args):
    """List running instances."""
    stdout, stderr, code = run_vastai_cmd(["show", "instances"])

    if code != 0:
        print(f"Error: {stderr}")
        return

    print(stdout)


def stop_instance(args):
    """Stop a running instance."""
    if not args.instance_id:
        print("ERROR: --instance_id required")
        return

    print(f"Stopping instance {args.instance_id}...")

    stdout, stderr, code = run_vastai_cmd(["destroy", "instance", str(args.instance_id)])

    if code != 0:
        print(f"Error: {stderr}")
        return

    print(stdout)


def build_image(args):
    """Build and push Docker image."""
    print("Building Docker image...")

    # Build
    build_result = subprocess.run([
        "docker", "build",
        "-t", DOCKER_IMAGE,
        "-f", "remote/Dockerfile",
        "."
    ])

    if build_result.returncode != 0:
        print("Build failed!")
        return

    print("\nPushing to Docker Hub...")
    print("NOTE: You need to be logged in: docker login")

    push_result = subprocess.run([
        "docker", "push", DOCKER_IMAGE
    ])

    if push_result.returncode != 0:
        print("Push failed! Make sure you're logged in to Docker Hub.")
        return

    print(f"\nImage pushed successfully: {DOCKER_IMAGE}")


def main():
    parser = argparse.ArgumentParser(description="Launch training on vast.ai")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for available instances")
    search_parser.add_argument("--gpu", type=str, help="GPU name filter (e.g., RTX4090, A100)")
    search_parser.add_argument("--max_price", type=float, help="Maximum price per hour")
    search_parser.add_argument("--min_disk", type=int, default=150, help="Minimum disk space in GB (default: 150 for Pi0)")
    search_parser.add_argument("--limit", type=int, default=10, help="Number of results")

    # Launch command
    launch_parser = subparsers.add_parser("launch", help="Launch a training instance")
    launch_parser.add_argument("--dataset", type=str, required=True, help="HuggingFace dataset ID")
    launch_parser.add_argument("--cameras", type=str, help="Comma-separated camera names")
    launch_parser.add_argument("--steps", type=int, default=50000, help="Training steps")
    launch_parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    launch_parser.add_argument("--save_freq", type=int, default=5000, help="Checkpoint save frequency")
    launch_parser.add_argument("--eval_episodes", type=int, default=30, help="Eval episodes per checkpoint")
    launch_parser.add_argument("--use_joint_actions", action="store_true", help="Use joint action space")
    launch_parser.add_argument("--run_name", type=str, help="WandB run name")
    launch_parser.add_argument("--upload_repo", type=str, help="HuggingFace repo for final model")
    launch_parser.add_argument("--gpu", type=str, help="GPU name filter")
    launch_parser.add_argument("--max_price", type=float, help="Maximum price per hour")
    launch_parser.add_argument("--min_disk", type=int, default=150, help="Minimum disk space in GB (default: 150 for Pi0)")
    launch_parser.add_argument("--disk", type=int, default=150, help="Disk space to request in GB (default: 150)")

    # List command
    list_parser = subparsers.add_parser("list", help="List running instances")

    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop a running instance")
    stop_parser.add_argument("--instance_id", type=int, help="Instance ID to stop")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build and push Docker image")

    args = parser.parse_args()

    if args.command == "search":
        search_instances(args)
    elif args.command == "launch":
        launch_instance(args)
    elif args.command == "list":
        list_instances(args)
    elif args.command == "stop":
        stop_instance(args)
    elif args.command == "build":
        build_image(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
