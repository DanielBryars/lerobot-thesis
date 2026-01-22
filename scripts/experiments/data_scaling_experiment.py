#!/usr/bin/env python
"""
Data Scaling Experiment: Train ACT with varying amounts of training data.

Trains ACT models with 1, 2, 5, 10, 20, 40, 60, 80, 100, 120, 140, 157 episodes
and evaluates ALL checkpoints for each, creating a results matrix.

Usage:
    python scripts/experiments/data_scaling_experiment.py

Results saved to: outputs/experiments/data_scaling/
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add project root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.configs.types import FeatureType
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

from utils.training import run_evaluation, save_checkpoint, CachedDataset, cycle, get_scene_metadata

# Experiment configuration
DATASET = "danbhf/sim_pick_place_157ep"
EPISODE_COUNTS = [1, 2, 5, 10, 20, 40, 60, 80, 100, 120, 140, 157]
TRAINING_STEPS = 45000  # Same as original model
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
CHUNK_SIZE = 100
SAVE_FREQ = 5000  # Checkpoint every 5k steps
EVAL_EPISODES = 20  # Episodes per evaluation

# Pre-computed episode boundaries (frames at end of each episode count)
# This avoids recomputing for each run
EPISODE_FRAME_COUNTS = {
    1: 163, 2: 290, 5: 648, 10: 1385, 20: 2776,
    40: 6559, 60: 9625, 80: 12421, 100: 15116,
    120: 17727, 140: 20336, 157: 22534
}


def get_episode_frame_count(max_episodes):
    """Get the number of frames in the first N episodes using pre-computed values."""
    return EPISODE_FRAME_COUNTS.get(max_episodes, EPISODE_FRAME_COUNTS[157])


def train_with_episodes(
    num_episodes: int,
    output_dir: Path,
    device: torch.device,
    dataset_metadata: LeRobotDatasetMetadata,
):
    """Train ACT with a specific number of episodes."""

    print(f"\n{'='*60}", flush=True)
    print(f"Training with {num_episodes} episodes", flush=True)
    print(f"{'='*60}", flush=True)

    # Get features for policy
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}

    # Remove action_joints if present
    if 'action_joints' in output_features:
        del output_features['action_joints']

    input_features = {key: ft for key, ft in features.items()
                      if key not in output_features and key != 'action_joints' and key != 'action'}

    # Create ACT config
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=CHUNK_SIZE,
        n_action_steps=CHUNK_SIZE,
    )

    # Create policy
    policy = ACTPolicy(cfg)
    policy.train()
    policy.to(device)

    # Create pre/post processors
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    # Set up delta timestamps for ACT
    delta_timestamps = {
        "action": [i / dataset_metadata.fps for i in range(CHUNK_SIZE)],
    }
    for key in input_features:
        delta_timestamps[key] = [0.0]

    # Load full dataset
    full_dataset = LeRobotDataset(DATASET, delta_timestamps=delta_timestamps)

    # Filter to first N episodes
    frame_count = get_episode_frame_count(num_episodes)
    if frame_count < len(full_dataset):
        # Create subset using torch Subset with sequential indices
        from torch.utils.data import Subset
        indices = list(range(frame_count))
        dataset = Subset(full_dataset, indices)
        print(f"  Using {len(dataset)} frames from {num_episodes} episodes", flush=True)
    else:
        dataset = full_dataset
        print(f"  Using full dataset: {len(dataset)} frames from {num_episodes} episodes", flush=True)

    # Create dataloader (num_workers=0 for Windows compatibility)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(policy.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAINING_STEPS, eta_min=1e-7)

    # Training metadata
    camera_names = [key.replace("observation.images.", "") for key in input_features.keys()
                    if key.startswith("observation.images.")]

    training_metadata = {
        "dataset_repo_id": DATASET,
        "num_episodes": num_episodes,
        "total_frames": len(dataset),
        "cameras": camera_names,
        "chunk_size": CHUNK_SIZE,
        "fps": dataset_metadata.fps,
    }

    # Training loop
    step = 0
    data_iter = cycle(dataloader)
    start_time = time.time()
    checkpoints_saved = []

    print(f"  Starting training for {TRAINING_STEPS} steps...", flush=True)

    while step < TRAINING_STEPS:
        batch = next(data_iter)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch = preprocessor(batch)

        # Fix tensor dimensions
        if "observation.state" in batch and batch["observation.state"].dim() == 3:
            batch["observation.state"] = batch["observation.state"].squeeze(1)

        # Forward pass
        loss, output_dict = policy.forward(batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()

        step += 1

        # Progress update
        if step % 1000 == 0:
            elapsed = time.time() - start_time
            eta = (TRAINING_STEPS - step) / (step / elapsed) / 60
            print(f"    Step {step}/{TRAINING_STEPS}, Loss: {loss.item():.4f}, ETA: {eta:.1f}m", flush=True)

        # Save checkpoint
        if step % SAVE_FREQ == 0:
            checkpoint_name = f"checkpoint_{step:06d}"
            save_checkpoint(
                policy, optimizer, scheduler, step, output_dir,
                training_metadata=training_metadata,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
            )
            checkpoints_saved.append(checkpoint_name)

    # Save final checkpoint
    save_checkpoint(
        policy, optimizer, scheduler, TRAINING_STEPS, output_dir,
        training_metadata=training_metadata,
        checkpoint_name="final",
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )
    checkpoints_saved.append("final")

    elapsed = time.time() - start_time
    print(f"  Training complete in {elapsed/60:.1f} minutes", flush=True)

    return checkpoints_saved, preprocessor, postprocessor


def evaluate_all_checkpoints(
    model_dir: Path,
    checkpoints: list,
    device: torch.device,
    eval_episodes: int = EVAL_EPISODES,
):
    """Evaluate all checkpoints and return results."""
    results = {}

    for checkpoint_name in checkpoints:
        checkpoint_path = model_dir / checkpoint_name
        print(f"    Evaluating {checkpoint_name}...", flush=True)

        try:
            # Load policy
            policy = ACTPolicy.from_pretrained(str(checkpoint_path))
            policy.eval()
            policy.to(device)

            # Load processors
            preprocessor, postprocessor = make_pre_post_processors(
                policy.config, pretrained_path=str(checkpoint_path)
            )

            # Run evaluation
            success_rate, avg_steps, avg_time, _, _, failure_summary = run_evaluation(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                device=device,
                num_episodes=eval_episodes,
                randomize=True,
                verbose=False,
                analyze_failures=True,
            )

            # Convert failure_summary to JSON-serializable format
            serializable_summary = {}
            if failure_summary:
                for key, value in failure_summary.items():
                    if isinstance(value, dict):
                        # Convert enum keys to strings
                        serializable_summary[key] = {str(k): v for k, v in value.items()}
                    else:
                        serializable_summary[key] = value

            results[checkpoint_name] = {
                "success_rate": success_rate,
                "avg_steps": avg_steps,
                "avg_time": avg_time,
                "failure_summary": serializable_summary,
            }

            print(f"      Success: {success_rate*100:.1f}%", flush=True)

            # Clean up
            del policy
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"      ERROR: {e}", flush=True)
            results[checkpoint_name] = {"error": str(e)}

    return results


def main():
    global TRAINING_STEPS, SAVE_FREQ, EVAL_EPISODES, EPISODE_COUNTS

    parser = argparse.ArgumentParser(description="Data scaling experiment")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume-from", type=int, default=None,
                        help="Resume from a specific episode count (skip earlier ones)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run evaluation on existing checkpoints")
    parser.add_argument("--test", action="store_true",
                        help="Quick test mode: 500 steps, 5 eval episodes, only 1 and 2 episode counts")
    args = parser.parse_args()

    # Test mode overrides
    if args.test:
        TRAINING_STEPS = 500
        SAVE_FREQ = 250
        EVAL_EPISODES = 5
        EPISODE_COUNTS = [1, 2]
        print("TEST MODE: Reduced settings for quick validation")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # Create output directory
    experiment_dir = REPO_ROOT / "outputs" / "experiments" / "data_scaling"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset metadata
    print(f"Loading dataset metadata: {DATASET}", flush=True)
    dataset_metadata = LeRobotDatasetMetadata(DATASET)
    print(f"Total episodes: {dataset_metadata.total_episodes}", flush=True)
    print(f"Total frames: {dataset_metadata.total_frames}", flush=True)

    # Results storage
    all_results = {}
    results_file = experiment_dir / "results.json"

    # Load existing results if resuming
    if results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)
        print(f"Loaded existing results for {len(all_results)} episode counts", flush=True)

    # Determine which episode counts to process
    episode_counts = EPISODE_COUNTS
    if args.resume_from:
        episode_counts = [n for n in EPISODE_COUNTS if n >= args.resume_from]
        print(f"Resuming from {args.resume_from} episodes", flush=True)

    experiment_start = time.time()

    for num_episodes in episode_counts:
        print(f"\n{'#'*60}", flush=True)
        print(f"# EPISODE COUNT: {num_episodes}", flush=True)
        print(f"{'#'*60}", flush=True)

        model_dir = experiment_dir / f"ep_{num_episodes:03d}"

        if args.eval_only:
            # Find existing checkpoints
            if model_dir.exists():
                checkpoints = sorted([d.name for d in model_dir.iterdir()
                                     if d.is_dir() and (d.name.startswith("checkpoint_") or d.name == "final")])
            else:
                print(f"  No model directory found, skipping", flush=True)
                continue
        else:
            # Train model
            model_dir.mkdir(parents=True, exist_ok=True)

            checkpoints, preprocessor, postprocessor = train_with_episodes(
                num_episodes=num_episodes,
                output_dir=model_dir,
                device=device,
                dataset_metadata=dataset_metadata,
            )

        # Evaluate all checkpoints
        print(f"\n  Evaluating {len(checkpoints)} checkpoints...", flush=True)
        eval_results = evaluate_all_checkpoints(
            model_dir=model_dir,
            checkpoints=checkpoints,
            device=device,
        )

        # Store results
        all_results[str(num_episodes)] = {
            "num_episodes": num_episodes,
            "checkpoints": eval_results,
            "timestamp": datetime.now().isoformat(),
        }

        # Save results after each episode count (in case of interruption)
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  Results saved to {results_file}", flush=True)

        # Create summary CSV
        create_summary_csv(all_results, experiment_dir)

    # Final summary
    total_time = time.time() - experiment_start
    print(f"\n{'='*60}", flush=True)
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}", flush=True)
    print(f"Total time: {total_time/3600:.1f} hours", flush=True)
    print(f"Results: {results_file}", flush=True)
    print(f"Summary CSV: {experiment_dir / 'summary.csv'}", flush=True)


def create_summary_csv(all_results: dict, output_dir: Path):
    """Create a summary CSV with success rates for each episode count and checkpoint."""

    # Get all checkpoint names
    checkpoint_names = set()
    for ep_data in all_results.values():
        if "checkpoints" in ep_data:
            checkpoint_names.update(ep_data["checkpoints"].keys())

    # Sort checkpoints
    def sort_key(name):
        if name == "final":
            return float('inf')
        try:
            return int(name.split("_")[1])
        except:
            return 0
    checkpoint_names = sorted(checkpoint_names, key=sort_key)

    # Create matrix
    rows = []
    for ep_str, ep_data in sorted(all_results.items(), key=lambda x: int(x[0])):
        row = {"episodes": int(ep_str)}
        if "checkpoints" in ep_data:
            for ckpt in checkpoint_names:
                if ckpt in ep_data["checkpoints"]:
                    ckpt_data = ep_data["checkpoints"][ckpt]
                    if "success_rate" in ckpt_data:
                        row[ckpt] = ckpt_data["success_rate"]
                    else:
                        row[ckpt] = None
                else:
                    row[ckpt] = None
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = output_dir / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Summary CSV saved to {csv_path}", flush=True)

    # Print summary table
    print("\n  Success Rate Matrix (episodes × checkpoints):")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
