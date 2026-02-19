#!/usr/bin/env python
"""
Convert a LeRobot dataset's actions from absolute to delta joint actions.

Creates a new HuggingFace dataset with delta actions:
- delta[0] = action[0] - state[0]  (first action relative to current state)
- delta[t] = action[t] - action[t-1]  (subsequent relative to previous)
- Gripper (dim 5) stays absolute

Usage:
    python scripts/tools/convert_to_delta_actions.py \
        danbhf/sim_pick_place_2pos_220ep_v2 \
        danbhf/sim_pick_place_2pos_220ep_v2_delta

    # Dry run (compute stats only, don't push)
    python scripts/tools/convert_to_delta_actions.py \
        danbhf/sim_pick_place_2pos_220ep_v2 \
        danbhf/sim_pick_place_2pos_220ep_v2_delta \
        --dry-run
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.constants import NUM_JOINTS, GRIPPER_IDX


def convert_episode_to_delta(actions: np.ndarray, states: np.ndarray, action_dim: int = 5) -> np.ndarray:
    """Convert absolute actions to delta actions for one episode.

    Args:
        actions: (T, D) absolute actions
        states: (T, D) observation states (joint positions)
        action_dim: Number of arm joint dimensions (excluding gripper)

    Returns:
        (T, D) delta actions (gripper stays absolute)
    """
    T, D = actions.shape
    deltas = np.zeros_like(actions)

    # First step: delta relative to current state
    deltas[0, :action_dim] = actions[0, :action_dim] - states[0, :action_dim]
    if D > action_dim:
        deltas[0, action_dim:] = actions[0, action_dim:]  # Gripper stays absolute

    # Subsequent steps: delta relative to previous action
    for t in range(1, T):
        deltas[t, :action_dim] = actions[t, :action_dim] - actions[t - 1, :action_dim]
        if D > action_dim:
            deltas[t, action_dim:] = actions[t, action_dim:]  # Gripper stays absolute

    return deltas


def main():
    parser = argparse.ArgumentParser(description="Convert dataset to delta joint actions")
    parser.add_argument("source", type=str, help="Source HuggingFace dataset ID")
    parser.add_argument("target", type=str, help="Target HuggingFace dataset ID")
    parser.add_argument("--action-dim", type=int, default=NUM_JOINTS - 1,
                        help=f"Arm joint dimensions to convert (default: {NUM_JOINTS - 1}, excludes gripper)")
    parser.add_argument("--dry-run", action="store_true", help="Compute stats only, don't push")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download, list_repo_files, HfApi

    print(f"Source dataset: {args.source}")
    print(f"Target dataset: {args.target}")
    print(f"Action dim (arm joints): {args.action_dim}")

    # List all files in source dataset
    files = list_repo_files(args.source, repo_type="dataset")
    parquet_files = sorted([f for f in files if f.endswith('.parquet') and 'data' in f])
    meta_files = [f for f in files if not f.endswith('.parquet') and 'data' not in f.split('/')[0]]

    print(f"Found {len(parquet_files)} parquet files, {len(meta_files)} meta files")

    # Process each parquet file
    all_delta_stats = []
    converted_parquets = []

    for pf in parquet_files:
        print(f"\nProcessing {pf}...")
        local_path = hf_hub_download(args.source, pf, repo_type="dataset")
        df = pd.read_parquet(local_path)

        actions = np.array(df['action'].tolist())
        states = np.array(df['observation.state'].tolist())
        episode_indices = np.array(df['episode_index'].tolist())

        print(f"  Shape: actions={actions.shape}, states={states.shape}")
        print(f"  Episodes: {np.unique(episode_indices)[:5]}... ({len(np.unique(episode_indices))} total)")

        # Convert per-episode
        deltas = np.zeros_like(actions)
        unique_episodes = np.unique(episode_indices)

        for ep_idx in unique_episodes:
            mask = episode_indices == ep_idx
            ep_actions = actions[mask]
            ep_states = states[mask]
            ep_deltas = convert_episode_to_delta(ep_actions, ep_states, args.action_dim)
            deltas[mask] = ep_deltas

        # Compute stats
        arm_deltas = deltas[:, :args.action_dim]
        print(f"  Delta arm stats:")
        print(f"    Mean: {arm_deltas.mean(axis=0)}")
        print(f"    Std:  {arm_deltas.std(axis=0)}")
        print(f"    Min:  {arm_deltas.min(axis=0)}")
        print(f"    Max:  {arm_deltas.max(axis=0)}")

        all_delta_stats.append(arm_deltas)

        # Replace actions in dataframe
        df['action'] = [deltas[i].tolist() for i in range(len(deltas))]
        converted_parquets.append((pf, df))

    # Overall stats
    all_deltas = np.concatenate(all_delta_stats, axis=0)
    print(f"\n{'='*60}")
    print(f"Overall delta stats ({len(all_deltas)} frames):")
    print(f"  Mean: {all_deltas.mean(axis=0)}")
    print(f"  Std:  {all_deltas.std(axis=0)}")
    print(f"  Min:  {all_deltas.min(axis=0)}")
    print(f"  Max:  {all_deltas.max(axis=0)}")
    print(f"{'='*60}")

    if args.dry_run:
        print("\nDry run complete. No data pushed.")
        return

    # Push to HuggingFace
    api = HfApi()
    print(f"\nCreating target dataset: {args.target}")
    api.create_repo(args.target, repo_type="dataset", exist_ok=True)

    # Upload converted parquet files
    import tempfile
    for pf, df in converted_parquets:
        print(f"  Uploading {pf}...")
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            df.to_parquet(tmp.name)
            api.upload_file(
                path_or_fileobj=tmp.name,
                path_in_repo=pf,
                repo_id=args.target,
                repo_type="dataset",
            )

    # Copy meta files from source
    for mf in meta_files:
        if mf.startswith('.') or mf == 'README.md':
            continue
        print(f"  Copying meta file: {mf}")
        local_path = hf_hub_download(args.source, mf, repo_type="dataset")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=mf,
            repo_id=args.target,
            repo_type="dataset",
        )

    print(f"\nDone! Delta dataset pushed to: {args.target}")


if __name__ == "__main__":
    main()
