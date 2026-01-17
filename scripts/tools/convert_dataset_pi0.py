#!/usr/bin/env python3
"""
Convert a LeRobot dataset for Pi0 training.

Performs these conversions:
1. Convert gripper values from [0-97] (degrees) to [0-1] range
   - Gripper is at index 5 (6th dimension) in both action and observation.state
   - Pi0 expects [0, 1] where 0=open and 1=closed

Usage:
    # Convert and save locally
    python scripts/tools/convert_dataset_pi0.py danbhf/sim_pick_place_157ep -o datasets/sim_pick_place_pi0

    # Convert, remove bad episodes, and upload
    python scripts/tools/convert_dataset_pi0.py danbhf/sim_pick_place_157ep -o datasets/sim_pick_place_pi0 --remove-episodes 22 --upload danbhf/sim_pick_place_157ep_pi0
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download, HfApi


GRIPPER_INDEX = 5  # Gripper is 6th dimension (index 5)
GRIPPER_MAX = 97.0  # Max gripper value in degrees


def convert_gripper_values(arr: np.ndarray) -> np.ndarray:
    """Convert gripper values from [0-97] to [0-1] range.

    Args:
        arr: Array of shape (N, D) where D >= GRIPPER_INDEX+1

    Returns:
        Array with gripper values normalized to [0, 1]
    """
    arr = arr.copy()
    # Normalize gripper: divide by max value, clip to [0, 1]
    arr[:, GRIPPER_INDEX] = np.clip(arr[:, GRIPPER_INDEX] / GRIPPER_MAX, 0.0, 1.0)
    return arr


def main():
    parser = argparse.ArgumentParser(description="Convert LeRobot dataset for Pi0 training")
    parser.add_argument("dataset", help="HuggingFace dataset ID (e.g., danbhf/sim_pick_place_157ep)")
    parser.add_argument("-o", "--output", required=True, help="Output directory for converted dataset")
    parser.add_argument("--remove-episodes", type=int, nargs="*", default=[],
                        help="Episode indices to remove before conversion")
    parser.add_argument("--upload", type=str, help="HuggingFace repo ID to upload converted dataset")
    args = parser.parse_args()

    episodes_to_remove = set(args.remove_episodes)

    # Download dataset
    print(f"Downloading {args.dataset}...")
    local_path = Path(snapshot_download(args.dataset, repo_type="dataset"))
    print(f"Downloaded to: {local_path}")

    # Setup output directory
    output_dir = Path(args.output)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    print(f"Output directory: {output_dir}")

    # Load data
    data_path = local_path / "data" / "chunk-000" / "file-000.parquet"
    data_df = pd.read_parquet(data_path)
    print(f"Loaded {len(data_df)} frames")

    # Load episodes metadata
    episodes_path = local_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episodes_df = pd.read_parquet(episodes_path)
    print(f"Loaded {len(episodes_df)} episodes")

    # Remove episodes if requested
    if episodes_to_remove:
        print(f"\nRemoving episodes: {sorted(episodes_to_remove)}")

        # Filter data and episodes
        keep_mask = ~data_df["episode_index"].isin(episodes_to_remove)
        data_df = data_df[keep_mask].copy()

        keep_mask = ~episodes_df["episode_index"].isin(episodes_to_remove)
        episodes_df = episodes_df[keep_mask].copy()

        # Create episode index mapping (old -> new)
        old_indices = sorted(episodes_df["episode_index"].unique())
        index_map = {old: new for new, old in enumerate(old_indices)}

        # Remap episode indices
        data_df["episode_index"] = data_df["episode_index"].map(index_map)
        episodes_df["episode_index"] = episodes_df["episode_index"].map(index_map)

        # Recalculate frame indices
        data_df = data_df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
        data_df["index"] = range(len(data_df))

        # Recalculate dataset_from_index and dataset_to_index
        new_from_indices = []
        new_to_indices = []
        current_idx = 0
        for _, row in episodes_df.iterrows():
            length = row["length"]
            new_from_indices.append(current_idx)
            new_to_indices.append(current_idx + length)
            current_idx += length

        episodes_df["dataset_from_index"] = new_from_indices
        episodes_df["dataset_to_index"] = new_to_indices

        print(f"Remaining: {len(episodes_df)} episodes, {len(data_df)} frames")

    # Convert gripper values
    print("\nConverting gripper values from [0-97] to [0-1]...")

    # Convert action column
    actions = np.array(data_df["action"].tolist())
    print(f"  Action shape: {actions.shape}")
    print(f"  Gripper before: min={actions[:, GRIPPER_INDEX].min():.2f}, max={actions[:, GRIPPER_INDEX].max():.2f}")
    actions = convert_gripper_values(actions)
    print(f"  Gripper after:  min={actions[:, GRIPPER_INDEX].min():.4f}, max={actions[:, GRIPPER_INDEX].max():.4f}")
    data_df["action"] = list(actions)

    # Convert observation.state column
    states = np.array(data_df["observation.state"].tolist())
    print(f"  State shape: {states.shape}")
    print(f"  Gripper before: min={states[:, GRIPPER_INDEX].min():.2f}, max={states[:, GRIPPER_INDEX].max():.2f}")
    states = convert_gripper_values(states)
    print(f"  Gripper after:  min={states[:, GRIPPER_INDEX].min():.4f}, max={states[:, GRIPPER_INDEX].max():.4f}")
    data_df["observation.state"] = list(states)

    # Calculate new statistics
    print("\nCalculating statistics...")
    stats = {}
    for col in ["action", "observation.state"]:
        values = np.array(data_df[col].tolist())
        stats[col] = {
            "min": values.min(axis=0).tolist(),
            "max": values.max(axis=0).tolist(),
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
        }

    # Print stats for verification
    print("  Action stats:")
    print(f"    min: {[f'{x:.4f}' for x in stats['action']['min']]}")
    print(f"    max: {[f'{x:.4f}' for x in stats['action']['max']]}")
    print("  State stats:")
    print(f"    min: {[f'{x:.4f}' for x in stats['observation.state']['min']]}")
    print(f"    max: {[f'{x:.4f}' for x in stats['observation.state']['max']]}")

    # Save converted dataset
    print("\nSaving converted dataset...")

    # Copy directory structure
    (output_dir / "data" / "chunk-000").mkdir(parents=True)
    (output_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True)

    # Save data parquet
    data_df.to_parquet(output_dir / "data" / "chunk-000" / "file-000.parquet")

    # Save episodes parquet
    episodes_df.to_parquet(output_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    # Copy and update info.json
    with open(local_path / "meta" / "info.json") as f:
        info = json.load(f)
    info["total_episodes"] = len(episodes_df)
    info["total_frames"] = len(data_df)
    info["splits"] = {"train": f"0:{len(episodes_df)}"}
    with open(output_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # Save stats
    with open(output_dir / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Copy tasks.parquet
    shutil.copy(local_path / "meta" / "tasks.parquet", output_dir / "meta" / "tasks.parquet")

    # Copy optional meta files
    for meta_file in ["recording_metadata.json", "episode_scenes.json"]:
        src = local_path / "meta" / meta_file
        if src.exists():
            if episodes_to_remove and meta_file == "episode_scenes.json":
                # Remap episode indices in scene data
                with open(src) as f:
                    scenes = json.load(f)
                new_scenes = {}
                for old_idx_str, scene_data in scenes.items():
                    old_idx = int(old_idx_str)
                    if old_idx not in episodes_to_remove:
                        new_idx = index_map[old_idx]
                        new_scenes[str(new_idx)] = scene_data
                with open(output_dir / "meta" / meta_file, "w") as f:
                    json.dump(new_scenes, f, indent=2)
            else:
                shutil.copy(src, output_dir / "meta" / meta_file)

    # Copy videos directory (no changes needed for videos)
    videos_src = local_path / "videos"
    if videos_src.exists():
        print("Copying videos...")
        shutil.copytree(videos_src, output_dir / "videos")

    print(f"\nConverted dataset saved to: {output_dir}")
    print(f"  Episodes: {len(episodes_df)}")
    print(f"  Frames: {len(data_df)}")

    # Upload if requested
    if args.upload:
        print(f"\nUploading to {args.upload}...")
        api = HfApi()
        api.create_repo(args.upload, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=args.upload,
            repo_type="dataset",
        )
        print(f"Uploaded to: https://huggingface.co/datasets/{args.upload}")


if __name__ == "__main__":
    main()
