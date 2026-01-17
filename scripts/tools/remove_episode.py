#!/usr/bin/env python
"""
Remove one or more episodes from a LeRobot dataset.

Usage:
    python scripts/tools/remove_episode.py danbhf/dataset_name --episodes 16
    python scripts/tools/remove_episode.py danbhf/dataset_name --episodes 5 10 16
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download, HfApi


def main():
    parser = argparse.ArgumentParser(description="Remove episodes from a LeRobot dataset")
    parser.add_argument("dataset", help="HuggingFace dataset ID (e.g., danbhf/sim_pick_place_xxx)")
    parser.add_argument("--episodes", type=int, nargs="+", required=True, help="Episode indices to remove")
    parser.add_argument("--output", help="Output directory (default: datasets/<name>_filtered)")
    parser.add_argument("--upload", action="store_true", help="Upload filtered dataset to HuggingFace")
    parser.add_argument("--new_repo", help="New repo name for upload (default: overwrites original)")
    args = parser.parse_args()

    episodes_to_remove = set(args.episodes)
    print(f"Removing episodes: {sorted(episodes_to_remove)}")

    # Download dataset
    print(f"\nDownloading {args.dataset}...")
    local_path = Path(snapshot_download(args.dataset, repo_type="dataset"))
    print(f"Downloaded to: {local_path}")

    # Setup output directory
    dataset_name = args.dataset.split("/")[-1]
    output_dir = Path(args.output) if args.output else Path("datasets") / f"{dataset_name}_filtered"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    print(f"Output directory: {output_dir}")

    # Load episodes metadata
    episodes_path = local_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episodes_df = pd.read_parquet(episodes_path)
    print(f"\nOriginal episodes: {len(episodes_df)}")

    # Filter episodes
    keep_mask = ~episodes_df["episode_index"].isin(episodes_to_remove)
    filtered_episodes = episodes_df[keep_mask].copy()
    removed_count = len(episodes_df) - len(filtered_episodes)
    print(f"Removing {removed_count} episodes, keeping {len(filtered_episodes)}")

    if len(filtered_episodes) == 0:
        print("Error: No episodes remaining!")
        sys.exit(1)

    # Load data parquet
    data_path = local_path / "data" / "chunk-000" / "file-000.parquet"
    data_df = pd.read_parquet(data_path)
    print(f"Original frames: {len(data_df)}")

    # Filter data
    data_filtered = data_df[~data_df["episode_index"].isin(episodes_to_remove)].copy()
    print(f"Remaining frames: {len(data_filtered)}")

    # Create episode index mapping (old -> new)
    old_indices = sorted(filtered_episodes["episode_index"].unique())
    index_map = {old: new for new, old in enumerate(old_indices)}
    print(f"Index mapping: {index_map}")

    # Remap episode indices in data
    data_filtered["episode_index"] = data_filtered["episode_index"].map(index_map)

    # Recalculate frame indices
    data_filtered = data_filtered.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    data_filtered["index"] = range(len(data_filtered))

    # Update episodes metadata
    filtered_episodes["episode_index"] = filtered_episodes["episode_index"].map(index_map)

    # Recalculate dataset_from_index and dataset_to_index
    new_from_indices = []
    new_to_indices = []
    current_idx = 0
    for _, row in filtered_episodes.iterrows():
        length = row["length"]
        new_from_indices.append(current_idx)
        new_to_indices.append(current_idx + length)
        current_idx += length

    filtered_episodes["dataset_from_index"] = new_from_indices
    filtered_episodes["dataset_to_index"] = new_to_indices

    # Process videos
    print("\nProcessing videos...")
    videos_dir = local_path / "videos"
    output_videos_dir = output_dir / "videos"

    for cam_dir in videos_dir.iterdir():
        if not cam_dir.is_dir():
            continue

        cam_name = cam_dir.name
        print(f"  Processing {cam_name}...")

        input_video = cam_dir / "chunk-000" / "file-000.mp4"
        if not input_video.exists():
            print(f"    Warning: {input_video} not found, skipping")
            continue

        # Create output directory
        out_cam_dir = output_videos_dir / cam_name / "chunk-000"
        out_cam_dir.mkdir(parents=True, exist_ok=True)
        output_video = out_cam_dir / "file-000.mp4"

        # Build ffmpeg filter to select frames
        # We need to extract specific frame ranges and concatenate
        # Get frame ranges to keep from original episodes metadata
        frame_ranges = []
        for _, row in episodes_df.iterrows():
            ep_idx = row["episode_index"]
            if ep_idx not in episodes_to_remove:
                # Get timestamp columns for this camera (has videos/ prefix)
                from_ts_col = f"videos/{cam_name}/from_timestamp"
                to_ts_col = f"videos/{cam_name}/to_timestamp"
                if from_ts_col in row.index and to_ts_col in row.index:
                    frame_ranges.append((row[from_ts_col], row[to_ts_col]))

        if not frame_ranges:
            print(f"    No frame ranges found for {cam_name}")
            continue

        # Use ffmpeg to extract and concatenate segments
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            segment_files = []

            for i, (start_ts, end_ts) in enumerate(frame_ranges):
                segment_file = tmpdir / f"segment_{i:04d}.mp4"
                duration = end_ts - start_ts

                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(start_ts),
                    "-i", str(input_video),
                    "-t", str(duration),
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "18",
                    "-an",
                    str(segment_file)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"    Error extracting segment {i}: {result.stderr[:200]}")
                    continue
                segment_files.append(segment_file)

            # Create concat file
            concat_file = tmpdir / "concat.txt"
            with open(concat_file, "w") as f:
                for seg in segment_files:
                    f.write(f"file '{seg}'\n")

            # Concatenate segments
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
                str(output_video)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"    Error concatenating: {result.stderr[:200]}")
            else:
                print(f"    Created {output_video}")

    # Update video timestamps in episodes metadata
    # Recalculate based on cumulative frame counts
    fps = 30  # Assuming 30 fps
    for cam_dir in videos_dir.iterdir():
        if not cam_dir.is_dir():
            continue
        cam_name = cam_dir.name
        from_col = f"videos/{cam_name}/from_timestamp"
        to_col = f"videos/{cam_name}/to_timestamp"

        if from_col in filtered_episodes.columns:
            current_ts = 0.0
            new_from_ts = []
            new_to_ts = []
            for _, row in filtered_episodes.iterrows():
                duration = row["length"] / fps
                new_from_ts.append(current_ts)
                new_to_ts.append(current_ts + duration)
                current_ts += duration
            filtered_episodes[from_col] = new_from_ts
            filtered_episodes[to_col] = new_to_ts

    # Save filtered data
    print("\nSaving filtered dataset...")

    # Data parquet
    data_out_dir = output_dir / "data" / "chunk-000"
    data_out_dir.mkdir(parents=True, exist_ok=True)
    data_filtered.to_parquet(data_out_dir / "file-000.parquet")

    # Episodes parquet
    episodes_out_dir = output_dir / "meta" / "episodes" / "chunk-000"
    episodes_out_dir.mkdir(parents=True, exist_ok=True)
    filtered_episodes.to_parquet(episodes_out_dir / "file-000.parquet")

    # Copy and update info.json
    info_path = local_path / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    info["total_episodes"] = len(filtered_episodes)
    info["total_frames"] = len(data_filtered)
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)
    with open(output_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # Copy other meta files
    for meta_file in ["stats.json", "tasks.parquet", "recording_metadata.json", "episode_scenes.json"]:
        src = local_path / "meta" / meta_file
        if src.exists():
            shutil.copy(src, output_dir / "meta" / meta_file)

    # Update episode_scenes.json if it exists
    scenes_path = output_dir / "meta" / "episode_scenes.json"
    if scenes_path.exists():
        with open(scenes_path) as f:
            scenes = json.load(f)
        # Filter and remap scenes
        new_scenes = {}
        for old_idx_str, scene_data in scenes.items():
            old_idx = int(old_idx_str)
            if old_idx not in episodes_to_remove:
                new_idx = index_map[old_idx]
                new_scenes[str(new_idx)] = scene_data
        with open(scenes_path, "w") as f:
            json.dump(new_scenes, f, indent=2)

    print(f"\nFiltered dataset saved to: {output_dir}")
    print(f"  Episodes: {len(filtered_episodes)}")
    print(f"  Frames: {len(data_filtered)}")

    # Upload if requested
    if args.upload:
        repo_id = args.new_repo or args.dataset
        print(f"\nUploading to {repo_id}...")
        api = HfApi()
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"Uploaded to: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
