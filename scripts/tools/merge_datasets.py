#!/usr/bin/env python3
"""
Merge multiple LeRobot datasets (local or HuggingFace) into a single dataset.

Usage:
    # Merge two HuggingFace datasets
    python scripts/merge_datasets.py danbhf/sim_pick_place_20251229_101340 danbhf/sim_pick_place_20251229_144730 -o datasets/merged_40ep

    # Merge and upload to HuggingFace
    python scripts/merge_datasets.py danbhf/sim_pick_place_20251229_101340 danbhf/sim_pick_place_20251229_144730 -o datasets/merged_40ep --upload danbhf/sim_pick_place_merged_40ep
"""

import argparse
import shutil
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa


def load_parquet_as_df(path):
    """Load a parquet file as a pandas DataFrame."""
    return pq.read_table(path).to_pandas()


def save_df_as_parquet(df, path):
    """Save a pandas DataFrame as a parquet file."""
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)


def get_dataset_path(dataset_id: str) -> Path:
    """Get local path for a dataset, downloading from HuggingFace if needed."""
    # Check if it's a local path
    local_path = Path(dataset_id)
    if local_path.exists():
        return local_path

    # Download from HuggingFace
    from huggingface_hub import snapshot_download

    print(f"Downloading {dataset_id} from HuggingFace...")
    cache_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / dataset_id.replace("/", "_")

    path = snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        local_dir=cache_dir,
    )
    return Path(path)


def get_video_keys(dataset_path: Path) -> list[str]:
    """Get list of video keys from a dataset."""
    videos_dir = dataset_path / "videos"
    if not videos_dir.exists():
        return []
    return [d.name for d in videos_dir.iterdir() if d.is_dir()]


def merge_datasets(dataset_paths: list[Path], output_path: Path):
    """Merge multiple datasets into one."""

    if output_path.exists():
        print(f"Removing existing {output_path}")
        shutil.rmtree(output_path)

    print(f"Merging {len(dataset_paths)} datasets into {output_path}")

    # Load all metadata
    infos = []
    for ds_path in dataset_paths:
        with open(ds_path / "meta" / "info.json") as f:
            infos.append(json.load(f))
        print(f"  {ds_path.name}: {infos[-1]['total_episodes']} episodes, {infos[-1]['total_frames']} frames")

    # Get video keys from first dataset
    video_keys = get_video_keys(dataset_paths[0])
    print(f"Video keys: {video_keys}")

    # Create output directory structure
    output_path.mkdir(parents=True)
    (output_path / "meta").mkdir()
    (output_path / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (output_path / "data" / "chunk-000").mkdir(parents=True)

    for video_key in video_keys:
        (output_path / "videos" / video_key / "chunk-000").mkdir(parents=True)

    # Track offsets
    episode_offset = 0
    frame_offset = 0
    chunk_idx = 0

    all_data_dfs = []
    all_episodes_dfs = []
    all_tasks = {}
    next_task_idx = 0

    # Track per-episode scene data and recording metadata
    merged_episode_scenes = {}
    merged_recording_metadata = []

    for ds_idx, ds_path in enumerate(dataset_paths):
        print(f"\nProcessing dataset {ds_idx + 1}: {ds_path.name}")

        # Load data
        data_df = load_parquet_as_df(ds_path / "data" / "chunk-000" / "file-000.parquet")
        episodes_df = load_parquet_as_df(ds_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
        tasks_df = load_parquet_as_df(ds_path / "meta" / "tasks.parquet")

        print(f"  Data shape: {data_df.shape}, Episodes: {len(episodes_df)}")

        # Load per-episode scene data if available
        episode_scenes_path = ds_path / "meta" / "episode_scenes.json"
        if episode_scenes_path.exists():
            with open(episode_scenes_path) as f:
                episode_scenes = json.load(f)
            # Remap episode indices and add to merged dict
            for old_ep_idx, scene_data in episode_scenes.items():
                new_ep_idx = str(int(old_ep_idx) + episode_offset)
                merged_episode_scenes[new_ep_idx] = scene_data
            print(f"  Loaded {len(episode_scenes)} episode scenes")

        # Load recording metadata if available
        recording_meta_path = ds_path / "meta" / "recording_metadata.json"
        if recording_meta_path.exists():
            with open(recording_meta_path) as f:
                recording_meta = json.load(f)
            recording_meta['_source_dataset'] = ds_path.name
            recording_meta['_episode_range'] = [episode_offset, episode_offset + infos[ds_idx]['total_episodes']]
            merged_recording_metadata.append(recording_meta)
            print(f"  Loaded recording metadata")

        # Build task mapping
        task_mapping = {}
        for _, row in tasks_df.iterrows():
            task_str = str(row.name) if hasattr(row, 'name') else str(row.get('task', ''))
            old_idx = row['task_index']
            if task_str in all_tasks:
                task_mapping[old_idx] = all_tasks[task_str]
            else:
                task_mapping[old_idx] = next_task_idx
                all_tasks[task_str] = next_task_idx
                next_task_idx += 1

        # Adjust data indices
        data_adjusted = data_df.copy()
        if episode_offset > 0 or frame_offset > 0:
            data_adjusted['episode_index'] = data_adjusted['episode_index'] + episode_offset
            data_adjusted['index'] = data_adjusted['index'] + frame_offset
            if 'task_index' in data_adjusted.columns:
                data_adjusted['task_index'] = data_adjusted['task_index'].map(task_mapping)

        all_data_dfs.append(data_adjusted)

        # Adjust episodes indices
        episodes_adjusted = episodes_df.copy()
        if episode_offset > 0 or frame_offset > 0:
            episodes_adjusted['episode_index'] = episodes_adjusted['episode_index'] + episode_offset
            episodes_adjusted['dataset_from_index'] = episodes_adjusted['dataset_from_index'] + frame_offset
            episodes_adjusted['dataset_to_index'] = episodes_adjusted['dataset_to_index'] + frame_offset

            # Update video chunk indices
            for video_key in video_keys:
                chunk_col = f'videos/{video_key}/chunk_index'
                if chunk_col in episodes_adjusted.columns:
                    episodes_adjusted[chunk_col] = chunk_idx

        all_episodes_dfs.append(episodes_adjusted)

        # Copy videos
        for video_key in video_keys:
            src = ds_path / "videos" / video_key / "chunk-000" / "file-000.mp4"
            if src.exists():
                if chunk_idx == 0:
                    dst = output_path / "videos" / video_key / "chunk-000" / "file-000.mp4"
                else:
                    chunk_dir = output_path / "videos" / video_key / f"chunk-{chunk_idx:03d}"
                    chunk_dir.mkdir(parents=True, exist_ok=True)
                    dst = chunk_dir / "file-000.mp4"
                shutil.copy2(src, dst)
                print(f"  Copied {video_key} to chunk-{chunk_idx:03d}")

        # Update offsets
        episode_offset += infos[ds_idx]['total_episodes']
        frame_offset += infos[ds_idx]['total_frames']
        chunk_idx += 1

    # Merge all dataframes
    print("\nMerging data...")
    merged_data = pd.concat(all_data_dfs, ignore_index=True)
    merged_episodes = pd.concat(all_episodes_dfs, ignore_index=True)

    print(f"Merged data shape: {merged_data.shape}")
    print(f"Merged episodes: {len(merged_episodes)}")

    # Save merged data
    save_df_as_parquet(merged_data, output_path / "data" / "chunk-000" / "file-000.parquet")
    save_df_as_parquet(merged_episodes, output_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    # Save tasks
    tasks_data = {'task_index': list(all_tasks.values())}
    tasks_df = pd.DataFrame(tasks_data)
    tasks_df.index = pd.Index(list(all_tasks.keys()))
    save_df_as_parquet(tasks_df, output_path / "meta" / "tasks.parquet")

    # Create merged info.json
    merged_info = infos[0].copy()
    merged_info['total_episodes'] = episode_offset
    merged_info['total_frames'] = frame_offset
    merged_info['total_tasks'] = len(all_tasks)
    merged_info['splits'] = {'train': f"0:{episode_offset}"}

    with open(output_path / "meta" / "info.json", 'w') as f:
        json.dump(merged_info, f, indent=4)

    # Calculate and save stats
    print("Calculating statistics...")
    stats = {}

    for col in merged_data.columns:
        if col in ['action', 'observation.state']:
            values = merged_data[col].tolist()
            arr = np.array(values)
            stats[col] = {
                'min': arr.min(axis=0).tolist(),
                'max': arr.max(axis=0).tolist(),
                'mean': arr.mean(axis=0).tolist(),
                'std': arr.std(axis=0).tolist(),
            }

    with open(output_path / "meta" / "stats.json", 'w') as f:
        json.dump(stats, f, indent=4)

    # Save merged episode scenes
    if merged_episode_scenes:
        with open(output_path / "meta" / "episode_scenes.json", 'w') as f:
            json.dump(merged_episode_scenes, f, indent=2)
        print(f"Saved {len(merged_episode_scenes)} episode scenes")

    # Save merged recording metadata
    if merged_recording_metadata:
        with open(output_path / "meta" / "recording_metadata.json", 'w') as f:
            json.dump(merged_recording_metadata, f, indent=2)
        print(f"Saved recording metadata from {len(merged_recording_metadata)} source datasets")

    print()
    print("=" * 60)
    print("Merge complete!")
    print(f"Output: {output_path}")
    print(f"Total episodes: {episode_offset}")
    print(f"Total frames: {frame_offset}")
    print(f"Total tasks: {len(all_tasks)}")
    print("=" * 60)

    return output_path


def upload_to_hub(local_path: Path, repo_id: str):
    """Upload merged dataset to HuggingFace Hub."""
    from huggingface_hub import HfApi

    print(f"\nUploading to {repo_id}...")
    api = HfApi()

    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(local_path),
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Uploaded to https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Merge LeRobot datasets")
    parser.add_argument("datasets", nargs="+", help="Dataset IDs or local paths to merge")
    parser.add_argument("-o", "--output", required=True, help="Output directory for merged dataset")
    parser.add_argument("--upload", type=str, help="HuggingFace repo ID to upload merged dataset")

    args = parser.parse_args()

    # Get dataset paths
    dataset_paths = [get_dataset_path(ds) for ds in args.datasets]
    output_path = Path(args.output)

    # Merge
    merge_datasets(dataset_paths, output_path)

    # Upload if requested
    if args.upload:
        upload_to_hub(output_path, args.upload)


if __name__ == "__main__":
    main()
