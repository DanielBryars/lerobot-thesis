#!/usr/bin/env python3
"""Split a dataset into per-subtask episodes.

Each source episode spanning all 4 subtask phases gets split into individual
segments (e.g., episode 5 frames 28-45 = PICK_UP becomes a new episode).

This enables training ACT on individual subtasks with shorter chunk sizes.

Usage:
    # Split all subtasks:
    python scripts/tools/split_subtask_dataset.py danbhf/sim_pick_place_2pos_220ep_v2 \
        -o datasets/sim_pick_place_220ep_subtasks

    # Only PICK_UP segments:
    python scripts/tools/split_subtask_dataset.py danbhf/sim_pick_place_2pos_220ep_v2 \
        -o datasets/sim_pick_place_220ep_pickup --subtask PICK_UP

    # Quick test with 5 episodes:
    python scripts/tools/split_subtask_dataset.py danbhf/sim_pick_place_2pos_220ep_v2 \
        -o datasets/test_split --max-episodes 5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add project root
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Subtask definitions (must match annotate_subtasks.py)
SUBTASK_NAMES = ["MOVE_TO_SOURCE", "PICK_UP", "MOVE_TO_DEST", "DROP"]
SUBTASK_TASK_NAMES = {
    0: "Move to the block",
    1: "Pick up the block",
    2: "Move to the bowl",
    3: "Drop the block in the bowl",
}

# Default features auto-added by LeRobotDataset.create()
DEFAULT_FEATURE_KEYS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def _find_meta_file(filename: str, source_dataset_id: str, local_root: Path = None) -> Path | None:
    """Search for a metadata file in local paths and HuggingFace.

    Search order:
    1. local_root/meta/ (dataset's own root, e.g. HF cache)
    2. REPO_ROOT/datasets/<dataset_name>/meta/ (local datasets dir)
    3. HuggingFace hub download
    """
    # 1. Dataset root (e.g. HF cache dir)
    if local_root:
        path = local_root / "meta" / filename
        if path.exists():
            return path

    # 2. Local datasets directory (datasets/<name>/meta/)
    dataset_name = source_dataset_id.split("/")[-1] if "/" in source_dataset_id else source_dataset_id
    local_datasets_path = REPO_ROOT / "datasets" / dataset_name / "meta" / filename
    if local_datasets_path.exists():
        return local_datasets_path

    # 3. HuggingFace
    try:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(source_dataset_id, f"meta/{filename}", repo_type="dataset"))
    except Exception:
        return None


def load_annotations(source_dataset_id: str, local_root: Path = None) -> dict:
    """Load subtask_annotations.json from local path or HuggingFace."""
    path = _find_meta_file("subtask_annotations.json", source_dataset_id, local_root)
    if path:
        with open(path) as f:
            return json.load(f)
    print("ERROR: Could not find subtask_annotations.json (checked dataset root, local datasets/, and HuggingFace)")
    sys.exit(1)


def load_episode_scenes(source_dataset_id: str, local_root: Path = None) -> dict:
    """Load episode_scenes.json from local path or HuggingFace."""
    path = _find_meta_file("episode_scenes.json", source_dataset_id, local_root)
    if path:
        with open(path) as f:
            return json.load(f)
    print("WARNING: Could not find episode_scenes.json")
    return {}


def find_subtask_segments(annotations: dict, episode_meta: dict) -> list:
    """Find contiguous subtask segments across all episodes.

    Returns list of dicts with source_episode, subtask_id, subtask_name,
    start_frame, end_frame (exclusive), length, global_start_idx.
    """
    segments = []

    for ep_idx_str, labels in sorted(annotations.items(), key=lambda x: int(x[0])):
        ep_idx = int(ep_idx_str)
        if ep_idx not in episode_meta:
            continue

        ep_global_start = episode_meta[ep_idx]["dataset_from_index"]

        if not labels:
            continue

        # Walk through labels finding contiguous runs
        seg_start = 0
        current_subtask = labels[0]

        for i in range(1, len(labels)):
            if labels[i] != current_subtask:
                segments.append({
                    "source_episode": ep_idx,
                    "subtask_id": current_subtask,
                    "subtask_name": SUBTASK_NAMES[current_subtask],
                    "start_frame": seg_start,
                    "end_frame": i,
                    "length": i - seg_start,
                    "global_start_idx": ep_global_start + seg_start,
                })
                seg_start = i
                current_subtask = labels[i]

        # Final segment
        segments.append({
            "source_episode": ep_idx,
            "subtask_id": current_subtask,
            "subtask_name": SUBTASK_NAMES[current_subtask],
            "start_frame": seg_start,
            "end_frame": len(labels),
            "length": len(labels) - seg_start,
            "global_start_idx": ep_global_start + seg_start,
        })

    return segments


def get_user_features(source_dataset) -> dict:
    """Extract user-defined features from source dataset, excluding defaults."""
    return {
        key: value
        for key, value in source_dataset.features.items()
        if key not in DEFAULT_FEATURE_KEYS
    }


def main():
    parser = argparse.ArgumentParser(
        description="Split dataset episodes into per-subtask segments"
    )
    parser.add_argument(
        "source_dataset",
        type=str,
        help="Source dataset (HuggingFace repo ID or local path)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output directory for new dataset",
    )
    parser.add_argument(
        "--subtask",
        type=str,
        default=None,
        help="Filter to one subtask: MOVE_TO_SOURCE, PICK_UP, MOVE_TO_DEST, DROP (or 0-3)",
    )
    parser.add_argument(
        "--pad-before",
        type=int,
        default=0,
        help="Include N extra frames before each segment (from previous subtask)",
    )
    parser.add_argument(
        "--pad-after",
        type=int,
        default=0,
        help="Include N extra frames after each segment (from next subtask)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=5,
        help="Skip segments shorter than N frames (default: 5)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Process only first N source episodes (for testing)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Repo ID for new dataset metadata (default: derived from output dir name)",
    )
    args = parser.parse_args()

    # Parse subtask filter
    subtask_filter = None
    if args.subtask is not None:
        if args.subtask.isdigit():
            subtask_filter = int(args.subtask)
            if subtask_filter not in range(4):
                print(f"ERROR: Subtask ID must be 0-3, got {subtask_filter}")
                sys.exit(1)
        else:
            subtask_upper = args.subtask.upper()
            if subtask_upper in SUBTASK_NAMES:
                subtask_filter = SUBTASK_NAMES.index(subtask_upper)
            else:
                print(f"ERROR: Unknown subtask '{args.subtask}'. Valid: {SUBTASK_NAMES} or 0-3")
                sys.exit(1)
        print(f"Filtering to subtask: {SUBTASK_NAMES[subtask_filter]} ({subtask_filter})")

    # Load source dataset
    print(f"Loading source dataset: {args.source_dataset}")
    source_dataset = LeRobotDataset(args.source_dataset)
    print(f"  Episodes: {source_dataset.meta.total_episodes}")
    print(f"  Frames: {len(source_dataset)}")
    print(f"  FPS: {source_dataset.fps}")

    # Determine local root for metadata files
    local_root = Path(source_dataset.root) if hasattr(source_dataset, 'root') else None

    # Load annotations and episode scenes
    print("Loading subtask annotations...")
    annotations = load_annotations(args.source_dataset, local_root)
    print(f"  Loaded annotations for {len(annotations)} episodes")

    print("Loading episode scenes...")
    episode_scenes = load_episode_scenes(args.source_dataset, local_root)
    print(f"  Loaded scenes for {len(episode_scenes)} episodes")

    # Build episode metadata lookup
    episode_meta = {}
    for ep_idx in range(source_dataset.meta.total_episodes):
        ep_info = source_dataset.meta.episodes[ep_idx]
        episode_meta[ep_idx] = {
            "dataset_from_index": ep_info["dataset_from_index"],
            "dataset_to_index": ep_info["dataset_to_index"],
        }

    # Limit episodes if requested
    if args.max_episodes is not None:
        limited = {k: v for k, v in annotations.items() if int(k) < args.max_episodes}
        annotations = limited
        print(f"  Limited to first {args.max_episodes} episodes")

    # Find all subtask segments
    segments = find_subtask_segments(annotations, episode_meta)
    print(f"\nFound {len(segments)} total segments")

    # Apply subtask filter
    if subtask_filter is not None:
        segments = [s for s in segments if s["subtask_id"] == subtask_filter]
        print(f"  After subtask filter: {len(segments)} segments")

    # Apply min-length filter (on original segment length, before padding)
    short_count = sum(1 for s in segments if s["length"] < args.min_length)
    segments = [s for s in segments if s["length"] >= args.min_length]
    if short_count > 0:
        print(f"  Skipped {short_count} segments shorter than {args.min_length} frames")

    # Apply padding: extend each segment's range, clamped to source episode boundaries
    if args.pad_before > 0 or args.pad_after > 0:
        print(f"  Padding: {args.pad_before} frames before, {args.pad_after} frames after")
        for seg in segments:
            ep_idx = seg["source_episode"]
            ep_start = episode_meta[ep_idx]["dataset_from_index"]
            ep_end = episode_meta[ep_idx]["dataset_to_index"]
            ep_length = ep_end - ep_start

            # Clamp padded range to episode boundaries (in local frame coords)
            new_start = max(0, seg["start_frame"] - args.pad_before)
            new_end = min(ep_length, seg["end_frame"] + args.pad_after)

            seg["start_frame"] = new_start
            seg["end_frame"] = new_end
            seg["length"] = new_end - new_start
            seg["global_start_idx"] = ep_start + new_start

    print(f"  Final: {len(segments)} segments to process")

    if not segments:
        print("ERROR: No segments to process!")
        sys.exit(1)

    # Print segment length summary per subtask
    print("\nSegment length summary:")
    for subtask_id, name in enumerate(SUBTASK_NAMES):
        subtask_segs = [s for s in segments if s["subtask_id"] == subtask_id]
        if subtask_segs:
            lengths = [s["length"] for s in subtask_segs]
            print(f"  {name}: {len(subtask_segs)} segments, "
                  f"length min={min(lengths)} max={max(lengths)} mean={np.mean(lengths):.1f}")

    # Create output dataset
    output_dir = Path(args.output)
    repo_id = args.repo_id or f"danbhf/{output_dir.name}"

    features = get_user_features(source_dataset)
    print(f"\nCreating output dataset: {repo_id}")
    print(f"  Output dir: {output_dir}")
    print(f"  Features: {list(features.keys())}")

    output_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=source_dataset.fps,
        root=str(output_dir),
        robot_type="so100_sim",
        features=features,
        image_writer_threads=4,
    )

    # Process segments
    new_episode_scenes = {}
    new_annotations = {}

    for seg_idx, segment in enumerate(tqdm(segments, desc="Processing segments")):
        src_ep = segment["source_episode"]
        subtask_id = segment["subtask_id"]
        global_start = segment["global_start_idx"]
        length = segment["length"]
        task_name = SUBTASK_TASK_NAMES[subtask_id]

        output_dataset.create_episode_buffer()

        for frame_offset in range(length):
            global_idx = global_start + frame_offset
            source_frame = source_dataset[global_idx]

            output_frame = {"task": task_name}

            for key in features:
                if key not in source_frame:
                    continue
                value = source_frame[key]

                # Convert images: CHW float [0,1] tensor -> HWC uint8 numpy
                if isinstance(value, torch.Tensor) and value.dim() == 3 and "images" in key:
                    img_np = (value.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
                    output_frame[key] = img_np
                elif isinstance(value, torch.Tensor):
                    output_frame[key] = value.numpy()
                else:
                    output_frame[key] = value

            output_dataset.add_frame(output_frame)

        output_dataset.save_episode()

        # Map episode scenes: output episode inherits source episode's scene info
        ep_key = str(src_ep)
        if ep_key in episode_scenes:
            scene_info = dict(episode_scenes[ep_key])
            scene_info["_source_episode"] = src_ep
            scene_info["_subtask"] = SUBTASK_NAMES[subtask_id]
            new_episode_scenes[str(seg_idx)] = scene_info

        # Build per-frame annotations from source (handles padding correctly)
        src_annotations = annotations.get(str(src_ep), [])
        seg_labels = []
        for frame_offset in range(length):
            src_frame_idx = segment["start_frame"] + frame_offset
            if src_frame_idx < len(src_annotations):
                seg_labels.append(src_annotations[src_frame_idx])
            else:
                seg_labels.append(subtask_id)
        new_annotations[str(seg_idx)] = seg_labels

    # Finalize dataset
    print("\nFinalizing dataset...")
    output_dataset.finalize()

    # Save metadata
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    with open(meta_dir / "episode_scenes.json", "w") as f:
        json.dump(new_episode_scenes, f, indent=2)
    print(f"Saved episode_scenes.json ({len(new_episode_scenes)} episodes)")

    with open(meta_dir / "subtask_annotations.json", "w") as f:
        json.dump(new_annotations, f, indent=2)
    print(f"Saved subtask_annotations.json ({len(new_annotations)} episodes)")

    # Final summary
    total_frames = sum(s["length"] for s in segments)
    print("\n" + "=" * 60)
    print("DATASET SPLIT COMPLETE")
    print("=" * 60)
    print(f"Source: {args.source_dataset}")
    print(f"Output: {output_dir}")
    print(f"Episodes: {len(segments)}")
    print(f"Total frames: {total_frames}")
    if subtask_filter is not None:
        print(f"Subtask filter: {SUBTASK_NAMES[subtask_filter]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
