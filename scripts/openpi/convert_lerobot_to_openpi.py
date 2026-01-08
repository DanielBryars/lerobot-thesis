#!/usr/bin/env python
"""
Convert a LeRobot dataset to openpi (RLDS) format.

Openpi (Physical Intelligence's Pi0/Pi0.5) uses RLDS format for datasets.
This script converts LeRobot datasets to the expected RLDS format.

RLDS format structure:
- Each episode is stored as a TFRecord with:
  - observation/state: proprioceptive state [state_dim]
  - observation/image: main camera image (H, W, 3) uint8
  - observation/wrist_image: wrist camera image (optional)
  - action: robot action [action_dim]
  - language_instruction: text instruction
  - is_first: bool (first step in episode)
  - is_last: bool (last step in episode)
  - is_terminal: bool (terminal state)

Usage:
    # Convert dataset with default cameras
    python convert_lerobot_to_openpi.py danbhf/sim_pick_place_merged_40ep output/openpi_dataset

    # Specify cameras explicitly
    python convert_lerobot_to_openpi.py danbhf/sim_pick_place_merged_40ep output/openpi_dataset \
        --main_camera overhead_cam --wrist_camera wrist_cam

    # Add language instruction
    python convert_lerobot_to_openpi.py danbhf/sim_pick_place_merged_40ep output/openpi_dataset \
        --language "Pick up the block and place it in the bowl"

Requirements:
    pip install tensorflow tensorflow-datasets
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import numpy as np
from tqdm import tqdm

# Add project root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def load_lerobot_dataset(repo_id: str, local_dir: Optional[str] = None):
    """Load a LeRobot dataset.

    Args:
        repo_id: HuggingFace dataset repo ID
        local_dir: Optional local directory to cache the dataset

    Returns:
        Tuple of (dataset, metadata)
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    print(f"Loading LeRobot dataset: {repo_id}")
    metadata = LeRobotDatasetMetadata(repo_id)
    dataset = LeRobotDataset(repo_id)

    return dataset, metadata


def get_episode_indices(dataset, metadata) -> List[Tuple[int, int]]:
    """Get start and end indices for each episode.

    Returns:
        List of (start_idx, end_idx) tuples for each episode
    """
    episodes = []
    for ep_idx in range(metadata.info["total_episodes"]):
        ep_data = metadata.episodes[ep_idx]
        # LeRobot v3.0 uses dataset_from_index/dataset_to_index
        start = ep_data.get("dataset_from_index", ep_data.get("index_from", 0))
        end = ep_data.get("dataset_to_index", ep_data.get("index_to", 0))
        episodes.append((start, end))
    return episodes


def extract_cameras_from_features(features: Dict) -> List[str]:
    """Extract camera names from dataset features."""
    cameras = []
    for key in features:
        if key.startswith("observation.images."):
            cam_name = key.replace("observation.images.", "")
            if not cam_name.endswith("_depth"):  # Skip depth cameras for RGB
                cameras.append(cam_name)
    return cameras


def convert_step_to_rlds(
    step: Dict,
    main_camera: str,
    wrist_camera: Optional[str],
    language: str,
    is_first: bool,
    is_last: bool,
    action_key: str = "action",
) -> Dict:
    """Convert a single LeRobot step to RLDS format.

    Args:
        step: LeRobot dataset step dictionary
        main_camera: Name of main camera
        wrist_camera: Name of wrist camera (optional)
        language: Language instruction
        is_first: Whether this is first step in episode
        is_last: Whether this is last step in episode
        action_key: Key for action in dataset

    Returns:
        Dict in RLDS format
    """
    rlds_step = {}

    # State (proprioceptive)
    if "observation.state" in step:
        state = step["observation.state"]
        if hasattr(state, "numpy"):
            state = state.numpy()
        rlds_step["observation/state"] = np.array(state, dtype=np.float32)
    else:
        # No state available, create empty
        rlds_step["observation/state"] = np.array([], dtype=np.float32)

    # Main camera image
    main_img_key = f"observation.images.{main_camera}"
    if main_img_key in step:
        img = step[main_img_key]
        if hasattr(img, "numpy"):
            img = img.numpy()
        # Convert from PyTorch (C, H, W) to RLDS (H, W, C) format if needed
        if img.ndim == 3 and img.shape[0] in [1, 3, 4]:  # Likely (C, H, W)
            img = np.transpose(img, (1, 2, 0))
        # Ensure uint8
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        rlds_step["observation/image"] = img
    else:
        raise ValueError(f"Main camera {main_camera} not found in step keys: {list(step.keys())}")

    # Wrist camera image (optional)
    if wrist_camera:
        wrist_img_key = f"observation.images.{wrist_camera}"
        if wrist_img_key in step:
            img = step[wrist_img_key]
            if hasattr(img, "numpy"):
                img = img.numpy()
            # Convert from PyTorch (C, H, W) to RLDS (H, W, C) format if needed
            if img.ndim == 3 and img.shape[0] in [1, 3, 4]:  # Likely (C, H, W)
                img = np.transpose(img, (1, 2, 0))
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            rlds_step["observation/wrist_image"] = img

    # Action
    if action_key in step:
        action = step[action_key]
        if hasattr(action, "numpy"):
            action = action.numpy()
        rlds_step["action"] = np.array(action, dtype=np.float32)
    else:
        raise ValueError(f"Action key {action_key} not found in step keys: {list(step.keys())}")

    # Metadata
    rlds_step["language_instruction"] = language
    rlds_step["is_first"] = is_first
    rlds_step["is_last"] = is_last
    rlds_step["is_terminal"] = is_last  # Terminal at episode end

    return rlds_step


def save_episode_as_tfrecord(
    episode_steps: List[Dict],
    output_path: Path,
    episode_idx: int,
):
    """Save an episode as a TFRecord file.

    Args:
        episode_steps: List of RLDS-format steps
        output_path: Output directory
        episode_idx: Episode index
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("tensorflow required for TFRecord output. Install with: pip install tensorflow")

    def _bytes_feature(value):
        if isinstance(value, str):
            value = value.encode("utf-8")
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten().tolist()))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))

    tfrecord_path = output_path / f"episode_{episode_idx:06d}.tfrecord"

    with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
        for step in episode_steps:
            feature = {
                "observation/state": _float_feature(step["observation/state"]),
                "observation/image": _bytes_feature(step["observation/image"].tobytes()),
                "observation/image_shape": _int64_feature(step["observation/image"].shape[0]),
                "action": _float_feature(step["action"]),
                "language_instruction": _bytes_feature(step["language_instruction"]),
                "is_first": _int64_feature(step["is_first"]),
                "is_last": _int64_feature(step["is_last"]),
                "is_terminal": _int64_feature(step["is_terminal"]),
            }

            # Optional wrist image
            if "observation/wrist_image" in step:
                feature["observation/wrist_image"] = _bytes_feature(
                    step["observation/wrist_image"].tobytes()
                )
                feature["observation/wrist_image_shape"] = _int64_feature(
                    step["observation/wrist_image"].shape[0]
                )

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())


def save_episode_as_npz(
    episode_steps: List[Dict],
    output_path: Path,
    episode_idx: int,
):
    """Save an episode as a numpy archive (alternative to TFRecord).

    Args:
        episode_steps: List of RLDS-format steps
        output_path: Output directory
        episode_idx: Episode index
    """
    # Organize into arrays per field
    episode_data = {
        "observation/state": np.stack([s["observation/state"] for s in episode_steps]),
        "observation/image": np.stack([s["observation/image"] for s in episode_steps]),
        "action": np.stack([s["action"] for s in episode_steps]),
        "is_first": np.array([s["is_first"] for s in episode_steps]),
        "is_last": np.array([s["is_last"] for s in episode_steps]),
        "is_terminal": np.array([s["is_terminal"] for s in episode_steps]),
        "language_instruction": episode_steps[0]["language_instruction"],
    }

    # Optional wrist image
    if "observation/wrist_image" in episode_steps[0]:
        episode_data["observation/wrist_image"] = np.stack(
            [s["observation/wrist_image"] for s in episode_steps]
        )

    npz_path = output_path / f"episode_{episode_idx:06d}.npz"
    np.savez_compressed(str(npz_path), **episode_data)


def convert_dataset(
    repo_id: str,
    output_dir: str,
    main_camera: Optional[str] = None,
    wrist_camera: Optional[str] = None,
    language: str = "Pick up the block and place it in the bowl",
    action_key: str = "action",
    output_format: str = "npz",
    max_episodes: Optional[int] = None,
) -> Dict:
    """Convert a LeRobot dataset to openpi format.

    Args:
        repo_id: HuggingFace dataset repo ID
        output_dir: Output directory for converted dataset
        main_camera: Main camera name (auto-detected if None)
        wrist_camera: Wrist camera name (auto-detected if None)
        language: Language instruction for all episodes
        action_key: Key for action in dataset ("action" or "action_joints")
        output_format: "npz" or "tfrecord"
        max_episodes: Maximum episodes to convert (None = all)

    Returns:
        Dict with conversion statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset, metadata = load_lerobot_dataset(repo_id)

    # Get features
    features = metadata.features
    print(f"Dataset features: {list(features.keys())}")

    # Auto-detect cameras
    cameras = extract_cameras_from_features(features)
    print(f"Available RGB cameras: {cameras}")

    if main_camera is None:
        # Prefer overhead camera
        if "overhead_cam" in cameras:
            main_camera = "overhead_cam"
        elif len(cameras) > 0:
            main_camera = cameras[0]
        else:
            raise ValueError("No RGB cameras found in dataset")
    print(f"Using main camera: {main_camera}")

    if wrist_camera is None:
        # Look for wrist camera
        for cam in cameras:
            if "wrist" in cam.lower():
                wrist_camera = cam
                break
    if wrist_camera:
        print(f"Using wrist camera: {wrist_camera}")
    else:
        print("No wrist camera detected")

    # Get action dimension
    if action_key in features:
        action_shape = features[action_key].get("shape", [7])
        print(f"Action dimension: {action_shape}")
    else:
        print(f"Warning: {action_key} not in features, will try at runtime")

    # Get state dimension
    if "observation.state" in features:
        state_shape = features["observation.state"].get("shape", [7])
        print(f"State dimension: {state_shape}")

    # Get episode ranges
    total_episodes = metadata.info["total_episodes"]
    if max_episodes:
        total_episodes = min(total_episodes, max_episodes)
    print(f"Converting {total_episodes} episodes")

    # Convert episodes
    stats = {
        "total_episodes": 0,
        "total_steps": 0,
        "skipped_episodes": 0,
        "errors": [],
    }

    for ep_idx in tqdm(range(total_episodes), desc="Converting episodes"):
        try:
            # Get episode data - LeRobot v3.0 uses dataset_from_index/dataset_to_index
            ep_info = metadata.episodes[ep_idx]
            start_idx = ep_info.get("dataset_from_index", ep_info.get("index_from", 0))
            end_idx = ep_info.get("dataset_to_index", ep_info.get("index_to", 0))
            ep_length = end_idx - start_idx

            if ep_length <= 0:
                stats["skipped_episodes"] += 1
                continue

            episode_steps = []
            for step_idx in range(start_idx, end_idx):
                step = dataset[step_idx]
                is_first = (step_idx == start_idx)
                is_last = (step_idx == end_idx - 1)

                rlds_step = convert_step_to_rlds(
                    step=step,
                    main_camera=main_camera,
                    wrist_camera=wrist_camera,
                    language=language,
                    is_first=is_first,
                    is_last=is_last,
                    action_key=action_key,
                )
                episode_steps.append(rlds_step)

            # Save episode
            if output_format == "tfrecord":
                save_episode_as_tfrecord(episode_steps, output_path, ep_idx)
            else:
                save_episode_as_npz(episode_steps, output_path, ep_idx)

            stats["total_episodes"] += 1
            stats["total_steps"] += len(episode_steps)

        except Exception as e:
            stats["errors"].append(f"Episode {ep_idx}: {str(e)}")
            stats["skipped_episodes"] += 1

    # Save metadata
    meta = {
        "source_dataset": repo_id,
        "main_camera": main_camera,
        "wrist_camera": wrist_camera,
        "language_instruction": language,
        "action_key": action_key,
        "format": output_format,
        "stats": stats,
    }

    meta_path = output_path / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nConversion complete!")
    print(f"  Episodes: {stats['total_episodes']}")
    print(f"  Steps: {stats['total_steps']}")
    print(f"  Skipped: {stats['skipped_episodes']}")
    if stats["errors"]:
        print(f"  Errors: {len(stats['errors'])}")
        for err in stats["errors"][:5]:
            print(f"    - {err}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot dataset to openpi (RLDS) format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("repo_id", type=str, help="HuggingFace dataset repo ID")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument("--main_camera", type=str, default=None,
                        help="Main camera name (auto-detected if not specified)")
    parser.add_argument("--wrist_camera", type=str, default=None,
                        help="Wrist camera name (auto-detected if not specified)")
    parser.add_argument("--language", type=str,
                        default="Pick up the block and place it in the bowl",
                        help="Language instruction for all episodes")
    parser.add_argument("--action_key", type=str, default="action",
                        choices=["action", "action_joints"],
                        help="Action field to use (default: action)")
    parser.add_argument("--format", type=str, default="npz",
                        choices=["npz", "tfrecord"],
                        help="Output format (default: npz)")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Maximum episodes to convert (default: all)")

    args = parser.parse_args()

    stats = convert_dataset(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        main_camera=args.main_camera,
        wrist_camera=args.wrist_camera,
        language=args.language,
        action_key=args.action_key,
        output_format=args.format,
        max_episodes=args.max_episodes,
    )

    # Exit with error if too many failures
    if stats["skipped_episodes"] > stats["total_episodes"] * 0.1:
        print("\nWarning: More than 10% of episodes failed to convert")
        sys.exit(1)


if __name__ == "__main__":
    main()
