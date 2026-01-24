#!/usr/bin/env python
"""
Convert a joint-space dataset to end-effector action space.

Takes an existing LeRobot dataset with joint angle actions and creates a new
dataset with end-effector pose actions (xyz + quaternion + gripper).

Usage:
    python recording/convert_to_ee_actions.py danbhf/sim_pick_place_merged_40ep
    python recording/convert_to_ee_actions.py danbhf/sim_pick_place_merged_40ep --output my_ee_dataset
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Add project paths
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "scripts"))

# Import shared utilities
from utils.conversions import normalized_to_radians, rotation_matrix_to_quaternion

# Import FK
from test_fk_ik import MuJoCoFK

# Import LeRobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def convert_dataset(input_repo_id: str, output_name: str = None, local: bool = False):
    """Convert a joint-space dataset to end-effector action space."""

    # Load input dataset
    print(f"Loading dataset: {input_repo_id}")
    if local:
        local_path = repo_root / "datasets" / input_repo_id
        if not local_path.exists():
            print(f"ERROR: Local dataset not found: {local_path}")
            return
        print(f"Using local path: {local_path}")
        dataset = LeRobotDataset(repo_id=input_repo_id, root=local_path)
    else:
        dataset = LeRobotDataset(repo_id=input_repo_id)

    num_episodes = dataset.num_episodes
    num_frames = len(dataset)
    fps = dataset.fps

    print(f"Loaded {num_episodes} episodes, {num_frames} frames at {fps} FPS")

    # Check action format
    sample = dataset[0]
    if "action" not in sample:
        print("ERROR: Dataset does not have 'action' key")
        return

    action_shape = sample["action"].shape
    print(f"Original action shape: {action_shape}")

    if action_shape[0] != 6:
        print(f"WARNING: Expected 6 joint actions, got {action_shape[0]}")

    # Initialize FK
    print("\nInitializing FK...")
    scene_xml = str(repo_root / "scenes" / "so101_with_wrist_cam.xml")
    fk = MuJoCoFK(scene_xml)

    # Convert all actions and collect all data
    print("\nConverting actions to end-effector space...")

    all_data = []
    prev_quat = None  # Track previous quaternion for continuity
    prev_episode = -1  # Track episode to reset quaternion continuity
    sign_flips = 0  # Count sign flips for debugging

    for idx in tqdm(range(num_frames), desc="Converting"):
        frame = dataset[idx]
        action = frame["action"]
        episode_idx = frame["episode_index"].item() if hasattr(frame["episode_index"], 'item') else int(frame["episode_index"])

        if hasattr(action, 'numpy'):
            action = action.numpy()

        # Convert normalized values to radians
        action_radians = normalized_to_radians(action)

        # Extract arm joints (first 5) and gripper (6th)
        arm_joints = action_radians[:5]
        gripper = action_radians[5]

        # FK: joint angles -> end-effector pose
        ee_pos, ee_rot = fk.forward(arm_joints)

        # Convert rotation matrix to quaternion [qw, qx, qy, qz]
        ee_quat = rotation_matrix_to_quaternion(ee_rot)

        # Ensure quaternion continuity within each episode
        # q and -q represent the same rotation, but we want smooth transitions
        if episode_idx != prev_episode:
            # New episode - reset quaternion tracking
            prev_quat = ee_quat.copy()
            prev_episode = episode_idx
        else:
            # Same episode - check if we need to flip the sign
            # If dot product is negative, quaternions are on opposite hemispheres
            if np.dot(prev_quat, ee_quat) < 0:
                ee_quat = -ee_quat
                sign_flips += 1
            prev_quat = ee_quat.copy()

        # New action: [x, y, z, qw, qx, qy, qz, gripper]
        ee_action = np.concatenate([ee_pos, ee_quat, [gripper]]).astype(np.float32)

        # Get observation state
        obs_state = frame["observation.state"]
        if hasattr(obs_state, 'numpy'):
            obs_state = obs_state.numpy()

        # Keep original joint action (normalized form) for comparison
        original_action = action.tolist() if not isinstance(action, list) else action

        # Build row data
        row = {
            "action": ee_action.tolist(),
            "action_joints": original_action,  # Original joint-space action for comparison
            "observation.state": obs_state.tolist(),
            "timestamp": frame["timestamp"].item() if hasattr(frame["timestamp"], 'item') else float(frame["timestamp"]),
            "frame_index": frame["frame_index"].item() if hasattr(frame["frame_index"], 'item') else int(frame["frame_index"]),
            "episode_index": frame["episode_index"].item() if hasattr(frame["episode_index"], 'item') else int(frame["episode_index"]),
            "index": frame["index"].item() if hasattr(frame["index"], 'item') else int(frame["index"]),
            "task_index": frame["task_index"].item() if hasattr(frame["task_index"], 'item') else int(frame["task_index"]),
        }
        all_data.append(row)

    print(f"\nQuaternion sign flips corrected: {sign_flips}")

    # Compute action statistics
    ee_actions = np.array([d["action"] for d in all_data], dtype=np.float32)
    action_mean = ee_actions.mean(axis=0)
    action_std = ee_actions.std(axis=0)
    action_min = ee_actions.min(axis=0)
    action_max = ee_actions.max(axis=0)

    print("\nAction statistics:")
    labels = ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"]
    for i, label in enumerate(labels):
        print(f"  {label:8s}: mean={action_mean[i]:7.4f}, std={action_std[i]:7.4f}, "
              f"min={action_min[i]:7.4f}, max={action_max[i]:7.4f}")

    # Create output dataset
    if output_name is None:
        output_name = input_repo_id.split("/")[-1] + "_ee"

    output_dir = repo_root / "datasets" / output_name

    if output_dir.exists():
        print(f"\nOutput directory exists: {output_dir}")
        response = input("Overwrite? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True)
    print(f"\nCreating output dataset at: {output_dir}")

    # Create directory structure
    (output_dir / "data" / "chunk-000").mkdir(parents=True)
    (output_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True)

    # Copy videos from source (use HF cache path)
    source_root = dataset.root
    print(f"Source dataset root: {source_root}")

    source_videos = source_root / "videos"
    if source_videos.exists():
        print("Copying videos...")
        shutil.copytree(source_videos, output_dir / "videos")

    # Create data parquet
    print("Creating data parquet file...")
    df = pd.DataFrame(all_data)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_dir / "data" / "chunk-000" / "file-000.parquet")

    # Copy and update episodes parquet
    source_episodes = source_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if source_episodes.exists():
        shutil.copy(source_episodes, output_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
        print("Copied episodes metadata")

    # Copy tasks parquet
    source_tasks = source_root / "meta" / "tasks.parquet"
    if source_tasks.exists():
        shutil.copy(source_tasks, output_dir / "meta" / "tasks.parquet")
        print("Copied tasks metadata")

    # Copy episode scenes if available
    source_episode_scenes = source_root / "meta" / "episode_scenes.json"
    if source_episode_scenes.exists():
        shutil.copy(source_episode_scenes, output_dir / "meta" / "episode_scenes.json")
        print("Copied per-episode scene data")

    # Copy recording metadata if available
    source_recording_meta = source_root / "meta" / "recording_metadata.json"
    if source_recording_meta.exists():
        shutil.copy(source_recording_meta, output_dir / "meta" / "recording_metadata.json")
        print("Copied recording metadata")

    # Create info.json
    info = {
        "codebase_version": "v3.0",
        "robot_type": "so100_sim",
        "total_episodes": num_episodes,
        "total_frames": num_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{num_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "action": {
                "dtype": "float32",
                "shape": [8],
                "names": ["ee.x", "ee.y", "ee.z", "ee.qw", "ee.qx", "ee.qy", "ee.qz", "gripper.pos"]
            },
            "action_joints": {
                "dtype": "float32",
                "shape": [6],
                "names": ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                         "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"],
                "description": "Original joint-space action (normalized form) for comparison"
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [6],
                "names": ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                         "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
            },
            "observation.images.wrist_cam": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": 480,
                    "video.width": 640,
                    "video.codec": "av1",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": fps,
                    "video.channels": 3,
                    "has_audio": False
                }
            },
            "observation.images.overhead_cam": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": 480,
                    "video.width": 640,
                    "video.codec": "av1",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": fps,
                    "video.channels": 3,
                    "has_audio": False
                }
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
        "action_space": "end_effector",
        "action_description": "End-effector pose: [x, y, z, qw, qx, qy, qz, gripper]",
        "converted_from": input_repo_id,
    }

    # Copy scene info from source dataset if available
    if 'scene' in dataset.meta.info:
        info['scene'] = dataset.meta.info['scene']
        print("Copied scene metadata (object positions)")

    with open(output_dir / "meta" / "info.json", 'w') as f:
        json.dump(info, f, indent=2)

    # Create stats.json for normalization (required by LeRobot preprocessor)
    # Get observation.state statistics
    obs_states = np.array([d["observation.state"] for d in all_data], dtype=np.float32)
    obs_mean = obs_states.mean(axis=0)
    obs_std = obs_states.std(axis=0)

    # Get action_joints statistics (for training with --use_joint_actions)
    joint_actions = np.array([d["action_joints"] for d in all_data], dtype=np.float32)
    joint_mean = joint_actions.mean(axis=0)
    joint_std = joint_actions.std(axis=0)

    stats = {
        "action": {
            "mean": action_mean.tolist(),
            "std": action_std.tolist(),
            "min": action_min.tolist(),
            "max": action_max.tolist(),
        },
        "action_joints": {
            "mean": joint_mean.tolist(),
            "std": joint_std.tolist(),
            "min": joint_actions.min(axis=0).tolist(),
            "max": joint_actions.max(axis=0).tolist(),
        },
        "observation.state": {
            "mean": obs_mean.tolist(),
            "std": obs_std.tolist(),
            "min": obs_states.min(axis=0).tolist(),
            "max": obs_states.max(axis=0).tolist(),
        },
    }

    with open(output_dir / "meta" / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    print("Created stats.json for normalization")

    print(f"\n{'='*60}")
    print(f"Dataset converted successfully!")
    print(f"{'='*60}")
    print(f"Output: {output_dir}")
    print(f"Action space: end-effector [x, y, z, qw, qx, qy, qz, gripper]")
    print(f"Action dimension: 8 (was 6)")
    print(f"\nTo upload to HuggingFace:")
    print(f"  huggingface-cli upload <username>/{output_name} datasets/{output_name} --repo-type dataset")
    print(f"\nThen tag with codebase version (required for LeRobot):")
    print(f'  python -c "from huggingface_hub import HfApi; HfApi().create_tag(\'<username>/{output_name}\', tag=\'v3.0\', repo_type=\'dataset\')"')


def main():
    parser = argparse.ArgumentParser(
        description="Convert joint-space dataset to end-effector action space"
    )
    parser.add_argument("dataset", type=str,
                        help="Input dataset (HuggingFace repo ID or local name)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output dataset name (default: <input>_ee)")
    parser.add_argument("--local", action="store_true",
                        help="Load from local datasets/ folder instead of HuggingFace")

    args = parser.parse_args()

    convert_dataset(
        input_repo_id=args.dataset,
        output_name=args.output,
        local=args.local,
    )


if __name__ == "__main__":
    main()
