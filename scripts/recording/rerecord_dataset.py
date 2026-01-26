#!/usr/bin/env python
"""
Re-record an existing dataset with a new scene/camera setup.

Plays back joint actions from an existing dataset in a new scene and records
new observations (including depth if enabled). This allows comparing training
with different camera configurations on identical movements.

Usage:
    python recording/rerecord_dataset.py danbhf/sim_pick_place_merged_40ep_ee_2 --depth
    python recording/rerecord_dataset.py danbhf/dataset --depth --output my_rgbd_dataset

The source dataset must have 'action_joints' or 'action' field (joint actions).
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Add project paths
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import hw_to_dataset_features, build_dataset_frame

# Import simulation
from lerobot_robot_sim import SO100Sim, SO100SimConfig, MOTOR_NAMES


def get_git_info() -> dict:
    """Get git repository information."""
    import subprocess

    def run_git(args):
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except:
            return None

    return {
        "commit_hash": run_git(["rev-parse", "HEAD"]),
        "commit_short": run_git(["rev-parse", "--short", "HEAD"]),
        "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Re-record dataset with new scene/cameras")
    parser.add_argument("source_dataset", type=str, help="Source dataset (HuggingFace repo ID or local path)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output dataset name (default: auto-generated)")
    parser.add_argument("--depth", action="store_true", help="Enable depth recording for overhead camera")
    parser.add_argument("--root", type=str, default="./datasets", help="Output root directory")
    parser.add_argument("--scene", type=str, default=None, help="Scene XML path (default: so101_rgbd.xml)")
    parser.add_argument("--fps", type=int, default=None, help="Override FPS (default: use source dataset FPS)")

    args = parser.parse_args()

    # Load source dataset
    print(f"Loading source dataset: {args.source_dataset}")
    source_dataset = LeRobotDataset(args.source_dataset)
    source_fps = source_dataset.fps
    fps = args.fps or source_fps

    # Check for action field (prefer action_joints, fall back to action)
    sample = source_dataset[0]
    if 'action_joints' in sample:
        action_key = 'action_joints'
    elif 'action' in sample:
        action_key = 'action'
    else:
        print("ERROR: Source dataset must have 'action_joints' or 'action' field")
        print("Available keys:", list(sample.keys()))
        sys.exit(1)
    print(f"  Using action field: {action_key}")

    print(f"  Episodes: {source_dataset.meta.total_episodes}")
    print(f"  Frames: {len(source_dataset)}")
    print(f"  FPS: {source_fps}")

    # Setup simulation
    print("\nInitializing simulation...")
    scene_xml = args.scene
    if scene_xml:
        scene_xml = str(repo_root / scene_xml) if not Path(scene_xml).is_absolute() else scene_xml

    depth_cameras = ["overhead_cam"] if args.depth else []
    sim_config = SO100SimConfig(
        id="rerecorder",
        sim_cameras=["wrist_cam", "overhead_cam"],
        depth_cameras=depth_cameras,
        camera_width=640,
        camera_height=480,
        enable_vr=False,
        n_sim_steps=10,
    )
    if scene_xml:
        sim_config.scene_xml = scene_xml

    sim_robot = SO100Sim(sim_config)
    sim_robot.connect()

    print(f"  Scene: {sim_robot.scene_xml}")
    print(f"  Cameras: {sim_config.sim_cameras}")
    print(f"  Depth cameras: {depth_cameras}")

    # Create output dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        repo_id = args.output if "/" in args.output else f"danbhf/{args.output}"
    else:
        depth_suffix = "_rgbd" if args.depth else ""
        repo_id = f"danbhf/sim_pick_place_rerecord{depth_suffix}_{timestamp}"

    root_dir = Path(args.root) / timestamp
    print(f"\nCreating output dataset: {repo_id}")
    print(f"  Storage: {root_dir}")

    # Get observation features from simulation (new camera views with depth)
    obs_features = hw_to_dataset_features(sim_robot.observation_features, "observation")

    # Copy action features directly from source dataset (preserves both action_joints and action/EE)
    action_features = {}
    for key in source_dataset.features:
        if key.startswith("action"):
            action_features[key] = source_dataset.features[key]

    features = {**obs_features, **action_features}

    output_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=root_dir,
        robot_type="so100_sim",
        features=features,
        image_writer_threads=4,
    )

    # Save metadata
    metadata = {
        "rerecord_info": {
            "source_dataset": args.source_dataset,
            "timestamp": timestamp,
            "depth_enabled": args.depth,
            "scene_xml": str(sim_robot.scene_xml),
            "fps": fps,
        },
        "git": get_git_info(),
        "simulation": {
            "sim_cameras": sim_config.sim_cameras,
            "depth_cameras": depth_cameras,
            "camera_width": sim_config.camera_width,
            "camera_height": sim_config.camera_height,
        },
    }

    meta_dir = root_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "rerecord_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Process each episode
    total_episodes = source_dataset.meta.total_episodes
    print(f"\nRe-recording {total_episodes} episodes...")

    for ep_idx in range(total_episodes):
        ep_meta = source_dataset.meta.episodes[ep_idx]
        from_idx = ep_meta['dataset_from_index']
        to_idx = ep_meta['dataset_to_index']
        num_frames = to_idx - from_idx

        # Get episode scene info if available
        task = ep_meta.get('task', 'Pick up the Duplo block and place it in the bowl')

        print(f"\nEpisode {ep_idx + 1}/{total_episodes} ({num_frames} frames)")

        # Reset simulation
        sim_robot.reset_scene(randomize=False)

        # Try to restore original duplo position from source dataset metadata
        # For now, just use default position

        output_dataset.create_episode_buffer()

        # Play through each frame
        for frame_offset in range(num_frames):
            frame_idx = from_idx + frame_offset
            source_frame = source_dataset[frame_idx]

            # Get joint action from source
            joint_action = source_frame[action_key].numpy()
            if joint_action.ndim > 1:
                joint_action = joint_action[0]  # Take first timestep if chunked

            # Convert to action dict (joint_action is in degrees)
            action_dict = {f"{MOTOR_NAMES[i]}.pos": float(joint_action[i]) for i in range(6)}

            # Send to sim
            sim_robot.send_action(action_dict)

            # Get new observation
            observation = sim_robot.get_observation()

            # Build observation frame from simulation (new camera views)
            obs_frame = build_dataset_frame(output_dataset.features, observation, prefix="observation")

            # Build output frame with new observations and original actions from source
            output_frame = {
                **obs_frame,
                "task": task,
            }
            # Copy all action fields from source
            for key in source_frame.keys():
                if key.startswith("action"):
                    output_frame[key] = source_frame[key]

            output_dataset.add_frame(output_frame)

            # Progress
            if (frame_offset + 1) % 50 == 0:
                print(f"  Frame {frame_offset + 1}/{num_frames}", end="\r")

        output_dataset.save_episode()
        print(f"  Saved episode {ep_idx + 1} ({num_frames} frames)")

    # Finalize
    print("\nFinalizing dataset...")
    output_dataset.finalize()

    print("\n" + "=" * 60)
    print("RE-RECORDING COMPLETE")
    print("=" * 60)
    print(f"Source: {args.source_dataset}")
    print(f"Output: {root_dir}")
    print(f"Episodes: {total_episodes}")
    print(f"Depth: {'enabled' if args.depth else 'disabled'}")
    print("=" * 60)

    # Cleanup
    sim_robot.disconnect()


if __name__ == "__main__":
    main()
