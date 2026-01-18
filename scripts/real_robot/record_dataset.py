#!/usr/bin/env python
"""
Record a dataset using SO100 arms with STS3250 motors.

This script uses the custom STS3250 classes with file-based calibration.

Usage:
    python record_dataset.py --repo-id your-username/dataset-name --task "Pick up the cube"
    python record_dataset.py --repo-id your-username/dataset-name --task "Pick up the cube" --num-episodes 10
"""
import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass, field

# Import custom classes FIRST - this registers them with lerobot
from SO100LeaderSTS3250 import SO100LeaderSTS3250, SO100LeaderSTS3250Config
from SO100FollowerSTS3250 import SO100FollowerSTS3250, SO100FollowerSTS3250Config

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.scripts.lerobot_record import record, RecordConfig, DatasetRecordConfig


def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    config_file = Path(__file__).parent / config_path
    with open(config_file, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Record dataset with SO100 STS3250 arms")
    parser.add_argument("--repo-id", "-r", type=str, required=True,
                        help="HuggingFace repo ID (e.g., username/dataset-name)")
    parser.add_argument("--task", "-t", type=str, required=True,
                        help="Task description (e.g., 'Pick up the cube')")
    parser.add_argument("--num-episodes", "-n", type=int, default=50,
                        help="Number of episodes to record (default: 50)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Recording FPS (default: 30)")
    parser.add_argument("--root", type=str, default=None,
                        help="Local data directory (default: data/<dataset-name>)")
    parser.add_argument("--no-push", action="store_true",
                        help="Don't push to HuggingFace hub")
    parser.add_argument("--display", action="store_true",
                        help="Display camera feeds during recording")
    args = parser.parse_args()

    # Auto-generate root from repo-id if not specified
    if args.root is None:
        dataset_name = args.repo_id.split("/")[-1]
        args.root = f"data/{dataset_name}"

    # Load hardware config
    hw_config = load_config()

    leader_port = hw_config["leader"]["port"]
    leader_id = hw_config["leader"]["id"]
    follower_port = hw_config["follower"]["port"]
    follower_id = hw_config["follower"]["id"]

    # Build camera configs from config.json
    camera_configs = {}
    for cam_name, cam_cfg in hw_config.get("cameras", {}).items():
        camera_configs[cam_name] = OpenCVCameraConfig(
            cam_cfg["index_or_path"],  # positional arg (was camera_index)
            fps=cam_cfg.get("fps", 30),
            width=cam_cfg.get("width", 640),
            height=cam_cfg.get("height", 480),
        )

    # Create configs for robot and teleoperator
    robot_config = SO100FollowerSTS3250Config(
        port=follower_port,
        id=follower_id,
        cameras=camera_configs,
    )

    teleop_config = SO100LeaderSTS3250Config(
        port=leader_port,
        id=leader_id,
    )

    # Create dataset config
    dataset_config = DatasetRecordConfig(
        repo_id=args.repo_id,
        root=args.root,
        fps=args.fps,
        num_episodes=args.num_episodes,
        single_task=args.task,
        push_to_hub=not args.no_push,
    )

    # Create record config
    record_config = RecordConfig(
        robot=robot_config,
        teleop=teleop_config,
        dataset=dataset_config,
        display_data=args.display,
    )

    print("=" * 60)
    print("SO100 STS3250 Dataset Recording")
    print("=" * 60)
    print(f"Repo ID: {args.repo_id}")
    print(f"Task: {args.task}")
    print(f"Episodes: {args.num_episodes}")
    print(f"FPS: {args.fps}")
    print(f"Leader: {leader_port} (id={leader_id})")
    print(f"Follower: {follower_port} (id={follower_id})")
    print(f"Cameras: {list(camera_configs.keys())}")
    print(f"Push to hub: {not args.no_push}")
    print("=" * 60)

    # Run recording
    record(record_config)


if __name__ == "__main__":
    main()
