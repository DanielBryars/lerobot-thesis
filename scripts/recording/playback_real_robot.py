#!/usr/bin/env python
"""
Play back recorded episodes on the real follower robot.

Loads a LeRobot dataset and replays the actions on the physical SO100 follower arm.

WARNING: This will move the real robot! Make sure the workspace is clear.

Usage:
    python playback_real_robot.py danbhf/sim_pick_place_20251228_230352
    python playback_real_robot.py danbhf/sim_pick_place_20251228_230352 --episode 0
    python playback_real_robot.py ./datasets/20251228_230352 --local

Controls:
    ENTER/SPACE - Play next episode
    R           - Replay current episode
    Q           - Quit
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add recording dir to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import msvcrt

# Import follower robot class
from SO100FollowerSTS3250 import SO100FollowerSTS3250, SO100FollowerSTS3250Config

# Import LeRobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Motor names in order (must match recording)
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def load_config():
    """Load config.json for follower port."""
    repo_root = Path(__file__).parent.parent
    config_paths = [
        repo_root / "configs" / "config.json",
        Path("config.json"),
    ]
    for path in config_paths:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


def find_local_dataset(dataset_id: str) -> Path | None:
    """Try to find a local dataset matching the ID."""
    repo_root = Path(__file__).parent.parent
    datasets_dir = repo_root / "datasets"

    # Check if it's already a path
    if Path(dataset_id).exists():
        return Path(dataset_id)

    # Extract timestamp from repo_id like "danbhf/sim_pick_place_20251228_213938"
    if "/" in dataset_id:
        name = dataset_id.split("/")[-1]
        parts = name.split("_")
        if len(parts) >= 2:
            timestamp_candidates = [
                datasets_dir / f"{parts[-2]}_{parts[-1]}",
                datasets_dir / parts[-1],
            ]
            for candidate in timestamp_candidates:
                if candidate.exists():
                    return candidate

    # Search datasets directory for matching folders
    if datasets_dir.exists():
        for folder in datasets_dir.iterdir():
            if folder.is_dir() and dataset_id.replace("/", "_") in folder.name:
                return folder
            if "/" in dataset_id:
                name_part = dataset_id.split("/")[-1]
                if name_part in folder.name or folder.name in name_part:
                    return folder

    return None


def check_key():
    """Non-blocking keyboard check."""
    if msvcrt.kbhit():
        key = msvcrt.getch()
        if key == b'\xe0':
            msvcrt.getch()
            return None
        return key
    return None


def play_episode(follower: SO100FollowerSTS3250, dataset: LeRobotDataset, episode_idx: int):
    """Play back a single episode on the real robot."""

    # Get episode data bounds
    ep_meta = dataset.meta.episodes[episode_idx]
    from_idx = ep_meta['dataset_from_index']
    to_idx = ep_meta['dataset_to_index']
    num_frames = to_idx - from_idx

    fps = dataset.fps
    frame_time = 1.0 / fps

    print(f"\nPlaying episode {episode_idx} ({num_frames} frames at {fps} FPS)")
    print("Press Q to stop, SPACE to pause/resume")

    # Show first action to verify format
    first_frame = dataset[from_idx]
    if "action" in first_frame:
        action_tensor = first_frame["action"]
        print(f"First action: {action_tensor}")

    paused = False
    frame_idx = 0

    while frame_idx < num_frames:
        loop_start = time.time()

        # Check keyboard
        key = check_key()
        if key == b'q':
            print("\nStopped")
            return False
        elif key == b' ':
            paused = not paused
            print("\nPaused" if paused else "\nResumed")
        elif key == b'r':
            # Restart episode
            frame_idx = 0
            print("\nRestarting episode...")
            continue

        if not paused:
            # Get action for this frame
            data_idx = from_idx + frame_idx
            frame_data = dataset[data_idx]

            # Build action dict from dataset
            action = {}
            if "action" in frame_data:
                # Single tensor format: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
                action_tensor = frame_data["action"]
                if hasattr(action_tensor, 'numpy'):
                    action_tensor = action_tensor.numpy()
                for i, motor in enumerate(MOTOR_NAMES):
                    if i < len(action_tensor):
                        action[f"{motor}.pos"] = float(action_tensor[i])
            else:
                # Individual key format: action.{motor}.pos
                for motor in MOTOR_NAMES:
                    key_name = f"action.{motor}.pos"
                    if key_name in dataset.features:
                        action[f"{motor}.pos"] = frame_data[key_name].item()

            # Send to real robot
            if action:
                follower.send_action(action)

            frame_idx += 1

            # Progress
            if frame_idx % 30 == 0:
                elapsed = frame_idx / fps
                print(f"  Frame {frame_idx}/{num_frames} ({elapsed:.1f}s)", end="\r")

        # Maintain frame rate
        elapsed = time.time() - loop_start
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)

    print(f"\nEpisode {episode_idx} complete")
    return True


def main():
    parser = argparse.ArgumentParser(description="Play back episodes on real follower robot")
    parser.add_argument("dataset", type=str, help="HuggingFace repo ID or local path")
    parser.add_argument("--episode", "-e", type=int, default=None, help="Episode index to play (default: all)")
    parser.add_argument("--local", action="store_true", help="Force load from local path")
    parser.add_argument("--loop", action="store_true", help="Loop playback continuously")
    parser.add_argument("--follower_port", type=str, default=None, help="Follower COM port")

    args = parser.parse_args()

    # Get follower config
    config = load_config()
    follower_port = args.follower_port
    if follower_port is None:
        if config and "follower" in config:
            follower_port = config["follower"]["port"]
        else:
            follower_port = "COM7"
    follower_id = config["follower"]["id"] if config and "follower" in config else "follower_so100"

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    local_path = find_local_dataset(args.dataset)

    if local_path:
        print(f"Found local dataset: {local_path}")
        dataset = LeRobotDataset(repo_id=args.dataset, root=local_path)
    elif args.local:
        print(f"ERROR: Could not find local dataset: {args.dataset}")
        datasets_dir = Path(__file__).parent.parent / "datasets"
        if datasets_dir.exists():
            print("Available datasets in ./datasets/:")
            for folder in sorted(datasets_dir.iterdir()):
                if folder.is_dir():
                    print(f"  {folder.name}")
        return
    else:
        # Try HuggingFace
        try:
            dataset = LeRobotDataset(repo_id=args.dataset)
        except Exception as e:
            print(f"ERROR loading from HuggingFace: {e}")
            local_path = find_local_dataset(args.dataset)
            if local_path:
                print(f"Found local: {local_path}")
                dataset = LeRobotDataset(repo_id=args.dataset, root=local_path)
            else:
                print("No local dataset found either.")
                return

    num_episodes = dataset.num_episodes
    fps = dataset.fps
    print(f"Loaded {num_episodes} episodes at {fps} FPS")

    # Show episode info
    print("\nEpisodes:")
    for i in range(num_episodes):
        ep_meta = dataset.meta.episodes[i]
        from_idx = ep_meta['dataset_from_index']
        to_idx = ep_meta['dataset_to_index']
        num_frames = to_idx - from_idx
        duration = num_frames / fps
        print(f"  [{i}] {num_frames} frames ({duration:.1f}s)")

    # Connect to follower robot
    print(f"\nConnecting to follower robot on {follower_port}...")
    follower_config = SO100FollowerSTS3250Config(port=follower_port, id=follower_id, cameras={})
    follower = SO100FollowerSTS3250(follower_config)

    try:
        follower.connect()
        print("Follower robot connected!")
    except Exception as e:
        print(f"ERROR: Failed to connect to follower: {e}")
        return

    print("\n" + "=" * 50)
    print("REAL ROBOT PLAYBACK")
    print("=" * 50)
    print("WARNING: Robot will move! Ensure workspace is clear.")
    print("")
    print("Controls:")
    print("  ENTER/SPACE - Start / Pause")
    print("  R           - Replay current episode")
    print("  N           - Next episode")
    print("  Q           - Quit")
    print("=" * 50)

    print("\nPress ENTER to start playback...")

    # Wait for ready
    ready = False
    while not ready:
        key = check_key()
        if key == b'\r' or key == b' ':
            ready = True
        elif key == b'q':
            follower.disconnect()
            return
        time.sleep(0.03)

    # Determine episodes to play
    if args.episode is not None:
        episodes = [args.episode]
    else:
        episodes = list(range(num_episodes))

    try:
        episode_cursor = 0
        while episode_cursor < len(episodes):
            ep_idx = episodes[episode_cursor]

            print(f"\n--- Episode {ep_idx} ---")

            # Play episode
            completed = play_episode(follower, dataset, ep_idx)

            if not completed:
                break

            episode_cursor += 1

            # Check for next action
            if episode_cursor < len(episodes):
                print(f"\nPress ENTER for next episode, R to replay, Q to quit...")
                waiting = True
                while waiting:
                    key = check_key()
                    if key == b'\r' or key == b' ' or key == b'n':
                        waiting = False
                    elif key == b'r':
                        episode_cursor -= 1  # Replay current
                        waiting = False
                    elif key == b'q':
                        episode_cursor = len(episodes)  # Exit
                        waiting = False
                    time.sleep(0.03)
            elif args.loop:
                episode_cursor = 0
                print("\nLooping...")
            else:
                # Last episode - wait before closing
                print(f"\nPlayback complete! Press Q to quit, R to replay...")
                waiting = True
                while waiting:
                    key = check_key()
                    if key == b'q' or key == b'\x1b':  # q or ESC
                        waiting = False
                    elif key == b'r':
                        episode_cursor -= 1  # Replay last
                        waiting = False
                    time.sleep(0.03)
                if episode_cursor < len(episodes):
                    continue  # Replay was requested

        print("\nDone!")

    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        follower.disconnect()
        print("Follower disconnected")


if __name__ == "__main__":
    main()
