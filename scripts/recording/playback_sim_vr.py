#!/usr/bin/env python
"""
Play back recorded episodes in simulation.

Loads a LeRobot dataset and replays the actions through the simulation.
Supports both VR and screen rendering modes.

Usage:
    python playback_sim_vr.py danbhf/sim_pick_place_20251228_213938
    python playback_sim_vr.py danbhf/sim_pick_place_20251228_213938 --episode 0
    python playback_sim_vr.py danbhf/sim_pick_place_20251228_213938 --no-vr
    python playback_sim_vr.py ./datasets/20251228_213938 --local

Controls:
    ENTER/SPACE - Play next episode
    R           - Replay current episode
    Q           - Quit
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import msvcrt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import sim
import lerobot_robot_sim
from lerobot_robot_sim import SO100Sim, SO100SimConfig, MOTOR_NAMES

# Import LeRobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset


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
        # Try to extract timestamp (last part after _)
        parts = name.split("_")
        if len(parts) >= 2:
            # Try timestamps like "20251228_213938"
            for i in range(len(parts) - 1):
                timestamp = f"{parts[i]}_{parts[i+1]}"
                if timestamp.isdigit() or (parts[i].isdigit() and parts[i+1].isdigit()):
                    candidate = datasets_dir / f"{parts[-2]}_{parts[-1]}"
                    if candidate.exists():
                        return candidate
            # Try just the last part as timestamp folder
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
            # Check if timestamp matches
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


def play_episode(sim_robot: SO100Sim, dataset: LeRobotDataset, episode_idx: int, use_vr: bool = True):
    """Play back a single episode."""

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

        # Render (VR or screen)
        if use_vr:
            sim_robot.render_vr()
        else:
            sim_robot.render()

        if not paused:
            # Get action for this frame
            data_idx = from_idx + frame_idx
            frame_data = dataset[data_idx]

            # Build action dict from dataset
            # Actions can be stored as tensor or as individual keys
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

            # Send to simulation
            if action:
                sim_robot.send_action(action)

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
    parser = argparse.ArgumentParser(description="Play back episodes in simulation")
    parser.add_argument("dataset", type=str, help="HuggingFace repo ID or local path")
    parser.add_argument("--episode", "-e", type=int, default=None, help="Episode index to play (default: all)")
    parser.add_argument("--local", action="store_true", help="Force load from local path")
    parser.add_argument("--no-vr", action="store_true", help="Render to screen instead of VR")
    parser.add_argument("--loop", action="store_true", help="Loop playback continuously")

    args = parser.parse_args()
    use_vr = not args.no_vr

    # Try to find local dataset first
    print(f"Loading dataset: {args.dataset}")
    local_path = find_local_dataset(args.dataset)

    if local_path:
        print(f"Found local dataset: {local_path}")
        dataset = LeRobotDataset(repo_id=args.dataset, root=local_path)
    elif args.local:
        print(f"ERROR: Could not find local dataset: {args.dataset}")
        print("Available datasets in ./datasets/:")
        datasets_dir = Path(__file__).parent.parent / "datasets"
        if datasets_dir.exists():
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
            print("\nTrying to find local dataset...")
            local_path = find_local_dataset(args.dataset)
            if local_path:
                print(f"Found local: {local_path}")
                dataset = LeRobotDataset(repo_id=args.dataset, root=local_path)
            else:
                print("No local dataset found either.")
                print("\nAvailable local datasets:")
                datasets_dir = Path(__file__).parent.parent / "datasets"
                if datasets_dir.exists():
                    for folder in sorted(datasets_dir.iterdir()):
                        if folder.is_dir():
                            print(f"  ./datasets/{folder.name}")
                return

    # Try to load per-episode scene info
    episode_scenes_path = dataset.root / "meta" / "episode_scenes.json"
    print(f"Looking for episode_scenes at: {episode_scenes_path}")
    if episode_scenes_path.exists():
        import json
        with open(episode_scenes_path) as f:
            dataset._episode_scenes = json.load(f)
        print(f"Loaded scene info for {len(dataset._episode_scenes)} episodes")
    else:
        dataset._episode_scenes = {}

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

    # Initialize simulation
    mode_str = "VR" if use_vr else "screen"
    print(f"\nInitializing {mode_str} simulation...")
    sim_config = SO100SimConfig(
        id="sim_playback",
        sim_cameras=["wrist_cam", "overhead_cam"],
        camera_width=640,
        camera_height=480,
        enable_vr=use_vr,
        n_sim_steps=10,
    )
    sim_robot = SO100Sim(sim_config)
    sim_robot.connect()

    print("\n" + "=" * 50)
    print(f"PLAYBACK ({mode_str.upper()})")
    print("=" * 50)
    print("Controls:")
    print("  ENTER/SPACE - Start / Pause")
    print("  R           - Replay current episode")
    print("  N           - Next episode")
    print("  Q           - Quit")
    print("=" * 50)

    if use_vr:
        print("\nPut on VR headset. Press ENTER to start playback...")
    else:
        print("\nPress ENTER to start playback...")

    # Wait for ready (with rendering)
    ready = False
    while not ready:
        if use_vr:
            sim_robot.render_vr()
        else:
            sim_robot.render()
        key = check_key()
        if key == b'\r' or key == b' ':
            ready = True
        elif key == b'q':
            sim_robot.disconnect()
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

            # Reset scene before episode
            print(f"\n--- Setting up episode {ep_idx} ---")
            sim_robot.reset_scene(randomize=False)

            # Initialize duplo position from per-episode scene info or dataset metadata
            scene_initialized = False

            # Try per-episode scene info first
            print(f"Looking for episode {ep_idx} in _episode_scenes (has {len(getattr(dataset, '_episode_scenes', {}))} entries)")
            if hasattr(dataset, '_episode_scenes') and dataset._episode_scenes:
                ep_scene = dataset._episode_scenes.get(str(ep_idx))
                if ep_scene and 'objects' in ep_scene and 'duplo' in ep_scene['objects']:
                    duplo_info = ep_scene['objects']['duplo']
                    duplo_pos = duplo_info['position']
                    # Get quaternion if available
                    quat = None
                    if 'quaternion' in duplo_info:
                        q = duplo_info['quaternion']
                        quat = [q['w'], q['x'], q['y'], q['z']]
                    sim_robot.set_duplo_position(duplo_pos['x'], duplo_pos['y'], duplo_pos.get('z'), quat=quat)
                    print(f"Duplo initialized at ({duplo_pos['x']:.3f}, {duplo_pos['y']:.3f}) [episode {ep_idx}]")
                    scene_initialized = True
                else:
                    print(f"No scene found for episode {ep_idx} (keys: {list(dataset._episode_scenes.keys())})")

            # Fall back to dataset-level scene info
            if not scene_initialized and 'scene' in dataset.meta.info:
                scene = dataset.meta.info['scene']
                if 'objects' in scene and 'duplo' in scene['objects']:
                    duplo_info = scene['objects']['duplo']
                    duplo_pos = duplo_info['position']
                    quat = None
                    if 'quaternion' in duplo_info:
                        q = duplo_info['quaternion']
                        quat = [q['w'], q['x'], q['y'], q['z']]
                    sim_robot.set_duplo_position(duplo_pos['x'], duplo_pos['y'], duplo_pos.get('z'), quat=quat)
                    print(f"Duplo initialized at ({duplo_pos['x']:.3f}, {duplo_pos['y']:.3f}) [dataset default]")

            # Small delay for scene to settle
            for _ in range(10):
                if use_vr:
                    sim_robot.render_vr()
                else:
                    sim_robot.render()
                time.sleep(0.03)

            # Play episode
            completed = play_episode(sim_robot, dataset, ep_idx, use_vr=use_vr)

            if not completed:
                break

            episode_cursor += 1

            # Check for next action
            if episode_cursor < len(episodes):
                print(f"\nPress ENTER for next episode, R to replay, Q to quit...")
                waiting = True
                while waiting:
                    if use_vr:
                        sim_robot.render_vr()
                    else:
                        sim_robot.render()
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
                    if use_vr:
                        sim_robot.render_vr()
                    else:
                        sim_robot.render()
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
        sim_robot.disconnect()
        print("Done")


if __name__ == "__main__":
    main()
