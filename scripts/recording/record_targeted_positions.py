#!/usr/bin/env python
"""
Record demonstrations at specific target positions from a JSON file.

For "few-shot gap filling" experiments - recording a small number of episodes
at strategic positions to fill coverage gaps in the training data.

Each position in the JSON file gets one episode recorded.
Supports reset/discard functionality to redo failed recordings.

Usage:
    python record_targeted_positions.py --positions configs/gap_filling_positions.json
    python record_targeted_positions.py --positions my_positions.json --no-randomize

JSON format:
    {
      "positions": [
        {"x": 0.27, "y": 0.12, "rotation": 0, "note": "optional description"},
        {"x": 0.30, "y": 0.08, "rotation": 45},
        ...
      ]
    }

Controls:
    ENTER - Start recording / Save episode and move to next position
    D     - Discard current episode (stay at same position to retry)
    R     - Reset scene (when not recording)
    S     - Skip to next position without recording
    Q     - Quit (saves current progress)
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
import msvcrt

# Add paths for imports
_editable_lerobot = Path(__file__).parent.parent.parent.parent / "lerobot" / "src"
if _editable_lerobot.exists():
    sys.path.insert(0, str(_editable_lerobot))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Text-to-speech
try:
    import pyttsx3
    _tts_available = True
except ImportError:
    _tts_available = False


def speak(text: str):
    """Speak text aloud (non-blocking) and print it."""
    print(f"\n{text}")
    if _tts_available:
        try:
            import subprocess
            import threading
            def _speak_thread():
                try:
                    subprocess.run(
                        ['powershell', '-Command', f'Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{text}")'],
                        creationflags=subprocess.CREATE_NO_WINDOW,
                        timeout=10
                    )
                except:
                    pass
            thread = threading.Thread(target=_speak_thread, daemon=True)
            thread.start()
        except:
            pass


import lerobot_robot_sim
from lerobot_robot_sim import SO100Sim, SO100SimConfig, MOTOR_NAMES
from SO100LeaderSTS3250 import SO100LeaderSTS3250, SO100LeaderSTS3250Config
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features, build_dataset_frame


class State(Enum):
    SETUP = "setup"
    READY = "ready"
    RECORDING = "recording"
    FINISHED = "finished"


def load_positions(filepath: str) -> list:
    """Load positions from JSON file."""
    with open(filepath) as f:
        data = json.load(f)

    positions = data.get("positions", data)  # Handle both {positions: [...]} and [...]

    # Validate
    for i, pos in enumerate(positions):
        if "x" not in pos or "y" not in pos:
            raise ValueError(f"Position {i} missing required 'x' or 'y' field")
        pos.setdefault("rotation", 0)
        pos.setdefault("note", "")

    return positions


def load_config():
    """Load config.json for leader arm port."""
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


def check_key():
    """Non-blocking keyboard check."""
    if msvcrt.kbhit():
        key = msvcrt.getch()
        if key == b'\xe0':
            msvcrt.getch()
            return None
        return key
    return None


def get_git_info() -> dict:
    """Get git repository information."""
    import subprocess
    repo_root = Path(__file__).parent.parent

    def run_git(args):
        try:
            result = subprocess.run(["git"] + args, cwd=repo_root, capture_output=True, text=True, timeout=5)
            return result.stdout.strip() if result.returncode == 0 else None
        except:
            return None

    return {
        "commit_hash": run_git(["rev-parse", "HEAD"]),
        "commit_short": run_git(["rev-parse", "--short", "HEAD"]),
        "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Record demos at targeted positions")
    parser.add_argument("--positions", "-p", type=str, required=True, help="JSON file with target positions")
    parser.add_argument("--task", "-t", type=str, default="Pick up the Duplo block and place it in the bowl")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--repo_id", type=str, default=None)
    parser.add_argument("--root", type=str, default="./datasets")
    parser.add_argument("--leader_port", type=str, default=None)
    parser.add_argument("--max_duration", type=float, default=60.0)
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--no-randomize", action="store_true", help="Disable small position noise around target")
    parser.add_argument("--pos_range", type=float, default=1.0, help="Small noise range in cm (default: 1cm)")
    parser.add_argument("--start-from", type=int, default=0, help="Start from position index (for resuming)")
    parser.add_argument("--depth", action="store_true", help="Enable depth rendering")

    args = parser.parse_args()

    # Load positions
    print(f"Loading positions from: {args.positions}")
    positions = load_positions(args.positions)
    print(f"Found {len(positions)} target positions")

    if args.start_from > 0:
        print(f"Starting from position {args.start_from}")
        positions = positions[args.start_from:]
        print(f"  {len(positions)} positions remaining")

    # Get leader config
    config = load_config()
    leader_port = args.leader_port
    if leader_port is None:
        if config and "leader" in config:
            leader_port = config["leader"]["port"]
        else:
            leader_port = "COM8"
    leader_id = config["leader"]["id"] if config and "leader" in config else "leader_so100"
    print(f"Leader port: {leader_port}")

    # Generate repo ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    repo_id = args.repo_id or f"danbhf/sim_pick_place_targeted_{timestamp}"
    root_dir = Path(args.root) / timestamp
    print(f"Dataset: {repo_id}")
    print(f"Storage: {root_dir}")

    # Initialize simulation
    print("\nInitializing simulation with VR...")
    depth_cameras = ["overhead_cam"] if args.depth else []
    sim_config = SO100SimConfig(
        id="sim_recorder",
        sim_cameras=["wrist_cam", "overhead_cam"],
        depth_cameras=depth_cameras,
        camera_width=640,
        camera_height=480,
        enable_vr=True,
        n_sim_steps=10,
    )
    sim_robot = SO100Sim(sim_config)
    sim_robot.connect()
    speak("Simulation ready")

    for _ in range(10):
        sim_robot.render_vr()

    # Connect leader arm
    print(f"\nConnecting leader arm...")
    leader_config = SO100LeaderSTS3250Config(port=leader_port, id=leader_id)
    leader = SO100LeaderSTS3250(leader_config)
    leader.connect()
    speak("Leader arm connected")

    for _ in range(10):
        sim_robot.render_vr()

    # Create dataset
    print(f"\nCreating dataset...")
    action_features = hw_to_dataset_features(sim_robot.action_features, "action")
    obs_features = hw_to_dataset_features(sim_robot.observation_features, "observation")
    features = {**action_features, **obs_features}

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=args.fps,
        root=root_dir,
        robot_type="so100_sim",
        features=features,
        image_writer_threads=4,
    )

    # Save metadata
    scene_info = sim_robot.get_scene_info()
    dataset.meta.info["scene"] = scene_info
    info_path = root_dir / "meta" / "info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(info_path, "w") as f:
        json.dump(dataset.meta.info, f, indent=4)

    # Save recording metadata
    recording_metadata = {
        "recording_type": "targeted_positions",
        "positions_file": str(args.positions),
        "timestamp": timestamp,
        "repo_id": repo_id,
        "task": args.task,
        "fps": args.fps,
        "total_positions": len(positions),
        "start_from": args.start_from,
        "git": get_git_info(),
        "target_positions": positions,
    }
    metadata_path = root_dir / "meta" / "recording_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(recording_metadata, f, indent=2)

    # Print instructions
    print("\n" + "=" * 60)
    print("TARGETED POSITION RECORDING")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Positions: {len(positions)}")
    print(f"FPS: {args.fps}")
    print("=" * 60)
    print("\nControls:")
    print("  ENTER - Start recording / Save episode")
    print("  D     - Discard episode (retry same position)")
    print("  R     - Reset scene")
    print("  S     - Skip to next position")
    print("  Q     - Quit")
    print("  SPACE - Recenter VR")
    print("=" * 60)

    for _ in range(30):
        sim_robot.render_vr()

    # State machine
    state = State.SETUP
    position_idx = 0
    successful_episodes = 0
    episode_positions = {}  # Map episode -> position info

    episode_start_time = None
    episode_frames = 0
    task_complete_frames = 0
    last_action = None
    consecutive_errors = 0
    frame_time = 1.0 / args.fps
    last_frame_time = time.time()

    def set_position(idx):
        """Set block to target position."""
        pos = positions[idx]
        rot_deg = pos.get("rotation", 0)
        note = pos.get("note", "")

        # Small rotation noise (±10°) when randomize enabled, exact rotation otherwise
        rot_noise = np.radians(10) if not args.no_randomize else 0

        sim_robot.reset_scene(
            randomize=not args.no_randomize,
            pos_range=args.pos_range / 100.0 if not args.no_randomize else 0,
            rot_range=rot_noise,
            pos_center_x=pos["x"],
            pos_center_y=pos["y"],
            rot_offset=np.radians(rot_deg),
        )

        print(f"\n[Position {idx + 1}/{len(positions)}] ({pos['x']:.3f}, {pos['y']:.3f}) rot={rot_deg}")
        if note:
            print(f"  Note: {note}")

    speak("Put on VR headset. Press ENTER when ready.")

    try:
        while state != State.FINISHED:
            loop_start = time.time()
            sim_robot.render_vr()

            # Read leader arm
            try:
                action = leader.get_action()
                last_action = action.copy()
                consecutive_errors = 0
            except ConnectionError:
                consecutive_errors += 1
                if consecutive_errors > 30:
                    print("\n[!] Leader arm connection lost!")
                action = last_action if last_action else None

            if action:
                sim_robot.send_action(action)

            key = check_key()

            # VR recenter
            if key == b' ':
                sim_robot.recenter_vr()
                key = None

            if state == State.SETUP:
                if key == b'\r':
                    state = State.READY
                    set_position(position_idx)
                    speak(f"Position 1 of {len(positions)}. Press ENTER to record.")
                elif key == b'q':
                    state = State.FINISHED

            elif state == State.READY:
                if key == b'\r':
                    state = State.RECORDING
                    dataset.create_episode_buffer()
                    episode_start_time = time.time()
                    episode_frames = 0
                    task_complete_frames = 0
                    speak("Recording")
                    print(f"Recording position {position_idx + 1}...")

                elif key == b'r':
                    set_position(position_idx)

                elif key == b's':
                    # Skip to next position
                    position_idx += 1
                    if position_idx >= len(positions):
                        state = State.FINISHED
                        speak("All positions done")
                    else:
                        set_position(position_idx)
                        speak(f"Skipped. Position {position_idx + 1} of {len(positions)}")

                elif key == b'q':
                    state = State.FINISHED

            elif state == State.RECORDING:
                elapsed = time.time() - episode_start_time

                # Record frame
                if time.time() - last_frame_time >= frame_time:
                    last_frame_time = time.time()
                    observation = sim_robot.get_observation()
                    obs_frame = build_dataset_frame(dataset.features, observation, prefix="observation")
                    action_frame = build_dataset_frame(dataset.features, action, prefix="action")
                    dataset.add_frame({**obs_frame, **action_frame, "task": args.task})
                    episode_frames += 1

                    if episode_frames % 30 == 0:
                        print(f"  Frames: {episode_frames:4d} | Time: {elapsed:5.1f}s", end="\r")

                # Check task completion
                if sim_robot.is_task_complete():
                    task_complete_frames += 1
                    if task_complete_frames >= 10:
                        speak("Task complete!")
                        episode_positions[successful_episodes] = positions[position_idx]
                        dataset.save_episode()
                        successful_episodes += 1
                        print(f"\n\nEpisode {successful_episodes} saved ({episode_frames} frames)")

                        position_idx += 1
                        if position_idx >= len(positions):
                            state = State.FINISHED
                            speak("All positions recorded!")
                        else:
                            state = State.READY
                            set_position(position_idx)
                            speak(f"Position {position_idx + 1} of {len(positions)}. Press ENTER.")
                else:
                    task_complete_frames = 0

                # Timeout
                if elapsed > args.max_duration:
                    speak("Timeout")
                    print(f"\n\nTimeout after {elapsed:.1f}s")
                    episode_positions[successful_episodes] = positions[position_idx]
                    dataset.save_episode()
                    successful_episodes += 1
                    print(f"Episode {successful_episodes} saved (incomplete)")

                    position_idx += 1
                    if position_idx >= len(positions):
                        state = State.FINISHED
                    else:
                        state = State.READY
                        set_position(position_idx)
                        speak(f"Position {position_idx + 1}. Press ENTER.")

                # Keys during recording
                if key == b'\r':
                    episode_positions[successful_episodes] = positions[position_idx]
                    dataset.save_episode()
                    successful_episodes += 1
                    print(f"\n\nEpisode {successful_episodes} saved ({episode_frames} frames)")

                    position_idx += 1
                    if position_idx >= len(positions):
                        state = State.FINISHED
                        speak("All positions recorded!")
                    else:
                        state = State.READY
                        set_position(position_idx)
                        speak(f"Position {position_idx + 1} of {len(positions)}. Press ENTER.")

                elif key == b'd':
                    # Discard - stay at same position for retry
                    dataset.clear_episode_buffer()
                    speak("Episode discarded. Retry this position.")
                    print("\n\nEpisode discarded")
                    state = State.READY
                    set_position(position_idx)  # Reset same position

                elif key == b'q':
                    dataset.clear_episode_buffer()
                    print("\n\nQuitting (current episode discarded)")
                    state = State.FINISHED

            elapsed = time.time() - loop_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        if state == State.RECORDING:
            dataset.clear_episode_buffer()

    finally:
        print("\n" + "=" * 60)
        print("RECORDING COMPLETE")
        print("=" * 60)
        print(f"Episodes saved: {successful_episodes} / {len(positions)}")
        print(f"Dataset: {root_dir}")

        if successful_episodes > 0:
            print("\nFinalizing dataset...")
            dataset.finalize()

            # Save episode-position mapping
            ep_positions_path = root_dir / "meta" / "episode_positions.json"
            with open(ep_positions_path, "w") as f:
                json.dump(episode_positions, f, indent=2)
            print(f"Saved position info for {len(episode_positions)} episodes")

            if not args.no_upload:
                speak("Uploading to HuggingFace")
                print("\nUploading to HuggingFace Hub...")
                try:
                    import subprocess
                    upload_script = Path(__file__).parent / "upload_dataset.py"
                    dataset_path = root_dir.resolve()
                    result = subprocess.run([sys.executable, str(upload_script), str(dataset_path), repo_id])
                    if result.returncode == 0:
                        speak("Upload complete")
                    else:
                        speak("Upload failed")
                except Exception as e:
                    speak("Upload failed")
                    print(f"Upload failed: {e}")

        sim_robot.disconnect()
        leader.disconnect()
        speak("Done")


if __name__ == "__main__":
    main()
