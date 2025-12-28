#!/usr/bin/env python
"""
Record pick-and-place demonstrations using SO100 simulation with VR.

Task: Pick up the Duplo block and place it in the bowl.
Episode automatically ends when task is complete (duplo in bowl).

Saves to LeRobot dataset format v3.0.

This version runs the simulation continuously (never freezes VR view).

Usage:
    python record_sim_vr_pickplace.py --task "Pick up Duplo" --num_episodes 10

Controls:
    ENTER - Start recording / Save episode and continue
    D     - Discard current episode
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

# Add src to path for lerobot_robot_sim
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Text-to-speech setup
try:
    import pyttsx3
    _tts_available = True
except ImportError:
    _tts_available = False
    logger.warning("pyttsx3 not available, using print-only announcements")


def speak(text: str):
    """Speak text aloud (non-blocking) and print it."""
    print(f"\n🔊 {text}")
    if _tts_available:
        try:
            import subprocess
            import threading
            # Run TTS in background thread so it doesn't block the main loop
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
        except Exception as e:
            print(f"  (TTS failed: {e})")


# Import the sim plugin
import lerobot_robot_sim
from lerobot_robot_sim import SO100Sim, SO100SimConfig, MOTOR_NAMES

# Import LeRobot dataset utilities
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features, build_dataset_frame


class State(Enum):
    """Recording state machine states."""
    SETUP = "setup"              # Initial setup, waiting for headset
    READY = "ready"              # Between episodes, waiting to start recording
    RECORDING = "recording"      # Actively recording an episode
    FINISHED = "finished"        # All done


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


def load_calibration(arm_id: str = "leader_so100"):
    """Load calibration from JSON file."""
    import draccus
    from lerobot.motors import MotorCalibration
    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION

    calib_path = HF_LEROBOT_CALIBRATION / "teleoperators" / "so100_leader_sts3250" / f"{arm_id}.json"
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration not found: {calib_path}")

    with open(calib_path) as f, draccus.config_type("json"):
        return draccus.load(dict[str, MotorCalibration], f)


def create_leader_bus(port: str):
    """Create motor bus for leader arm."""
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    bus = FeetechMotorsBus(
        port=port,
        motors={
            "shoulder_pan": Motor(1, "sts3250", MotorNormMode.RANGE_M100_100),
            "shoulder_lift": Motor(2, "sts3250", MotorNormMode.RANGE_M100_100),
            "elbow_flex": Motor(3, "sts3250", MotorNormMode.RANGE_M100_100),
            "wrist_flex": Motor(4, "sts3250", MotorNormMode.RANGE_M100_100),
            "wrist_roll": Motor(5, "sts3250", MotorNormMode.RANGE_M100_100),
            "gripper": Motor(6, "sts3250", MotorNormMode.RANGE_0_100),
        },
    )
    return bus


def check_key():
    """Non-blocking keyboard check. Returns key or None."""
    if msvcrt.kbhit():
        key = msvcrt.getch()
        # Handle special keys (arrows, etc.)
        if key == b'\xe0':
            msvcrt.getch()  # Consume the second byte
            return None
        return key
    return None


def main():
    parser = argparse.ArgumentParser(description="Record pick-place demos with VR (continuous sim)")
    parser.add_argument("--task", "-t", type=str, default="Pick up the Duplo block and place it in the bowl")
    parser.add_argument("--num_episodes", "-n", type=int, default=10)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--repo_id", type=str, default=None)
    parser.add_argument("--root", type=str, default="./datasets")
    parser.add_argument("--leader_port", type=str, default=None)
    parser.add_argument("--max_duration", type=float, default=60.0, help="Max episode duration in seconds")
    parser.add_argument("--no-upload", action="store_true", help="Don't upload to HuggingFace")
    parser.add_argument("--pos_range", type=float, default=2.0, help="Position randomization range in cm")
    parser.add_argument("--rot_range", type=float, default=180.0, help="Rotation randomization range in degrees")
    parser.add_argument("--no-randomize", action="store_true", help="Disable position/rotation randomization")

    args = parser.parse_args()

    # Get leader port
    leader_port = args.leader_port
    if leader_port is None:
        config = load_config()
        if config and "leader" in config:
            leader_port = config["leader"]["port"]
        else:
            leader_port = "COM8"
    print(f"Leader port: {leader_port}")

    # Generate repo ID with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    repo_id = args.repo_id or f"danbhf/sim_pick_place_{timestamp}"
    print(f"Dataset: {repo_id}")

    # Use timestamped root directory
    root_dir = Path(args.root) / timestamp
    print(f"Storage: {root_dir}")

    # Create simulation with VR
    print("\nInitializing simulation with VR...")
    sim_config = SO100SimConfig(
        id="sim_recorder",
        sim_cameras=["wrist_cam", "overhead_cam"],
        camera_width=640,
        camera_height=480,
        enable_vr=True,
        n_sim_steps=10,
    )
    sim_robot = SO100Sim(sim_config)
    sim_robot.connect()
    speak("Simulation ready")

    # Connect leader arm
    print(f"\nConnecting leader arm on {leader_port}...")
    leader_bus = create_leader_bus(leader_port)
    leader_bus.connect()
    leader_bus.calibration = load_calibration("leader_so100")
    leader_bus.disable_torque()
    speak("Leader arm connected")

    # Create dataset
    print(f"\nCreating dataset: {repo_id}")
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

    # Add scene info to metadata
    scene_info = sim_robot.get_scene_info()
    dataset.meta.info["scene"] = scene_info
    # LeRobot stores files at root_dir directly (repo_id is just HF metadata)
    info_path = root_dir / "meta" / "info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(info_path, "w") as f:
        json.dump(dataset.meta.info, f, indent=4)

    print("Dataset created (LeRobot v3.0 format)")

    # Print instructions
    print("\n" + "=" * 60)
    print("CONTINUOUS VR RECORDING")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Episodes: {args.num_episodes}")
    print(f"FPS: {args.fps}")
    if not args.no_randomize:
        print(f"Randomization: ±{args.pos_range}cm position, ±{args.rot_range}° rotation")
    print("=" * 60)
    print("\nControls:")
    print("  ENTER - Start recording / Save episode")
    print("  D     - Discard current episode")
    print("  R     - Reset scene (when not recording)")
    print("  Q     - Quit")
    print("\nTask auto-completes when Duplo lands in bowl!")
    print("=" * 60)

    # State machine variables
    state = State.SETUP
    successful_episodes = 0
    completed_tasks = 0

    # Recording state
    episode_start_time = None
    episode_frames = 0
    task_complete_frames = 0
    last_action = None
    consecutive_errors = 0

    frame_time = 1.0 / args.fps
    last_frame_time = time.time()

    speak("Put on VR headset. Press ENTER when ready.")

    try:
        while state != State.FINISHED:
            loop_start = time.time()

            # Always render VR and step simulation
            sim_robot.render_vr()

            # Read leader arm (always, for live preview)
            try:
                positions = leader_bus.sync_read("Present_Position")
                action = {f"{motor}.pos": positions[motor] for motor in MOTOR_NAMES}
                last_action = action.copy()
                consecutive_errors = 0
            except ConnectionError:
                consecutive_errors += 1
                if consecutive_errors > 30:
                    print("\n[!] Leader arm connection lost!")
                if last_action:
                    action = last_action
                else:
                    action = None

            # Send action to simulation (live preview)
            if action:
                sim_robot.send_action(action)

            # Check keyboard input
            key = check_key()

            # State machine
            if state == State.SETUP:
                # Waiting for user to put on headset and press Enter
                if key == b'\r':
                    state = State.READY
                    # Reset scene for first episode
                    sim_robot.reset_scene(
                        randomize=not args.no_randomize,
                        pos_range=args.pos_range / 100.0,
                        rot_range=np.radians(args.rot_range)
                    )
                    scene = sim_robot.get_scene_info()
                    duplo_pos = scene['objects']['duplo']['position']
                    print(f"\nDuplo at: ({duplo_pos['x']:.3f}, {duplo_pos['y']:.3f})")
                    speak(f"Ready. Episode {successful_episodes + 1} of {args.num_episodes}. Press ENTER to record.")
                elif key == b'q':
                    state = State.FINISHED

            elif state == State.READY:
                # Between episodes, waiting to start recording
                if key == b'\r':
                    # Start recording
                    state = State.RECORDING
                    dataset.create_episode_buffer()
                    episode_start_time = time.time()
                    episode_frames = 0
                    task_complete_frames = 0
                    speak("Recording")
                    print(f"\nRecording episode {successful_episodes + 1}...")
                elif key == b'r':
                    # Reset scene
                    sim_robot.reset_scene(
                        randomize=not args.no_randomize,
                        pos_range=args.pos_range / 100.0,
                        rot_range=np.radians(args.rot_range)
                    )
                    scene = sim_robot.get_scene_info()
                    duplo_pos = scene['objects']['duplo']['position']
                    print(f"\nScene reset. Duplo at: ({duplo_pos['x']:.3f}, {duplo_pos['y']:.3f})")
                elif key == b'q':
                    state = State.FINISHED

            elif state == State.RECORDING:
                elapsed = time.time() - episode_start_time

                # Record frame at target FPS
                if time.time() - last_frame_time >= frame_time:
                    last_frame_time = time.time()

                    # Get observation
                    observation = sim_robot.get_observation()

                    # Build and add frame
                    obs_frame = build_dataset_frame(dataset.features, observation, prefix="observation")
                    action_frame = build_dataset_frame(dataset.features, action, prefix="action")
                    dataset.add_frame({
                        **obs_frame,
                        **action_frame,
                        "task": args.task,
                    })
                    episode_frames += 1

                    # Progress display
                    if episode_frames % 30 == 0:
                        print(f"  Frames: {episode_frames:4d} | Time: {elapsed:5.1f}s", end="\r")

                # Check task completion
                if sim_robot.is_task_complete():
                    task_complete_frames += 1
                    if task_complete_frames >= 10:  # Debounce
                        speak("Task complete!")
                        completed_tasks += 1
                        # Save episode
                        dataset.save_episode()
                        successful_episodes += 1
                        print(f"\n\nEpisode {successful_episodes} saved ({episode_frames} frames, task completed)")

                        if successful_episodes >= args.num_episodes:
                            state = State.FINISHED
                            speak("All episodes recorded!")
                        else:
                            state = State.READY
                            # Reset for next episode
                            sim_robot.reset_scene(
                                randomize=not args.no_randomize,
                                pos_range=args.pos_range / 100.0,
                                rot_range=np.radians(args.rot_range)
                            )
                            scene = sim_robot.get_scene_info()
                            duplo_pos = scene['objects']['duplo']['position']
                            print(f"Duplo at: ({duplo_pos['x']:.3f}, {duplo_pos['y']:.3f})")
                            speak(f"Episode {successful_episodes + 1} of {args.num_episodes}. Press ENTER to record.")
                else:
                    task_complete_frames = 0

                # Check timeout
                if elapsed > args.max_duration:
                    speak("Timeout")
                    print(f"\n\nTimeout after {elapsed:.1f}s")
                    # Ask to save (but non-blocking style - just save incomplete)
                    dataset.save_episode()
                    successful_episodes += 1
                    print(f"Episode {successful_episodes} saved ({episode_frames} frames, incomplete)")

                    if successful_episodes >= args.num_episodes:
                        state = State.FINISHED
                    else:
                        state = State.READY
                        sim_robot.reset_scene(
                            randomize=not args.no_randomize,
                            pos_range=args.pos_range / 100.0,
                            rot_range=np.radians(args.rot_range)
                        )
                        speak(f"Episode {successful_episodes + 1}. Press ENTER to record.")

                # Handle keys during recording
                if key == b'\r':
                    # Manual stop - save episode
                    dataset.save_episode()
                    successful_episodes += 1
                    print(f"\n\nEpisode {successful_episodes} saved ({episode_frames} frames, manual stop)")

                    if successful_episodes >= args.num_episodes:
                        state = State.FINISHED
                        speak("All episodes recorded!")
                    else:
                        state = State.READY
                        sim_robot.reset_scene(
                            randomize=not args.no_randomize,
                            pos_range=args.pos_range / 100.0,
                            rot_range=np.radians(args.rot_range)
                        )
                        scene = sim_robot.get_scene_info()
                        duplo_pos = scene['objects']['duplo']['position']
                        print(f"Duplo at: ({duplo_pos['x']:.3f}, {duplo_pos['y']:.3f})")
                        speak(f"Episode {successful_episodes + 1}. Press ENTER to record.")

                elif key == b'd':
                    # Discard episode
                    dataset.clear_episode_buffer()
                    speak("Episode discarded")
                    print("\n\nEpisode discarded")
                    state = State.READY
                    sim_robot.reset_scene(
                        randomize=not args.no_randomize,
                        pos_range=args.pos_range / 100.0,
                        rot_range=np.radians(args.rot_range)
                    )
                    speak(f"Press ENTER to record episode {successful_episodes + 1}.")

                elif key == b'q':
                    # Quit - discard current and exit
                    dataset.clear_episode_buffer()
                    print("\n\nQuitting (current episode discarded)")
                    state = State.FINISHED

            # Maintain roughly consistent loop timing
            elapsed = time.time() - loop_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        if state == State.RECORDING:
            dataset.clear_episode_buffer()

    finally:
        # Summary
        print("\n" + "=" * 60)
        print("RECORDING COMPLETE")
        print("=" * 60)
        print(f"Episodes saved: {successful_episodes}")
        print(f"Tasks completed: {completed_tasks}")
        print(f"Dataset: {root_dir}")

        # Finalize and upload
        if successful_episodes > 0:
            print("\nFinalizing dataset...")
            dataset.finalize()
            print("Dataset finalized.")

            if not args.no_upload:
                speak("Uploading to HuggingFace")
                print("\nUploading to HuggingFace Hub...")
                try:
                    import subprocess
                    upload_script = Path(__file__).parent / "upload_dataset.py"
                    # LeRobot stores files at root_dir directly (repo_id is just HF metadata)
                    dataset_path = root_dir.resolve()
                    result = subprocess.run(
                        [sys.executable, str(upload_script), str(dataset_path), repo_id],
                    )
                    if result.returncode == 0:
                        speak("Upload complete")
                    else:
                        speak("Upload failed")
                except Exception as e:
                    speak("Upload failed")
                    print(f"Upload failed: {e}")
                    print(f"Upload manually: python upload_dataset.py {dataset_path} {repo_id}")

        # Cleanup
        sim_robot.disconnect()
        leader_bus.disconnect()
        speak("Done")


if __name__ == "__main__":
    main()
