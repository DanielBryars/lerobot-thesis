#!/usr/bin/env python
"""
Record pick-and-place demonstrations using SO100 simulation with VR.

Task: Pick up the Duplo block and place it in the bowl.
Episode automatically ends when task is complete (duplo in bowl).

Saves to LeRobot dataset format v3.0.

Usage:
    python record_sim_vr_pickplace.py --task "Pick up Duplo" --num_episodes 10
"""

import argparse
import json
import logging
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

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
    """Speak text aloud and print it."""
    print(f"\n🔊 {text}")
    if _tts_available:
        try:
            # Use subprocess to avoid COM threading conflicts with VR/OpenGL
            import subprocess
            # Use run() to wait for completion (avoids overlapping speech)
            subprocess.run(
                ['powershell', '-Command', f'Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{text}")'],
                creationflags=subprocess.CREATE_NO_WINDOW,
                timeout=10
            )
        except Exception as e:
            # Fallback to pyttsx3
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 180)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e2:
                print(f"  (TTS failed: {e2})")


# Import the sim plugin (registers so100_sim)
import lerobot_robot_sim
from lerobot_robot_sim import SO100Sim, SO100SimConfig, MOTOR_NAMES

# Import LeRobot dataset utilities
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features, build_dataset_frame


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


def record_episode(
    sim_robot: SO100Sim,
    leader_bus,
    dataset: LeRobotDataset,
    episode_idx: int,
    task: str,
    fps: int = 30,
    max_duration: float = 60.0
):
    """Record a single episode until task completion or timeout."""
    print(f"\n{'='*50}")
    print(f"Episode {episode_idx + 1}")
    print(f"{'='*50}")

    speak("Move leader arm to starting position")
    speak("Press Enter to start recording")
    input()

    # Start episode buffer
    dataset.create_episode_buffer()

    frame_time = 1.0 / fps
    start_time = time.time()

    speak("Recording started")
    print("(auto-stops when Duplo lands in bowl)")
    print("Press 'q' to stop, 'd' to discard")

    # For keyboard input
    stop_flag = threading.Event()
    discard_flag = threading.Event()

    def check_keyboard():
        import msvcrt
        while not stop_flag.is_set():
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
                if key == 'q':
                    stop_flag.set()
                elif key == 'd':
                    discard_flag.set()
                    stop_flag.set()
            time.sleep(0.05)

    kb_thread = threading.Thread(target=check_keyboard, daemon=True)
    kb_thread.start()

    step = 0
    task_complete = False
    task_complete_frames = 0
    consecutive_errors = 0
    last_action = None

    while not stop_flag.is_set():
        loop_start = time.time()
        elapsed_total = loop_start - start_time

        # Check timeout
        if elapsed_total > max_duration:
            speak("Timeout")
            break

        # Read leader arm (with error recovery)
        try:
            positions = leader_bus.sync_read("Present_Position")
            action = {f"{motor}.pos": positions[motor] for motor in MOTOR_NAMES}
            last_action = action.copy()
            consecutive_errors = 0
        except ConnectionError:
            consecutive_errors += 1
            if consecutive_errors == 1:
                print("\n[!] Leader arm read failed, using last action...")
            if consecutive_errors > 30:
                print("\n[!] Too many errors, stopping...")
                break
            if last_action:
                action = last_action
            else:
                continue

        # Send to simulation
        sim_robot.send_action(action)

        # Get observation from simulation
        observation = sim_robot.get_observation()

        # Build frames using LeRobot utilities
        obs_frame = build_dataset_frame(dataset.features, observation, prefix="observation")
        action_frame = build_dataset_frame(dataset.features, action, prefix="action")

        # Add frame to dataset
        dataset.add_frame({
            **obs_frame,
            **action_frame,
            "task": task,
        })

        step += 1

        # Check task completion
        if sim_robot.is_task_complete():
            task_complete_frames += 1
            if task_complete_frames >= 10:  # Debounce: 10 frames (~0.3s)
                task_complete = True
                speak("Task complete! Duplo is in the bowl!")
                # Record a few more frames then stop
                for _ in range(15):  # ~0.5s extra
                    time.sleep(frame_time)
                    try:
                        positions = leader_bus.sync_read("Present_Position")
                        action = {f"{motor}.pos": positions[motor] for motor in MOTOR_NAMES}
                    except:
                        pass
                    sim_robot.send_action(action)
                    observation = sim_robot.get_observation()
                    obs_frame = build_dataset_frame(dataset.features, observation, prefix="observation")
                    action_frame = build_dataset_frame(dataset.features, action, prefix="action")
                    dataset.add_frame({**obs_frame, **action_frame, "task": task})
                    step += 1
                break
        else:
            task_complete_frames = 0

        # Progress update
        if step % 30 == 0:
            print(f"  Step {step:4d} | Time: {elapsed_total:5.1f}s", end="\r")

        # Maintain frame rate
        elapsed = time.time() - loop_start
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)

    stop_flag.set()
    print()

    # Check if discarded
    if discard_flag.is_set():
        speak("Episode discarded")
        dataset.clear_episode_buffer()
        return False, False

    print(f"\nRecorded {step} frames ({step/fps:.1f}s)")
    print(f"Task completed: {'YES' if task_complete else 'NO'}")

    # Ask to save if task incomplete
    if not task_complete:
        speak("Task incomplete. Save anyway?")
        save = input("Save? [y/N]: ").strip().lower()
        if save != 'y':
            speak("Episode discarded")
            dataset.clear_episode_buffer()
            return False, False

    # Save episode
    dataset.save_episode()
    speak("Episode saved")
    return True, task_complete


def main():
    parser = argparse.ArgumentParser(description="Record pick-place demos with VR")
    parser.add_argument("--task", "-t", type=str, default="Pick up the Duplo block and place it in the bowl")
    parser.add_argument("--num_episodes", "-n", type=int, default=10)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--repo_id", type=str, default=None)
    parser.add_argument("--root", type=str, default="./datasets")
    parser.add_argument("--leader_port", type=str, default=None)
    parser.add_argument("--max_duration", type=float, default=60.0, help="Max episode duration in seconds")
    parser.add_argument("--no-upload", action="store_true", help="Don't upload to HuggingFace")
    parser.add_argument("--pos_range", type=float, default=2.0, help="Position randomization range in cm (default 2)")
    parser.add_argument("--rot_range", type=float, default=180.0, help="Rotation randomization range in degrees (default 180)")
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
    repo_id = args.repo_id
    if repo_id is None:
        repo_id = f"danbhf/sim_pick_place_{timestamp}"
    print(f"Dataset: {repo_id}")

    # Use timestamped root directory to avoid conflicts
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

    # Create dataset using LeRobot v3.0 API
    print(f"\nCreating dataset: {repo_id}")

    # Get features from robot
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
    # Save updated info.json
    info_path = root_dir / "meta" / "info.json"
    with open(info_path, "w") as f:
        json.dump(dataset.meta.info, f, indent=4)
    print("Dataset created (LeRobot v3.0 format)")
    print(f"Scene: duplo at ({scene_info['objects']['duplo']['position']['x']:.3f}, "
          f"{scene_info['objects']['duplo']['position']['y']:.3f}, "
          f"{scene_info['objects']['duplo']['position']['z']:.3f})")

    print("\n" + "="*60)
    print(f"RECORDING: {args.task}")
    print(f"Episodes: {args.num_episodes}")
    print(f"FPS: {args.fps}")
    if not args.no_randomize:
        print(f"Randomization: ±{args.pos_range}cm position, ±{args.rot_range}° rotation")
    else:
        print("Randomization: disabled")
    print("="*60)
    print("\nControls during recording:")
    print("  'q' - Stop recording (keep episode)")
    print("  'd' - Discard current episode")
    print("\nTask auto-completes when Duplo lands in bowl!")
    print("="*60)

    # Step 1: Put headset on
    speak("Put headset on")
    print("\nPut your VR headset on.")
    speak("Press ENTER when headset is on to start the simulation")
    input("Press ENTER when headset is on...")

    # Step 2: Get into position (with rendering so user can see)
    speak("Position the view and then")
    print("\nSimulation is running - get into position.")
    speak("Press Enter when ready to start recording")
    print("Press ENTER when ready to start recording...")

    import msvcrt
    ready = False
    while not ready:
        # Keep VR rendering
        sim_robot.render_vr()
        time.sleep(0.03)  # ~30fps
        # Check for Enter key (non-blocking)
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b'\r':  # Enter key
                ready = True

    speak("Starting")

    # Record episodes
    successful_episodes = 0
    completed_tasks = 0

    try:
        ep_idx = 0
        while successful_episodes < args.num_episodes:
            # Reset scene for new episode (with optional randomization)
            sim_robot.reset_scene(
                randomize=not args.no_randomize,
                pos_range=args.pos_range / 100.0,  # cm to meters
                rot_range=np.radians(args.rot_range)  # degrees to radians
            )

            # Get and display the starting position
            scene = sim_robot.get_scene_info()
            duplo_pos = scene['objects']['duplo']['position']
            print(f"\nDuplo start: ({duplo_pos['x']:.3f}, {duplo_pos['y']:.3f})")

            saved, task_complete = record_episode(
                sim_robot, leader_bus, dataset, ep_idx,
                task=args.task,
                fps=args.fps,
                max_duration=args.max_duration
            )

            if saved:
                successful_episodes += 1
                if task_complete:
                    completed_tasks += 1
                print(f"\nProgress: {successful_episodes}/{args.num_episodes} episodes")
                print(f"Tasks completed: {completed_tasks}/{successful_episodes}")

            ep_idx += 1

            if successful_episodes < args.num_episodes:
                speak(f"Episode {successful_episodes} of {args.num_episodes} complete. Press Enter for next episode.")
                cont = input("\nPress ENTER for next episode, 'q' to finish: ").strip().lower()
                if cont == 'q':
                    break

    except KeyboardInterrupt:
        print("\n\nRecording interrupted.")

    finally:
        # Summary
        print("\n" + "="*60)
        print("RECORDING COMPLETE")
        print("="*60)
        print(f"Episodes saved: {successful_episodes}")
        print(f"Tasks completed: {completed_tasks}")
        print(f"Dataset: {root_dir / repo_id}")

        # Finalize dataset (writes all metadata files including episodes parquet)
        if successful_episodes > 0:
            print("\nFinalizing dataset...")
            dataset.finalize()
            print("Dataset finalized.")

        # Auto-upload to HuggingFace
        if successful_episodes > 0 and not args.no_upload:
            speak("Uploading to HuggingFace")
            print("\nUploading to HuggingFace Hub...")
            try:
                import subprocess
                import sys
                upload_script = Path(__file__).parent / "upload_dataset.py"
                result = subprocess.run(
                    [sys.executable, str(upload_script), str(root_dir), repo_id],
                    cwd=Path(__file__).parent
                )
                if result.returncode == 0:
                    speak("Upload complete")
                else:
                    speak("Upload failed")
            except Exception as e:
                speak("Upload failed")
                print(f"❌ Upload failed: {e}")
                print(f"Upload manually with:")
                print(f"  python upload_dataset.py {root_dir} {repo_id}")
        elif successful_episodes == 0:
            print("\nNo episodes to upload.")

        # Cleanup
        sim_robot.disconnect()
        leader_bus.disconnect()
        speak("Done")


if __name__ == "__main__":
    main()
