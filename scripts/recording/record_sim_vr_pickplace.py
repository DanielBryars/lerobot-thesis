#!/usr/bin/env python
"""
Record pick-and-place demonstrations using SO100 simulation with VR.

Task: Pick up the Duplo block and place it in the bowl.
Episode automatically ends when task is complete (duplo in bowl).

Saves to LeRobot dataset format v3.0.

This version runs the simulation continuously (never freezes VR view).

Usage:
    python record_sim_vr_pickplace.py --task "Pick up the Duplo block and place it in the bowl" --num_episodes 10

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

# Add editable lerobot install if available (fixes circular import in some venv installs)
_editable_lerobot = Path(__file__).parent.parent.parent.parent / "lerobot" / "src"
if _editable_lerobot.exists():
    sys.path.insert(0, str(_editable_lerobot))
# Add src to path for lerobot_robot_sim
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
# Add recording dir to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

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

# Import leader arm (same class used by teleoperate_so100.py)
from SO100LeaderSTS3250 import SO100LeaderSTS3250, SO100LeaderSTS3250Config

# Import LeRobot dataset utilities
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features, build_dataset_frame


class State(Enum):
    """Recording state machine states."""
    SETUP = "setup"              # Initial setup, waiting for headset
    READY = "ready"              # Between episodes, waiting to start recording
    RECORDING = "recording"      # Actively recording an episode
    FINISHED = "finished"        # All done


def get_git_info() -> dict:
    """Get git repository information."""
    import subprocess

    repo_root = Path(__file__).parent.parent

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
        "remote_url": run_git(["remote", "get-url", "origin"]),
        "is_dirty": run_git(["status", "--porcelain"]) != "",
        "last_commit_message": run_git(["log", "-1", "--pretty=%s"]),
        "last_commit_date": run_git(["log", "-1", "--pretty=%ci"]),
    }


def read_motor_eeprom(bus, motor_name: str) -> dict:
    """Read EEPROM values from a motor."""
    try:
        eeprom = {}
        # Key EEPROM registers for calibration
        registers = [
            "Homing_Offset",
            "Min_Position_Limit",
            "Max_Position_Limit",
            "Max_Torque_Limit",
            "Protection_Current",
            "Operating_Mode",
            "P_Coefficient",
            "I_Coefficient",
            "D_Coefficient",
        ]
        for reg in registers:
            try:
                value = bus.read(reg, motor_name)
                eeprom[reg] = int(value) if value is not None else None
            except:
                eeprom[reg] = None

        # Also read current position
        try:
            pos = bus.read("Present_Position", motor_name)
            eeprom["Present_Position"] = int(pos) if pos is not None else None
        except:
            eeprom["Present_Position"] = None

        return eeprom
    except Exception as e:
        return {"error": str(e)}


def get_leader_metadata(leader: SO100LeaderSTS3250) -> dict:
    """Get comprehensive metadata from leader arm."""
    metadata = {
        "type": leader.name,
        "port": leader.config.port,
        "id": leader.config.id,
        "calibration": {},
        "eeprom": {},
        "motor_models": {},
    }

    # Get calibration data
    if hasattr(leader, 'calibration') and leader.calibration:
        for motor, calib in leader.calibration.items():
            metadata["calibration"][motor] = {
                "id": calib.id,
                "drive_mode": calib.drive_mode,
                "homing_offset": calib.homing_offset,
                "range_min": calib.range_min,
                "range_max": calib.range_max,
            }

    # Get EEPROM values
    if hasattr(leader, 'bus') and leader.bus:
        for motor in MOTOR_NAMES:
            metadata["eeprom"][motor] = read_motor_eeprom(leader.bus, motor)
            # Get motor model info
            if motor in leader.bus.motors:
                m = leader.bus.motors[motor]
                metadata["motor_models"][motor] = {
                    "id": m.id,
                    "model": m.model,
                    "norm_mode": str(m.norm_mode),
                }

    return metadata


def get_recording_metadata(
    leader: SO100LeaderSTS3250,
    sim_robot: SO100Sim,
    args,
    repo_id: str,
    timestamp: str,
) -> dict:
    """Gather all metadata for the recording session."""

    # Read scene XML content
    scene_xml_path = sim_robot.scene_xml
    scene_xml_content = None
    if scene_xml_path.exists():
        try:
            scene_xml_content = scene_xml_path.read_text()
        except:
            pass

    metadata = {
        "recording_info": {
            "timestamp": timestamp,
            "repo_id": repo_id,
            "task": args.task,
            "fps": args.fps,
            "max_duration": args.max_duration,
            "num_episodes_target": args.num_episodes,
            "randomization": {
                "enabled": not args.no_randomize,
                "pos_range_cm": args.pos_range,
                "rot_range_deg": args.rot_range,
            },
            "block_center": {
                "x": args.block_x if args.block_x is not None else 0.217,
                "y": args.block_y if args.block_y is not None else 0.225,
                "custom": args.block_x is not None or args.block_y is not None,
            },
        },
        "git": get_git_info(),
        "leader_arm": get_leader_metadata(leader),
        "simulation": {
            "type": sim_robot.name,
            "scene_xml_path": str(scene_xml_path),
            "scene_xml_content": scene_xml_content,
            "config": {
                "n_sim_steps": sim_robot.config.n_sim_steps,
                "camera_width": sim_robot.config.camera_width,
                "camera_height": sim_robot.config.camera_height,
                "sim_cameras": sim_robot.config.sim_cameras,
                "depth_cameras": sim_robot.config.depth_cameras,
                "use_degrees": sim_robot.config.use_degrees,
            },
            "action_space": {
                "low": [-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.17453],
                "high": [1.91986, 1.74533, 1.69, 1.65806, 2.84121, 1.74533],
                "motor_names": MOTOR_NAMES,
            },
        },
        "environment": {
            "platform": sys.platform,
            "python_version": sys.version,
            "working_directory": str(Path.cwd()),
        },
    }

    return metadata


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
    parser.add_argument("--pos_range", type=float, default=4.0, help="Position randomization range in cm")
    parser.add_argument("--rot_range", type=float, default=180.0, help="Rotation randomization range in degrees")
    parser.add_argument("--no-randomize", action="store_true", help="Disable position/rotation randomization")
    parser.add_argument("--block-x", type=float, default=None, help="Block center X position in meters (default: 0.217)")
    parser.add_argument("--block-y", type=float, default=None, help="Block center Y position in meters (default: 0.225)")
    parser.add_argument("--depth", action="store_true", help="Enable depth rendering for overhead camera")
    parser.add_argument("--lift_stop", type=float, default=None,
                        help="Auto-stop episode when block is lifted by this many cm (e.g. --lift_stop 5 for 5cm). "
                             "For recording pickup-only demonstrations.")
    parser.add_argument("--show-fov", action="store_true",
                        help="Show wrist camera FOV projection on the table (light red overlay in VR)")

    args = parser.parse_args()

    # Get leader config
    config = load_config()
    leader_port = args.leader_port
    if leader_port is None:
        if config and "leader" in config:
            leader_port = config["leader"]["port"]
        else:
            leader_port = "COM8"
    leader_id = config["leader"]["id"] if config and "leader" in config else "leader_so100"
    print(f"Leader port: {leader_port}, ID: {leader_id}")

    # Generate repo ID with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    repo_id = args.repo_id or f"danbhf/sim_pick_place_{timestamp}"
    print(f"Dataset: {repo_id}")

    # Use timestamped root directory
    root_dir = Path(args.root) / timestamp
    print(f"Storage: {root_dir}")

    # Create simulation with VR
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
    if args.depth:
        print("  Depth rendering enabled for overhead camera")
    if args.show_fov and sim_robot.vr_renderer is not None:
        sim_robot.vr_renderer.show_wrist_cam_fov = True
        print("  Wrist camera FOV overlay enabled")
    speak("Simulation ready")

    # Keep VR alive during initialization
    for _ in range(10):
        sim_robot.render_vr()

    # Connect leader arm (using same class as teleoperate_so100.py for consistency)
    print(f"\nConnecting leader arm on {leader_port}...")
    leader_config = SO100LeaderSTS3250Config(port=leader_port, id=leader_id)
    leader = SO100LeaderSTS3250(leader_config)
    leader.connect()
    speak("Leader arm connected")

    # Keep VR alive during initialization
    for _ in range(10):
        sim_robot.render_vr()

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

    # Gather and save comprehensive recording metadata
    print("Gathering recording metadata (calibration, EEPROM, git info)...")
    recording_metadata = get_recording_metadata(leader, sim_robot, args, repo_id, timestamp)
    metadata_path = root_dir / "meta" / "recording_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(recording_metadata, f, indent=2)
    print(f"  Git: {recording_metadata['git']['commit_short']} ({recording_metadata['git']['branch']})")
    print(f"  Leader calibration: {len(recording_metadata['leader_arm']['calibration'])} motors")
    print(f"  Scene XML: {recording_metadata['simulation']['scene_xml_path']}")

    # Keep VR alive during initialization
    for _ in range(10):
        sim_robot.render_vr()

    # Print instructions
    print("\n" + "=" * 60)
    print("CONTINUOUS VR RECORDING")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Episodes: {args.num_episodes}")
    print(f"FPS: {args.fps}")
    if not args.no_randomize:
        print(f"Randomization: ±{args.pos_range}cm position, ±{args.rot_range}° rotation")
    if args.block_x is not None or args.block_y is not None:
        bx = args.block_x if args.block_x is not None else 0.217
        by = args.block_y if args.block_y is not None else 0.225
        print(f"Block center: ({bx:.3f}, {by:.3f})")
    else:
        print(f"Block center: (0.217, 0.225) [default]")
    print("=" * 60)
    print("\nControls:")
    print("  ENTER - Start recording / Save episode")
    print("  D     - Discard current episode")
    print("  R     - Reset scene (when not recording)")
    print("  Q     - Quit")
    print("")
    print("VR Controls:")
    print("  SPACEBAR    - Recenter robot in front of you")
    print("  Thumbsticks - Move/rotate view (if VR controllers working)")
    if args.lift_stop:
        print(f"\nTask auto-completes when block is lifted {args.lift_stop:.0f}cm!")
    else:
        print("\nTask auto-completes when Duplo lands in bowl!")
    print("=" * 60)

    # Keep VR alive before entering main loop
    for _ in range(30):
        sim_robot.render_vr()

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

    # Lift-stop tracking
    import mujoco as _mj
    duplo_body_id = _mj.mj_name2id(sim_robot.mj_model, _mj.mjtObj.mjOBJ_BODY, "duplo")
    initial_duplo_z = None
    lift_threshold = args.lift_stop / 100.0 if args.lift_stop else None  # Convert cm to m

    # Per-episode scene info tracking
    episode_scenes = {}
    current_episode_scene = None

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
                action = leader.get_action()
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

            # Handle VR recenter (spacebar) - works in any state
            if key == b' ':
                print("Spacebar pressed - recentering VR view...")
                sim_robot.recenter_vr()
                key = None  # Consume the key so it doesn't trigger other actions

            # State machine
            if state == State.SETUP:
                # Waiting for user to put on headset and press Enter
                if key == b'\r':
                    state = State.READY
                    # Reset scene for first episode
                    sim_robot.reset_scene(
                        randomize=not args.no_randomize,
                        pos_range=args.pos_range / 100.0,
                        rot_range=np.radians(args.rot_range),
                        pos_center_x=args.block_x,
                        pos_center_y=args.block_y
                    )
                    current_episode_scene = sim_robot.get_scene_info()
                    duplo_pos = current_episode_scene['objects']['duplo']['position']
                    print(f"\nDuplo at: ({duplo_pos['x']:.3f}, {duplo_pos['y']:.3f})")
                    speak(f"Ready. Episode {successful_episodes + 1} of {args.num_episodes}. Press ENTER to record.")
                elif key == b'q':
                    state = State.FINISHED

            elif state == State.READY:
                # Between episodes, waiting to start recording
                if key == b'\r':
                    # Start recording - capture scene info for this episode
                    state = State.RECORDING
                    dataset.create_episode_buffer()
                    episode_start_time = time.time()
                    episode_frames = 0
                    task_complete_frames = 0
                    # Capture initial duplo height for lift-stop
                    initial_duplo_z = sim_robot.mj_data.xpos[duplo_body_id][2]
                    speak("Recording")
                    print(f"\nRecording episode {successful_episodes + 1}...")
                    if lift_threshold:
                        print(f"  Auto-stop: lift block {args.lift_stop:.0f}cm (z > {initial_duplo_z + lift_threshold:.3f}m)")
                elif key == b'r':
                    # Reset scene
                    sim_robot.reset_scene(
                        randomize=not args.no_randomize,
                        pos_range=args.pos_range / 100.0,
                        rot_range=np.radians(args.rot_range),
                        pos_center_x=args.block_x,
                        pos_center_y=args.block_y
                    )
                    current_episode_scene = sim_robot.get_scene_info()
                    duplo_pos = current_episode_scene['objects']['duplo']['position']
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

                # Check task completion (lift-stop or bowl completion)
                duplo_z = sim_robot.mj_data.xpos[duplo_body_id][2]
                if lift_threshold and initial_duplo_z is not None:
                    task_done = (duplo_z - initial_duplo_z) >= lift_threshold
                else:
                    task_done = sim_robot.is_task_complete()

                if task_done:
                    task_complete_frames += 1
                    if task_complete_frames >= 10:  # Debounce
                        speak("Task complete!")
                        completed_tasks += 1
                        # Save episode and scene info
                        episode_scenes[successful_episodes] = current_episode_scene
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
                                rot_range=np.radians(args.rot_range),
                                pos_center_x=args.block_x,
                                pos_center_y=args.block_y
                            )
                            current_episode_scene = sim_robot.get_scene_info()
                            duplo_pos = current_episode_scene['objects']['duplo']['position']
                            print(f"Duplo at: ({duplo_pos['x']:.3f}, {duplo_pos['y']:.3f})")
                            speak(f"Episode {successful_episodes + 1} of {args.num_episodes}. Press ENTER to record.")
                else:
                    task_complete_frames = 0

                # Check timeout
                if elapsed > args.max_duration:
                    speak("Timeout")
                    print(f"\n\nTimeout after {elapsed:.1f}s")
                    # Save episode and scene info
                    episode_scenes[successful_episodes] = current_episode_scene
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
                            rot_range=np.radians(args.rot_range),
                            pos_center_x=args.block_x,
                            pos_center_y=args.block_y
                        )
                        current_episode_scene = sim_robot.get_scene_info()
                        speak(f"Episode {successful_episodes + 1}. Press ENTER to record.")

                # Handle keys during recording
                if key == b'\r':
                    # Manual stop - save episode and scene info
                    episode_scenes[successful_episodes] = current_episode_scene
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
                            rot_range=np.radians(args.rot_range),
                            pos_center_x=args.block_x,
                            pos_center_y=args.block_y
                        )
                        current_episode_scene = sim_robot.get_scene_info()
                        duplo_pos = current_episode_scene['objects']['duplo']['position']
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
                        rot_range=np.radians(args.rot_range),
                        pos_center_x=args.block_x,
                        pos_center_y=args.block_y
                    )
                    current_episode_scene = sim_robot.get_scene_info()
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

            # Save per-episode scene info
            episode_scenes_path = root_dir / "meta" / "episode_scenes.json"
            with open(episode_scenes_path, "w") as f:
                json.dump(episode_scenes, f, indent=2)
            print(f"Saved scene info for {len(episode_scenes)} episodes")
            print("Dataset finalized.")
            print("\nMetadata saved:")
            print(f"  - recording_metadata.json (calibration, EEPROM, git, scene XML)")
            print(f"  - episode_scenes.json (per-episode object positions)")
            print(f"  - info.json (LeRobot dataset info)")

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
        leader.disconnect()
        speak("Done")


if __name__ == "__main__":
    main()
