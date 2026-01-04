#!/usr/bin/env python
"""
Play back end-effector action space recordings in simulation.

Loads a LeRobot dataset with EE actions (xyz + quat + gripper) and replays
them through IK to control the simulation.

Usage:
    python playback_ee_sim.py danbhf/sim_pick_place_merged_40ep_ee
    python playback_ee_sim.py danbhf/sim_pick_place_merged_40ep_ee --episode 0
    python playback_ee_sim.py danbhf/sim_pick_place_merged_40ep_ee --no-vr
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

try:
    import msvcrt
    _msvcrt_available = True
except ImportError:
    _msvcrt_available = False

# Add project paths
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "scripts"))

# Import shared utilities
from utils.constants import SIM_ACTION_LOW, SIM_ACTION_HIGH
from utils.conversions import (
    normalized_to_radians,
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
)

# Import FK/IK
from test_fk_ik import MuJoCoFK, MuJoCoIK

# Import LeRobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def check_key():
    """Non-blocking keyboard check."""
    if not _msvcrt_available:
        return None
    if msvcrt.kbhit():
        key = msvcrt.getch()
        if key == b'\xe0':
            msvcrt.getch()
            return None
        return key
    return None


def play_episode(
    mj_model,
    mj_data,
    viewer,
    fk: MuJoCoFK,
    ik: MuJoCoIK,
    dataset: LeRobotDataset,
    episode_idx: int,
    use_vr: bool = False,
    use_joints: bool = False,
):
    """Play back a single episode using IK to convert EE actions to joint angles.

    If use_joints=True, uses original joint actions directly (from action_joints field)
    instead of IK, for comparison purposes.
    """

    # Get episode data bounds
    ep_meta = dataset.meta.episodes[episode_idx]
    from_idx = ep_meta['dataset_from_index']
    to_idx = ep_meta['dataset_to_index']
    num_frames = to_idx - from_idx

    fps = dataset.fps
    frame_time = 1.0 / fps
    n_sim_steps = 10

    mode_str = "DIRECT JOINTS" if use_joints else "EE → IK"
    print(f"\nPlaying episode {episode_idx} ({num_frames} frames at {fps} FPS) [{mode_str}]")
    print("Press Q to stop, SPACE to pause/resume")

    paused = False
    frame_idx = 0

    # Track IK stats
    ik_successes = 0
    ik_attempts = 0

    # Keep track of last joint angles for IK initial guess
    last_joints = np.zeros(5)

    # Debug: print first few frames
    debug_frames = 5

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
            frame_idx = 0
            last_joints = np.zeros(5)
            print("\nRestarting episode...")
            continue

        # Render
        viewer.sync()

        if not paused:
            # Get action for this frame
            data_idx = from_idx + frame_idx
            frame_data = dataset[data_idx]

            if use_joints:
                # Use original joint actions directly (for comparison)
                if "action_joints" in frame_data:
                    joint_action = frame_data["action_joints"]
                else:
                    # Fall back to main action if no action_joints field
                    joint_action = frame_data["action"]

                if hasattr(joint_action, 'numpy'):
                    joint_action = joint_action.numpy()

                # Convert from normalized to radians
                sim_joints = normalized_to_radians(joint_action)

                # Debug: print first few frames
                if frame_idx < debug_frames:
                    print(f"\nFrame {frame_idx} [DIRECT JOINTS]:")
                    print(f"  Normalized: [{joint_action[0]:.1f}, {joint_action[1]:.1f}, {joint_action[2]:.1f}, {joint_action[3]:.1f}, {joint_action[4]:.1f}, {joint_action[5]:.1f}]")
                    print(f"  Radians:    [{sim_joints[0]:.3f}, {sim_joints[1]:.3f}, {sim_joints[2]:.3f}, {sim_joints[3]:.3f}, {sim_joints[4]:.3f}, {sim_joints[5]:.3f}]")
                    # Also compute FK to show the EE pose
                    fk_pos, fk_rot = fk.forward(sim_joints[:5])
                    fk_quat = rotation_matrix_to_quaternion(fk_rot)
                    print(f"  FK pos:     [{fk_pos[0]:.4f}, {fk_pos[1]:.4f}, {fk_pos[2]:.4f}]")
                    print(f"  FK quat:    [{fk_quat[0]:.3f}, {fk_quat[1]:.3f}, {fk_quat[2]:.3f}, {fk_quat[3]:.3f}]")

            else:
                # Get EE action: [x, y, z, qw, qx, qy, qz, gripper]
                action = frame_data["action"]
                if hasattr(action, 'numpy'):
                    action = action.numpy()

                # Extract EE pose and gripper
                ee_pos = action[:3]
                ee_quat = action[3:7]  # [qw, qx, qy, qz]
                gripper = action[7] if len(action) > 7 else 0.0

                # Convert quaternion to rotation matrix
                ee_rot = quaternion_to_rotation_matrix(ee_quat)

                # IK: EE pose -> joint angles
                ik_attempts += 1
                ik_joints, ik_success, ik_error = ik.solve(
                    target_pos=ee_pos,
                    target_rot=ee_rot,
                    initial_angles=last_joints,
                    max_iterations=100,
                    pos_tolerance=1e-3,
                )

                if ik_success:
                    ik_successes += 1
                    last_joints = ik_joints.copy()
                else:
                    # Use last known good joints if IK fails
                    ik_joints = last_joints

                # Debug: print first few frames to check for inversion
                if frame_idx < debug_frames:
                    # Check what position IK actually achieves
                    actual_pos, actual_rot = fk.forward(ik_joints)
                    print(f"\nFrame {frame_idx} [EE → IK]:")
                    print(f"  Target pos: [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]")
                    print(f"  Actual pos: [{actual_pos[0]:.4f}, {actual_pos[1]:.4f}, {actual_pos[2]:.4f}]")
                    print(f"  Pos error:  [{actual_pos[0]-ee_pos[0]:.4f}, {actual_pos[1]-ee_pos[1]:.4f}, {actual_pos[2]-ee_pos[2]:.4f}]")
                    print(f"  Target quat: [{ee_quat[0]:.3f}, {ee_quat[1]:.3f}, {ee_quat[2]:.3f}, {ee_quat[3]:.3f}]")
                    # Convert actual rotation to quaternion for comparison
                    actual_quat = rotation_matrix_to_quaternion(actual_rot)
                    print(f"  Actual quat: [{actual_quat[0]:.3f}, {actual_quat[1]:.3f}, {actual_quat[2]:.3f}, {actual_quat[3]:.3f}]")
                    print(f"  IK joints:  [{ik_joints[0]:.3f}, {ik_joints[1]:.3f}, {ik_joints[2]:.3f}, {ik_joints[3]:.3f}, {ik_joints[4]:.3f}]")

                # Combine with gripper
                sim_joints = np.concatenate([ik_joints, [gripper]])
            sim_joints = np.clip(sim_joints, SIM_ACTION_LOW, SIM_ACTION_HIGH)

            # Apply to simulation
            mj_data.ctrl[:] = sim_joints
            for _ in range(n_sim_steps):
                mujoco.mj_step(mj_model, mj_data)

            frame_idx += 1

            # Progress
            if frame_idx % 30 == 0:
                elapsed = frame_idx / fps
                ik_rate = 100.0 * ik_successes / ik_attempts if ik_attempts > 0 else 0
                print(f"  Frame {frame_idx}/{num_frames} ({elapsed:.1f}s) IK: {ik_rate:.0f}%", end="\r")

        # Maintain frame rate
        elapsed = time.time() - loop_start
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)

    ik_rate = 100.0 * ik_successes / ik_attempts if ik_attempts > 0 else 0
    print(f"\nEpisode {episode_idx} complete (IK success: {ik_rate:.1f}%)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Play back EE action episodes in simulation")
    parser.add_argument("dataset", type=str, help="HuggingFace repo ID or local path")
    parser.add_argument("--episode", "-e", type=int, default=None, help="Episode index (default: all)")
    parser.add_argument("--no-vr", action="store_true", help="Render to screen instead of VR")
    parser.add_argument("--loop", action="store_true", help="Loop playback continuously")
    parser.add_argument("--local", action="store_true", help="Load from local datasets/ folder")
    parser.add_argument("--use-joints", action="store_true",
                        help="Use original joint actions directly (skip IK) for comparison")

    args = parser.parse_args()
    use_vr = not args.no_vr

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    if args.local:
        local_path = repo_root / "datasets" / args.dataset
        if not local_path.exists():
            print(f"ERROR: Local dataset not found: {local_path}")
            return
        print(f"Using local path: {local_path}")
        dataset = LeRobotDataset(repo_id=args.dataset, root=local_path)
    else:
        dataset = LeRobotDataset(repo_id=args.dataset)

    num_episodes = dataset.num_episodes
    fps = dataset.fps

    # Verify dataset format
    sample = dataset[0]
    action_shape = sample["action"].shape
    has_action_joints = "action_joints" in sample

    if action_shape[0] != 8:
        print(f"WARNING: Expected 8 EE actions, got {action_shape[0]}")
        print("This script is for end-effector action datasets.")
        print("For joint-space datasets, use playback_sim_vr.py instead.")
        return

    print(f"Loaded {num_episodes} episodes at {fps} FPS")
    print(f"Action space: end-effector (8 dims)")
    if has_action_joints:
        print(f"Original joint actions: available (action_joints field)")
    if args.use_joints:
        if has_action_joints:
            print(f"MODE: Using original joint actions directly (bypassing IK)")
        else:
            print(f"WARNING: --use-joints specified but action_joints field not found!")
            print(f"         Will fall back to main action field.")

    # Show episode info
    print("\nEpisodes:")
    for i in range(min(num_episodes, 10)):
        ep_meta = dataset.meta.episodes[i]
        from_idx = ep_meta['dataset_from_index']
        to_idx = ep_meta['dataset_to_index']
        num_frames = to_idx - from_idx
        duration = num_frames / fps
        print(f"  [{i}] {num_frames} frames ({duration:.1f}s)")
    if num_episodes > 10:
        print(f"  ... and {num_episodes - 10} more episodes")

    # Initialize FK/IK
    print("\nInitializing FK/IK...")
    scene_xml = str(repo_root / "scenes" / "so101_with_wrist_cam.xml")
    fk = MuJoCoFK(scene_xml)
    ik = MuJoCoIK(fk)

    # Create separate model/data for simulation
    mj_model = mujoco.MjModel.from_xml_path(scene_xml)
    mj_data = mujoco.MjData(mj_model)

    # Initialize simulation
    for _ in range(100):
        mujoco.mj_step(mj_model, mj_data)

    # Get duplo body id for positioning
    duplo_body_id = None
    try:
        duplo_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "duplo")
        print(f"Found duplo body (id={duplo_body_id})")
    except:
        print("WARNING: Could not find duplo body in scene")

    # Load per-episode scene data if available
    episode_scenes = {}
    episode_scenes_path = dataset.root / "meta" / "episode_scenes.json"
    if episode_scenes_path.exists():
        with open(episode_scenes_path) as f:
            episode_scenes = json.load(f)
        print(f"Loaded per-episode scene data ({len(episode_scenes)} episodes)")

    # Fall back to dataset-level scene info
    scene_info = dataset.meta.info.get('scene', {})
    if scene_info:
        print(f"Dataset-level scene info available as fallback")

    # Launch viewer
    print("Launching viewer...")
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

    print("\n" + "=" * 50)
    print("EE PLAYBACK (with IK)")
    print("=" * 50)
    print("Controls:")
    print("  SPACE - Pause/Resume")
    print("  R     - Replay current episode")
    print("  Q     - Quit")
    print("=" * 50)

    print("\nPress ENTER to start playback...")
    while True:
        viewer.sync()
        key = check_key()
        if key == b'\r' or key == b' ':
            break
        elif key == b'q':
            viewer.close()
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

            # Reset simulation
            mujoco.mj_resetData(mj_model, mj_data)

            # Reset duplo position from per-episode scene data (or fallback to dataset-level)
            if duplo_body_id is not None:
                # Try per-episode scene first
                ep_scene = episode_scenes.get(str(ep_idx), {})
                if not ep_scene:
                    ep_scene = scene_info  # Fallback to dataset-level

                duplo_info = ep_scene.get('objects', {}).get('duplo', {})
                if 'position' in duplo_info:
                    duplo_pos = duplo_info['position']
                    joint_id = mj_model.body_jntadr[duplo_body_id]
                    if joint_id >= 0:
                        qpos_adr = mj_model.jnt_qposadr[joint_id]
                        mj_data.qpos[qpos_adr:qpos_adr+3] = [duplo_pos['x'], duplo_pos['y'], duplo_pos.get('z', 0.01)]
                        if 'quaternion' in duplo_info:
                            q = duplo_info['quaternion']
                            mj_data.qpos[qpos_adr+3:qpos_adr+7] = [q['w'], q['x'], q['y'], q['z']]
                        print(f"Duplo at ({duplo_pos['x']:.3f}, {duplo_pos['y']:.3f}) for episode {ep_idx}")

            for _ in range(100):
                mujoco.mj_step(mj_model, mj_data)

            # Play episode
            completed = play_episode(
                mj_model, mj_data, viewer,
                fk, ik, dataset, ep_idx,
                use_vr=use_vr,
                use_joints=args.use_joints
            )

            if not completed:
                break

            episode_cursor += 1

            # Check for next action
            if episode_cursor < len(episodes):
                print(f"\nPress ENTER for next episode, R to replay, Q to quit...")
                waiting = True
                while waiting and viewer.is_running():
                    viewer.sync()
                    key = check_key()
                    if key == b'\r' or key == b' ' or key == b'n':
                        waiting = False
                    elif key == b'r':
                        episode_cursor -= 1
                        waiting = False
                    elif key == b'q':
                        episode_cursor = len(episodes)
                        waiting = False
                    time.sleep(0.03)
            elif args.loop:
                episode_cursor = 0
                print("\nLooping...")

        print("\nPlayback complete!")

    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        viewer.close()
        print("Done")


if __name__ == "__main__":
    main()
