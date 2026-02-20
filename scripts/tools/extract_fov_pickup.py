#!/usr/bin/env python3
"""Extract pickup episodes using wrist camera FOV and lift detection.

Replays each episode through MuJoCo to detect:
  - Start: first frame where the duplo block is visible in the wrist camera FOV
  - End: first frame where the block is lifted ≥ lift_height above its initial z

This produces pickup segments that begin exactly when the block enters the
gripper camera's view — more natural than hand-annotated boundaries.

Usage:
    python scripts/tools/extract_fov_pickup.py danbhf/sim_pick_place_2pos_220ep_v2 \
        -o datasets/sim_pick_place_220ep_fov_pickup

    # Quick test:
    python scripts/tools/extract_fov_pickup.py danbhf/sim_pick_place_2pos_220ep_v2 \
        -o datasets/test_fov_pickup --max-episodes 5
"""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch
from tqdm import tqdm

# Add project root
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_robot_sim import SO100Sim, SO100SimConfig, MOTOR_NAMES

# Default features auto-added by LeRobotDataset.create()
DEFAULT_FEATURE_KEYS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def _find_meta_file(filename: str, source_dataset_id: str, local_root: Path = None) -> Path | None:
    """Search for a metadata file in local paths and HuggingFace."""
    if local_root:
        path = local_root / "meta" / filename
        if path.exists():
            return path

    dataset_name = source_dataset_id.split("/")[-1] if "/" in source_dataset_id else source_dataset_id
    local_datasets_path = REPO_ROOT / "datasets" / dataset_name / "meta" / filename
    if local_datasets_path.exists():
        return local_datasets_path

    try:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(source_dataset_id, f"meta/{filename}", repo_type="dataset"))
    except Exception:
        return None


def load_episode_scenes(source_dataset_id: str, local_root: Path = None) -> dict:
    """Load episode_scenes.json from local path or HuggingFace."""
    path = _find_meta_file("episode_scenes.json", source_dataset_id, local_root)
    if path:
        with open(path) as f:
            return json.load(f)
    print("WARNING: Could not find episode_scenes.json")
    return {}


def is_in_fov(cam_pos, cam_mat, block_pos, model, cam_id, max_distance=0.15):
    """Check if block_pos is inside the wrist camera's FOV cone and close enough.

    Args:
        max_distance: Maximum distance from camera to block (meters). Prevents
            triggering when the block is technically in the cone but far away.
    """
    fovy_rad = np.radians(model.cam_fovy[cam_id])
    aspect = 640.0 / 480.0
    half_fovy = fovy_rad / 2
    half_fovx = np.arctan(np.tan(half_fovy) * aspect)

    # Transform block to camera frame
    # cam_mat columns: X=right, Y=up, -Z=forward
    R = cam_mat.reshape(3, 3)
    delta = block_pos - cam_pos
    local = R.T @ delta  # [right, up, forward_neg]

    # Block must be in front of camera (negative Z in cam frame = forward)
    if local[2] >= 0:
        return False

    # Check distance
    depth = -local[2]
    dist = np.linalg.norm(delta)
    if dist > max_distance:
        return False

    # Check angular bounds
    return (abs(local[0]) / depth < np.tan(half_fovx) and
            abs(local[1]) / depth < np.tan(half_fovy))


def get_user_features(source_dataset) -> dict:
    """Extract user-defined features from source dataset, excluding defaults."""
    return {
        key: value
        for key, value in source_dataset.features.items()
        if key not in DEFAULT_FEATURE_KEYS
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract pickup episodes using wrist camera FOV + lift detection"
    )
    parser.add_argument(
        "source_dataset",
        type=str,
        help="Source dataset (HuggingFace repo ID or local path)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output directory for new dataset",
    )
    parser.add_argument(
        "--lift-height",
        type=float,
        default=0.03,
        help="Block must be lifted this many meters above initial z (default: 0.03)",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=0.15,
        help="Max distance (m) from wrist cam to block for FOV trigger (default: 0.15)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Process only first N source episodes (for testing)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Repo ID for new dataset metadata (default: derived from output dir name)",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Scene XML path (default: use SO100Sim default)",
    )
    args = parser.parse_args()

    # Load source dataset
    print(f"Loading source dataset: {args.source_dataset}")
    source_dataset = LeRobotDataset(args.source_dataset)
    print(f"  Episodes: {source_dataset.meta.total_episodes}")
    print(f"  Frames: {len(source_dataset)}")
    print(f"  FPS: {source_dataset.fps}")

    # Determine local root for metadata files
    local_root = Path(source_dataset.root) if hasattr(source_dataset, 'root') else None

    # Check for action field
    sample = source_dataset[0]
    if 'action_joints' in sample:
        action_key = 'action_joints'
    elif 'action' in sample:
        action_key = 'action'
    else:
        print("ERROR: Source dataset must have 'action_joints' or 'action' field")
        sys.exit(1)
    print(f"  Action field: {action_key}")

    # Load episode scenes
    print("Loading episode scenes...")
    episode_scenes = load_episode_scenes(args.source_dataset, local_root)
    if not episode_scenes:
        print("ERROR: episode_scenes.json is required for block positions")
        sys.exit(1)
    print(f"  Loaded scenes for {len(episode_scenes)} episodes")

    # Setup simulation (only for replay — no rendering needed)
    print("\nInitializing simulation...")
    scene_xml = args.scene
    if scene_xml:
        scene_xml = str(REPO_ROOT / scene_xml) if not Path(scene_xml).is_absolute() else scene_xml

    sim_config = SO100SimConfig(
        id="fov_extractor",
        sim_cameras=["wrist_cam"],
        depth_cameras=[],
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

    # Get wrist camera ID
    wrist_cam_id = mujoco.mj_name2id(sim_robot.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    if wrist_cam_id < 0:
        print("ERROR: 'wrist_cam' not found in scene")
        sys.exit(1)
    print(f"  Wrist camera ID: {wrist_cam_id}")

    # Determine episodes to process
    total_episodes = source_dataset.meta.total_episodes
    if args.max_episodes is not None:
        total_episodes = min(total_episodes, args.max_episodes)
        print(f"  Limited to first {total_episodes} episodes")

    # Phase 1: Replay all episodes to find FOV start + lift end frames
    print(f"\n{'='*60}")
    print("PHASE 1: Replay episodes to detect FOV entry and lift")
    print(f"{'='*60}")

    segments = []  # (ep_idx, start_frame, end_frame)

    for ep_idx in range(total_episodes):
        ep_meta = source_dataset.meta.episodes[ep_idx]
        from_idx = ep_meta['dataset_from_index']
        to_idx = ep_meta['dataset_to_index']
        num_frames = to_idx - from_idx

        # Reset simulation
        sim_robot.reset_scene(randomize=False)

        # Restore duplo position from episode_scenes
        ep_key = str(ep_idx)
        initial_block_z = None
        if ep_key in episode_scenes:
            scene_info = episode_scenes[ep_key]
            if 'objects' in scene_info and 'duplo' in scene_info['objects']:
                duplo_info = scene_info['objects']['duplo']
                pos = duplo_info['position']
                quat = duplo_info['quaternion']

                sim_robot.mj_data.qpos[0] = pos['x']
                sim_robot.mj_data.qpos[1] = pos['y']
                sim_robot.mj_data.qpos[2] = pos['z']
                sim_robot.mj_data.qpos[3] = quat['w']
                sim_robot.mj_data.qpos[4] = quat['x']
                sim_robot.mj_data.qpos[5] = quat['y']
                sim_robot.mj_data.qpos[6] = quat['z']

                mujoco.mj_forward(sim_robot.mj_model, sim_robot.mj_data)
                initial_block_z = pos['z']
        else:
            print(f"  Episode {ep_idx}: no scene info, skipping")
            continue

        if initial_block_z is None:
            print(f"  Episode {ep_idx}: no block position, skipping")
            continue

        # Replay frame-by-frame
        start_frame = None
        end_frame = None

        for frame_offset in range(num_frames):
            frame_idx = from_idx + frame_offset
            source_frame = source_dataset[frame_idx]

            # Get joint action
            joint_action = source_frame[action_key].numpy()
            if joint_action.ndim > 1:
                joint_action = joint_action[0]

            # Send action to sim
            action_dict = {f"{MOTOR_NAMES[i]}.pos": float(joint_action[i]) for i in range(6)}
            sim_robot.send_action(action_dict)

            # Read block position
            block_pos = sim_robot.mj_data.qpos[0:3].copy()

            # Check FOV
            if start_frame is None:
                cam_pos = sim_robot.mj_data.cam_xpos[wrist_cam_id].copy()
                cam_mat = sim_robot.mj_data.cam_xmat[wrist_cam_id].copy()
                if is_in_fov(cam_pos, cam_mat, block_pos, sim_robot.mj_model, wrist_cam_id, args.max_distance):
                    start_frame = frame_offset

            # Check lift (only after FOV entry)
            if start_frame is not None and end_frame is None:
                if block_pos[2] >= initial_block_z + args.lift_height:
                    end_frame = frame_offset
                    break  # No need to continue replay

        # Report
        if start_frame is not None and end_frame is not None:
            length = end_frame - start_frame
            print(f"  Episode {ep_idx:3d}: FOV entry frame {start_frame:4d}, "
                  f"lift frame {end_frame:4d}, length {length:3d} frames, "
                  f"block z0={initial_block_z:.3f}")
            segments.append({
                "source_episode": ep_idx,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "length": length,
                "global_start_idx": from_idx + start_frame,
                "initial_block_z": initial_block_z,
            })
        else:
            reason = "no FOV entry" if start_frame is None else "no lift detected"
            print(f"  Episode {ep_idx:3d}: SKIPPED ({reason})")

    print(f"\nFound {len(segments)} valid pickup segments out of {total_episodes} episodes")

    if not segments:
        print("ERROR: No valid segments found!")
        sim_robot.disconnect()
        sys.exit(1)

    # Print summary
    lengths = [s["length"] for s in segments]
    print(f"  Length: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")

    # Phase 2: Create output dataset by copying source frames
    print(f"\n{'='*60}")
    print("PHASE 2: Create output dataset")
    print(f"{'='*60}")

    output_dir = Path(args.output)
    repo_id = args.repo_id or f"danbhf/{output_dir.name}"

    features = get_user_features(source_dataset)
    print(f"Creating output dataset: {repo_id}")
    print(f"  Output dir: {output_dir}")
    print(f"  Features: {list(features.keys())}")

    output_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=source_dataset.fps,
        root=str(output_dir),
        robot_type="so100_sim",
        features=features,
        image_writer_threads=4,
    )

    new_episode_scenes = {}

    for seg_idx, segment in enumerate(tqdm(segments, desc="Copying segments")):
        src_ep = segment["source_episode"]
        global_start = segment["global_start_idx"]
        length = segment["length"]

        output_dataset.create_episode_buffer()

        for frame_offset in range(length):
            global_idx = global_start + frame_offset
            source_frame = source_dataset[global_idx]

            output_frame = {"task": "Pick up the block"}

            for key in features:
                if key not in source_frame:
                    continue
                value = source_frame[key]

                # Convert images: CHW float [0,1] tensor -> HWC uint8 numpy
                if isinstance(value, torch.Tensor) and value.dim() == 3 and "images" in key:
                    img_np = (value.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
                    output_frame[key] = img_np
                elif isinstance(value, torch.Tensor):
                    output_frame[key] = value.numpy()
                else:
                    output_frame[key] = value

            output_dataset.add_frame(output_frame)

        output_dataset.save_episode()

        # Save scene info for this episode
        ep_key = str(src_ep)
        if ep_key in episode_scenes:
            scene_info = dict(episode_scenes[ep_key])
            scene_info["_source_episode"] = src_ep
            scene_info["_fov_start_frame"] = segment["start_frame"]
            scene_info["_fov_end_frame"] = segment["end_frame"]
            new_episode_scenes[str(seg_idx)] = scene_info

    # Finalize
    print("\nFinalizing dataset...")
    output_dataset.finalize()

    # Save episode scenes metadata
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "episode_scenes.json", "w") as f:
        json.dump(new_episode_scenes, f, indent=2)
    print(f"Saved episode_scenes.json ({len(new_episode_scenes)} episodes)")

    # Final summary
    total_frames = sum(s["length"] for s in segments)
    print(f"\n{'='*60}")
    print("FOV PICKUP EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Source: {args.source_dataset}")
    print(f"Output: {output_dir}")
    print(f"Episodes: {len(segments)} (from {total_episodes} source episodes)")
    print(f"Total frames: {total_frames}")
    print(f"Lift height threshold: {args.lift_height}m")
    print(f"Segment lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
    print(f"{'='*60}")

    # Cleanup
    sim_robot.disconnect()


if __name__ == "__main__":
    main()
