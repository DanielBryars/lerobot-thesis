#!/usr/bin/env python3
"""Annotate dataset episodes with subtask labels.

Uses a forward-only state machine based on end-effector distance to block/bowl:

    MOVE_TO_SOURCE (0) -> PICK_UP (1) -> MOVE_TO_DEST (2) -> DROP (3)

Transitions:
    - MOVE_TO_SOURCE -> PICK_UP: when EE near block
    - PICK_UP -> MOVE_TO_DEST: when EE far from block (lifted)
    - MOVE_TO_DEST -> DROP: when EE near bowl

Usage:
    python scripts/tools/annotate_subtasks.py --dataset datasets/sim_pick_place_157ep
    python scripts/tools/annotate_subtasks.py --dataset datasets/sim_pick_place_157ep --visualize
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from utils.conversions import normalized_to_radians


# Subtask labels
MOVE_TO_SOURCE = 0
PICK_UP = 1
MOVE_TO_DEST = 2
DROP = 3

SUBTASK_NAMES = ["MOVE_TO_SOURCE", "PICK_UP", "MOVE_TO_DEST", "DROP"]

# Distance thresholds (meters)
NEAR_THRESHOLD = 0.06   # 6cm - close enough to interact
FAR_THRESHOLD = 0.12    # 12cm - lifted away from surface


class FKSolver:
    """Forward kinematics solver for end-effector position."""

    def __init__(self, scene_xml: str):
        """Initialize with MuJoCo scene.

        Args:
            scene_xml: Path to MuJoCo XML scene file
        """
        self.mj_model = mujoco.MjModel.from_xml_path(scene_xml)
        self.mj_data = mujoco.MjData(self.mj_model)

        # Find end-effector site
        for name in ["gripperframe", "gripper_site", "ee_site", "end_effector"]:
            self.ee_site_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_SITE, name
            )
            if self.ee_site_id != -1:
                break

        if self.ee_site_id == -1:
            raise ValueError("Could not find end-effector site in scene")

        # Find arm joint start position
        shoulder_joint_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_pan"
        )
        if shoulder_joint_id != -1:
            self.arm_joint_start = self.mj_model.jnt_qposadr[shoulder_joint_id]
        else:
            # Fallback: count free joints
            num_free = sum(
                1 for i in range(self.mj_model.njnt)
                if self.mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
            )
            self.arm_joint_start = num_free * 7

    def compute_ee_position(self, joint_radians: np.ndarray) -> np.ndarray:
        """Compute end-effector position from joint angles.

        Args:
            joint_radians: 6-dim array of joint angles in radians

        Returns:
            3-dim array [x, y, z] of EE position
        """
        self.mj_data.qpos[self.arm_joint_start:self.arm_joint_start + 6] = joint_radians
        mujoco.mj_forward(self.mj_model, self.mj_data)
        return self.mj_data.site_xpos[self.ee_site_id].copy()


def load_episode_scenes(dataset_path: Path) -> dict:
    """Load episode scene metadata."""
    scenes_file = dataset_path / "meta" / "episode_scenes.json"
    if not scenes_file.exists():
        raise FileNotFoundError(f"episode_scenes.json not found at {scenes_file}")

    with open(scenes_file) as f:
        return json.load(f)


def annotate_episode(
    actions: np.ndarray,
    block_pos: np.ndarray,
    bowl_pos: np.ndarray,
    fk_solver: FKSolver,
    near_threshold: float = NEAR_THRESHOLD,
    far_threshold: float = FAR_THRESHOLD,
) -> np.ndarray:
    """Annotate a single episode with subtask labels.

    Args:
        actions: (N, 6) array of actions in normalized [-100,100] format
        block_pos: (3,) array of block [x, y, z]
        bowl_pos: (3,) array of bowl [x, y, z]
        fk_solver: FK solver instance
        near_threshold: Distance to consider "near" target
        far_threshold: Distance to consider "far" from block (lifted)

    Returns:
        (N,) array of subtask labels (0-3)
    """
    n_frames = len(actions)
    subtasks = np.zeros(n_frames, dtype=np.int32)

    current_state = MOVE_TO_SOURCE

    for i, action in enumerate(actions):
        # Convert normalized action to radians for FK
        joint_radians = normalized_to_radians(action)

        # Compute EE position
        ee_pos = fk_solver.compute_ee_position(joint_radians)

        # Compute distances (XY only for block approach, full 3D for lift detection)
        dist_to_block_xy = np.linalg.norm(ee_pos[:2] - block_pos[:2])
        dist_to_block_3d = np.linalg.norm(ee_pos - block_pos)
        dist_to_bowl_xy = np.linalg.norm(ee_pos[:2] - bowl_pos[:2])

        # State machine transitions (forward only)
        if current_state == MOVE_TO_SOURCE:
            if dist_to_block_xy < near_threshold:
                current_state = PICK_UP

        elif current_state == PICK_UP:
            # Use 3D distance to detect lift (Z increases when lifted)
            if dist_to_block_3d > far_threshold:
                current_state = MOVE_TO_DEST

        elif current_state == MOVE_TO_DEST:
            if dist_to_bowl_xy < near_threshold:
                current_state = DROP

        # DROP is terminal - stays until end

        subtasks[i] = current_state

    return subtasks


def annotate_dataset(
    dataset_path: Path,
    visualize: bool = False,
    near_threshold: float = NEAR_THRESHOLD,
    far_threshold: float = FAR_THRESHOLD,
) -> dict:
    """Annotate all episodes in a dataset.

    Args:
        dataset_path: Path to dataset directory
        visualize: If True, print per-episode stats
        near_threshold: Distance to consider "near" target
        far_threshold: Distance to consider "far" from block (lifted)

    Returns:
        Dict mapping episode_index -> subtask array
    """
    # Load episode scenes
    episode_scenes = load_episode_scenes(dataset_path)

    # Load parquet data
    data_dir = dataset_path / "data"
    parquet_files = list(data_dir.glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    # Read all data
    tables = [pq.read_table(f) for f in parquet_files]
    table = pa.concat_tables(tables)

    # Get scene XML from first episode
    first_scene = episode_scenes["0"]["scene_xml"]
    # Handle potential path differences
    scene_path = Path(first_scene)
    if not scene_path.exists():
        # Try relative path
        scene_path = Path("scenes") / scene_path.name

    print(f"Using scene: {scene_path}")
    fk_solver = FKSolver(str(scene_path))

    # Group by episode
    episode_indices = table["episode_index"].to_numpy()
    # Use actions (normalized [-100,100]) not observation.state
    # Actions represent commanded joint targets which give correct FK positions
    actions = np.array([a.as_py() for a in table["action"]])

    unique_episodes = np.unique(episode_indices)
    print(f"Found {len(unique_episodes)} episodes")

    annotations = {}
    subtask_counts = {name: 0 for name in SUBTASK_NAMES}

    for ep_idx in unique_episodes:
        ep_key = str(ep_idx)
        if ep_key not in episode_scenes:
            print(f"Warning: Episode {ep_idx} not in episode_scenes.json, skipping")
            continue

        # Get episode data
        mask = episode_indices == ep_idx
        ep_actions = actions[mask]

        # Get block and bowl positions
        scene_info = episode_scenes[ep_key]
        block_pos = np.array([
            scene_info["objects"]["duplo"]["position"]["x"],
            scene_info["objects"]["duplo"]["position"]["y"],
            scene_info["objects"]["duplo"]["position"]["z"],
        ])
        bowl_pos = np.array([
            scene_info["objects"]["bowl"]["position"]["x"],
            scene_info["objects"]["bowl"]["position"]["y"],
            scene_info["objects"]["bowl"]["position"]["z"],
        ])

        # Annotate
        subtasks = annotate_episode(
            ep_actions, block_pos, bowl_pos, fk_solver,
            near_threshold=near_threshold, far_threshold=far_threshold
        )
        annotations[int(ep_idx)] = subtasks.tolist()

        # Count subtask frames
        for i, name in enumerate(SUBTASK_NAMES):
            subtask_counts[name] += np.sum(subtasks == i)

        if visualize:
            # Find transition points
            transitions = []
            for i in range(1, len(subtasks)):
                if subtasks[i] != subtasks[i-1]:
                    transitions.append((i, SUBTASK_NAMES[subtasks[i]]))

            trans_str = ", ".join([f"{t[0]}:{t[1]}" for t in transitions])
            print(f"  Episode {ep_idx}: {len(ep_actions)} frames, transitions: {trans_str}")

    # Print summary
    total_frames = sum(subtask_counts.values())
    print(f"\nSubtask distribution ({total_frames} total frames):")
    for name, count in subtask_counts.items():
        pct = 100 * count / total_frames if total_frames > 0 else 0
        print(f"  {name}: {count} frames ({pct:.1f}%)")

    return annotations


def save_annotations(annotations: dict, dataset_path: Path):
    """Save annotations to JSON file."""
    output_file = dataset_path / "meta" / "subtask_annotations.json"

    with open(output_file, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"\nSaved annotations to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Annotate dataset with subtask labels")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Print per-episode annotation details",
    )
    parser.add_argument(
        "--near-threshold",
        type=float,
        default=NEAR_THRESHOLD,
        help=f"Distance threshold for 'near' (default: {NEAR_THRESHOLD}m)",
    )
    parser.add_argument(
        "--far-threshold",
        type=float,
        default=FAR_THRESHOLD,
        help=f"Distance threshold for 'far' (default: {FAR_THRESHOLD}m)",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"Annotating dataset: {dataset_path}")
    print(f"Thresholds: NEAR={args.near_threshold}m, FAR={args.far_threshold}m")

    annotations = annotate_dataset(
        dataset_path,
        visualize=args.visualize,
        near_threshold=args.near_threshold,
        far_threshold=args.far_threshold,
    )
    save_annotations(annotations, dataset_path)


if __name__ == "__main__":
    main()
