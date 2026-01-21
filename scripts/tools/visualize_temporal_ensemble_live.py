#!/usr/bin/env python3
"""
Live visualization of ACT temporal ensembling in MuJoCo.

Shows overlapping chunk predictions as whiskers converging to the ensembled action.

Usage:
    python scripts/tools/visualize_temporal_ensemble_live.py outputs/train/act_20260118_155135 --checkpoint checkpoint_045000
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

# Add project paths
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot_robot_sim import SO100Sim, SO100SimConfig

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def load_act_policy(model_path: Path, device: torch.device):
    """Load ACT policy."""
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    print(f"Loading ACT from {model_path}...")
    policy = ACTPolicy.from_pretrained(str(model_path))
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(model_path)
    )

    return policy, preprocessor, postprocessor


def prepare_obs(obs: dict, device: str = "cuda") -> dict:
    """Convert sim observation to policy input format."""
    batch = {}

    state = np.array([obs[m + ".pos"] for m in MOTOR_NAMES], dtype=np.float32)
    batch["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(device)

    for key, value in obs.items():
        if isinstance(value, np.ndarray) and value.ndim == 3:
            img = torch.from_numpy(value).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            batch[f"observation.images.{key}"] = img.to(device)

    return batch


class TemporalEnsembler:
    """Temporal ensembler for action chunks."""

    def __init__(self, coeff: float, chunk_size: int):
        self.coeff = coeff
        self.chunk_size = chunk_size
        self.weights = np.exp(-coeff * np.arange(chunk_size))
        self.chunk_history = deque(maxlen=chunk_size)
        self.step = 0

    def reset(self):
        self.chunk_history.clear()
        self.step = 0

    def update(self, chunk: np.ndarray):
        """Add new chunk and compute ensembled action."""
        self.chunk_history.append((self.step, chunk.copy()))

        # Collect predictions for current step
        predictions = []
        weights = []

        chunk_list = list(self.chunk_history)
        for i, (chunk_start, chunk_actions) in enumerate(chunk_list):
            idx_in_chunk = self.step - chunk_start
            if 0 <= idx_in_chunk < len(chunk_actions):
                predictions.append(chunk_actions[idx_in_chunk])
                age = len(chunk_list) - 1 - i
                weights.append(self.weights[min(age, len(self.weights)-1)])

        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()

        ensembled = (predictions * weights[:, None]).sum(axis=0)
        self.step += 1

        return ensembled, predictions, weights

    def get_future_predictions(self, n_future: int = 20):
        """Get all chunk predictions for the next n_future steps."""
        futures = []  # List of (chunk_idx, future_actions)

        chunk_list = list(self.chunk_history)
        for i, (chunk_start, chunk_actions) in enumerate(chunk_list[-5:]):  # Last 5 chunks
            # Get future portion of this chunk
            current_idx = self.step - chunk_start
            if current_idx < len(chunk_actions):
                future_actions = chunk_actions[current_idx:current_idx + n_future]
                if len(future_actions) > 0:
                    futures.append((i, future_actions))

        return futures


def draw_whisker(viewer, positions, color, alpha=0.8, radius=0.003):
    """Draw a trajectory whisker as connected spheres."""
    for i, pos in enumerate(positions):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom - 1:
            break
        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, 0, 0], dtype=np.float64),
            np.array(pos, dtype=np.float64),
            np.eye(3, dtype=np.float64).flatten(),
            np.array([*color, alpha], dtype=np.float64),
        )
        viewer.user_scn.ngeom += 1


def compute_ee_positions(mj_model, mj_data, joint_angles_sequence, ee_site_id):
    """Compute EE positions using MuJoCo FK.

    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data
        joint_angles_sequence: Joint angles in DEGREES, shape (N, 6)
        ee_site_id: Site ID for end-effector

    Returns:
        EE positions, shape (N, 3)
    """
    saved_qpos = mj_data.qpos.copy()
    saved_qvel = mj_data.qvel.copy()

    positions = []
    arm_joint_start = 7  # After duplo's free joint

    for joint_angles in joint_angles_sequence:
        mj_data.qpos[arm_joint_start:arm_joint_start+6] = np.radians(joint_angles[:6])
        mujoco.mj_forward(mj_model, mj_data)
        positions.append(mj_data.site_xpos[ee_site_id].copy())

    # Restore state
    mj_data.qpos[:] = saved_qpos
    mj_data.qvel[:] = saved_qvel
    mujoco.mj_forward(mj_model, mj_data)

    return np.array(positions)


def run_live(
    policy,
    preprocessor,
    postprocessor,
    device: torch.device,
    max_steps: int = 300,
    ensemble_coeff: float = 0.01,
):
    """Run with live MuJoCo visualization of temporal ensembling."""

    # Create simulation
    scene_path = REPO_ROOT / "scenes" / "so101_with_wrist_cam.xml"
    sim_config = SO100SimConfig(
        scene_xml=str(scene_path),
        sim_cameras=["overhead_cam", "wrist_cam"],
        camera_width=640,
        camera_height=480,
    )
    sim = SO100Sim(sim_config)
    sim.connect()
    sim.reset_scene(randomize=False)

    chunk_size = policy.config.chunk_size
    ensembler = TemporalEnsembler(ensemble_coeff, chunk_size)

    # Find EE site for FK
    ee_site_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    print(f"EE site ID: {ee_site_id} (gripperframe)")

    print("\n" + "="*60)
    print("TEMPORAL ENSEMBLING LIVE VISUALIZATION")
    print("="*60)
    print(f"Chunk size: {chunk_size}")
    print(f"Ensemble coefficient: {ensemble_coeff}")
    print("\nWhisker colors:")
    print("  GREEN = Ensembled trajectory (what we execute)")
    print("  GREY  = Individual chunk predictions (overlapping)")
    print("\nPress ESC or Q to quit")
    print("="*60 + "\n")

    step = 0
    success = False

    with mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data) as viewer:
        while viewer.is_running() and step < max_steps:
            # Get observation and predict
            obs = sim.get_observation()
            batch = prepare_obs(obs, device)
            batch = preprocessor(batch)

            with torch.no_grad():
                chunk = policy.predict_action_chunk(batch)
                chunk = postprocessor(chunk)
                chunk_np = chunk.cpu().numpy()[0]

            # Update ensembler
            ensembled_action, predictions, weights = ensembler.update(chunk_np)

            # Get future predictions for visualization
            futures = ensembler.get_future_predictions(n_future=30)

            # Execute ensembled action
            action_dict = {m + ".pos": float(ensembled_action[i]) for i, m in enumerate(MOTOR_NAMES)}
            sim.send_action(action_dict)

            # Check success
            if sim.is_task_complete():
                success = True
                print(f"\n*** SUCCESS at step {step}! ***")
                print(f"Chunks contributing at success: {len(predictions)}")
                # Keep viewer open briefly
                for _ in range(60):
                    viewer.sync()
                    time.sleep(0.05)
                break

            # Visualization
            with viewer.lock():
                viewer.user_scn.ngeom = 0

                # Draw individual chunk predictions (grey whiskers)
                for chunk_idx, future_actions in futures:
                    if len(future_actions) > 0 and ee_site_id >= 0:
                        positions = compute_ee_positions(
                            sim.mj_model, sim.mj_data,
                            future_actions, ee_site_id
                        )
                        grey = 0.4 + 0.1 * chunk_idx  # Newer chunks slightly lighter
                        draw_whisker(viewer, positions, (grey, grey, grey), alpha=0.4, radius=0.002)

                # Draw ensembled trajectory (green whisker)
                # Compute ensembled future by averaging chunk futures
                if futures and ee_site_id >= 0:
                    max_len = max(len(f[1]) for f in futures)
                    ensembled_future = []

                    for t in range(min(max_len, 30)):
                        preds_at_t = []
                        w_at_t = []
                        for i, (chunk_idx, future_actions) in enumerate(futures):
                            if t < len(future_actions):
                                preds_at_t.append(future_actions[t])
                                w_at_t.append(ensembler.weights[min(i, len(ensembler.weights)-1)])

                        if preds_at_t:
                            preds_at_t = np.array(preds_at_t)
                            w_at_t = np.array(w_at_t)
                            w_at_t = w_at_t / w_at_t.sum()
                            ensembled_at_t = (preds_at_t * w_at_t[:, None]).sum(axis=0)
                            ensembled_future.append(ensembled_at_t)

                    if ensembled_future:
                        ensembled_future = np.array(ensembled_future)
                        ensembled_positions = compute_ee_positions(
                            sim.mj_model, sim.mj_data,
                            ensembled_future, ee_site_id
                        )
                        draw_whisker(viewer, ensembled_positions, (0.0, 0.9, 0.0), alpha=0.9, radius=0.004)

            viewer.sync()
            step += 1
            time.sleep(0.02)

    sim.disconnect()

    print(f"\nEpisode ended at step {step}")
    print(f"Result: {'SUCCESS' if success else 'FAILURE'}")

    return success, step


def main():
    parser = argparse.ArgumentParser(description="Live temporal ensembling visualization")
    parser.add_argument("model_path", type=str, help="Path to model directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_045000")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--coeff", type=float, default=0.01, help="Ensemble coefficient")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model_path = Path(args.model_path) / args.checkpoint
    policy, preprocessor, postprocessor = load_act_policy(model_path, device)

    success, steps = run_live(
        policy, preprocessor, postprocessor, device,
        max_steps=args.max_steps,
        ensemble_coeff=args.coeff,
    )


if __name__ == "__main__":
    main()
