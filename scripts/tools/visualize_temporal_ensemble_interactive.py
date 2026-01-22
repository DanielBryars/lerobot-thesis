#!/usr/bin/env python3
"""
Interactive frame-by-frame viewer for ACT temporal ensembling.

Step through the episode one frame at a time to see how predictions evolve.

Usage:
    python scripts/tools/visualize_temporal_ensemble_interactive.py outputs/train/act_20260118_155135 --checkpoint checkpoint_045000

Controls:
    SPACE / RIGHT: Next frame
    LEFT: Previous frame (rewinds and replays to that point)
    1-6: Switch camera viewpoints (6 = tracking camera)
    R: Toggle recording
    P: Play/Pause continuous playback
    ESC/Q: Quit
"""

import argparse
import sys
import time
from collections import deque
from datetime import datetime
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

# Camera viewpoint presets (camera 6 is tracking)
CAMERA_PRESETS = {
    1: {"name": "Overview", "azimuth": 135, "elevation": -25, "distance": 1.2, "lookat": [0.2, 0.0, 0.1]},
    2: {"name": "Side View", "azimuth": 90, "elevation": -15, "distance": 0.9, "lookat": [0.25, 0.0, 0.15]},
    3: {"name": "Front View", "azimuth": 180, "elevation": -20, "distance": 0.8, "lookat": [0.25, 0.0, 0.1]},
    4: {"name": "Top Down", "azimuth": 90, "elevation": -89, "distance": 0.8, "lookat": [0.22, 0.0, 0.0]},
    5: {"name": "Close Up", "azimuth": 120, "elevation": -10, "distance": 0.5, "lookat": [0.25, 0.1, 0.15]},
    6: {"name": "Tracking", "azimuth": 135, "elevation": -20, "distance": 0.35, "lookat": None, "tracking": True},
}


class SmoothTracker:
    """Smooth camera tracking with exponential moving average."""

    def __init__(self, smoothing: float = 0.15):
        self.smoothing = smoothing
        self.current_lookat = None

    def update(self, target_pos: np.ndarray) -> np.ndarray:
        if self.current_lookat is None:
            self.current_lookat = target_pos.copy()
        else:
            self.current_lookat = (1 - self.smoothing) * self.current_lookat + self.smoothing * target_pos
        return self.current_lookat.copy()

    def reset(self):
        self.current_lookat = None


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
        futures = []

        chunk_list = list(self.chunk_history)
        for i, (chunk_start, chunk_actions) in enumerate(chunk_list[-5:]):
            current_idx = self.step - chunk_start
            if current_idx < len(chunk_actions):
                future_actions = chunk_actions[current_idx:current_idx + n_future]
                if len(future_actions) > 0:
                    futures.append((i, future_actions))

        return futures


def draw_whisker_to_scene(scene, positions, color, alpha=0.8, radius=0.003):
    """Draw a trajectory whisker as connected spheres to a mjvScene."""
    for i, pos in enumerate(positions):
        if scene.ngeom >= scene.maxgeom - 1:
            break
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, 0, 0], dtype=np.float64),
            np.array(pos, dtype=np.float64),
            np.eye(3, dtype=np.float64).flatten(),
            np.array([*color, alpha], dtype=np.float64),
        )
        scene.ngeom += 1


def compute_ee_positions(mj_model, mj_data, joint_angles_sequence, ee_site_id):
    """Compute EE positions using MuJoCo FK."""
    saved_qpos = mj_data.qpos.copy()
    saved_qvel = mj_data.qvel.copy()

    positions = []
    arm_joint_start = 7

    for joint_angles in joint_angles_sequence:
        mj_data.qpos[arm_joint_start:arm_joint_start+6] = np.radians(joint_angles[:6])
        mujoco.mj_forward(mj_model, mj_data)
        positions.append(mj_data.site_xpos[ee_site_id].copy())

    mj_data.qpos[:] = saved_qpos
    mj_data.qvel[:] = saved_qvel
    mujoco.mj_forward(mj_model, mj_data)

    return np.array(positions)


def draw_all_whiskers(scene, futures, ensembler, mj_model, mj_data, ee_site_id):
    """Draw all whiskers (grey chunks + green ensembled) to a scene."""
    # Draw individual chunk predictions (grey whiskers)
    for chunk_idx, future_actions in futures:
        if len(future_actions) > 0 and ee_site_id >= 0:
            positions = compute_ee_positions(mj_model, mj_data, future_actions, ee_site_id)
            grey = 0.4 + 0.1 * chunk_idx
            draw_whisker_to_scene(scene, positions, (grey, grey, grey), alpha=0.4, radius=0.002)

    # Draw ensembled trajectory (green whisker)
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
            ensembled_positions = compute_ee_positions(mj_model, mj_data, ensembled_future, ee_site_id)
            draw_whisker_to_scene(scene, ensembled_positions, (0.0, 0.9, 0.0), alpha=0.9, radius=0.004)


def set_camera(viewer, preset_num, lookat_override=None):
    """Set camera to a preset viewpoint."""
    if preset_num not in CAMERA_PRESETS:
        return
    preset = CAMERA_PRESETS[preset_num]
    viewer.cam.azimuth = preset["azimuth"]
    viewer.cam.elevation = preset["elevation"]
    viewer.cam.distance = preset["distance"]

    if lookat_override is not None:
        viewer.cam.lookat[:] = lookat_override
    elif preset.get("lookat") is not None:
        viewer.cam.lookat[:] = preset["lookat"]

    print(f"  Camera: {preset['name']}" + (" (tracking gripper)" if preset.get("tracking") else ""))


class FrameState:
    """Stores the state at each frame for rewinding."""
    def __init__(self):
        self.qpos = None
        self.qvel = None
        self.chunk_history = None
        self.ensembler_step = 0
        self.futures = None
        self.n_chunks = 0
        self.success = False


def run_interactive(
    policy,
    preprocessor,
    postprocessor,
    device: torch.device,
    max_steps: int = 300,
    ensemble_coeff: float = 0.01,
    record: bool = False,
    output_dir: Path = None,
):
    """Run with interactive frame-by-frame control."""

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

    # Smooth tracker for tracking camera
    tracker = SmoothTracker(smoothing=0.15)

    # Recording setup
    recording = record
    if output_dir is None:
        output_dir = REPO_ROOT / "outputs" / "recordings" / f"temporal_interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    camera_dirs = {}
    if recording:
        output_dir.mkdir(parents=True, exist_ok=True)
        for cam_num, cam_preset in CAMERA_PRESETS.items():
            cam_dir = output_dir / cam_preset["name"].lower().replace(" ", "_")
            cam_dir.mkdir(parents=True, exist_ok=True)
            camera_dirs[cam_num] = cam_dir
        print(f"Recording to: {output_dir}")

    # Offscreen renderer
    render_width, render_height = 1280, 720
    offscreen_renderer = None
    if recording:
        offscreen_renderer = mujoco.Renderer(sim.mj_model, height=render_height, width=render_width)

    print("\n" + "="*60)
    print("INTERACTIVE TEMPORAL ENSEMBLING VIEWER")
    print("="*60)
    print(f"Chunk size: {chunk_size}")
    print(f"Ensemble coefficient: {ensemble_coeff}")
    print("\nControls:")
    print("  SPACE/RIGHT : Step forward one frame")
    print("  LEFT        : Step backward (rewind)")
    print("  P           : Play/Pause continuous playback")
    print("  1-6         : Switch camera viewpoints (6 = tracking)")
    print("  R           : Toggle recording" + (" (ON)" if recording else ""))
    print("  ESC/Q       : Quit")
    print("\nWhisker colors:")
    print("  GREEN = Ensembled trajectory")
    print("  GREY  = Individual chunk predictions")
    print("="*60 + "\n")

    # Frame history for rewinding
    frame_history = []
    current_frame = 0
    playing = False
    success = False
    frame_count = 0

    # Store initial state
    initial_qpos = sim.mj_data.qpos.copy()
    initial_qvel = sim.mj_data.qvel.copy()

    # Key state
    key_state = {
        "advance": False,
        "rewind": False,
        "camera": None,
        "toggle_record": False,
        "toggle_play": False,
    }

    def key_callback(keycode):
        nonlocal recording
        if keycode == 32 or keycode == 262:  # SPACE or RIGHT
            key_state["advance"] = True
        elif keycode == 263:  # LEFT
            key_state["rewind"] = True
        elif 49 <= keycode <= 54:  # 1-6
            key_state["camera"] = keycode - 48
        elif keycode == 82:  # R
            key_state["toggle_record"] = True
        elif keycode == 80:  # P
            key_state["toggle_play"] = True

    def compute_frame(frame_idx):
        """Compute or retrieve frame state."""
        nonlocal success

        # If we have this frame cached, restore it
        if frame_idx < len(frame_history):
            state = frame_history[frame_idx]
            sim.mj_data.qpos[:] = state.qpos
            sim.mj_data.qvel[:] = state.qvel
            mujoco.mj_forward(sim.mj_model, sim.mj_data)

            # Restore ensembler state
            ensembler.chunk_history = deque(state.chunk_history, maxlen=chunk_size)
            ensembler.step = state.ensembler_step

            return state.futures, state.n_chunks, state.success

        # Need to compute new frames up to frame_idx
        while len(frame_history) <= frame_idx:
            step = len(frame_history)

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
            futures = ensembler.get_future_predictions(n_future=30)

            # Store state before executing action
            state = FrameState()
            state.qpos = sim.mj_data.qpos.copy()
            state.qvel = sim.mj_data.qvel.copy()
            state.chunk_history = list(ensembler.chunk_history)
            state.ensembler_step = ensembler.step
            state.futures = futures
            state.n_chunks = len(predictions)
            state.success = sim.is_task_complete()

            frame_history.append(state)

            # Execute action
            action_dict = {m + ".pos": float(ensembled_action[i]) for i, m in enumerate(MOTOR_NAMES)}
            sim.send_action(action_dict)

            if state.success:
                success = True

        state = frame_history[frame_idx]
        return state.futures, state.n_chunks, state.success

    with mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data, key_callback=key_callback) as viewer:
        set_camera(viewer, 1)

        # Compute initial frame
        futures, n_chunks, frame_success = compute_frame(0)

        current_camera = 1
        while viewer.is_running() and current_frame < max_steps:
            # Get current EE position for tracking camera
            current_ee_pos = sim.mj_data.site_xpos[ee_site_id].copy()
            tracked_pos = tracker.update(current_ee_pos)

            # Handle key presses
            if key_state["camera"] is not None:
                current_camera = key_state["camera"]
                set_camera(viewer, current_camera, lookat_override=tracked_pos if current_camera == 6 else None)
                key_state["camera"] = None

            # Update tracking camera continuously if selected
            if current_camera == 6:
                preset = CAMERA_PRESETS[6]
                viewer.cam.lookat[:] = tracked_pos

            if key_state["toggle_record"]:
                recording = not recording
                if recording and not output_dir.exists():
                    output_dir.mkdir(parents=True, exist_ok=True)
                    for cam_num, cam_preset in CAMERA_PRESETS.items():
                        cam_dir = output_dir / cam_preset["name"].lower().replace(" ", "_")
                        cam_dir.mkdir(parents=True, exist_ok=True)
                        camera_dirs[cam_num] = cam_dir
                print(f"  Recording: {'ON' if recording else 'OFF'}")
                key_state["toggle_record"] = False

            if key_state["toggle_play"]:
                playing = not playing
                print(f"  Playback: {'PLAYING' if playing else 'PAUSED'}")
                key_state["toggle_play"] = False

            # Advance frame
            if key_state["advance"] or playing:
                key_state["advance"] = False
                if current_frame < max_steps - 1 and not success:
                    current_frame += 1
                    futures, n_chunks, frame_success = compute_frame(current_frame)
                    print(f"  Frame {current_frame}: {n_chunks} chunks contributing" +
                          (" - SUCCESS!" if frame_success else ""))

                    # Record if enabled
                    if recording and offscreen_renderer is not None:
                        try:
                            import PIL.Image as Image
                            for cam_num, cam_preset in CAMERA_PRESETS.items():
                                cam = mujoco.MjvCamera()
                                cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                                cam.azimuth = cam_preset["azimuth"]
                                cam.elevation = cam_preset["elevation"]
                                cam.distance = cam_preset["distance"]

                                # Use tracked position for tracking camera
                                if cam_preset.get("tracking"):
                                    cam.lookat[:] = tracked_pos
                                else:
                                    cam.lookat[:] = cam_preset["lookat"]

                                offscreen_renderer.update_scene(sim.mj_data, camera=cam)
                                draw_all_whiskers(offscreen_renderer.scene, futures, ensembler,
                                                 sim.mj_model, sim.mj_data, ee_site_id)

                                pixels = offscreen_renderer.render()
                                frame_path = camera_dirs[cam_num] / f"frame_{frame_count:05d}.png"
                                Image.fromarray(pixels).save(frame_path)
                            frame_count += 1
                        except Exception as e:
                            print(f"  Recording error: {e}")

                    if frame_success:
                        playing = False
                        print("\n*** SUCCESS! Episode complete. ***\n")

            # Rewind frame
            if key_state["rewind"]:
                key_state["rewind"] = False
                if current_frame > 0:
                    current_frame -= 1
                    futures, n_chunks, frame_success = compute_frame(current_frame)
                    print(f"  Frame {current_frame}: {n_chunks} chunks contributing (rewound)")

            # Draw whiskers
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                draw_all_whiskers(viewer.user_scn, futures, ensembler,
                                 sim.mj_model, sim.mj_data, ee_site_id)

            viewer.sync()

            if playing:
                time.sleep(0.05)  # Slower playback for visibility
            else:
                time.sleep(0.02)

    sim.disconnect()

    print(f"\nSession ended at frame {current_frame}")
    print(f"Total frames computed: {len(frame_history)}")
    if frame_count > 0:
        print(f"Recorded {frame_count} frames to {output_dir}")

    return success, current_frame


def main():
    parser = argparse.ArgumentParser(description="Interactive temporal ensembling viewer")
    parser.add_argument("model_path", type=str, help="Path to model directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_045000")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--coeff", type=float, default=0.01, help="Ensemble coefficient")
    parser.add_argument("--record", action="store_true", help="Enable recording")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for recordings")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model_path = Path(args.model_path) / args.checkpoint
    policy, preprocessor, postprocessor = load_act_policy(model_path, device)

    output_dir = Path(args.output_dir) if args.output_dir else None

    success, steps = run_interactive(
        policy, preprocessor, postprocessor, device,
        max_steps=args.max_steps,
        ensemble_coeff=args.coeff,
        record=args.record,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
