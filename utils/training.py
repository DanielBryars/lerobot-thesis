"""Shared training utilities for ACT and SmolVLA policies.

This module provides common functionality used across different policy training scripts:
- Dataset caching for faster training
- Observation preparation for policies
- Evaluation in simulation
- Training loop utilities
"""

import re
import time
from pathlib import Path
from typing import Optional, Callable, Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.constants import MOTOR_NAMES, NUM_JOINTS
from utils.ik_solver import IKSolver


# Default scene XML path
REPO_ROOT = Path(__file__).parent.parent
DEFAULT_SCENE_XML = REPO_ROOT / "scenes" / "so101_with_wrist_cam.xml"


def get_scene_metadata(scene_xml: Path = None) -> dict:
    """Extract metadata from a MuJoCo scene XML file.

    Args:
        scene_xml: Path to scene XML. Uses default if None.

    Returns:
        Dict with scene info including camera FOVs, positions, etc.
    """
    if scene_xml is None:
        scene_xml = DEFAULT_SCENE_XML

    scene_xml = Path(scene_xml)
    if not scene_xml.exists():
        return {"scene_xml": str(scene_xml), "error": "file not found"}

    content = scene_xml.read_text()

    # Extract camera info using regex (simple approach, works for our XMLs)
    cameras = {}
    # Match: <camera name="xxx" ... fovy="yyy" .../>
    camera_pattern = r'<camera\s+name="([^"]+)"[^>]*fovy="([^"]+)"[^>]*/>'
    for match in re.finditer(camera_pattern, content):
        cam_name = match.group(1)
        fovy = float(match.group(2))
        cameras[cam_name] = {"fovy": fovy}

    # Also try to get position if available
    camera_pos_pattern = r'<camera\s+name="([^"]+)"[^>]*pos="([^"]+)"[^>]*/>'
    for match in re.finditer(camera_pos_pattern, content):
        cam_name = match.group(1)
        pos_str = match.group(2)
        if cam_name in cameras:
            cameras[cam_name]["pos"] = [float(x) for x in pos_str.split()]

    return {
        "scene_xml": scene_xml.name,
        "scene_path": str(scene_xml),
        "cameras": cameras,
    }


class CachedDataset(torch.utils.data.Dataset):
    """Wrapper that pre-caches all dataset samples in memory for fast access.

    This eliminates GPU idle time caused by video decoding during training.
    All frames are decoded once at startup and kept in CPU memory.

    Args:
        dataset: The underlying dataset to cache
        resize_images_to: Optional size to resize images (e.g., 224 for 224x224)
    """

    def __init__(self, dataset, resize_images_to: Optional[int] = None):
        self.resize_to = resize_images_to
        print(f"\nPre-caching {len(dataset)} samples to memory...")
        if resize_images_to:
            print(f"  Resizing images to {resize_images_to}x{resize_images_to}")

        # Pre-load all samples
        self.samples = []
        for i in tqdm(range(len(dataset)), desc="Caching dataset"):
            sample = dataset[i]
            cached_sample = {}
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    # Resize images if requested
                    if resize_images_to and "images" in k and v.dim() >= 3:
                        if v.dim() == 4:  # [n_obs_steps, C, H, W]
                            v = F.interpolate(v, size=(resize_images_to, resize_images_to),
                                            mode='bilinear', align_corners=False)
                        elif v.dim() == 3:  # [C, H, W]
                            v = F.interpolate(v.unsqueeze(0), size=(resize_images_to, resize_images_to),
                                            mode='bilinear', align_corners=False).squeeze(0)
                    cached_sample[k] = v.clone()
                else:
                    cached_sample[k] = v
            self.samples.append(cached_sample)

        print(f"Cached {len(self.samples)} samples")

        # Estimate memory usage
        total_bytes = 0
        for sample in self.samples[:1]:
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    total_bytes += v.element_size() * v.numel()
        total_gb = (total_bytes * len(self.samples)) / (1024**3)
        print(f"Estimated cache size: {total_gb:.2f} GB")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class DiskCachedDataset(torch.utils.data.Dataset):
    """Disk-cached dataset wrapper for fast training without high RAM usage.

    On first run, decodes all video frames and saves to disk as torch tensors.
    On subsequent runs, loads directly from disk (~10-100x faster than video decode).

    Unlike CachedDataset, this:
    - Uses disk instead of RAM (works with large datasets)
    - Supports num_workers > 0 for parallel loading
    - Persists across training runs (no re-decoding)

    Args:
        dataset: The underlying LeRobotDataset to cache
        cache_dir: Directory to store cache (default: ~/.cache/lerobot_disk_cache/{dataset_name})
        resize_images_to: Optional size to resize images during caching (e.g., 224)
        force_rebuild: If True, rebuild cache even if it exists
    """

    def __init__(
        self,
        dataset,
        cache_dir: Path = None,
        resize_images_to: Optional[int] = None,
        force_rebuild: bool = False,
    ):
        self.dataset = dataset
        self.resize_to = resize_images_to
        self._length = len(dataset)

        # Determine cache directory
        if cache_dir is None:
            dataset_name = getattr(dataset, 'repo_id', None)
            if dataset_name is None:
                dataset_name = getattr(dataset, 'root', Path('unknown')).name
            dataset_name = str(dataset_name).replace('/', '_').replace('\\', '_')

            # Include size suffix for uniqueness
            size_suffix = f"_{self._length}samples"
            if resize_images_to:
                size_suffix += f"_{resize_images_to}px"

            cache_dir = Path.home() / '.cache' / 'lerobot_disk_cache' / (dataset_name + size_suffix)

        self.cache_dir = Path(cache_dir)
        self.samples_dir = self.cache_dir / 'samples'

        # Check if cache is complete
        complete_marker = self.cache_dir / '.complete'
        cache_complete = complete_marker.exists() and not force_rebuild

        if not cache_complete:
            self._build_cache()
            complete_marker.touch()
        else:
            print(f"Using existing disk cache: {self.cache_dir}")
            # Load valid indices mapping
            indices_file = self.cache_dir / 'valid_indices.pt'
            if indices_file.exists():
                self.valid_indices = torch.load(indices_file, weights_only=True)
            else:
                # Legacy cache without indices file - assume all valid
                self.valid_indices = list(range(self._length))
            print(f"  {len(self.valid_indices)} cached samples")

    def _build_cache(self):
        """Build the disk cache by decoding all frames."""
        import shutil

        # Clear existing incomplete cache
        if self.cache_dir.exists():
            print(f"Removing incomplete cache at {self.cache_dir}")
            shutil.rmtree(self.cache_dir)

        self.samples_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nBuilding disk cache at {self.cache_dir}")
        if self.resize_to:
            print(f"  Resizing images to {self.resize_to}x{self.resize_to}")

        total_bytes = 0
        skipped = 0
        valid_indices = []

        for i in tqdm(range(self._length), desc="Caching to disk"):
            cache_file = self.samples_dir / f'{i:07d}.pt'

            try:
                sample = self.dataset[i]
            except (AssertionError, Exception) as e:
                # Skip samples with timestamp/decoding issues (common in merged datasets)
                skipped += 1
                if skipped <= 3:
                    tqdm.write(f"  Skipping sample {i}: {type(e).__name__}")
                elif skipped == 4:
                    tqdm.write(f"  (suppressing further skip messages...)")
                continue

            cached_sample = {}

            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    # Resize images if requested
                    if self.resize_to and "images" in k and v.dim() >= 3:
                        if v.dim() == 4:  # [n_obs_steps, C, H, W]
                            v = F.interpolate(v, size=(self.resize_to, self.resize_to),
                                            mode='bilinear', align_corners=False)
                        elif v.dim() == 3:  # [C, H, W]
                            v = F.interpolate(v.unsqueeze(0), size=(self.resize_to, self.resize_to),
                                            mode='bilinear', align_corners=False).squeeze(0)
                    cached_sample[k] = v
                    if len(valid_indices) == 0:
                        total_bytes += v.element_size() * v.numel()
                else:
                    cached_sample[k] = v

            # Save as torch file (pickle-based, fast to load)
            torch.save(cached_sample, cache_file)
            valid_indices.append(i)

        # Save valid indices mapping
        self.valid_indices = valid_indices
        torch.save(valid_indices, self.cache_dir / 'valid_indices.pt')

        if skipped > 0:
            print(f"Skipped {skipped} samples with decoding errors")

        # Estimate disk usage
        total_gb = (total_bytes * self._length) / (1024**3)
        print(f"Disk cache complete. Estimated size: {total_gb:.2f} GB")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Map to valid index (handles skipped samples)
        actual_idx = self.valid_indices[idx]
        cache_file = self.samples_dir / f'{actual_idx:07d}.pt'
        return torch.load(cache_file, weights_only=False)


def cycle(dataloader):
    """Infinite dataloader iterator."""
    while True:
        for batch in dataloader:
            yield batch


def prepare_obs_for_policy(obs: dict, device: torch.device, depth_cameras: list = None) -> dict:
    """Convert simulation observation to policy input format.

    Args:
        obs: Dictionary of observations from simulation
        device: Target device for tensors
        depth_cameras: List of BASE camera names that have depth (e.g., ["overhead_cam"])
                       The actual depth observation key will have _depth suffix

    Returns:
        Dictionary formatted for policy input
    """
    depth_cameras = depth_cameras or []
    batch = {}

    # Extract state (joint positions)
    state = []
    for motor in MOTOR_NAMES:
        key = f"{motor}.pos"
        state.append(obs.get(key, 0.0))
    batch["observation.state"] = torch.tensor([state], dtype=torch.float32, device=device)

    # Camera images
    for key, value in obs.items():
        if isinstance(value, np.ndarray) and value.ndim == 3:
            # Check if this is a depth image (has _depth suffix in the key)
            is_depth = "_depth" in key

            if is_depth:
                # Depth: take first channel, normalize to 0-1, repeat to 3 channels for CNN
                # Stored as 0-255 = 0-2m range
                depth_uint8 = value[:, :, 0]  # Take first channel
                img = torch.from_numpy(depth_uint8).unsqueeze(0).unsqueeze(0).float() / 255.0
                img = img.repeat(1, 3, 1, 1)  # [1, 1, H, W] -> [1, 3, H, W] for CNN compatibility
            else:
                # RGB image: normalize to [0, 1]
                img = torch.from_numpy(value).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            batch[f"observation.images.{key}"] = img.to(device)

    return batch


def run_evaluation(
    policy,
    preprocessor: Callable,
    postprocessor: Callable,
    device: torch.device,
    num_episodes: int,
    randomize: bool = True,
    fps: int = 30,
    action_dim: int = None,
    depth_cameras: list = None,
    language_instruction: str = None,
    max_steps: int = 300,
    verbose: bool = True,
    analyze_failures: bool = True,
    visualize: bool = False,
    mujoco_viewer: bool = False,
    dataset_repo_id: str = None,
    temporal_ensemble_coeff: float = None,
    block_x: float = None,
    block_y: float = None,
) -> tuple:
    """Run evaluation episodes in simulation.

    Args:
        policy: The policy model to evaluate
        preprocessor: Function to preprocess observations
        postprocessor: Function to postprocess actions
        device: Device to run inference on
        num_episodes: Number of episodes to run
        randomize: Whether to randomize object positions
        fps: Frames per second for simulation
        action_dim: Action dimension (6 for joint, 8 for EE)
        depth_cameras: List of depth camera names
        language_instruction: Optional language instruction for VLA models
        max_steps: Maximum steps per episode
        verbose: Whether to print progress
        analyze_failures: Whether to track and analyze failures
        visualize: Whether to show live visualization (OpenCV window showing camera feeds)
        mujoco_viewer: Whether to open the MuJoCo 3D viewer window
        dataset_repo_id: Optional dataset repo ID to show training info
        temporal_ensemble_coeff: If set, enables temporal ensembling with this coefficient
            (e.g., 0.01). Predicts every step and averages overlapping chunks.
        block_x: Optional X position for block center (default: scene XML default)
        block_y: Optional Y position for block center (default: scene XML default)

    Returns:
        Tuple of (success_rate, avg_steps, avg_time, ik_failure_rate, avg_ik_error, failure_summary)
    """
    import mujoco
    from lerobot_robot_sim import SO100SimConfig, SO100Sim
    from utils.failure_analysis import (
        Outcome, EpisodeAnalysis, analyze_trajectory,
        compute_analysis_summary,
    )

    # Tokenize language instruction for SmolVLA models
    lang_tokens = None
    lang_attention_mask = None
    if language_instruction is not None:
        # Check if this is a SmolVLA model by looking for the tokenizer
        try:
            tokenizer = policy.model.vlm_with_expert.processor.tokenizer
            encoding = tokenizer(
                language_instruction,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512,
            )
            lang_tokens = encoding["input_ids"].to(device)
            lang_attention_mask = encoding["attention_mask"].bool().to(device)
            if verbose:
                print(f"  Tokenized language instruction: '{language_instruction}'")
        except AttributeError:
            # Not a SmolVLA model, skip tokenization
            pass

    # Temporal ensembling setup
    use_ensemble = temporal_ensemble_coeff is not None
    ensemble_weights = None
    chunk_size = None
    if use_ensemble:
        chunk_size = policy.config.chunk_size
        ensemble_weights = np.exp(-temporal_ensemble_coeff * np.arange(chunk_size))
        if verbose:
            print(f"  Temporal ensembling ENABLED (coeff={temporal_ensemble_coeff}, chunk_size={chunk_size})")

    # Determine if using EE action space based on action dimension
    if action_dim is None:
        # Try to get from policy config
        try:
            action_dim = policy.config.output_features['action'].shape[0]
        except:
            action_dim = 6  # Default to joint space

    is_ee_action_space = action_dim == 8

    if verbose:
        if is_ee_action_space:
            print(f"  Using EE action space (8-dim) with IK conversion")
        else:
            print(f"  Using joint action space ({action_dim}-dim)")

    # Detect cameras from policy config and extract training info
    sim_cameras = []
    depth_camera_names = []
    trained_camera_info = {}  # camera_name -> (height, width, channels)

    try:
        for key, feature in policy.config.input_features.items():
            if key.startswith("observation.images."):
                cam_name = key.replace("observation.images.", "")
                # Extract shape info (channels, height, width)
                shape = feature.shape if hasattr(feature, 'shape') else None
                if shape and len(shape) == 3:
                    trained_camera_info[cam_name] = (shape[1], shape[2], shape[0])  # H, W, C

                if "_depth" in cam_name:
                    # This is a depth camera - extract base name
                    base_cam = cam_name.replace("_depth", "")
                    depth_camera_names.append(base_cam)
                    if base_cam not in sim_cameras:
                        sim_cameras.append(base_cam)
                else:
                    if cam_name not in sim_cameras:
                        sim_cameras.append(cam_name)
    except:
        sim_cameras = ["wrist_cam"]  # fallback

    # Override with provided depth_cameras if specified
    if depth_cameras:
        depth_camera_names = depth_cameras

    # Evaluation settings
    eval_camera_width = 640
    eval_camera_height = 480
    # Use RGBD scene if model uses depth cameras (different overhead FOV: 58° vs 52°)
    if depth_camera_names:
        eval_scene = "so101_rgbd.xml"
    else:
        eval_scene = "so101_with_wrist_cam.xml"

    # Try to load training metadata from checkpoint
    training_meta = {}
    pretrained_path = getattr(getattr(policy, 'config', None), 'pretrained_path', None)
    if pretrained_path:
        meta_path = Path(pretrained_path) / "training_metadata.json"
        if meta_path.exists():
            import json
            with open(meta_path) as f:
                training_meta = json.load(f)

    if verbose:
        print(f"\n  --- TRAINING CONFIG ---")
        if training_meta:
            print(f"    Dataset: {training_meta.get('dataset_repo_id', 'Unknown')}")
            print(f"    Scene: {training_meta.get('scene', 'Unknown')}")
            print(f"    Cameras: {training_meta.get('cameras', [])}")
            # Show camera details with FOV
            scene_cameras = training_meta.get('scene_cameras', {})
            cam_resolutions = training_meta.get('camera_resolutions', {})
            for cam_name in training_meta.get('cameras', []):
                res = cam_resolutions.get(cam_name, '?')
                fov = scene_cameras.get(cam_name, {}).get('fovy', '?')
                print(f"      {cam_name}: {res}, FOV={fov}°")
        else:
            print(f"    (No training_metadata.json found)")
            for cam_name, (h, w, c) in trained_camera_info.items():
                print(f"      {cam_name}: {w}x{h} ({c} ch)")
        print(f"\n  --- EVALUATION CONFIG ---")
        print(f"    Scene: {eval_scene}")
        print(f"    Cameras: {sim_cameras}")
        print(f"    Resolution: {eval_camera_width}x{eval_camera_height}")
        if depth_camera_names:
            print(f"    Depth cameras: {depth_camera_names}")

    # Initialize IK solver if using EE actions
    ik_solver = None
    if is_ee_action_space:
        ik_solver = IKSolver()

    # Create simulation
    scene_path = REPO_ROOT / "scenes" / eval_scene
    config = SO100SimConfig(
        scene_xml=str(scene_path),
        sim_cameras=sim_cameras,
        depth_cameras=depth_camera_names,
        enable_vr=False,
        camera_width=eval_camera_width,
        camera_height=eval_camera_height,
    )
    sim_robot = SO100Sim(config)
    sim_robot.connect()

    # Setup visualization if requested
    if visualize:
        import cv2
        cv2.namedWindow("Policy Evaluation", cv2.WINDOW_NORMAL)
        print("  Camera visualization enabled - press Q to quit, R to reset episode")
    if mujoco_viewer:
        print("  MuJoCo 3D viewer enabled")

    # Goal position (bowl center) for failure analysis
    BOWL_POSITION = np.array([0.217, -0.225])

    successes = 0
    total_steps = 0
    total_time = 0
    episode_analyses = []

    policy.eval()

    for ep in range(num_episodes):
        print(f"  Episode {ep+1}/{num_episodes}...", end=" ", flush=True)
        policy.reset()  # Reset action chunking state
        sim_robot.reset_scene(randomize=randomize, pos_range=0.04, rot_range=np.pi,
                              pos_center_x=block_x, pos_center_y=block_y)

        # Reset temporal ensembling state for this episode
        if use_ensemble:
            from collections import deque
            chunk_history = deque(maxlen=chunk_size)
            ensemble_step = 0
        ep_start = time.time()
        trajectory = []  # Track object position for failure analysis
        task_completed = False

        for step in range(max_steps):
            # Track duplo position for failure analysis
            if analyze_failures:
                try:
                    duplo_body_id = mujoco.mj_name2id(sim_robot.mj_model, mujoco.mjtObj.mjOBJ_BODY, "duplo")
                    duplo_pos = sim_robot.mj_data.xpos[duplo_body_id].copy()
                    trajectory.append(duplo_pos)
                except:
                    pass  # Skip tracking if duplo not found

            obs = sim_robot.get_observation()
            
            # MuJoCo 3D viewer
            if mujoco_viewer:
                if not sim_robot.render():
                    print("\nMuJoCo viewer closed")
                    sim_robot.disconnect()
                    if visualize:
                        import cv2
                        cv2.destroyAllWindows()
                    return success_rate if 'success_rate' in dir() else 0, 0, 0, None, None, None

            # Camera feed visualization
            if visualize:
                import cv2
                frames = []
                # RGB cameras
                for cam in sim_cameras:
                    if cam in obs and isinstance(obs[cam], np.ndarray):
                        frame = cv2.cvtColor(obs[cam], cv2.COLOR_RGB2BGR)
                        cv2.putText(frame, f"{cam} | Ep {ep+1} Step {step}", (10, 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        frames.append(frame)
                # Depth cameras (normalize to 0-255 for display)
                for cam in depth_camera_names:
                    depth_key = f"{cam}_depth"
                    if depth_key in obs and isinstance(obs[depth_key], np.ndarray):
                        depth = obs[depth_key]
                        # Normalize depth to 0-255 (clip to reasonable range first)
                        depth_clipped = np.clip(depth, 0, 2.0)  # 0-2m range
                        depth_norm = (depth_clipped / 2.0 * 255).astype(np.uint8)
                        # Convert to 3-channel for display
                        if depth_norm.ndim == 2:
                            depth_vis = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                        else:
                            depth_vis = cv2.applyColorMap(depth_norm.squeeze(), cv2.COLORMAP_JET)
                        cv2.putText(depth_vis, f"{cam}_depth", (10, 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        frames.append(depth_vis)
                if frames:
                    display = np.hstack(frames) if len(frames) > 1 else frames[0]
                    cv2.imshow("Policy Evaluation", display)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        print("\nQuit requested")
                        sim_robot.disconnect()
                        cv2.destroyAllWindows()
                        return success_rate if 'success_rate' in dir() else 0, 0, 0, None, None, None
                    elif key == ord('r'):
                        print(" [RESET]", end="")
                        break  # Break inner loop to reset episode

            batch = prepare_obs_for_policy(obs, device, depth_camera_names)

            # Add language tokens if provided (for VLA models like SmolVLA)
            if lang_tokens is not None:
                batch["observation.language.tokens"] = lang_tokens
                batch["observation.language.attention_mask"] = lang_attention_mask
                # Also add task for preprocessor compatibility
                batch["task"] = language_instruction

            batch = preprocessor(batch)

            with torch.no_grad():
                if use_ensemble:
                    # Temporal ensembling: predict full chunk and average with history
                    chunk = policy.predict_action_chunk(batch)
                    chunk = postprocessor(chunk)
                    chunk_np = chunk.cpu().numpy()[0]  # (chunk_size, action_dim)

                    # Add chunk to history
                    chunk_history.append((ensemble_step, chunk_np.copy()))

                    # Compute ensembled action for current step
                    predictions = []
                    weights = []
                    chunk_list = list(chunk_history)
                    for i, (chunk_start, chunk_actions) in enumerate(chunk_list):
                        idx_in_chunk = ensemble_step - chunk_start
                        if 0 <= idx_in_chunk < len(chunk_actions):
                            predictions.append(chunk_actions[idx_in_chunk])
                            age = len(chunk_list) - 1 - i
                            weights.append(ensemble_weights[min(age, len(ensemble_weights)-1)])

                    predictions = np.array(predictions)
                    weights = np.array(weights)
                    weights = weights / weights.sum()
                    action_np = (predictions * weights[:, None]).sum(axis=0)
                    ensemble_step += 1

                    # Skip postprocessor since we already applied it to the chunk
                    action = torch.from_numpy(action_np)
                else:
                    action = policy.select_action(batch)
                    # Apply postprocessor (denormalizes action)
                    action = postprocessor(action)

            # Convert action to numpy
            action_np = action.cpu().numpy()
            if action_np.ndim > 1:
                action_np = action_np.flatten()

            # Convert EE actions to joint actions if needed
            if is_ee_action_space:
                action_np = action_np[:8]  # Take first 8 values (EE action)

                # Clamp EE position to approximate workspace bounds to reduce IK failures
                # Z: minimum 3cm above table to avoid unreachable low positions
                # This keeps robot moving in approximately right direction even if target is out of bounds
                action_np[2] = max(action_np[2], 0.03)  # Z minimum

                joint_action, _, ik_success = ik_solver.ee_to_joint_action(action_np, return_normalized=True)
                # Always use the IK solution (even if failed) - it's the solver's best attempt
                # Holding position causes policy/robot state mismatch and cascading failures
            else:
                joint_action = action_np[:NUM_JOINTS]

            action_dict = {f"{MOTOR_NAMES[i]}.pos": float(joint_action[i]) for i in range(NUM_JOINTS)}
            sim_robot.send_action(action_dict)

            if sim_robot.is_task_complete():
                task_completed = True
                successes += 1
                total_steps += step + 1
                total_time += time.time() - ep_start
                break
        else:
            total_steps += max_steps
            total_time += time.time() - ep_start

        # Print episode result
        status = "OK" if task_completed else "FAIL"
        print(f"{status} ({time.time() - ep_start:.1f}s)")

        # Analyze this episode
        if analyze_failures and trajectory:
            ep_duration = time.time() - ep_start
            outcome, metrics = analyze_trajectory(
                trajectory, task_completed, BOWL_POSITION
            )
            heights = [pos[2] for pos in trajectory]
            max_height = max(heights)
            was_lifted = max_height > 0.05
            was_dropped = was_lifted and heights[-1] < 0.03

            final_pos = trajectory[-1] if trajectory else np.zeros(3)
            final_xy = np.array([final_pos[0], final_pos[1]])
            final_distance = np.linalg.norm(final_xy - BOWL_POSITION)

            analysis = EpisodeAnalysis(
                outcome=outcome,
                steps=step + 1 if task_completed else max_steps,
                duration=ep_duration,
                max_height=max_height,
                was_lifted=was_lifted,
                was_dropped=was_dropped,
                final_distance_to_goal=final_distance,
                trajectory=trajectory
            )
            episode_analyses.append(analysis)

    sim_robot.disconnect()
    if visualize:
        import cv2
        cv2.destroyAllWindows()

    success_rate = successes / num_episodes
    avg_steps = total_steps / num_episodes
    avg_time = total_time / num_episodes

    # Compute failure analysis summary
    failure_summary = None
    if analyze_failures and episode_analyses:
        failure_summary = compute_analysis_summary(episode_analyses)
        # Print brief failure breakdown
        outcome_counts = failure_summary.get("outcome_counts", {})
        if any(outcome_counts.get(o, 0) > 0 for o in Outcome if o != Outcome.SUCCESS):
            failures = {o.value: outcome_counts.get(o, 0) for o in Outcome if o != Outcome.SUCCESS and outcome_counts.get(o, 0) > 0}
            if verbose:
                print(f"  Failure breakdown: {failures}")

    # Report IK stats if using EE action space
    if is_ee_action_space and ik_solver:
        stats = ik_solver.get_stats()
        if stats["total_count"] > 0 and verbose:
            print(f"  IK stats: {stats['failure_count']}/{stats['total_count']} failures "
                  f"({100*stats['failure_rate']:.2f}%), avg error: {stats['avg_error_mm']:.2f}mm")
            return success_rate, avg_steps, avg_time, stats["failure_rate"], stats["avg_error_mm"], failure_summary

    return success_rate, avg_steps, avg_time, None, None, failure_summary


def get_action_space_info(output_features: dict) -> tuple:
    """Determine action space type from output features.

    Args:
        output_features: Dictionary of output features from dataset

    Returns:
        Tuple of (action_dim, action_space_name)
    """
    action_feature = output_features.get('action')
    if action_feature:
        action_shape = list(action_feature.shape) if hasattr(action_feature, 'shape') else action_feature.shape
        action_dim = action_shape[0] if action_shape else 0
        if action_dim == 8:
            return action_dim, "end-effector (8-dim: xyz + quat + gripper)"
        elif action_dim == 6:
            return action_dim, "joint (6-dim: normalized joints)"
        else:
            return action_dim, f"unknown ({action_dim}-dim)"
    return 0, "unknown"


def get_camera_names(input_features: dict) -> list:
    """Extract camera names from input features.

    Args:
        input_features: Dictionary of input features

    Returns:
        List of camera names
    """
    return [key.replace("observation.images.", "")
            for key in input_features.keys()
            if key.startswith("observation.images.")]


def create_output_dir(base_dir: str = "outputs/train", prefix: str = "train") -> Path:
    """Create timestamped output directory.

    Args:
        base_dir: Base directory for outputs
        prefix: Prefix for the directory name

    Returns:
        Path to the created directory
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"{base_dir}/{prefix}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_checkpoint(
    policy,
    optimizer,
    scheduler,
    step: int,
    output_dir: Path,
    training_metadata: dict,
    checkpoint_name: str = None,
    preprocessor=None,
    postprocessor=None,
    best_loss: float = None,
):
    """Save a training checkpoint.

    Args:
        policy: The policy model
        optimizer: The optimizer
        scheduler: The learning rate scheduler
        step: Current training step
        output_dir: Directory to save checkpoint
        training_metadata: Dict with training info (dataset_repo_id, scene, cameras, etc.) - REQUIRED
        checkpoint_name: Optional name for checkpoint (default: checkpoint_{step:06d})
        preprocessor: Optional preprocessor to save
        postprocessor: Optional postprocessor to save
        best_loss: Optional best loss value to save in training state
    """
    import json

    if checkpoint_name is None:
        checkpoint_name = f"checkpoint_{step:06d}"

    checkpoint_dir = output_dir / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save policy
    policy.save_pretrained(str(checkpoint_dir))

    # Save preprocessor/postprocessor if provided
    if preprocessor is not None:
        preprocessor.save_pretrained(str(checkpoint_dir))
    if postprocessor is not None:
        postprocessor.save_pretrained(str(checkpoint_dir))

    # Save optimizer and scheduler state
    state = {
        'step': step,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }
    if best_loss is not None:
        state['best_loss'] = best_loss
    torch.save(state, checkpoint_dir / "training_state.pt")

    # Save training metadata (dataset, scene, cameras, etc.)
    with open(checkpoint_dir / "training_metadata.json", "w") as f:
        json.dump(training_metadata, f, indent=2)


def load_checkpoint(
    checkpoint_dir: Path,
    policy,
    optimizer=None,
    scheduler=None,
):
    """Load a training checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoint
        policy: The policy model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into

    Returns:
        Step number from checkpoint
    """
    # Load training state
    state_path = checkpoint_dir / "training_state.pt"
    if state_path.exists():
        state = torch.load(state_path)
        step = state['step']

        if optimizer and 'optimizer_state_dict' in state:
            optimizer.load_state_dict(state['optimizer_state_dict'])

        if scheduler and 'scheduler_state_dict' in state and state['scheduler_state_dict']:
            scheduler.load_state_dict(state['scheduler_state_dict'])

        return step

    return 0
