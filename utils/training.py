"""Shared training utilities for ACT and SmolVLA policies.

This module provides common functionality used across different policy training scripts:
- Dataset caching for faster training
- Observation preparation for policies
- Evaluation in simulation
- Training loop utilities
"""

import time
from pathlib import Path
from typing import Optional, Callable, Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.constants import MOTOR_NAMES, NUM_JOINTS
from utils.ik_solver import IKSolver


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

    Returns:
        Tuple of (success_rate, avg_steps, avg_time, ik_failure_rate, avg_ik_error, failure_summary)
    """
    import mujoco
    from lerobot_robot_sim import SO100SimConfig, SO100Sim
    from utils.failure_analysis import (
        Outcome, EpisodeAnalysis, analyze_trajectory,
        compute_analysis_summary,
    )

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

    # Detect cameras from policy config
    sim_cameras = []
    depth_camera_names = []
    try:
        for key in policy.config.input_features.keys():
            if key.startswith("observation.images."):
                cam_name = key.replace("observation.images.", "")
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

    if verbose:
        print(f"  Sim cameras: {sim_cameras}")
        if depth_camera_names:
            print(f"  Depth cameras: {depth_camera_names}")

    # Initialize IK solver if using EE actions
    ik_solver = None
    if is_ee_action_space:
        ik_solver = IKSolver()

    # Create simulation
    config = SO100SimConfig(
        sim_cameras=sim_cameras,
        depth_cameras=depth_camera_names,
        enable_vr=False,
        camera_width=640,
        camera_height=480,
    )
    sim_robot = SO100Sim(config)
    sim_robot.connect()

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
        sim_robot.reset_scene(randomize=randomize, pos_range=0.04, rot_range=np.pi)
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
            batch = prepare_obs_for_policy(obs, device, depth_cameras)

            # Add language instruction if provided (for VLA models)
            if language_instruction is not None:
                batch["observation.language"] = language_instruction

            batch = preprocessor(batch)

            with torch.no_grad():
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
                joint_action, _, _ = ik_solver.ee_to_joint_action(action_np, return_normalized=True)
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
    checkpoint_name: str = None
):
    """Save a training checkpoint.

    Args:
        policy: The policy model
        optimizer: The optimizer
        scheduler: The learning rate scheduler
        step: Current training step
        output_dir: Directory to save checkpoint
        checkpoint_name: Optional name for checkpoint (default: checkpoint_{step:06d})
    """
    if checkpoint_name is None:
        checkpoint_name = f"checkpoint_{step:06d}"

    checkpoint_dir = output_dir / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save policy
    policy.save_pretrained(str(checkpoint_dir))

    # Save optimizer and scheduler state
    torch.save({
        'step': step,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }, checkpoint_dir / "training_state.pt")


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
