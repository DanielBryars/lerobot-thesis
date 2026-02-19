"""Shared training utilities for ACT and SmolVLA policies.

This module provides common functionality used across different policy training scripts:
- Dataset caching for faster training
- Observation preparation for policies
- Evaluation in simulation
- Training loop utilities
"""

import json
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

    OBSOLETE: This class has frame alignment issues with merged/multi-episode datasets.
    Use no caching (direct dataset access) instead, or CachedDataset if RAM permits.

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


class EpisodeFilterDataset(torch.utils.data.Dataset):
    """Dataset wrapper that filters to specific episode indices.

    Args:
        dataset: The underlying dataset to wrap (LeRobotDataset or wrapper)
        episode_indices: Set of episode indices to include
    """

    def __init__(self, dataset, episode_indices: set):
        self.dataset = dataset
        self.episode_indices = episode_indices

        # Get episode indices efficiently from underlying hf_dataset
        # Navigate through any wrappers to find the LeRobotDataset
        base_dataset = dataset
        while hasattr(base_dataset, 'dataset'):
            base_dataset = base_dataset.dataset

        if hasattr(base_dataset, 'hf_dataset'):
            # Fast path: read episode_index column directly
            all_episode_indices = base_dataset.hf_dataset['episode_index']
            self.index_map = []
            for i, ep_idx in enumerate(all_episode_indices):
                if isinstance(ep_idx, torch.Tensor):
                    ep_idx = ep_idx.item()
                if ep_idx in episode_indices:
                    self.index_map.append(i)
        else:
            # Slow fallback: iterate through dataset
            self.index_map = []
            for i in range(len(dataset)):
                item = dataset[i]
                if 'episode_index' in item:
                    ep_idx = item['episode_index']
                    if isinstance(ep_idx, torch.Tensor):
                        ep_idx = ep_idx.item()
                    if ep_idx in episode_indices:
                        self.index_map.append(i)

        print(f"EpisodeFilterDataset: {len(self.index_map)} samples from {len(episode_indices)} episodes")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        return self.dataset[self.index_map[idx]]


class PickupCoordinateDataset(torch.utils.data.Dataset):
    """Dataset wrapper that adds pickup coordinates from episode_scenes.json.

    This enables spatial conditioning for the ACT model by providing the
    target pickup location (duplo block position) as additional input.

    The coordinates are normalized to [-1, 1] based on workspace bounds.

    Args:
        dataset: The underlying dataset to wrap
        episode_scenes: Dict mapping episode_index (str) -> scene info with objects.duplo.position
        x_bounds: (min, max) tuple for X normalization
        y_bounds: (min, max) tuple for Y normalization
    """

    # Full workspace bounds for generalization
    # Covers both duplo range and confuser workspace
    DEFAULT_X_BOUNDS = (0.10, 0.38)  # 28cm range (confuser min to duplo max)
    DEFAULT_Y_BOUNDS = (-0.28, 0.27)  # 55cm range (confuser min to duplo max)

    def __init__(
        self,
        dataset,
        episode_scenes: dict,
        x_bounds: tuple = None,
        y_bounds: tuple = None,
    ):
        self.dataset = dataset
        self.episode_scenes = episode_scenes
        self.x_bounds = x_bounds or self.DEFAULT_X_BOUNDS
        self.y_bounds = y_bounds or self.DEFAULT_Y_BOUNDS

        # Pre-compute normalized coordinates for each episode
        self.episode_coords = {}
        for ep_idx_str, scene_info in episode_scenes.items():
            ep_idx = int(ep_idx_str)
            try:
                pos = scene_info['objects']['duplo']['position']
                x, y = pos['x'], pos['y']
                # Normalize to [-1, 1]
                x_norm = 2 * (x - self.x_bounds[0]) / (self.x_bounds[1] - self.x_bounds[0]) - 1
                y_norm = 2 * (y - self.y_bounds[0]) / (self.y_bounds[1] - self.y_bounds[0]) - 1
                # Clamp to [-1, 1] in case of slight out-of-bounds
                x_norm = max(-1, min(1, x_norm))
                y_norm = max(-1, min(1, y_norm))
                self.episode_coords[ep_idx] = torch.tensor([x_norm, y_norm], dtype=torch.float32)
            except (KeyError, TypeError):
                # Missing position data - use zeros
                self.episode_coords[ep_idx] = torch.tensor([0.0, 0.0], dtype=torch.float32)

        print(f"PickupCoordinateDataset: loaded coordinates for {len(self.episode_coords)} episodes")
        print(f"  X bounds: {self.x_bounds}, Y bounds: {self.y_bounds}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Get episode index from the item
        ep_idx = item['episode_index']
        if isinstance(ep_idx, torch.Tensor):
            ep_idx = ep_idx.item()

        # Look up pickup coordinates
        if ep_idx in self.episode_coords:
            coords = self.episode_coords[ep_idx]
        else:
            # Fallback for missing episodes
            coords = torch.tensor([0.0, 0.0], dtype=torch.float32)

        # Add as observation.environment_state (ACT model already supports this)
        item['observation.environment_state'] = coords

        return item

    @classmethod
    def load_episode_scenes(cls, dataset_repo_id: str) -> dict:
        """Load episode_scenes.json from local cache or HuggingFace dataset.

        Args:
            dataset_repo_id: HuggingFace dataset ID (e.g., 'danbhf/sim_pick_place_2pos_220ep')

        Returns:
            Dict mapping episode_index (str) -> scene info
        """
        # Try local cache first
        try:
            from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
            local_path = HF_LEROBOT_HOME / dataset_repo_id / 'meta' / 'episode_scenes.json'
            if local_path.exists():
                with open(local_path) as f:
                    return json.load(f)
        except Exception:
            pass

        # Fall back to HuggingFace download
        from huggingface_hub import hf_hub_download
        try:
            scenes_path = hf_hub_download(
                dataset_repo_id,
                'meta/episode_scenes.json',
                repo_type='dataset'
            )
            with open(scenes_path) as f:
                return json.load(f)
        except Exception as e:
            print(f"WARNING: Could not load episode_scenes.json: {e}")
            return {}

    @classmethod
    def compute_stats(cls, episode_scenes: dict, x_bounds: tuple = None, y_bounds: tuple = None) -> dict:
        """Compute normalization statistics for pickup coordinates.

        Returns stats dict compatible with LeRobot's normalization pipeline.
        Since we pre-normalize to [-1, 1], we return mean=0, std=1 so the
        preprocessor doesn't modify the values (no-op normalization).

        Args:
            episode_scenes: Dict from load_episode_scenes()
            x_bounds: (min, max) for X normalization (unused, kept for API compatibility)
            y_bounds: (min, max) for Y normalization (unused, kept for API compatibility)

        Returns:
            Dict with 'mean' and 'std' tensors for observation.environment_state
        """
        # We pre-normalize to [-1, 1] in the dataset wrapper, so return
        # mean=0, std=1 to make the preprocessor a no-op for this feature
        return {
            'observation.environment_state': {
                'mean': torch.tensor([0.0, 0.0], dtype=torch.float32),
                'std': torch.tensor([1.0, 1.0], dtype=torch.float32),
            }
        }


class SubtaskDataset(torch.utils.data.Dataset):
    """Dataset wrapper that adds subtask labels from subtask_annotations.json.

    This enables subtask conditioning for the ACT model by providing the
    current subtask phase as additional input.

    Subtask labels:
        0: MOVE_TO_SOURCE - Approaching the target block
        1: PICK_UP - Near block, executing grasp
        2: MOVE_TO_DEST - Transporting to destination
        3: DROP - Near destination, releasing

    Args:
        dataset: The underlying dataset to wrap
        subtask_annotations: Dict mapping episode_index (int) -> list of subtask labels per frame
        use_one_hot: If True, return one-hot encoding (4 dims). If False, return integer label.
    """

    NUM_SUBTASKS = 4
    SUBTASK_NAMES = ["MOVE_TO_SOURCE", "PICK_UP", "MOVE_TO_DEST", "DROP"]

    def __init__(
        self,
        dataset,
        subtask_annotations: dict,
        use_one_hot: bool = True,
    ):
        self.dataset = dataset
        self.subtask_annotations = subtask_annotations
        self.use_one_hot = use_one_hot

        # Convert string keys to int if needed
        self.subtask_annotations = {
            int(k): v for k, v in subtask_annotations.items()
        }

        print(f"SubtaskDataset: loaded annotations for {len(self.subtask_annotations)} episodes")
        print(f"  Output format: {'one-hot (4 dims)' if use_one_hot else 'integer label'}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Get episode and frame index from the item
        ep_idx = item['episode_index']
        if isinstance(ep_idx, torch.Tensor):
            ep_idx = ep_idx.item()

        frame_idx = item['frame_index']
        if isinstance(frame_idx, torch.Tensor):
            frame_idx = frame_idx.item()

        # Look up subtask label
        if ep_idx in self.subtask_annotations:
            annotations = self.subtask_annotations[ep_idx]
            if frame_idx < len(annotations):
                subtask = annotations[frame_idx]
            else:
                # Frame index out of range - use last known subtask
                subtask = annotations[-1] if annotations else 0
        else:
            # Fallback for missing episodes - assume MOVE_TO_SOURCE
            subtask = 0

        # Convert to tensor
        if self.use_one_hot:
            subtask_tensor = torch.zeros(self.NUM_SUBTASKS, dtype=torch.float32)
            subtask_tensor[subtask] = 1.0
        else:
            subtask_tensor = torch.tensor([subtask], dtype=torch.float32)

        # Add subtask to observation.environment_state (concatenate if exists)
        # This allows the model to use subtask through the existing env_state pathway
        if 'observation.environment_state' in item:
            # Concatenate with existing env state (e.g., pickup coords)
            existing = item['observation.environment_state']
            item['observation.environment_state'] = torch.cat([existing, subtask_tensor])
        else:
            # Create new env state from subtask alone
            item['observation.environment_state'] = subtask_tensor

        return item

    @classmethod
    def load_annotations(cls, dataset_repo_id: str) -> dict:
        """Load subtask_annotations.json from HuggingFace dataset.

        Args:
            dataset_repo_id: HuggingFace dataset ID (e.g., 'danbhf/sim_pick_place_157ep')

        Returns:
            Dict mapping episode_index (int) -> list of subtask labels per frame
        """
        from huggingface_hub import hf_hub_download

        try:
            annotations_path = hf_hub_download(
                dataset_repo_id,
                'meta/subtask_annotations.json',
                repo_type='dataset'
            )
            with open(annotations_path) as f:
                return json.load(f)
        except Exception as e:
            print(f"WARNING: Could not load subtask_annotations.json: {e}")
            return {}

    @classmethod
    def load_annotations_local(cls, dataset_path: str) -> dict:
        """Load subtask_annotations.json from local dataset path.

        Args:
            dataset_path: Path to local dataset directory

        Returns:
            Dict mapping episode_index (int) -> list of subtask labels per frame
        """
        from pathlib import Path
        annotations_file = Path(dataset_path) / 'meta' / 'subtask_annotations.json'
        if annotations_file.exists():
            with open(annotations_file) as f:
                return json.load(f)
        else:
            print(f"WARNING: subtask_annotations.json not found at {annotations_file}")
            return {}

    @classmethod
    def compute_stats(cls, use_one_hot: bool = True) -> dict:
        """Compute normalization statistics for subtask labels.

        Returns stats dict compatible with LeRobot's normalization pipeline.
        For one-hot encoding, we return mean=0, std=1 (no-op normalization).

        Note: Subtask is added to observation.environment_state, so we return
        stats under that key.

        Args:
            use_one_hot: Whether using one-hot encoding

        Returns:
            Dict with 'mean' and 'std' tensors for observation.environment_state
        """
        if use_one_hot:
            return {
                'observation.environment_state': {
                    'mean': torch.zeros(cls.NUM_SUBTASKS, dtype=torch.float32),
                    'std': torch.ones(cls.NUM_SUBTASKS, dtype=torch.float32),
                }
            }
        else:
            return {
                'observation.environment_state': {
                    'mean': torch.tensor([0.0], dtype=torch.float32),
                    'std': torch.tensor([1.0], dtype=torch.float32),
                }
            }


class DeltaActionDataset(torch.utils.data.Dataset):
    """Dataset wrapper that converts absolute actions to frame-to-frame deltas.

    For action chunks of shape (chunk_size, action_dim):
    - delta[0] = action[0] - observation.state[:action_dim]  (first target relative to current)
    - delta[i] = action[i] - action[i-1] for i > 0  (subsequent relative to previous)

    At inference time, use cumsum to convert back:
    - target[0] = current_state + delta[0]
    - target[i] = target[i-1] + delta[i]

    This representation has ~99.7% lower variance than absolute actions,
    potentially improving generalization to novel positions.
    """

    def __init__(self, dataset, action_dim: int = 6):
        """
        Args:
            dataset: Base dataset that returns action chunks
            action_dim: Number of action dimensions (default: 6 for joint control)
        """
        self.dataset = dataset
        self.action_dim = action_dim
        print(f"DeltaActionDataset: converting actions to frame-to-frame deltas")
        print(f"  Action dimension: {action_dim}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Get action chunk and current state
        actions = item['action']  # Shape: (chunk_size, action_dim) or (action_dim,)
        state = item['observation.state']  # Shape: (state_dim,)

        # Handle both chunked and single-step cases
        if actions.dim() == 1:
            # Single action: delta = action - state
            delta = actions[:self.action_dim] - state[:self.action_dim]
            item['action'] = delta
        else:
            # Action chunk: convert to deltas
            chunk_size = actions.shape[0]
            deltas = torch.zeros_like(actions)

            # First delta: relative to current state
            deltas[0] = actions[0, :self.action_dim] - state[:self.action_dim]
            if actions.shape[1] > self.action_dim:
                # Keep gripper action as absolute (last dim)
                deltas[0, self.action_dim:] = actions[0, self.action_dim:]

            # Subsequent deltas: relative to previous action
            for i in range(1, chunk_size):
                deltas[i, :self.action_dim] = actions[i, :self.action_dim] - actions[i-1, :self.action_dim]
                if actions.shape[1] > self.action_dim:
                    deltas[i, self.action_dim:] = actions[i, self.action_dim:]

            item['action'] = deltas

        return item

    @staticmethod
    def deltas_to_absolute(deltas: torch.Tensor, initial_state: torch.Tensor, action_dim: int = 6) -> torch.Tensor:
        """Convert delta actions back to absolute positions for execution.

        Args:
            deltas: Delta actions, shape (chunk_size, action_dim) or (batch, chunk_size, action_dim)
            initial_state: Current joint state, shape (action_dim,) or (batch, action_dim)
            action_dim: Number of joint dimensions (gripper kept absolute)

        Returns:
            Absolute action targets with same shape as deltas
        """
        if deltas.dim() == 2:
            # Single batch: (chunk_size, action_dim)
            absolute = torch.zeros_like(deltas)
            absolute[0, :action_dim] = initial_state[:action_dim] + deltas[0, :action_dim]
            if deltas.shape[1] > action_dim:
                absolute[0, action_dim:] = deltas[0, action_dim:]

            for i in range(1, deltas.shape[0]):
                absolute[i, :action_dim] = absolute[i-1, :action_dim] + deltas[i, :action_dim]
                if deltas.shape[1] > action_dim:
                    absolute[i, action_dim:] = deltas[i, action_dim:]

            return absolute
        else:
            # Batched: (batch, chunk_size, action_dim)
            batch_size = deltas.shape[0]
            absolute = torch.zeros_like(deltas)

            for b in range(batch_size):
                absolute[b] = DeltaActionDataset.deltas_to_absolute(
                    deltas[b], initial_state[b], action_dim
                )

            return absolute

    @classmethod
    def compute_stats(cls, base_dataset_or_repo_id, action_dim: int = 6, num_samples: int = 10000) -> dict:
        """Compute normalization statistics for delta actions.

        This uses a fast path that reads directly from parquet files when
        a repo_id string is provided, avoiding slow video decoding.

        Args:
            base_dataset_or_repo_id: Either a dataset object or HuggingFace repo ID string
            action_dim: Number of action dimensions
            num_samples: Number of samples to use for stats computation

        Returns:
            Dict with 'mean' and 'std' tensors for 'action'
        """
        import pandas as pd
        from pathlib import Path

        # Fast path: load directly from parquet if repo_id provided
        if isinstance(base_dataset_or_repo_id, str):
            from huggingface_hub import hf_hub_download, list_repo_files

            repo_id = base_dataset_or_repo_id
            print(f"  Loading delta stats from parquet files (fast path)...")

            # Find parquet files
            files = list_repo_files(repo_id, repo_type="dataset")
            parquet_files = [f for f in files if f.endswith('.parquet') and 'data' in f]

            # Download and load
            all_actions = []
            all_states = []
            for pf in parquet_files[:3]:  # Limit to first 3 chunks for speed
                local_path = hf_hub_download(repo_id, pf, repo_type="dataset")
                df = pd.read_parquet(local_path)
                all_actions.extend(df['action'].tolist())
                all_states.extend(df['observation.state'].tolist())

            actions = np.array(all_actions)
            states = np.array(all_states)

            # Compute frame-to-frame deltas
            # Group by episode and compute deltas within each
            # For simplicity, just compute consecutive deltas (ignoring episode boundaries)
            # This is approximate but good enough for normalization stats
            deltas = actions[1:, :action_dim] - actions[:-1, :action_dim]

            # Sample if too many
            if len(deltas) > num_samples:
                indices = np.random.choice(len(deltas), num_samples, replace=False)
                deltas = deltas[indices]

            mean = torch.tensor(deltas.mean(axis=0), dtype=torch.float32)
            std = torch.tensor(deltas.std(axis=0), dtype=torch.float32)
            std = torch.clamp(std, min=1e-6)

            # Pad to full action dim if needed (for gripper)
            if len(mean) < action_dim:
                mean = torch.cat([mean, torch.zeros(action_dim - len(mean))])
                std = torch.cat([std, torch.ones(action_dim - len(std))])

            print(f"  Delta stats (from {len(deltas)} frame-to-frame deltas):")
            print(f"    Mean: {mean.numpy()}")
            print(f"    Std:  {std.numpy()}")

            return {
                'action': {
                    'mean': mean,
                    'std': std,
                }
            }

        # Slow path: iterate through dataset
        base_dataset = base_dataset_or_repo_id
        delta_ds = cls(base_dataset, action_dim=action_dim)

        # Sample deltas
        indices = torch.randperm(len(delta_ds))[:num_samples]
        deltas = []
        for idx in indices:
            item = delta_ds[idx.item()]
            deltas.append(item['action'])

        deltas = torch.stack(deltas)

        # Handle chunked vs single
        if deltas.dim() == 3:
            # (num_samples, chunk_size, action_dim) -> flatten to (N, action_dim)
            deltas = deltas.reshape(-1, deltas.shape[-1])

        mean = deltas.mean(dim=0)
        std = deltas.std(dim=0)
        std = torch.clamp(std, min=1e-6)  # Avoid division by zero

        print(f"DeltaActionDataset stats (from {num_samples} samples):")
        print(f"  Mean: {mean[:action_dim].numpy()}")
        print(f"  Std:  {std[:action_dim].numpy()}")

        return {
            'action': {
                'mean': mean,
                'std': std,
            }
        }


class SubtaskChunkDataset(torch.utils.data.Dataset):
    """Truncates action chunks at subtask boundaries and adds completion progress labels.

    For each training sample at frame t in subtask S:
    - Computes `remaining` = number of frames until subtask S ends
    - action_mask: 1.0 for action indices within current subtask, 0.0 beyond
    - completion_progress: (i+1)/remaining for i < remaining, 1.0 beyond

    This enables:
    1. Clean subtask training: action loss only supervises within the current subtask
    2. Completion prediction: model learns to predict when the subtask will end

    Must be applied AFTER SubtaskDataset (needs subtask annotations loaded).

    Args:
        dataset: The underlying dataset (must already have subtask annotations applied)
        subtask_annotations: Dict mapping episode_index -> list of per-frame subtask labels
        chunk_size: Action chunk size (must match model config)
    """

    def __init__(self, dataset, subtask_annotations: dict, chunk_size: int = 100, mask_actions: bool = True):
        self.dataset = dataset
        self.subtask_annotations = {int(k): v for k, v in subtask_annotations.items()}
        self.chunk_size = chunk_size
        self.mask_actions = mask_actions
        mode = "action masking + completion head" if mask_actions else "completion head only (no action masking)"
        print(f"SubtaskChunkDataset: {mode} (chunk_size={chunk_size})")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        ep_idx = item['episode_index']
        if isinstance(ep_idx, torch.Tensor):
            ep_idx = ep_idx.item()

        frame_idx = item['frame_index']
        if isinstance(frame_idx, torch.Tensor):
            frame_idx = frame_idx.item()

        annotations = self.subtask_annotations.get(ep_idx, [])

        if frame_idx < len(annotations):
            current_subtask = annotations[frame_idx]
        else:
            current_subtask = annotations[-1] if annotations else 0

        # Count remaining frames in the current subtask
        remaining = 0
        for i in range(self.chunk_size):
            future_frame = frame_idx + i
            if future_frame < len(annotations) and annotations[future_frame] == current_subtask:
                remaining += 1
            else:
                break

        remaining = max(remaining, 1)  # At least 1 step

        # Action mask: 1 within subtask, 0 beyond
        action_mask = torch.zeros(self.chunk_size, dtype=torch.float32)
        action_mask[:remaining] = 1.0

        # Completion progress: ramps from ~0 to 1.0 within subtask, stays 1.0 after
        completion_progress = torch.ones(self.chunk_size, dtype=torch.float32)
        for i in range(remaining):
            completion_progress[i] = (i + 1) / remaining

        if self.mask_actions:
            item['action_mask'] = action_mask
        item['completion_progress'] = completion_progress

        return item


class FixedStateDataset(torch.utils.data.Dataset):
    """Replaces buggy observation.state (duplo position) with action values (commanded joints).

    The original dataset has a bug where observation.state contains the duplo block's
    freejoint position (qpos[:6]) instead of robot joint positions (qpos[7:13]).

    Since the position controller tracks targets closely at 30fps, action[0] (the
    commanded joint position at the current timestep) approximates actual joint position.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        print("FixedStateDataset: replacing observation.state with action[0] (commanded joint positions)")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        item['observation.state'] = item['action'][0].clone()
        return item


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
    scene: str = None,
    pickup_coords: bool = False,
    subtask: bool = False,
    selective_coords: bool = False,
    delta_actions: bool = False,
    blinkering: bool = False,
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
        scene: Optional scene XML filename override (e.g., "so101_with_confuser.xml")
        pickup_coords: If True, inject pickup coordinates (from block_x, block_y) into batch
            as observation.environment_state. Required for models trained with --pickup_coords.
        subtask: If True, compute and inject subtask phase (one-hot) into batch.
            Uses FK state machine: MOVE_TO_SOURCE -> PICK_UP -> MOVE_TO_DEST -> DROP

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
    if scene:
        eval_scene = scene
    elif depth_camera_names:
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

    # Detect completion head
    has_completion_head = (hasattr(policy, 'model') and
                          hasattr(policy.model, 'use_completion_head') and
                          policy.model.use_completion_head)
    COMPLETION_THRESHOLD = 2.0  # Disabled by default: completion resetting hurts open-loop performance

    # Enable/disable blinkering on model
    if blinkering and hasattr(policy, 'model') and hasattr(policy.model, 'blinkering'):
        policy.model.blinkering = True
        if verbose:
            print(f"  Blinkering ENABLED on model")
    elif hasattr(policy, 'model') and hasattr(policy.model, 'blinkering'):
        policy.model.blinkering = False

    if has_completion_head and verbose:
        print(f"  Completion head ENABLED (threshold={COMPLETION_THRESHOLD})")

    for ep in range(num_episodes):
        print(f"  Episode {ep+1}/{num_episodes}...", end=" ", flush=True)
        policy.reset()  # Reset action chunking state
        # Reset delta action accumulator
        if hasattr(policy, '_delta_prev_target'):
            delattr(policy, '_delta_prev_target')
        # If explicit block position provided, disable randomization
        fixed_pos = block_x is not None and block_y is not None
        sim_robot.reset_scene(randomize=randomize,
                              pos_range=0.0 if fixed_pos else 0.04,
                              rot_range=np.pi,
                              pos_center_x=block_x, pos_center_y=block_y)

        # Get actual block position for pickup coordinate conditioning
        pickup_coord_tensor = None
        if pickup_coords:
            try:
                duplo_body_id = mujoco.mj_name2id(sim_robot.mj_model, mujoco.mjtObj.mjOBJ_BODY, "duplo")
                actual_block_x = sim_robot.mj_data.xpos[duplo_body_id][0]
                actual_block_y = sim_robot.mj_data.xpos[duplo_body_id][1]
                # Normalize to [-1, 1] using same bounds as PickupCoordinateDataset
                x_bounds = PickupCoordinateDataset.DEFAULT_X_BOUNDS
                y_bounds = PickupCoordinateDataset.DEFAULT_Y_BOUNDS
                x_norm = 2 * (actual_block_x - x_bounds[0]) / (x_bounds[1] - x_bounds[0]) - 1
                y_norm = 2 * (actual_block_y - y_bounds[0]) / (y_bounds[1] - y_bounds[0]) - 1
                x_norm = max(-1, min(1, x_norm))
                y_norm = max(-1, min(1, y_norm))
                pickup_coord_tensor = torch.tensor([[x_norm, y_norm]], dtype=torch.float32, device=device)
                if verbose and ep == 0:
                    print(f"[pickup_coords: ({actual_block_x:.3f}, {actual_block_y:.3f}) -> ({x_norm:.2f}, {y_norm:.2f})]", end=" ")
            except Exception as e:
                if verbose:
                    print(f"[pickup_coords error: {e}]", end=" ")

        # Setup subtask state machine for this episode
        subtask_state = 0  # Start at MOVE_TO_SOURCE
        block_pos_3d = None
        bowl_pos_3d = np.array([0.217, -0.225, 0.0])  # Fixed bowl position
        ee_site_id = None
        NEAR_THRESHOLD = 0.06  # 6cm
        FAR_THRESHOLD = 0.12   # 12cm

        if subtask:
            try:
                duplo_body_id = mujoco.mj_name2id(sim_robot.mj_model, mujoco.mjtObj.mjOBJ_BODY, "duplo")
                block_pos_3d = sim_robot.mj_data.xpos[duplo_body_id].copy()
                # Find EE site
                for site_name in ["gripperframe", "gripper_site", "ee_site"]:
                    ee_site_id = mujoco.mj_name2id(sim_robot.mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                    if ee_site_id != -1:
                        break
                if verbose and ep == 0:
                    print(f"[subtask: block=({block_pos_3d[0]:.2f}, {block_pos_3d[1]:.2f})]", end=" ")
            except Exception as e:
                if verbose:
                    print(f"[subtask error: {e}]", end=" ")

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

            # Save raw state before preprocessing (needed for delta action conversion)
            raw_state = batch["observation.state"].clone()

            # Compute subtask state if enabled
            subtask_tensor = None
            if subtask and ee_site_id is not None and block_pos_3d is not None:
                # Get current EE position
                ee_pos = sim_robot.mj_data.site_xpos[ee_site_id].copy()

                # Compute distances
                dist_to_block_xy = np.linalg.norm(ee_pos[:2] - block_pos_3d[:2])
                dist_to_block_3d = np.linalg.norm(ee_pos - block_pos_3d)
                dist_to_bowl_xy = np.linalg.norm(ee_pos[:2] - bowl_pos_3d[:2])

                # Forward-only state machine
                if subtask_state == 0:  # MOVE_TO_SOURCE
                    if dist_to_block_xy < NEAR_THRESHOLD:
                        subtask_state = 1  # -> PICK_UP
                elif subtask_state == 1:  # PICK_UP
                    if dist_to_block_3d > FAR_THRESHOLD:
                        subtask_state = 2  # -> MOVE_TO_DEST
                elif subtask_state == 2:  # MOVE_TO_DEST
                    if dist_to_bowl_xy < NEAR_THRESHOLD:
                        subtask_state = 3  # -> DROP
                # subtask_state 3 (DROP) is terminal

                # Create one-hot tensor
                subtask_onehot = torch.zeros(4, dtype=torch.float32, device=device)
                subtask_onehot[subtask_state] = 1.0
                subtask_tensor = subtask_onehot.unsqueeze(0)  # (1, 4)

            # Add pickup coordinates and/or subtask to environment_state
            if pickup_coord_tensor is not None and subtask_tensor is not None:
                # Both: concatenate coords (2) + subtask (4) = 6 dims
                # If selective_coords is enabled, zero out coords during PICK_UP (1) and DROP (3)
                if selective_coords and subtask_state in (1, 3):
                    zeroed_coords = torch.zeros_like(pickup_coord_tensor)
                    batch["observation.environment_state"] = torch.cat([zeroed_coords, subtask_tensor], dim=1)
                else:
                    batch["observation.environment_state"] = torch.cat([pickup_coord_tensor, subtask_tensor], dim=1)
            elif pickup_coord_tensor is not None:
                batch["observation.environment_state"] = pickup_coord_tensor
            elif subtask_tensor is not None:
                batch["observation.environment_state"] = subtask_tensor

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
                elif has_completion_head:
                    action, progress = policy.select_action_with_completion(batch)
                    # Apply postprocessor (denormalizes action)
                    action = postprocessor(action)
                    # Reset action queue when subtask is predicted complete
                    if progress is not None and progress > COMPLETION_THRESHOLD:
                        policy.reset()
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

            # Convert delta actions to absolute if needed
            # For delta actions with chunking: deltas are relative to previous action in chunk
            # delta[0] = target[0] - initial_state
            # delta[i] = target[i] - target[i-1]
            # So we need to accumulate: target[i] = initial_state + sum(delta[0:i+1])
            if delta_actions and not is_ee_action_space:
                if step == 0 or not hasattr(policy, '_delta_prev_target'):
                    # First step: use current state as base
                    policy._delta_prev_target = raw_state.cpu().numpy().flatten()[:NUM_JOINTS].copy()

                # Accumulate delta to previous target (not current state)
                policy._delta_prev_target = policy._delta_prev_target + joint_action
                joint_action = policy._delta_prev_target.copy()

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
        checkpoint_dir: Directory containing checkpoint, or parent directory with checkpoint_* subdirs
        policy: The policy model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into

    Returns:
        Step number from checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)

    # If this is a parent directory, find the latest checkpoint subdirectory
    state_path = checkpoint_dir / "training_state.pt"
    if not state_path.exists():
        # Look for checkpoint_* subdirectories and find the latest one
        checkpoints = sorted(
            [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint_")],
            key=lambda x: int(x.name.split("_")[1])
        )
        if checkpoints:
            checkpoint_dir = checkpoints[-1]  # Use latest checkpoint
            state_path = checkpoint_dir / "training_state.pt"
            print(f"  Found latest checkpoint: {checkpoint_dir.name}")

    if state_path.exists():
        state = torch.load(state_path)
        step = state['step']

        # Load policy weights
        model_path = checkpoint_dir / "model.safetensors"
        if model_path.exists():
            from safetensors.torch import load_file
            policy_state = load_file(model_path)
            policy.load_state_dict(policy_state)
            print(f"  Loaded policy weights from {model_path}")

        if optimizer and 'optimizer_state_dict' in state:
            optimizer.load_state_dict(state['optimizer_state_dict'])

        if scheduler and 'scheduler_state_dict' in state and state['scheduler_state_dict']:
            scheduler.load_state_dict(state['scheduler_state_dict'])

        return step

    return 0
