"""Dataset bridge: LeRobot HuggingFace dataset -> Octo batch format.

Loads data directly from HuggingFace (parquet + MP4 videos).
No lerobot library dependency — uses imageio for video decoding.

Key mappings:
- observation.images.overhead_cam -> image_primary (256x256, stochastic crop + color jitter)
- observation.images.wrist_cam -> image_wrist (128x128, resize only)
- observation.state (6-dim) -> proprio
- action -> delta joint actions (converted on-the-fly)
- Language: fixed instruction via t5-base tokenizer

Observation history: T=2 frames (current + 1 previous)

Performance: Images are pre-resized during caching to minimize per-step CPU work.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm


class OctoDataset(Dataset):
    """Bridge HuggingFace LeRobot dataset to Octo-compatible batch format.

    Pre-caches all frames in memory at target resolution for fast training.
    Loads images from MP4 video files and tabular data from parquet files.
    Converts absolute actions to delta format on-the-fly.
    """

    def __init__(
        self,
        dataset_repo_id: str,
        action_horizon: int = 4,
        primary_image_size: int = 256,
        wrist_image_size: int = 128,
        use_wrist_cam: bool = True,
        use_proprio: bool = True,
        augment: bool = True,
        instruction: str = "Pick up the block and place it in the bowl",
        fix_state: bool = True,
        action_dim: int = 5,
    ):
        self.action_horizon = action_horizon
        self.primary_image_size = primary_image_size
        self.wrist_image_size = wrist_image_size
        self.use_wrist_cam = use_wrist_cam
        self.use_proprio = use_proprio
        self.augment = augment
        self.instruction = instruction
        self.fix_state = fix_state
        self.action_dim = action_dim

        # Pre-resize target: slightly larger for augmentation crop
        if augment:
            self._primary_cache_size = int(primary_image_size * 1.1)  # 282 for 256
        else:
            self._primary_cache_size = primary_image_size

        # Load and cache data
        self._load_and_cache(dataset_repo_id)

        # Pre-compute episode boundaries
        self._compute_episode_boundaries()

        # Tokenize instruction
        self._tokenize_instruction()

        print(f"OctoDataset: {len(self)} samples, {len(self.episode_starts)} episodes")
        print(f"  Action horizon: {action_horizon}")
        print(f"  Primary image: {primary_image_size}x{primary_image_size} (cached at {self._primary_cache_size}x{self._primary_cache_size})")
        if use_wrist_cam:
            print(f"  Wrist image: {wrist_image_size}x{wrist_image_size}")
        print(f"  Proprio: {'enabled' if use_proprio else 'disabled'}")
        print(f"  Augmentation: {'enabled' if augment else 'disabled'}")
        print(f"  Fix state: {'enabled' if fix_state else 'disabled'}")
        print(f"  Delta action dim: {action_dim} (gripper stays absolute)")

    def _load_and_cache(self, repo_id: str):
        """Load all data from HuggingFace dataset and cache in memory.

        Images are pre-resized to target resolution during caching to
        minimize per-step CPU work during training.
        """
        import pandas as pd
        from huggingface_hub import hf_hub_download, list_repo_files

        print(f"Loading dataset from {repo_id}...")

        # List all files
        files = list_repo_files(repo_id, repo_type="dataset")
        parquet_files = sorted([f for f in files if f.endswith('.parquet') and 'data' in f])
        video_files = sorted([f for f in files if f.endswith('.mp4')])

        # Identify camera video files
        overhead_videos = sorted([f for f in video_files if 'overhead_cam' in f])
        wrist_videos = sorted([f for f in video_files if 'wrist_cam' in f])

        print(f"  {len(parquet_files)} parquet files, {len(overhead_videos)} overhead videos, {len(wrist_videos)} wrist videos")

        if not overhead_videos:
            raise ValueError(f"No overhead camera videos found in {repo_id}")
        if self.use_wrist_cam and not wrist_videos:
            print("  WARNING: No wrist camera videos found, disabling")
            self.use_wrist_cam = False

        # Load parquet data
        print("  Loading tabular data from parquet...")
        dfs = []
        for pf in parquet_files:
            local_path = hf_hub_download(repo_id, pf, repo_type="dataset")
            dfs.append(pd.read_parquet(local_path))

        df = pd.concat(dfs, ignore_index=True)

        self.actions = np.array(df['action'].tolist(), dtype=np.float32)
        self.states = np.array(df['observation.state'].tolist(), dtype=np.float32)
        self.episode_indices = np.array(df['episode_index'].tolist(), dtype=np.int64)
        self.frame_indices = np.array(df['frame_index'].tolist(), dtype=np.int64)

        n_samples = len(self.actions)
        print(f"  {n_samples} frames, {self.actions.shape[1]} action dims")

        # Fix state bug: replace duplo position with commanded joints
        if self.fix_state:
            self.states = self.actions.copy()
            print("  State fix applied: using action values as state")

        # Load and pre-resize overhead camera images
        print(f"  Decoding + resizing overhead camera videos to {self._primary_cache_size}x{self._primary_cache_size}...")
        self.overhead_images = self._decode_and_resize_videos(
            overhead_videos, repo_id, n_samples, self._primary_cache_size
        )

        # Load and pre-resize wrist camera images
        if self.use_wrist_cam and wrist_videos:
            print(f"  Decoding + resizing wrist camera videos to {self.wrist_image_size}x{self.wrist_image_size}...")
            self.wrist_images = self._decode_and_resize_videos(
                wrist_videos, repo_id, n_samples, self.wrist_image_size
            )
        else:
            self.wrist_images = None

        # Estimate memory usage
        total_bytes = self.overhead_images.nbytes + self.actions.nbytes + self.states.nbytes
        if self.wrist_images is not None:
            total_bytes += self.wrist_images.nbytes
        print(f"  Cache size: {total_bytes / 1024**3:.2f} GB")

    def _decode_and_resize_videos(
        self, video_paths: list, repo_id: str, expected_total: int, target_size: int
    ) -> np.ndarray:
        """Download, decode, and resize all video files into a pre-allocated numpy array.

        Pre-allocates the output array to avoid doubling memory during np.stack().

        Args:
            video_paths: List of video file paths in the HF repo
            repo_id: HuggingFace repo ID
            expected_total: Expected total number of frames across all videos
            target_size: Target image size (square)

        Returns:
            np.ndarray of shape (N, target_size, target_size, 3) uint8
        """
        import imageio.v3 as iio
        import cv2
        from huggingface_hub import hf_hub_download

        # Pre-allocate output array to avoid memory doubling
        output = np.empty((expected_total, target_size, target_size, 3), dtype=np.uint8)
        frame_idx = 0

        for vp in tqdm(video_paths, desc="  Videos"):
            local_path = hf_hub_download(repo_id, vp, repo_type="dataset")
            frames = iio.imread(local_path, plugin="pyav")
            for frame in frames:
                if frame_idx < expected_total:
                    # Resize directly into pre-allocated array
                    output[frame_idx] = cv2.resize(frame, (target_size, target_size),
                                                   interpolation=cv2.INTER_LINEAR)
                    frame_idx += 1

        print(f"    Decoded and resized {frame_idx} frames (expected {expected_total})")

        if frame_idx != expected_total:
            print(f"    WARNING: Frame count mismatch! Got {frame_idx}, expected {expected_total}")
            if frame_idx < expected_total and frame_idx > 0:
                # Repeat last frame for missing ones
                for i in range(frame_idx, expected_total):
                    output[i] = output[frame_idx - 1]

        return output

    def _compute_episode_boundaries(self):
        """Pre-compute start/end indices for each episode."""
        self.episode_starts = {}
        self.episode_ends = {}

        unique_episodes = np.unique(self.episode_indices)
        for ep_idx in unique_episodes:
            mask = np.where(self.episode_indices == ep_idx)[0]
            self.episode_starts[int(ep_idx)] = int(mask[0])
            self.episode_ends[int(ep_idx)] = int(mask[-1])

    def _tokenize_instruction(self):
        """Tokenize language instruction with T5 tokenizer (matches Octo)."""
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("t5-base")
            tokens = tokenizer(
                self.instruction,
                return_tensors="np",
                padding="max_length",
                truncation=True,
                max_length=16,
            )
            self.language_tokens = tokens["input_ids"][0].astype(np.int64)
            self.language_mask = tokens["attention_mask"][0].astype(np.float32)
        except Exception as e:
            print(f"  WARNING: Could not tokenize instruction: {e}")
            self.language_tokens = np.zeros(16, dtype=np.int64)
            self.language_mask = np.zeros(16, dtype=np.float32)

    def __len__(self):
        return len(self.actions)

    def _get_history_index(self, idx: int) -> int:
        """Get index of previous frame (for T=2 history). Repeats first frame at episode start."""
        ep_idx = int(self.episode_indices[idx])
        ep_start = self.episode_starts[ep_idx]
        if idx <= ep_start:
            return idx
        return idx - 1

    def _convert_to_delta(self, actions: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Convert an action horizon chunk from absolute to delta.

        Args:
            actions: (horizon, D) absolute actions
            state: (D,) current state (joint positions)

        Returns:
            (horizon, D) delta actions (gripper stays absolute)
        """
        deltas = np.zeros_like(actions)
        ad = self.action_dim

        # First step: delta relative to current state
        deltas[0, :ad] = actions[0, :ad] - state[:ad]
        if actions.shape[1] > ad:
            deltas[0, ad:] = actions[0, ad:]  # Gripper absolute

        # Subsequent steps: delta relative to previous action
        for t in range(1, len(actions)):
            deltas[t, :ad] = actions[t, :ad] - actions[t - 1, :ad]
            if actions.shape[1] > ad:
                deltas[t, ad:] = actions[t, ad:]

        return deltas

    def _process_primary_image(self, img: np.ndarray) -> torch.Tensor:
        """Process pre-resized overhead camera image: optional crop + jitter.

        Images are already resized to _primary_cache_size during caching.
        This only does cheap augmentation (crop + brightness jitter).

        Output: (3, primary_image_size, primary_image_size) float32 in [0, 1]
        """
        # Convert to float tensor: (3, H, W) in [0, 1]
        tensor = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0

        if self.augment:
            # Random crop from _primary_cache_size to primary_image_size
            cache_sz = self._primary_cache_size
            target_sz = self.primary_image_size
            if cache_sz > target_sz:
                i = torch.randint(0, cache_sz - target_sz + 1, (1,)).item()
                j = torch.randint(0, cache_sz - target_sz + 1, (1,)).item()
                tensor = tensor[:, i:i + target_sz, j:j + target_sz]

            # Color jitter (simple: random brightness)
            brightness = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
            tensor = torch.clamp(tensor * brightness, 0, 1)

        return tensor

    def _process_wrist_image(self, img: np.ndarray) -> torch.Tensor:
        """Process pre-resized wrist camera image.

        Images are already resized to wrist_image_size during caching.
        No augmentation per Octo paper.

        Output: (3, wrist_image_size, wrist_image_size) float32 in [0, 1]
        """
        return torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, idx):
        """Return an Octo-format training sample.

        Returns dict with:
            image_primary: (2, 3, H, W) - T=2 history of primary camera
            image_wrist: (2, 3, H, W) - T=2 history of wrist camera (if enabled)
            proprio: (2, D) - T=2 history of proprioception (if enabled)
            action: (action_horizon, D) - future delta actions
            language_tokens: (L,) - tokenized instruction
            language_mask: (L,) - attention mask for instruction
            obs_pad_mask: (2,) - observation padding mask
            action_pad_mask: (horizon,) - action padding mask
        """
        ep_idx = int(self.episode_indices[idx])
        ep_end = self.episode_ends[ep_idx]
        hist_idx = self._get_history_index(idx)

        # --- Observations (T=2 history) ---
        img_curr = self._process_primary_image(self.overhead_images[idx])
        img_prev = self._process_primary_image(self.overhead_images[hist_idx])
        image_primary = torch.stack([img_prev, img_curr], dim=0)

        sample = {"image_primary": image_primary}

        if self.use_wrist_cam and self.wrist_images is not None:
            wrist_curr = self._process_wrist_image(self.wrist_images[idx])
            wrist_prev = self._process_wrist_image(self.wrist_images[hist_idx])
            sample["image_wrist"] = torch.stack([wrist_prev, wrist_curr], dim=0)

        if self.use_proprio:
            state_curr = torch.from_numpy(self.states[idx]).float()
            state_prev = torch.from_numpy(self.states[hist_idx]).float()
            sample["proprio"] = torch.stack([state_prev, state_curr], dim=0)

        # --- Actions (future horizon, converted to delta) ---
        abs_actions = []
        for h in range(self.action_horizon):
            action_idx = min(idx + h, ep_end)
            abs_actions.append(self.actions[action_idx])
        abs_actions = np.stack(abs_actions, axis=0)

        # Convert to delta actions
        current_state = self.states[idx]
        delta_actions = self._convert_to_delta(abs_actions, current_state)
        sample["action"] = torch.from_numpy(delta_actions).float()

        # Action padding mask
        action_pad = torch.ones(self.action_horizon, dtype=torch.float32)
        remaining = ep_end - idx + 1
        if remaining < self.action_horizon:
            action_pad[remaining:] = 0.0
        sample["action_pad_mask"] = action_pad

        # --- Language ---
        sample["language_tokens"] = torch.from_numpy(self.language_tokens.copy())
        sample["language_mask"] = torch.from_numpy(self.language_mask.copy())

        # --- Observation pad mask ---
        obs_pad = torch.ones(2, dtype=torch.float32)
        if idx == self.episode_starts[ep_idx]:
            obs_pad[0] = 0.0
        sample["obs_pad_mask"] = obs_pad

        return sample

    def compute_action_stats(self) -> dict:
        """Compute mean/std of delta actions for normalization."""
        all_deltas = []
        unique_eps = np.unique(self.episode_indices)
        for ep_idx in unique_eps:
            ep_start = self.episode_starts[int(ep_idx)]
            ep_end = self.episode_ends[int(ep_idx)]
            ep_actions = self.actions[ep_start:ep_end + 1]
            ep_states = self.states[ep_start:ep_end + 1]

            # Convert full episode to deltas
            deltas = np.zeros_like(ep_actions)
            ad = self.action_dim
            deltas[0, :ad] = ep_actions[0, :ad] - ep_states[0, :ad]
            if ep_actions.shape[1] > ad:
                deltas[0, ad:] = ep_actions[0, ad:]
            for t in range(1, len(ep_actions)):
                deltas[t, :ad] = ep_actions[t, :ad] - ep_actions[t - 1, :ad]
                if ep_actions.shape[1] > ad:
                    deltas[t, ad:] = ep_actions[t, ad:]
            all_deltas.append(deltas)

        all_deltas = np.concatenate(all_deltas, axis=0)
        mean = all_deltas.mean(axis=0)
        std = all_deltas.std(axis=0)
        std = np.maximum(std, 1e-6)
        return {
            "action_mean": torch.from_numpy(mean).float(),
            "action_std": torch.from_numpy(std).float(),
        }

    def compute_proprio_stats(self) -> dict:
        """Compute mean/std of proprioception for normalization."""
        mean = self.states.mean(axis=0)
        std = self.states.std(axis=0)
        std = np.maximum(std, 1e-6)
        return {
            "proprio_mean": torch.from_numpy(mean).float(),
            "proprio_std": torch.from_numpy(std).float(),
        }
