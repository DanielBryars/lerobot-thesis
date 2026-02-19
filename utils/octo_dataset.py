"""Dataset bridge: LeRobot HuggingFace dataset -> Octo batch format.

No RLDS/TensorFlow dependency — loads directly from parquet files.

Key mappings:
- observation.images.overhead_cam -> image_primary (256x256, stochastic crop + color jitter)
- observation.images.wrist_cam -> image_wrist (128x128, resize only)
- observation.state (6-dim) -> proprio
- action (6-dim delta joints) -> target actions with configurable horizon
- Language: fixed instruction via t5-base tokenizer

Observation history: T=2 frames (current + 1 previous)
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path


class OctoDataset(Dataset):
    """Bridge LeRobot parquet dataset to Octo-compatible batch format.

    Handles:
    - Multi-frame observation history (T=2)
    - Image augmentation (stochastic crop, color jitter for primary cam)
    - Action horizon chunking
    - Language instruction tokenization

    Args:
        dataset_repo_id: HuggingFace dataset ID
        action_horizon: Number of future action steps to predict (default: 4)
        primary_image_size: Size for primary (overhead) camera image (default: 256)
        wrist_image_size: Size for wrist camera image (default: 128)
        use_wrist_cam: Whether to include wrist camera (default: True)
        use_proprio: Whether to include proprioception (default: True)
        augment: Whether to apply image augmentations (default: True)
        instruction: Language instruction string
        fix_state: Replace buggy state (duplo pos) with action[0] (commanded joints)
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
    ):
        self.action_horizon = action_horizon
        self.primary_image_size = primary_image_size
        self.wrist_image_size = wrist_image_size
        self.use_wrist_cam = use_wrist_cam
        self.use_proprio = use_proprio
        self.augment = augment
        self.instruction = instruction
        self.fix_state = fix_state

        # Load data from parquet files
        self._load_parquet(dataset_repo_id)

        # Pre-compute episode boundaries for history/horizon lookups
        self._compute_episode_boundaries()

        # Tokenize language instruction (cached, done once)
        self._tokenize_instruction()

        print(f"OctoDataset: {len(self)} samples, {len(self.episode_starts)} episodes")
        print(f"  Action horizon: {action_horizon}")
        print(f"  Primary image: {primary_image_size}x{primary_image_size}")
        if use_wrist_cam:
            print(f"  Wrist image: {wrist_image_size}x{wrist_image_size}")
        print(f"  Proprio: {'enabled' if use_proprio else 'disabled'}")
        print(f"  Augmentation: {'enabled' if augment else 'disabled'}")
        print(f"  Fix state: {'enabled' if fix_state else 'disabled'}")

    def _load_parquet(self, repo_id: str):
        """Load all data from HuggingFace parquet files into memory."""
        import pandas as pd
        from huggingface_hub import hf_hub_download, list_repo_files

        print(f"Loading dataset from {repo_id}...")
        files = list_repo_files(repo_id, repo_type="dataset")
        parquet_files = sorted([f for f in files if f.endswith('.parquet') and 'data' in f])

        dfs = []
        for pf in parquet_files:
            local_path = hf_hub_download(repo_id, pf, repo_type="dataset")
            dfs.append(pd.read_parquet(local_path))

        df = pd.concat(dfs, ignore_index=True)

        # Extract arrays
        self.actions = np.array(df['action'].tolist(), dtype=np.float32)
        self.states = np.array(df['observation.state'].tolist(), dtype=np.float32)
        self.episode_indices = np.array(df['episode_index'].tolist(), dtype=np.int64)
        self.frame_indices = np.array(df['frame_index'].tolist(), dtype=np.int64)

        # Fix state bug: replace duplo position with commanded joints
        if self.fix_state:
            self.states = self.actions.copy()
            print("  State fix applied: using action values as state")

        # Load images - stored as dicts with 'path' and 'bytes' in parquet
        self.overhead_images = self._load_images(df, 'observation.images.overhead_cam')
        if self.use_wrist_cam and 'observation.images.wrist_cam' in df.columns:
            self.wrist_images = self._load_images(df, 'observation.images.wrist_cam')
        else:
            self.wrist_images = None
            if self.use_wrist_cam:
                print("  WARNING: wrist_cam not found in dataset, disabling")
                self.use_wrist_cam = False

        print(f"  Loaded {len(self.actions)} frames, {self.actions.shape[1]} action dims")

    def _load_images(self, df, column: str) -> list:
        """Load images from parquet column (stored as bytes)."""
        from PIL import Image
        import io

        images = []
        for i, row in enumerate(df[column]):
            if isinstance(row, dict) and 'bytes' in row:
                img = Image.open(io.BytesIO(row['bytes'])).convert('RGB')
                images.append(np.array(img))
            elif isinstance(row, bytes):
                img = Image.open(io.BytesIO(row)).convert('RGB')
                images.append(np.array(img))
            else:
                # Placeholder if image loading fails
                images.append(np.zeros((480, 640, 3), dtype=np.uint8))
        return images

    def _compute_episode_boundaries(self):
        """Pre-compute start/end indices for each episode."""
        self.episode_starts = {}
        self.episode_ends = {}

        unique_episodes = np.unique(self.episode_indices)
        for ep_idx in unique_episodes:
            mask = np.where(self.episode_indices == ep_idx)[0]
            self.episode_starts[ep_idx] = mask[0]
            self.episode_ends[ep_idx] = mask[-1]

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
        """Get the index of the previous frame (for T=2 history).

        Returns the same index if this is the first frame of an episode.
        """
        ep_idx = self.episode_indices[idx]
        ep_start = self.episode_starts[ep_idx]
        if idx <= ep_start:
            return idx  # First frame: repeat current
        return idx - 1

    def _process_primary_image(self, img: np.ndarray) -> torch.Tensor:
        """Process overhead camera image: resize + optional augmentation.

        Output: (3, primary_image_size, primary_image_size) float32 in [0, 1]
        """
        # Convert to tensor [C, H, W] float [0, 1]
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        if self.augment:
            # Stochastic crop: resize to slightly larger, then random crop
            crop_scale = 1.1
            resize_size = int(self.primary_image_size * crop_scale)
            tensor = F.interpolate(tensor.unsqueeze(0), size=(resize_size, resize_size),
                                   mode='bilinear', align_corners=False)[0]
            # Random crop
            i = torch.randint(0, resize_size - self.primary_image_size + 1, (1,)).item()
            j = torch.randint(0, resize_size - self.primary_image_size + 1, (1,)).item()
            tensor = tensor[:, i:i + self.primary_image_size, j:j + self.primary_image_size]

            # Color jitter (simple: random brightness/contrast)
            brightness = 1.0 + (torch.rand(1).item() - 0.5) * 0.2  # +/- 10%
            tensor = torch.clamp(tensor * brightness, 0, 1)
        else:
            tensor = F.interpolate(tensor.unsqueeze(0),
                                   size=(self.primary_image_size, self.primary_image_size),
                                   mode='bilinear', align_corners=False)[0]

        return tensor

    def _process_wrist_image(self, img: np.ndarray) -> torch.Tensor:
        """Process wrist camera image: resize only (no crop per paper).

        Output: (3, wrist_image_size, wrist_image_size) float32 in [0, 1]
        """
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        tensor = F.interpolate(tensor.unsqueeze(0),
                               size=(self.wrist_image_size, self.wrist_image_size),
                               mode='bilinear', align_corners=False)[0]
        return tensor

    def __getitem__(self, idx):
        """Return an Octo-format training sample.

        Returns dict with:
            image_primary: (2, 3, H, W) - T=2 history of primary camera
            image_wrist: (2, 3, H, W) - T=2 history of wrist camera (if enabled)
            proprio: (2, D) - T=2 history of proprioception (if enabled)
            action: (action_horizon, D) - future delta actions
            language_tokens: (L,) - tokenized instruction
            language_mask: (L,) - attention mask for instruction
            pad_mask_dict: dict of padding masks for each observation
        """
        ep_idx = self.episode_indices[idx]
        ep_end = self.episode_ends[ep_idx]
        hist_idx = self._get_history_index(idx)

        # --- Observations (T=2 history) ---

        # Primary camera
        img_curr = self._process_primary_image(self.overhead_images[idx])
        img_prev = self._process_primary_image(self.overhead_images[hist_idx])
        image_primary = torch.stack([img_prev, img_curr], dim=0)  # (2, 3, H, W)

        sample = {
            "image_primary": image_primary,
        }

        # Wrist camera
        if self.use_wrist_cam and self.wrist_images is not None:
            wrist_curr = self._process_wrist_image(self.wrist_images[idx])
            wrist_prev = self._process_wrist_image(self.wrist_images[hist_idx])
            sample["image_wrist"] = torch.stack([wrist_prev, wrist_curr], dim=0)

        # Proprioception
        if self.use_proprio:
            state_curr = torch.from_numpy(self.states[idx]).float()
            state_prev = torch.from_numpy(self.states[hist_idx]).float()
            sample["proprio"] = torch.stack([state_prev, state_curr], dim=0)  # (2, D)

        # --- Actions (future horizon) ---
        actions = []
        for h in range(self.action_horizon):
            action_idx = min(idx + h, ep_end)
            actions.append(self.actions[action_idx])
        sample["action"] = torch.from_numpy(np.stack(actions, axis=0)).float()  # (horizon, D)

        # Action padding mask (1 for valid actions, 0 for padding at episode end)
        action_pad = torch.ones(self.action_horizon, dtype=torch.float32)
        remaining = ep_end - idx + 1
        if remaining < self.action_horizon:
            action_pad[remaining:] = 0.0
        sample["action_pad_mask"] = action_pad

        # --- Language ---
        sample["language_tokens"] = torch.from_numpy(self.language_tokens.copy())
        sample["language_mask"] = torch.from_numpy(self.language_mask.copy())

        # --- Observation pad masks ---
        # For T=2 history, first frame may be padded (repeated) at episode start
        obs_pad = torch.ones(2, dtype=torch.float32)
        if idx == self.episode_starts[ep_idx]:
            obs_pad[0] = 0.0  # First history frame is padded (repeated current)
        sample["obs_pad_mask"] = obs_pad

        return sample

    def compute_action_stats(self) -> dict:
        """Compute mean/std of actions for normalization."""
        mean = self.actions.mean(axis=0)
        std = self.actions.std(axis=0)
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
