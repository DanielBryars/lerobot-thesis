"""
SO-101 training configuration for openpi Pi0/Pi0.5.

This module defines training configs for the SO-101 robot using openpi's
training infrastructure. Based on pi0_aloha_sim config structure.

Usage:
    # Compute normalization stats first
    cd /app/openpi && uv run scripts/compute_norm_stats.py \
        --config-name=pi0_so101

    # Train
    cd /app/openpi && XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
        uv run scripts/train.py pi0_so101 \
        --exp-name=so101_pick_place \
        --num-train-steps=20000

To register these configs with openpi, add to openpi/src/openpi/training/config.py:
    from scripts.pi0.so101_config import get_so101_configs
    _CONFIGS.extend(get_so101_configs())
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

# These imports work when openpi is installed
try:
    from openpi.training.config import (
        TrainConfig,
        LeRobotDataConfig,
        AssetsConfig,
    )
    from openpi.models import pi0_config
    from openpi.training import weight_loaders
    from openpi.shared import normalize
    from openpi import transforms
    OPENPI_AVAILABLE = True
except ImportError:
    OPENPI_AVAILABLE = False
    print("Warning: openpi not installed, config validation disabled")


# SO-101 robot specifications
# Our dataset has 5 joints + 1 gripper = 6 dimensions
SO101_STATE_DIM = 6   # 5 joints + gripper
SO101_ACTION_DIM = 6  # 5 joints + gripper

# Joint names for reference (matching our dataset)
SO101_JOINT_NAMES = [
    "rotation_base",
    "pitch_shoulder",
    "pitch_elbow",
    "pitch_wrist",
    "roll_wrist",
    "gripper",
]

# Default LeRobot dataset (Pi0-ready with normalized gripper [0-1])
SO101_DEFAULT_DATASET = "danbhf/sim_pick_place_157ep_pi0"


@dataclass
class LeRobotSO101DataConfig:
    """Data configuration for SO-101 LeRobot datasets.

    This config tells openpi how to load and transform data from a LeRobot
    dataset with SO-101 robot format.
    """
    # LeRobot dataset repository ID on HuggingFace
    repo_id: str = SO101_DEFAULT_DATASET

    # Default language prompt when not provided in dataset
    default_prompt: str = "Pick up the block and place it in the bowl"

    # Whether to use delta actions (action = target - current)
    # Pi0 expects delta actions for joints, absolute for gripper
    use_delta_joint_actions: bool = True

    # Assets for normalization statistics
    assets: Optional[AssetsConfig] = None

    # Image keys in the dataset (mapped to openpi format)
    image_keys: List[str] = field(default_factory=lambda: [
        "observation.images.overhead_cam",  # overhead camera
        "observation.images.wrist_cam",     # wrist camera
    ])

    # State key in the dataset
    state_key: str = "observation.state"

    # Action key in the dataset
    action_key: str = "action"

    def create_data_transforms(self) -> Sequence:
        """Create transforms to map SO-101 LeRobot data to openpi format."""
        if not OPENPI_AVAILABLE:
            return []

        return [
            # Remap LeRobot keys to openpi format
            transforms.RepackTransform({
                # Map overhead camera to main image
                "observation.images.overhead_cam": "observation/image",
                # Map wrist camera
                "observation.images.wrist_cam": "observation/wrist_image",
                # Map state
                "observation.state": "observation/state",
                # Map action
                "action": "actions",
                # Map language instruction
                "task": "prompt",
            }),
            # Resize images to 224x224 for Pi0
            transforms.ResizeTransform(
                image_keys=["observation/image", "observation/wrist_image"],
                size=(224, 224),
            ),
        ]


def make_pi0_so101_config(
    repo_id: str = SO101_DEFAULT_DATASET,
    num_train_steps: int = 5000,  # 5k steps typically sufficient for Pi0 finetuning
    batch_size: int = 16,
    exp_name: str = "so101_experiment",
) -> "TrainConfig":
    """Create Pi0 training config for SO-101.

    Args:
        repo_id: HuggingFace LeRobot dataset ID
        num_train_steps: Number of training steps
        batch_size: Training batch size
        exp_name: Experiment name for logging

    Returns:
        TrainConfig for Pi0 training on SO-101
    """
    if not OPENPI_AVAILABLE:
        raise ImportError("openpi required for training config")

    return TrainConfig(
        name="pi0_so101",
        project_name="lerobot-thesis",
        exp_name=exp_name,

        # Pi0 model
        model=pi0_config.Pi0Config(
            action_dim=SO101_ACTION_DIM,
            action_horizon=30,  # Action chunk size (matching Ilia's config)
        ),

        # Data configuration using LeRobot loader
        data=LeRobotDataConfig(
            repo_id=repo_id,
            default_prompt="Pick up the block and place it in the bowl",
            use_delta_joint_actions=True,  # Pi0 expects delta actions for joints
        ),

        # Load base Pi0 weights
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params"
        ),

        # Training hyperparameters
        num_train_steps=num_train_steps,
        batch_size=batch_size,

        # Checkpointing
        save_interval=2000,
        log_interval=100,

        # WandB logging
        wandb_enabled=True,
    )


def make_pi05_so101_config(
    repo_id: str = SO101_DEFAULT_DATASET,
    num_train_steps: int = 5000,  # 5k steps typically sufficient for Pi0 finetuning
    batch_size: int = 8,  # Smaller batch for Pi0.5 (more memory)
    exp_name: str = "so101_pi05_experiment",
) -> "TrainConfig":
    """Create Pi0.5 training config for SO-101.

    Pi0.5 is the larger model variant requiring more GPU memory.

    Args:
        repo_id: HuggingFace LeRobot dataset ID
        num_train_steps: Number of training steps
        batch_size: Training batch size (smaller due to memory)
        exp_name: Experiment name for logging

    Returns:
        TrainConfig for Pi0.5 training on SO-101
    """
    if not OPENPI_AVAILABLE:
        raise ImportError("openpi required for training config")

    return TrainConfig(
        name="pi0_5_so101",
        project_name="lerobot-thesis",
        exp_name=exp_name,

        # Pi0.5 model (larger)
        model=pi0_config.Pi05Config(
            action_dim=SO101_ACTION_DIM,
            action_horizon=30,  # Action chunk size
        ),

        # Data configuration
        data=LeRobotDataConfig(
            repo_id=repo_id,
            default_prompt="Pick up the block and place it in the bowl",
            use_delta_joint_actions=True,  # Pi0 expects delta actions for joints
        ),

        # Load base Pi0.5 weights
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),

        # Training hyperparameters
        num_train_steps=num_train_steps,
        batch_size=batch_size,

        # Checkpointing
        save_interval=2000,
        log_interval=100,

        # WandB logging
        wandb_enabled=True,
    )


def get_so101_configs() -> list:
    """Get all SO-101 training configs.

    Returns:
        List of TrainConfig instances for registration with openpi
    """
    if not OPENPI_AVAILABLE:
        return []

    return [
        make_pi0_so101_config(),
        make_pi05_so101_config(),
    ]


# Config registry for easy access
CONFIGS = {
    "pi0_so101": make_pi0_so101_config,
    "pi0_5_so101": make_pi05_so101_config,
}


if __name__ == "__main__":
    print("SO-101 OpenPi Configs")
    print("=" * 50)
    print(f"Default dataset: {SO101_DEFAULT_DATASET}")
    print(f"State dim: {SO101_STATE_DIM}")
    print(f"Action dim: {SO101_ACTION_DIM}")
    print(f"Joint names: {SO101_JOINT_NAMES}")
    print()
    print("Available configs:")
    for name in CONFIGS:
        print(f"  - {name}")
    print()

    if OPENPI_AVAILABLE:
        print("OpenPi is installed, configs can be created.")
        config = make_pi0_so101_config()
        print(f"\nExample config: {config.name}")
        print(f"  Model: {type(config.model).__name__}")
        print(f"  Steps: {config.num_train_steps}")
        print(f"  Batch size: {config.batch_size}")
    else:
        print("OpenPi not installed. Install to create configs:")
        print("  git clone https://github.com/Physical-Intelligence/openpi.git")
        print("  cd openpi && pip install -e .")
