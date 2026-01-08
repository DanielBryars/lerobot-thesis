#!/usr/bin/env python
"""
SO-101 policy configuration for Physical Intelligence's openpi.

This module defines the input/output transforms for SO-101 robot to work with
openpi's Pi0/Pi0.5 models. Based on examples from openpi/examples/libero.

SO-101 specs:
- 6 DoF arm (rotation_base, pitch_shoulder, pitch_elbow, roll_wrist, pitch_wrist, roll_gripper_base)
- 1 gripper (parallel jaw, single value 0-1)
- Total: 7 action dimensions (6 joints + gripper)

The transforms handle:
- Proprioceptive state normalization
- Image resizing to model requirements (224x224 or 384x384)
- Action denormalization from model output to robot commands

Usage:
    from scripts.openpi.so101_policy import make_so101_policy_config

    config = make_so101_policy_config(
        action_dim=7,
        state_dim=7,
        cameras=["wrist_cam", "overhead_cam"],
    )
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class SO101Config:
    """Configuration for SO-101 robot in openpi."""

    # Robot specifications
    action_dim: int = 7  # 6 joints + gripper
    state_dim: int = 7   # 6 joint positions + gripper position

    # Joint limits (radians, from SO-101 URDF)
    joint_limits_low: List[float] = field(default_factory=lambda: [
        -3.14159,  # rotation_base
        -1.5708,   # pitch_shoulder
        -1.5708,   # pitch_elbow
        -3.14159,  # roll_wrist
        -1.5708,   # pitch_wrist
        -3.14159,  # roll_gripper_base (often treated as gripper rotation)
    ])
    joint_limits_high: List[float] = field(default_factory=lambda: [
        3.14159,   # rotation_base
        1.5708,    # pitch_shoulder
        1.5708,    # pitch_elbow
        3.14159,   # roll_wrist
        1.5708,    # pitch_wrist
        3.14159,   # roll_gripper_base
    ])

    # Gripper limits (0 = closed, 1 = open, but may vary by implementation)
    gripper_limits: Tuple[float, float] = (0.0, 1.0)

    # Camera configuration
    cameras: List[str] = field(default_factory=lambda: ["wrist_cam", "overhead_cam"])
    image_size: Tuple[int, int] = (224, 224)  # Pi0 default

    # Action scaling
    action_scale: float = 1.0  # Scale factor for action outputs

    # Language instruction
    default_language: str = "Pick up the block and place it in the bowl"


class SO101InputTransform:
    """Transform observations from SO-101 to openpi format.

    Openpi expects:
    - observation/state: proprioceptive state (normalized)
    - observation/image: main camera image (224x224 or 384x384)
    - observation/wrist_image: wrist camera image (optional)
    - prompt: language instruction string
    """

    def __init__(self, config: SO101Config):
        self.config = config

        # Compute normalization parameters from joint limits
        self.state_low = np.array(config.joint_limits_low + [config.gripper_limits[0]])
        self.state_high = np.array(config.joint_limits_high + [config.gripper_limits[1]])
        self.state_mean = (self.state_low + self.state_high) / 2
        self.state_range = (self.state_high - self.state_low) / 2

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state to [-1, 1] range."""
        return (state - self.state_mean) / self.state_range

    def __call__(
        self,
        state: np.ndarray,
        images: Dict[str, np.ndarray],
        language: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Transform SO-101 observation to openpi format.

        Args:
            state: Joint positions + gripper state [7]
            images: Dict of camera name -> image (H, W, C) uint8
            language: Optional language instruction

        Returns:
            Dict with keys:
                observation/state: Normalized state [7]
                observation/image: Main camera image (224, 224, 3)
                observation/wrist_image: Wrist camera image if available
                prompt: Language instruction string
        """
        import cv2

        result = {}

        # Normalize proprioceptive state
        result["observation/state"] = self.normalize_state(state).astype(np.float32)

        # Process images
        h, w = self.config.image_size

        # Main image (overhead camera preferred)
        if "overhead_cam" in images:
            main_image = images["overhead_cam"]
        elif len(images) > 0:
            # Use first available camera
            main_image = list(images.values())[0]
        else:
            raise ValueError("No images provided")

        # Resize to target size
        main_image = cv2.resize(main_image, (w, h))
        result["observation/image"] = main_image.astype(np.uint8)

        # Wrist image if available
        if "wrist_cam" in images:
            wrist_image = cv2.resize(images["wrist_cam"], (w, h))
            result["observation/wrist_image"] = wrist_image.astype(np.uint8)

        # Language instruction
        result["prompt"] = language or self.config.default_language

        return result


class SO101OutputTransform:
    """Transform actions from openpi format to SO-101 commands.

    Openpi outputs normalized actions in [-1, 1] range.
    This transform denormalizes to actual joint angles.
    """

    def __init__(self, config: SO101Config):
        self.config = config

        # Action bounds (same as state for position control)
        self.action_low = np.array(config.joint_limits_low + [config.gripper_limits[0]])
        self.action_high = np.array(config.joint_limits_high + [config.gripper_limits[1]])
        self.action_mean = (self.action_low + self.action_high) / 2
        self.action_range = (self.action_high - self.action_low) / 2

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Denormalize action from [-1, 1] to robot command range."""
        return action * self.action_range * self.config.action_scale + self.action_mean

    def clip_action(self, action: np.ndarray) -> np.ndarray:
        """Clip action to valid joint limits."""
        return np.clip(action, self.action_low, self.action_high)

    def __call__(self, action: np.ndarray) -> np.ndarray:
        """Transform openpi action to SO-101 command.

        Args:
            action: Normalized action from model [-1, 1] x [7]

        Returns:
            Denormalized joint angles + gripper [7]
        """
        denorm_action = self.denormalize_action(action)
        return self.clip_action(denorm_action)


def make_so101_policy_config(
    action_dim: int = 7,
    state_dim: int = 7,
    cameras: Optional[List[str]] = None,
    image_size: Tuple[int, int] = (224, 224),
    language: str = "Pick up the block and place it in the bowl",
) -> SO101Config:
    """Create SO-101 policy configuration for openpi.

    Args:
        action_dim: Action dimension (default 7: 6 joints + gripper)
        state_dim: State dimension (default 7: 6 joint pos + gripper)
        cameras: List of camera names
        image_size: Target image size (224x224 for Pi0, 384x384 for Pi0.5)
        language: Default language instruction

    Returns:
        SO101Config instance
    """
    config = SO101Config(
        action_dim=action_dim,
        state_dim=state_dim,
        cameras=cameras or ["wrist_cam", "overhead_cam"],
        image_size=image_size,
        default_language=language,
    )
    return config


def create_transforms(config: SO101Config) -> Tuple[SO101InputTransform, SO101OutputTransform]:
    """Create input/output transforms for SO-101.

    Args:
        config: SO101Config instance

    Returns:
        Tuple of (input_transform, output_transform)
    """
    return SO101InputTransform(config), SO101OutputTransform(config)


if __name__ == "__main__":
    # Test the transforms
    print("Testing SO-101 policy transforms...")

    config = make_so101_policy_config()
    input_transform, output_transform = create_transforms(config)

    # Fake observation
    state = np.array([0.0, 0.3, -0.5, 0.0, 0.2, 0.0, 0.5])  # 6 joints + gripper
    images = {
        "overhead_cam": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        "wrist_cam": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
    }

    # Transform observation
    obs = input_transform(state, images, "Pick up the red block")
    print(f"Observation keys: {list(obs.keys())}")
    print(f"State shape: {obs['observation/state'].shape}")
    print(f"Image shape: {obs['observation/image'].shape}")
    print(f"Wrist image shape: {obs['observation/wrist_image'].shape}")
    print(f"Prompt: {obs['prompt']}")

    # Fake model output
    model_action = np.random.uniform(-1, 1, 7)
    robot_action = output_transform(model_action)
    print(f"\nModel action (normalized): {model_action}")
    print(f"Robot action (denormalized): {robot_action}")

    print("\nSO-101 policy transforms test passed!")
