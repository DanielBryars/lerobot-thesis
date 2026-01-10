"""
Openpi integration for SO-101 robot.

This module provides tools to:
1. Convert LeRobot datasets to openpi (RLDS) format
2. Define SO-101 robot transforms for Pi0/Pi0.5 models

Usage:
    # Convert a dataset
    python scripts/openpi/convert_lerobot_to_openpi.py danbhf/sim_pick_place_merged_40ep output/openpi_data

    # Use SO-101 transforms
    from scripts.openpi.so101_policy import make_so101_policy_config, create_transforms
    config = make_so101_policy_config()
    input_transform, output_transform = create_transforms(config)
"""

from .so101_policy import (
    SO101Config,
    SO101InputTransform,
    SO101OutputTransform,
    make_so101_policy_config,
    create_transforms,
)

__all__ = [
    "SO101Config",
    "SO101InputTransform",
    "SO101OutputTransform",
    "make_so101_policy_config",
    "create_transforms",
]
