#!/usr/bin/env python3
"""
Compatibility shim for lerobot API changes between 0.1.x and 0.4.x.

openpi expects lerobot.common.datasets.lerobot_dataset which doesn't exist
in lerobot 0.4.x. This script patches sys.modules to provide a stub that
satisfies imports for inference (we don't need actual data loading).

Usage:
    import lerobot_compat  # Run this before importing openpi
    from openpi.policies import policy_config  # Now works
"""

import sys
from types import ModuleType


def patch_lerobot_common():
    """Create stub modules for lerobot.common.* to satisfy openpi imports."""

    # Check if we need the patch (lerobot 0.4.x doesn't have lerobot.common)
    try:
        import lerobot.common
        return  # Old lerobot, no patch needed
    except ImportError:
        pass

    # Create stub modules
    lerobot_common = ModuleType("lerobot.common")
    lerobot_common_datasets = ModuleType("lerobot.common.datasets")
    lerobot_common_datasets_lerobot_dataset = ModuleType("lerobot.common.datasets.lerobot_dataset")

    # Create a stub LeRobotDataset class
    class LeRobotDataset:
        """Stub class - not used for inference."""
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "LeRobotDataset stub - this is a compatibility shim for inference. "
                "Actual data loading requires lerobot 0.1.x or patched openpi."
            )

    # Wire up the module hierarchy
    lerobot_common_datasets_lerobot_dataset.LeRobotDataset = LeRobotDataset
    lerobot_common_datasets.lerobot_dataset = lerobot_common_datasets_lerobot_dataset
    lerobot_common.datasets = lerobot_common_datasets

    # Register in sys.modules
    sys.modules["lerobot.common"] = lerobot_common
    sys.modules["lerobot.common.datasets"] = lerobot_common_datasets
    sys.modules["lerobot.common.datasets.lerobot_dataset"] = lerobot_common_datasets_lerobot_dataset

    print("Applied lerobot.common compatibility patch for lerobot 0.4.x")


# Auto-apply patch on import
patch_lerobot_common()
