#!/usr/bin/env python3
"""
Patch openpi's config.py to add SO-101 training configuration.

This script adds a custom SO-101 data config class and training config
for training Pi0/Pi0.5 on SO-101 datasets in LeRobot format.

Usage:
    python patch_openpi_so101.py

This will patch /app/openpi/src/openpi/training/config.py
"""

import sys

# The SO-101 DataConfig class to add (based on LeRobotLiberoDataConfig but with default_prompt)
SO101_DATA_CONFIG = '''
@dataclasses.dataclass(frozen=True)
class LeRobotSO101DataConfig(DataConfigFactory):
    """
    Config for training on SO-101 robot datasets in LeRobot format.
    Based on LeRobotLiberoDataConfig but with support for default_prompt.

    SO-101 is a single-arm robot with 7 DoF (6 joints + gripper).
    """

    # Default language prompt when not provided in dataset
    default_prompt: str | None = None

    # Whether to convert actions to delta format (relative to first state in chunk)
    use_delta_actions: bool = False

    # Action key in LeRobot dataset (singular "action" not plural "actions")
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Repack transform maps LeRobot dataset keys to openpi expected keys
        # These mappings match the SO-101 LeRobot dataset format
        # Note: RepackTransform expects {new_key: old_key} format
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        # Map cameras - format is {new_key: old_key}
                        "observation/image": "observation.images.overhead_cam",
                        "observation/wrist_image": "observation.images.wrist_cam",
                        # State and actions
                        "observation/state": "observation.state",
                        "actions": "action",
                        # Language prompt
                        "prompt": "task",
                    }
                )
            ]
        )

        # Use Libero policy transforms since SO-101 is also a 7-DoF single-arm robot
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # Optionally convert absolute actions to delta actions
        if self.use_delta_actions:
            # Apply delta conversion to first 5 joints, leave gripper (dim 6) as absolute
            # SO-101 has 5 arm joints + 1 gripper = 6 total dimensions
            delta_action_mask = _transforms.make_bool_mask(5, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms with optional default prompt
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )

'''

# The training config to add to _CONFIGS list
SO101_TRAIN_CONFIG = '''    # SO-101 Pick and Place config (Pi0-ready dataset with normalized gripper)
    TrainConfig(
        name="pi0_so101",
        model=pi0_config.Pi0Config(action_dim=6, action_horizon=30),
        data=LeRobotSO101DataConfig(
            repo_id="danbhf/sim_pick_place_157ep_pi0",
            default_prompt="Pick up the block and place it in the bowl",
            use_delta_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params"
        ),
        num_train_steps=5_000,
    ),
'''


def patch_config_file(config_path: str = "/app/openpi/src/openpi/training/config.py"):
    """Patch the openpi config.py to add SO-101 support."""

    print(f"Reading {config_path}...")
    with open(config_path, 'r') as f:
        content = f.read()

    # Check if already patched
    if "LeRobotSO101DataConfig" in content:
        print("Already patched! LeRobotSO101DataConfig found.")
        return True

    # Find where to insert the SO-101 DataConfig class (after LeRobotLiberoDataConfig)
    marker = "class LeRobotLiberoDataConfig(DataConfigFactory):"
    if marker not in content:
        print(f"ERROR: Could not find {marker}")
        return False

    # Find the end of LeRobotLiberoDataConfig class
    # We look for the next @dataclasses.dataclass or class definition
    import re

    # Find position after LeRobotLiberoDataConfig class ends
    # We'll insert before the next dataclass decorator
    pattern = r'(class LeRobotLiberoDataConfig\(DataConfigFactory\):.*?)((?=\n@dataclasses\.dataclass)|\nclass )'
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        # Alternative: find by looking for next class that's not part of LeRobotLiberoDataConfig
        # Insert after the create() method return statement
        libero_end = content.find("class LeRobotLiberoDataConfig")
        # Find the next class after that
        next_class = content.find("\n@dataclasses.dataclass", libero_end + 100)
        if next_class == -1:
            next_class = content.find("\nclass ", libero_end + 100)

        if next_class == -1:
            print("ERROR: Could not find insertion point for SO101DataConfig")
            return False

        insert_pos = next_class
    else:
        insert_pos = match.end(1)

    # Insert the SO101 DataConfig class
    content = content[:insert_pos] + "\n\n" + SO101_DATA_CONFIG + content[insert_pos:]
    print("Added LeRobotSO101DataConfig class")

    # Now find where to insert the training config (in _CONFIGS list)
    # Find the start of _CONFIGS list
    configs_start = content.find("_CONFIGS = [")
    if configs_start == -1:
        print("ERROR: Could not find _CONFIGS list")
        return False

    # Insert the training config right after the opening bracket
    # Find the first TrainConfig after _CONFIGS = [
    first_config = content.find("TrainConfig(", configs_start)
    if first_config == -1:
        print("ERROR: Could not find TrainConfig in _CONFIGS")
        return False

    # Find the beginning of line before first TrainConfig
    line_start = content.rfind("\n", configs_start, first_config) + 1

    # Insert our config before the first config
    content = content[:line_start] + SO101_TRAIN_CONFIG + "\n" + content[line_start:]
    print("Added pi0_so101 training config")

    # Write the patched file
    print(f"Writing patched config to {config_path}...")
    with open(config_path, 'w') as f:
        f.write(content)

    print("Patch complete!")
    return True


def verify_patch(config_path: str = "/app/openpi/src/openpi/training/config.py"):
    """Verify the patch was applied correctly."""

    print(f"\nVerifying patch in {config_path}...")
    with open(config_path, 'r') as f:
        content = f.read()

    checks = [
        ("LeRobotSO101DataConfig", "SO-101 DataConfig class"),
        ("pi0_so101", "pi0_so101 training config"),
        ("danbhf/sim_pick_place_merged_40ep", "SO-101 dataset reference"),
    ]

    all_ok = True
    for pattern, description in checks:
        if pattern in content:
            print(f"  [OK] {description}")
        else:
            print(f"  [MISSING] {description} NOT FOUND")
            all_ok = False

    return all_ok


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Patch openpi for SO-101 support")
    parser.add_argument("--config-path", default="/app/openpi/src/openpi/training/config.py",
                        help="Path to openpi config.py")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify, don't patch")
    args = parser.parse_args()

    if args.verify_only:
        success = verify_patch(args.config_path)
    else:
        success = patch_config_file(args.config_path)
        if success:
            verify_patch(args.config_path)

    sys.exit(0 if success else 1)
