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

# Imports to add at the top of config.py
SO101_IMPORTS = '''import numpy as np
import einops
'''

# SO-101 Input/Output transforms (pad 6-dim to model's action_dim, typically 32)
SO101_POLICY = '''
def _so101_parse_image(image) -> np.ndarray:
    """Parse image to correct format for SO-101."""
    if image is None:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class SO101Inputs(_transforms.DataTransformFn):
    """Transform inputs for SO-101 robot (6 DoF: 5 joints + gripper)."""
    # Model action dimension (typically 32 for Pi0 base)
    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # Pad state from 6 to model action_dim
        state = _transforms.pad_to_dim(data["observation/state"], self.action_dim)

        # Get images - we map overhead_cam to base, wrist_cam to left_wrist
        # For SO-101 we only have 2 cameras, so right_wrist uses overhead as fallback
        base_image = _so101_parse_image(data.get("observation/image"))
        wrist_image = _so101_parse_image(data.get("observation/wrist_image", data.get("observation/image")))

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": base_image,  # Use base image as fallback
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Pad actions to model action_dim (only during training)
        if "actions" in data:
            actions = _transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class SO101Outputs(_transforms.DataTransformFn):
    """Transform outputs back to SO-101's 6-dim action space."""
    def __call__(self, data: dict) -> dict:
        # Only return the first 6 actions (5 joints + gripper)
        return {"actions": np.asarray(data["actions"][:, :6])}

'''

# The SO-101 DataConfig class to add
SO101_DATA_CONFIG = '''
@dataclasses.dataclass(frozen=True)
class LeRobotSO101DataConfig(DataConfigFactory):
    """
    Config for training on SO-101 robot datasets in LeRobot format.
    SO-101 is a single-arm robot with 6 DoF (5 joints + gripper).
    Actions are padded from 6 to model's action_dim (32) during training.
    """

    # Default language prompt when not provided in dataset
    default_prompt: str | None = None

    # Whether to convert actions to delta format
    use_delta_actions: bool = False

    # Action key in LeRobot dataset
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Repack transform maps LeRobot dataset keys to openpi expected keys
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.overhead_cam",
                        "observation/wrist_image": "observation.images.wrist_cam",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "task",
                    }
                )
            ]
        )

        # SO-101 specific transforms - pad 6-dim to model's action_dim
        data_transforms = _transforms.Group(
            inputs=[SO101Inputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[SO101Outputs()],
        )

        # Optionally convert absolute actions to delta actions
        if self.use_delta_actions:
            # Apply delta conversion to first 5 joints, leave gripper as absolute
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
# NOTE: Do NOT set action_dim in model config - use default (32) to match base weights
# Actions are padded from 6 to 32 by SO101Inputs transform
SO101_TRAIN_CONFIG = '''    # SO-101 Pick and Place config (Pi0-ready dataset with normalized gripper)
    TrainConfig(
        name="pi0_so101",
        model=pi0_config.Pi0Config(action_horizon=30),  # Use default action_dim=32
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
    import re

    print(f"Reading {config_path}...")
    with open(config_path, 'r') as f:
        content = f.read()

    # Check if already patched
    if "LeRobotSO101DataConfig" in content:
        print("Already patched! LeRobotSO101DataConfig found.")
        return True

    # First, add imports at the top of the file (after existing imports)
    # Find the last import statement block
    import_insert_pos = 0
    for match in re.finditer(r'^(from |import )[^\n]+\n', content, re.MULTILINE):
        import_insert_pos = match.end()

    if import_insert_pos > 0:
        # Check if numpy/einops already imported
        if "import numpy as np" not in content:
            content = content[:import_insert_pos] + SO101_IMPORTS + content[import_insert_pos:]
            print("Added numpy and einops imports at top of file")
        else:
            # Just add einops if numpy already there
            if "import einops" not in content:
                content = content[:import_insert_pos] + "import einops\n" + content[import_insert_pos:]
                print("Added einops import at top of file")

    # Find where to insert the SO-101 DataConfig class (after LeRobotLiberoDataConfig)
    marker = "class LeRobotLiberoDataConfig(DataConfigFactory):"
    if marker not in content:
        print(f"ERROR: Could not find {marker}")
        return False

    # Find the end of LeRobotLiberoDataConfig class
    # We look for the next @dataclasses.dataclass or class definition

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

    # Insert the SO101 policy classes first (includes its own imports)
    content = content[:insert_pos] + "\n\n" + SO101_POLICY + content[insert_pos:]
    print("Added SO101Inputs/SO101Outputs classes")

    # Recalculate insert position after adding policy classes
    insert_pos = content.find("class SO101Outputs") + len("class SO101Outputs")
    insert_pos = content.find("\n\n", insert_pos) + 2

    # Insert the SO101 DataConfig class
    content = content[:insert_pos] + "\n" + SO101_DATA_CONFIG + content[insert_pos:]
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
        ("SO101Inputs", "SO-101 Inputs transform"),
        ("SO101Outputs", "SO-101 Outputs transform"),
        ("LeRobotSO101DataConfig", "SO-101 DataConfig class"),
        ("pi0_so101", "pi0_so101 training config"),
        ("danbhf/sim_pick_place_157ep_pi0", "SO-101 Pi0-ready dataset reference"),
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
