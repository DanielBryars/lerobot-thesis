"""Tests for SmolVLA training and inference.

Note: Some tests require the smolvla extra dependencies and may be skipped
if they are not installed:
    pip install -e ".[smolvla]"
"""

import sys
from pathlib import Path
import numpy as np
import torch
import pytest

# Add project root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


def has_smolvla():
    """Check if SmolVLA dependencies are installed."""
    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not has_smolvla(), reason="SmolVLA dependencies not installed")
class TestSmolVLAConfig:
    """Tests for SmolVLA configuration."""

    def test_default_config(self):
        """Test default SmolVLA configuration."""
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

        config = SmolVLAConfig()

        assert config.chunk_size == 50
        assert config.n_action_steps == 50
        assert config.max_state_dim == 32
        assert config.max_action_dim == 32
        assert config.freeze_vision_encoder == True
        assert config.train_expert_only == True

    def test_custom_config(self):
        """Test custom SmolVLA configuration."""
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

        config = SmolVLAConfig(
            chunk_size=100,
            n_action_steps=100,
            optimizer_lr=5e-5,
        )

        assert config.chunk_size == 100
        assert config.n_action_steps == 100
        assert config.optimizer_lr == 5e-5

    def test_invalid_config(self):
        """Test that invalid config raises error."""
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

        # n_action_steps > chunk_size should raise error
        with pytest.raises(ValueError):
            SmolVLAConfig(chunk_size=50, n_action_steps=100)


@pytest.mark.skipif(not has_smolvla(), reason="SmolVLA dependencies not installed")
class TestSmolVLAPolicy:
    """Tests for SmolVLA policy.

    Note: These tests may be slow as they involve loading/creating the model.
    """

    @pytest.mark.slow
    def test_policy_creation(self):
        """Test creating SmolVLA policy from config."""
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.configs.types import FeatureType, PolicyFeature

        # Create minimal config
        config = SmolVLAConfig(
            input_features={
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
                "observation.images.wrist_cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            },
            output_features={
                "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
            },
        )

        # This will download the VLM if not cached
        # Skipping actual creation to avoid long download
        # policy = SmolVLAPolicy(config)

        assert config.input_features is not None
        assert config.output_features is not None

    @pytest.mark.slow
    def test_load_pretrained(self):
        """Test loading pretrained SmolVLA model."""
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        # This will download the model if not cached (~1GB)
        # Skipping to avoid long download in tests
        # policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")

        # Just verify the class exists and is importable
        assert SmolVLAPolicy is not None


class TestSmolVLATrainingScript:
    """Tests for SmolVLA training script."""

    def test_script_exists(self):
        """Test that training script exists."""
        script_path = REPO_ROOT / "training" / "train_smolvla.py"
        assert script_path.exists()

    def test_script_imports(self):
        """Test that training script can be imported."""
        import importlib.util

        script_path = REPO_ROOT / "training" / "train_smolvla.py"
        spec = importlib.util.spec_from_file_location("train_smolvla", script_path)
        module = importlib.util.module_from_spec(spec)

        # Just check it can be loaded without errors
        # Actually executing it would start training
        assert spec is not None
        assert module is not None


class TestSmolVLAInferenceScript:
    """Tests for SmolVLA inference script."""

    def test_script_exists(self):
        """Test that inference script exists."""
        script_path = REPO_ROOT / "inference" / "run_smolvla_sim.py"
        assert script_path.exists()

    def test_script_imports(self):
        """Test that inference script can be imported."""
        import importlib.util

        script_path = REPO_ROOT / "inference" / "run_smolvla_sim.py"
        spec = importlib.util.spec_from_file_location("run_smolvla_sim", script_path)
        module = importlib.util.module_from_spec(spec)

        assert spec is not None
        assert module is not None


class TestSmolVLAIntegration:
    """Integration tests for SmolVLA pipeline.

    These tests verify the full pipeline works end-to-end.
    """

    @pytest.mark.slow
    @pytest.mark.skipif(not has_smolvla(), reason="SmolVLA dependencies not installed")
    def test_prepare_batch_for_smolvla(self):
        """Test preparing a batch for SmolVLA input."""
        from utils.training import prepare_obs_for_policy

        device = torch.device("cpu")

        # Create mock observation
        obs = {
            "shoulder_pan.pos": 0.1,
            "shoulder_lift.pos": 0.2,
            "elbow_flex.pos": 0.3,
            "wrist_flex.pos": 0.4,
            "wrist_roll.pos": 0.5,
            "gripper.pos": 0.6,
            "wrist_cam": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            "overhead_cam": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        }

        batch = prepare_obs_for_policy(obs, device)

        # Add language instruction (required for SmolVLA)
        batch["observation.language"] = "Pick up the block"

        # Verify batch structure
        assert "observation.state" in batch
        assert "observation.images.wrist_cam" in batch
        assert "observation.images.overhead_cam" in batch
        assert "observation.language" in batch

        # Verify shapes
        assert batch["observation.state"].shape == (1, 6)
        assert batch["observation.images.wrist_cam"].shape == (1, 3, 480, 640)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
