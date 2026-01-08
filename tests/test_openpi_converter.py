"""Unit tests for openpi conversion functions.

Run with: python tests/test_openpi_converter.py
Or: pytest tests/test_openpi_converter.py -v
"""

import unittest
import tempfile
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSO101Policy(unittest.TestCase):
    """Test SO101 policy transforms."""

    def setUp(self):
        """Import the module (done here to handle import errors gracefully)."""
        from scripts.openpi.so101_policy import (
            SO101Config,
            SO101InputTransform,
            SO101OutputTransform,
            make_so101_policy_config,
            create_transforms,
        )
        self.SO101Config = SO101Config
        self.SO101InputTransform = SO101InputTransform
        self.SO101OutputTransform = SO101OutputTransform
        self.make_so101_policy_config = make_so101_policy_config
        self.create_transforms = create_transforms

    def test_config_defaults(self):
        """Test default configuration values."""
        config = self.SO101Config()
        self.assertEqual(config.action_dim, 7)
        self.assertEqual(config.state_dim, 7)
        self.assertEqual(len(config.joint_limits_low), 6)
        self.assertEqual(len(config.joint_limits_high), 6)
        self.assertEqual(config.image_size, (224, 224))

    def test_make_config(self):
        """Test config factory function."""
        config = self.make_so101_policy_config(
            action_dim=7,
            state_dim=7,
            cameras=["cam1", "cam2"],
            image_size=(384, 384),
        )
        self.assertEqual(config.cameras, ["cam1", "cam2"])
        self.assertEqual(config.image_size, (384, 384))

    def test_input_transform_state_normalization(self):
        """Test state normalization to [-1, 1] range."""
        config = self.SO101Config()
        transform = self.SO101InputTransform(config)

        # Test center state (should normalize to ~0)
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
        normalized = transform.normalize_state(state)

        # State should be in [-1, 1] range
        self.assertTrue(np.all(normalized >= -1.1))  # Small tolerance
        self.assertTrue(np.all(normalized <= 1.1))

    def test_input_transform_call(self):
        """Test full input transform."""
        config = self.SO101Config()
        transform = self.SO101InputTransform(config)

        state = np.array([0.0, 0.3, -0.5, 0.0, 0.2, 0.0, 0.5])
        images = {
            "overhead_cam": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            "wrist_cam": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        }

        result = transform(state, images, "Test instruction")

        # Check keys
        self.assertIn("observation/state", result)
        self.assertIn("observation/image", result)
        self.assertIn("observation/wrist_image", result)
        self.assertIn("prompt", result)

        # Check shapes
        self.assertEqual(result["observation/state"].shape, (7,))
        self.assertEqual(result["observation/image"].shape, (224, 224, 3))
        self.assertEqual(result["observation/wrist_image"].shape, (224, 224, 3))
        self.assertEqual(result["prompt"], "Test instruction")

    def test_input_transform_no_wrist(self):
        """Test input transform without wrist camera."""
        config = self.SO101Config()
        transform = self.SO101InputTransform(config)

        state = np.zeros(7)
        images = {
            "overhead_cam": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        }

        result = transform(state, images)

        self.assertIn("observation/image", result)
        self.assertNotIn("observation/wrist_image", result)

    def test_output_transform_denormalization(self):
        """Test action denormalization from [-1, 1]."""
        config = self.SO101Config()
        transform = self.SO101OutputTransform(config)

        # Test normalized action at limits
        action_min = np.array([-1.0] * 7)
        action_max = np.array([1.0] * 7)
        action_zero = np.array([0.0] * 7)

        denorm_min = transform(action_min)
        denorm_max = transform(action_max)
        denorm_zero = transform(action_zero)

        # Should be within joint limits
        expected_low = np.array(config.joint_limits_low + [config.gripper_limits[0]])
        expected_high = np.array(config.joint_limits_high + [config.gripper_limits[1]])

        np.testing.assert_allclose(denorm_min, expected_low, atol=1e-5)
        np.testing.assert_allclose(denorm_max, expected_high, atol=1e-5)

    def test_output_transform_clipping(self):
        """Test action clipping to valid range."""
        config = self.SO101Config()
        transform = self.SO101OutputTransform(config)

        # Action beyond limits should be clipped
        action_extreme = np.array([2.0] * 7)  # Beyond [-1, 1]
        result = transform(action_extreme)

        # Result should be clipped to max limits
        expected_high = np.array(config.joint_limits_high + [config.gripper_limits[1]])
        np.testing.assert_allclose(result, expected_high, atol=1e-5)

    def test_create_transforms(self):
        """Test transform creation utility."""
        config = self.SO101Config()
        input_t, output_t = self.create_transforms(config)

        self.assertIsInstance(input_t, self.SO101InputTransform)
        self.assertIsInstance(output_t, self.SO101OutputTransform)


class TestConvertStepToRLDS(unittest.TestCase):
    """Test step conversion to RLDS format."""

    def setUp(self):
        """Import the converter module."""
        from scripts.openpi.convert_lerobot_to_openpi import convert_step_to_rlds
        self.convert_step_to_rlds = convert_step_to_rlds

    def test_basic_conversion(self):
        """Test basic step conversion."""
        step = {
            "observation.state": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            "observation.images.overhead_cam": np.random.randint(
                0, 255, (480, 640, 3), dtype=np.uint8
            ),
            "action": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
        }

        result = self.convert_step_to_rlds(
            step=step,
            main_camera="overhead_cam",
            wrist_camera=None,
            language="Test task",
            is_first=True,
            is_last=False,
        )

        # Check required fields
        self.assertIn("observation/state", result)
        self.assertIn("observation/image", result)
        self.assertIn("action", result)
        self.assertIn("language_instruction", result)
        self.assertIn("is_first", result)
        self.assertIn("is_last", result)
        self.assertIn("is_terminal", result)

        # Check values
        self.assertEqual(result["language_instruction"], "Test task")
        self.assertTrue(result["is_first"])
        self.assertFalse(result["is_last"])
        self.assertFalse(result["is_terminal"])

    def test_conversion_with_wrist(self):
        """Test conversion with wrist camera."""
        step = {
            "observation.state": np.zeros(7),
            "observation.images.overhead_cam": np.zeros((480, 640, 3), dtype=np.uint8),
            "observation.images.wrist_cam": np.zeros((480, 640, 3), dtype=np.uint8),
            "action": np.zeros(7),
        }

        result = self.convert_step_to_rlds(
            step=step,
            main_camera="overhead_cam",
            wrist_camera="wrist_cam",
            language="Test",
            is_first=False,
            is_last=True,
        )

        self.assertIn("observation/wrist_image", result)
        self.assertTrue(result["is_last"])
        self.assertTrue(result["is_terminal"])

    def test_tensor_conversion(self):
        """Test conversion handles torch tensors."""
        import torch

        step = {
            "observation.state": torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            "observation.images.cam": torch.randint(0, 255, (480, 640, 3), dtype=torch.uint8),
            "action": torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
        }

        result = self.convert_step_to_rlds(
            step=step,
            main_camera="cam",
            wrist_camera=None,
            language="Test",
            is_first=True,
            is_last=True,
        )

        # Should convert to numpy
        self.assertIsInstance(result["observation/state"], np.ndarray)
        self.assertIsInstance(result["action"], np.ndarray)

    def test_missing_camera_raises(self):
        """Test that missing camera raises error."""
        step = {
            "observation.state": np.zeros(7),
            "observation.images.other_cam": np.zeros((480, 640, 3), dtype=np.uint8),
            "action": np.zeros(7),
        }

        with self.assertRaises(ValueError) as ctx:
            self.convert_step_to_rlds(
                step=step,
                main_camera="missing_cam",
                wrist_camera=None,
                language="Test",
                is_first=True,
                is_last=True,
            )
        self.assertIn("missing_cam", str(ctx.exception))


class TestExtractCameras(unittest.TestCase):
    """Test camera extraction from features."""

    def test_extract_rgb_cameras(self):
        """Test extracting RGB cameras from features."""
        from scripts.openpi.convert_lerobot_to_openpi import extract_cameras_from_features

        features = {
            "observation.state": {},
            "observation.images.overhead_cam": {},
            "observation.images.wrist_cam": {},
            "observation.images.overhead_cam_depth": {},
            "action": {},
        }

        cameras = extract_cameras_from_features(features)

        self.assertIn("overhead_cam", cameras)
        self.assertIn("wrist_cam", cameras)
        self.assertNotIn("overhead_cam_depth", cameras)


class TestSaveEpisodeAsNpz(unittest.TestCase):
    """Test NPZ episode saving."""

    def setUp(self):
        """Create temp directory for test outputs."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp directory."""
        shutil.rmtree(self.temp_dir)

    def test_save_episode(self):
        """Test saving episode as NPZ."""
        from scripts.openpi.convert_lerobot_to_openpi import save_episode_as_npz

        episode_steps = [
            {
                "observation/state": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
                "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                "action": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
                "is_first": True,
                "is_last": False,
                "is_terminal": False,
                "language_instruction": "Test task",
            },
            {
                "observation/state": np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
                "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                "action": np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
                "is_first": False,
                "is_last": True,
                "is_terminal": True,
                "language_instruction": "Test task",
            },
        ]

        output_path = Path(self.temp_dir)
        save_episode_as_npz(episode_steps, output_path, episode_idx=0)

        # Check file was created
        npz_path = output_path / "episode_000000.npz"
        self.assertTrue(npz_path.exists())

        # Load and verify
        data = np.load(str(npz_path))
        self.assertEqual(data["observation/state"].shape, (2, 7))
        self.assertEqual(data["observation/image"].shape, (2, 224, 224, 3))
        self.assertEqual(data["action"].shape, (2, 7))
        self.assertEqual(data["is_first"].tolist(), [True, False])
        self.assertEqual(data["is_last"].tolist(), [False, True])


class TestIntegration(unittest.TestCase):
    """Integration tests (require actual dataset - skipped in CI)."""

    @unittest.skip("Integration test - requires dataset download")
    def test_convert_real_dataset(self):
        """Test converting a real LeRobot dataset."""
        from scripts.openpi.convert_lerobot_to_openpi import convert_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            stats = convert_dataset(
                repo_id="danbhf/sim_pick_place_merged_40ep",
                output_dir=temp_dir,
                max_episodes=2,  # Only convert 2 episodes for speed
            )

            self.assertGreater(stats["total_episodes"], 0)
            self.assertGreater(stats["total_steps"], 0)


if __name__ == "__main__":
    unittest.main()
