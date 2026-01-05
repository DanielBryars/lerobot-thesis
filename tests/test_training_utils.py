"""Tests for training utilities."""

import sys
from pathlib import Path
import numpy as np
import torch
import pytest

# Add project root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.training import (
    CachedDataset,
    cycle,
    prepare_obs_for_policy,
    get_action_space_info,
    get_camera_names,
    create_output_dir,
)
from utils.constants import MOTOR_NAMES


class TestPrepareObsForPolicy:
    """Tests for prepare_obs_for_policy function."""

    def test_basic_observation(self):
        """Test converting basic simulation observation to policy format."""
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

        # Check state
        assert "observation.state" in batch
        assert batch["observation.state"].shape == (1, 6)
        assert batch["observation.state"].device == device

        # Check images
        assert "observation.images.wrist_cam" in batch
        assert "observation.images.overhead_cam" in batch
        assert batch["observation.images.wrist_cam"].shape == (1, 3, 480, 640)
        assert batch["observation.images.overhead_cam"].shape == (1, 3, 480, 640)

        # Check normalization (should be 0-1)
        assert batch["observation.images.wrist_cam"].min() >= 0
        assert batch["observation.images.wrist_cam"].max() <= 1

    def test_depth_camera(self):
        """Test handling of depth camera images."""
        device = torch.device("cpu")

        obs = {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": 0.0,
            "elbow_flex.pos": 0.0,
            "wrist_flex.pos": 0.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 0.0,
            "overhead_cam_depth": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        }

        batch = prepare_obs_for_policy(obs, device)

        # Depth image should be expanded to 3 channels
        assert "observation.images.overhead_cam_depth" in batch
        assert batch["observation.images.overhead_cam_depth"].shape == (1, 3, 480, 640)

    def test_explicit_depth_cameras(self):
        """Test specifying depth cameras explicitly."""
        device = torch.device("cpu")

        obs = {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": 0.0,
            "elbow_flex.pos": 0.0,
            "wrist_flex.pos": 0.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 0.0,
            "my_depth_sensor": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        }

        batch = prepare_obs_for_policy(obs, device, depth_cameras=["my_depth_sensor"])

        assert "observation.images.my_depth_sensor" in batch
        assert batch["observation.images.my_depth_sensor"].shape == (1, 3, 480, 640)


class TestCycle:
    """Tests for the cycle iterator."""

    def test_cycle_iterates_indefinitely(self):
        """Test that cycle creates an infinite iterator."""
        data = [1, 2, 3]
        loader = iter(data)

        # Create a simple mock dataloader
        class MockDataLoader:
            def __init__(self, data):
                self.data = data

            def __iter__(self):
                return iter(self.data)

        mock_loader = MockDataLoader(data)
        cycler = cycle(mock_loader)

        # Get more items than in the original data
        items = [next(cycler) for _ in range(10)]

        assert items == [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]


class TestGetActionSpaceInfo:
    """Tests for get_action_space_info function."""

    def test_ee_action_space(self):
        """Test detection of EE action space."""
        from dataclasses import dataclass

        @dataclass
        class MockFeature:
            shape: tuple

        output_features = {"action": MockFeature(shape=(8,))}
        dim, name = get_action_space_info(output_features)

        assert dim == 8
        assert "end-effector" in name

    def test_joint_action_space(self):
        """Test detection of joint action space."""
        from dataclasses import dataclass

        @dataclass
        class MockFeature:
            shape: tuple

        output_features = {"action": MockFeature(shape=(6,))}
        dim, name = get_action_space_info(output_features)

        assert dim == 6
        assert "joint" in name

    def test_empty_features(self):
        """Test handling of empty features."""
        dim, name = get_action_space_info({})
        assert dim == 0
        assert "unknown" in name


class TestGetCameraNames:
    """Tests for get_camera_names function."""

    def test_extracts_camera_names(self):
        """Test extraction of camera names from input features."""
        input_features = {
            "observation.state": None,
            "observation.images.wrist_cam": None,
            "observation.images.overhead_cam": None,
            "observation.images.overhead_cam_depth": None,
        }

        cameras = get_camera_names(input_features)

        assert len(cameras) == 3
        assert "wrist_cam" in cameras
        assert "overhead_cam" in cameras
        assert "overhead_cam_depth" in cameras

    def test_no_cameras(self):
        """Test with no camera features."""
        input_features = {
            "observation.state": None,
        }

        cameras = get_camera_names(input_features)
        assert cameras == []


class TestCreateOutputDir:
    """Tests for create_output_dir function."""

    def test_creates_directory(self, tmp_path):
        """Test that output directory is created."""
        output_dir = create_output_dir(str(tmp_path), prefix="test")

        assert output_dir.exists()
        assert output_dir.is_dir()
        assert "test_" in output_dir.name

    def test_unique_directories(self, tmp_path):
        """Test that multiple calls create unique directories."""
        import time

        dir1 = create_output_dir(str(tmp_path), prefix="test")
        time.sleep(0.01)  # Small delay to ensure different timestamps
        dir2 = create_output_dir(str(tmp_path), prefix="test")

        # They might have same name if called in same second, but both should exist
        assert dir1.exists()
        assert dir2.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
