"""Unit tests for conversion functions.

Run with: python tests/test_conversions.py
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.constants import (
    SIM_ACTION_LOW,
    SIM_ACTION_HIGH,
    NUM_JOINTS,
    GRIPPER_IDX,
)
from utils.conversions import (
    radians_to_normalized,
    normalized_to_radians,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    gripper_normalized_to_radians,
    clip_joints_to_limits,
)


class TestRadiansNormalizedConversion(unittest.TestCase):
    """Test radians <-> normalized conversions."""

    def test_roundtrip_zeros(self):
        """Converting zeros should roundtrip correctly."""
        radians = np.zeros(NUM_JOINTS)
        normalized = radians_to_normalized(radians)
        back = normalized_to_radians(normalized)
        np.testing.assert_allclose(back, radians, atol=1e-6)

    def test_roundtrip_random(self):
        """Random values in valid range should roundtrip."""
        np.random.seed(42)
        for _ in range(10):
            # Generate random radians in valid range
            t = np.random.random(NUM_JOINTS)
            radians = SIM_ACTION_LOW + t * (SIM_ACTION_HIGH - SIM_ACTION_LOW)

            normalized = radians_to_normalized(radians)
            back = normalized_to_radians(normalized)
            np.testing.assert_allclose(back, radians, atol=1e-5)

    def test_min_values(self):
        """Minimum radians should map to -100 (joints) or 0 (gripper)."""
        normalized = radians_to_normalized(SIM_ACTION_LOW)

        for i in range(NUM_JOINTS):
            if i == GRIPPER_IDX:
                self.assertAlmostEqual(normalized[i], 0.0, places=5)
            else:
                self.assertAlmostEqual(normalized[i], -100.0, places=5)

    def test_max_values(self):
        """Maximum radians should map to +100."""
        normalized = radians_to_normalized(SIM_ACTION_HIGH)

        for i in range(NUM_JOINTS):
            self.assertAlmostEqual(normalized[i], 100.0, places=5)

    def test_center_values(self):
        """Center of range should map to 0 (joints) or 50 (gripper)."""
        center = (SIM_ACTION_LOW + SIM_ACTION_HIGH) / 2
        normalized = radians_to_normalized(center)

        for i in range(NUM_JOINTS):
            if i == GRIPPER_IDX:
                self.assertAlmostEqual(normalized[i], 50.0, places=5)
            else:
                self.assertAlmostEqual(normalized[i], 0.0, places=5)

    def test_wrong_length_raises(self):
        """Wrong input length should raise ValueError."""
        with self.assertRaises(ValueError):
            radians_to_normalized(np.zeros(5))
        with self.assertRaises(ValueError):
            normalized_to_radians(np.zeros(7))


class TestQuaternionToRotationMatrix(unittest.TestCase):
    """Test quaternion to rotation matrix conversion."""

    def test_identity_quaternion(self):
        """Identity quaternion [1, 0, 0, 0] should give identity matrix."""
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        R = quaternion_to_rotation_matrix(quat)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_180_rotation_z(self):
        """180 degree rotation around Z axis."""
        quat = np.array([0.0, 0.0, 0.0, 1.0])  # 180 deg around Z
        R = quaternion_to_rotation_matrix(quat)
        expected = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
        ], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_90_rotation_x(self):
        """90 degree rotation around X axis."""
        angle = np.pi / 2
        quat = np.array([np.cos(angle/2), np.sin(angle/2), 0, 0])
        R = quaternion_to_rotation_matrix(quat)
        expected = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_orthogonality(self):
        """Resulting matrix should be orthogonal (R^T @ R = I)."""
        np.random.seed(42)
        for _ in range(10):
            quat = np.random.randn(4)
            quat = quat / np.linalg.norm(quat)
            R = quaternion_to_rotation_matrix(quat)

            # Check orthogonality
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10)

            # Check determinant is 1 (proper rotation)
            self.assertAlmostEqual(np.linalg.det(R), 1.0, places=10)

    def test_normalization(self):
        """Non-unit quaternions should be normalized internally."""
        quat = np.array([2.0, 0.0, 0.0, 0.0])  # Not unit length
        R = quaternion_to_rotation_matrix(quat)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_wrong_length_raises(self):
        """Wrong quaternion length should raise ValueError."""
        with self.assertRaises(ValueError):
            quaternion_to_rotation_matrix(np.zeros(3))


class TestRotationMatrixToQuaternion(unittest.TestCase):
    """Test rotation matrix to quaternion conversion."""

    def test_identity_matrix(self):
        """Identity matrix should give identity quaternion [1, 0, 0, 0]."""
        R = np.eye(3)
        quat = rotation_matrix_to_quaternion(R)
        np.testing.assert_allclose(quat, [1, 0, 0, 0], atol=1e-10)

    def test_180_rotation_z(self):
        """180 degree rotation around Z axis."""
        R = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
        ], dtype=float)
        quat = rotation_matrix_to_quaternion(R)
        # 180 deg around Z: quat = [0, 0, 0, 1] or [0, 0, 0, -1]
        np.testing.assert_allclose(np.abs(quat), [0, 0, 0, 1], atol=1e-10)

    def test_roundtrip_random(self):
        """quat -> R -> quat should give same quaternion (or negated)."""
        np.random.seed(42)
        for _ in range(10):
            quat_orig = np.random.randn(4)
            quat_orig = quat_orig / np.linalg.norm(quat_orig)

            R = quaternion_to_rotation_matrix(quat_orig)
            quat_back = rotation_matrix_to_quaternion(R)

            # Quaternions q and -q represent same rotation
            dot = np.abs(np.dot(quat_orig, quat_back))
            self.assertAlmostEqual(dot, 1.0, places=10)

    def test_unit_length(self):
        """Resulting quaternion should have unit length."""
        np.random.seed(42)
        for _ in range(10):
            # Random rotation matrix from random quaternion
            quat = np.random.randn(4)
            quat = quat / np.linalg.norm(quat)
            R = quaternion_to_rotation_matrix(quat)

            quat_out = rotation_matrix_to_quaternion(R)
            self.assertAlmostEqual(np.linalg.norm(quat_out), 1.0, places=10)

    def test_wrong_shape_raises(self):
        """Wrong matrix shape should raise ValueError."""
        with self.assertRaises(ValueError):
            rotation_matrix_to_quaternion(np.zeros((2, 3)))
        with self.assertRaises(ValueError):
            rotation_matrix_to_quaternion(np.zeros((4, 4)))


class TestGripperConversion(unittest.TestCase):
    """Test gripper-specific conversions."""

    def test_gripper_min(self):
        """Gripper 0 should map to minimum radians."""
        rad = gripper_normalized_to_radians(0.0)
        self.assertAlmostEqual(rad, SIM_ACTION_LOW[GRIPPER_IDX], places=6)

    def test_gripper_max(self):
        """Gripper 1 should map to maximum radians."""
        rad = gripper_normalized_to_radians(1.0)
        self.assertAlmostEqual(rad, SIM_ACTION_HIGH[GRIPPER_IDX], places=6)

    def test_gripper_half(self):
        """Gripper 0.5 should map to center of range."""
        rad = gripper_normalized_to_radians(0.5)
        expected = (SIM_ACTION_LOW[GRIPPER_IDX] + SIM_ACTION_HIGH[GRIPPER_IDX]) / 2
        self.assertAlmostEqual(rad, expected, places=6)


class TestClipJoints(unittest.TestCase):
    """Test joint clipping."""

    def test_within_limits(self):
        """Values within limits should pass through unchanged."""
        joints = np.zeros(NUM_JOINTS)
        clipped = clip_joints_to_limits(joints)
        np.testing.assert_array_equal(clipped, joints)

    def test_clip_low(self):
        """Values below minimum should be clipped."""
        joints = SIM_ACTION_LOW - 1.0
        clipped = clip_joints_to_limits(joints)
        np.testing.assert_array_equal(clipped, SIM_ACTION_LOW)

    def test_clip_high(self):
        """Values above maximum should be clipped."""
        joints = SIM_ACTION_HIGH + 1.0
        clipped = clip_joints_to_limits(joints)
        np.testing.assert_array_equal(clipped, SIM_ACTION_HIGH)


if __name__ == "__main__":
    # Run tests with verbosity
    unittest.main(verbosity=2)
