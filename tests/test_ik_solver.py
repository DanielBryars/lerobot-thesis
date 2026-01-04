"""Unit tests for IK solver wrapper.

These tests require MuJoCo and the scene XML file.
Run with: python tests/test_ik_solver.py
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from utils.constants import NUM_ARM_JOINTS, NUM_JOINTS, SIM_ACTION_LOW, SIM_ACTION_HIGH


# Check if MuJoCo scene exists
SCENE_PATH = Path(__file__).parent.parent / "scenes" / "so101_with_wrist_cam.xml"
MUJOCO_AVAILABLE = SCENE_PATH.exists()


def skip_if_no_mujoco(test_func):
    """Decorator to skip tests if MuJoCo scene is not available."""
    def wrapper(self):
        if not MUJOCO_AVAILABLE:
            self.skipTest("MuJoCo scene not available")
        return test_func(self)
    return wrapper


class TestIKSolverInit(unittest.TestCase):
    """Test IK solver initialization."""

    @skip_if_no_mujoco
    def test_lazy_init(self):
        """Solver should not initialize until first use."""
        from utils.ik_solver import IKSolver
        solver = IKSolver()
        self.assertIsNone(solver._ik)
        self.assertIsNone(solver._fk)

    @skip_if_no_mujoco
    def test_init_on_solve(self):
        """Solver should initialize on first solve."""
        from utils.ik_solver import IKSolver
        solver = IKSolver()

        # Provide a valid target pose
        target_pos = np.array([0.3, 0.0, 0.1])
        target_rot = np.eye(3)

        solver.solve_ik(target_pos, target_rot)

        self.assertIsNotNone(solver._ik)
        self.assertIsNotNone(solver._fk)


class TestIKSolverStats(unittest.TestCase):
    """Test IK statistics tracking."""

    @skip_if_no_mujoco
    def test_stats_initial(self):
        """Initial stats should be zero."""
        from utils.ik_solver import IKSolver
        solver = IKSolver()

        stats = solver.get_stats()
        self.assertEqual(stats["failure_count"], 0)
        self.assertEqual(stats["total_count"], 0)
        self.assertEqual(stats["failure_rate"], 0.0)

    @skip_if_no_mujoco
    def test_stats_after_solve(self):
        """Stats should update after solve."""
        from utils.ik_solver import IKSolver
        solver = IKSolver()

        target_pos = np.array([0.3, 0.0, 0.1])
        target_rot = np.eye(3)

        solver.solve_ik(target_pos, target_rot)

        stats = solver.get_stats()
        self.assertEqual(stats["total_count"], 1)

    @skip_if_no_mujoco
    def test_stats_reset(self):
        """Reset should clear all stats."""
        from utils.ik_solver import IKSolver
        solver = IKSolver()

        target_pos = np.array([0.3, 0.0, 0.1])
        target_rot = np.eye(3)

        solver.solve_ik(target_pos, target_rot)
        solver.reset_stats()

        stats = solver.get_stats()
        self.assertEqual(stats["total_count"], 0)
        self.assertEqual(stats["failure_count"], 0)


class TestIKSolve(unittest.TestCase):
    """Test IK solving functionality."""

    @skip_if_no_mujoco
    def test_solve_returns_correct_shape(self):
        """solve_ik should return 5 joint angles."""
        from utils.ik_solver import IKSolver
        solver = IKSolver()

        target_pos = np.array([0.3, 0.0, 0.1])
        target_rot = np.eye(3)

        joints, success, error = solver.solve_ik(target_pos, target_rot)

        self.assertEqual(joints.shape, (NUM_ARM_JOINTS,))
        self.assertIsInstance(success, bool)
        self.assertIsInstance(error, float)

    @skip_if_no_mujoco
    def test_fk_ik_roundtrip(self):
        """FK then IK should return to approximately same joints."""
        from utils.ik_solver import IKSolver
        solver = IKSolver()

        # Start with known joint angles
        original_joints = np.array([0.1, 0.2, -0.3, 0.1, 0.0])

        # Compute FK to get EE pose
        pos, rot = solver.forward_kinematics(original_joints)

        # Compute IK to get joints back
        recovered_joints, success, error = solver.solve_ik(
            pos, rot, initial_angles=original_joints
        )

        # Should be very close (not exact due to IK tolerance)
        self.assertTrue(success, f"IK failed with error {error*1000:.2f}mm")
        np.testing.assert_allclose(recovered_joints, original_joints, atol=0.1)


class TestEEToJointAction(unittest.TestCase):
    """Test EE action to joint action conversion."""

    @skip_if_no_mujoco
    def test_ee_to_joint_normalized(self):
        """ee_to_joint_action with normalized=True should return normalized values."""
        from utils.ik_solver import IKSolver
        solver = IKSolver()

        ee_action = np.array([0.3, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 0.5])  # xyz, quat, gripper

        joint_action, ik_rad, success = solver.ee_to_joint_action(
            ee_action, return_normalized=True
        )

        self.assertEqual(joint_action.shape, (NUM_JOINTS,))
        self.assertEqual(ik_rad.shape, (NUM_ARM_JOINTS,))
        # Normalized values should be in reasonable range
        self.assertTrue(np.all(joint_action[:5] >= -100) and np.all(joint_action[:5] <= 100))
        self.assertTrue(joint_action[5] >= 0 and joint_action[5] <= 100)  # gripper

    @skip_if_no_mujoco
    def test_ee_to_joint_radians(self):
        """ee_to_joint_action with normalized=False should return radians."""
        from utils.ik_solver import IKSolver
        solver = IKSolver()

        ee_action = np.array([0.3, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 0.5])

        joint_action, ik_rad, success = solver.ee_to_joint_action(
            ee_action, return_normalized=False
        )

        self.assertEqual(joint_action.shape, (NUM_JOINTS,))
        # Radians should be in joint limits
        self.assertTrue(np.all(joint_action >= SIM_ACTION_LOW))
        self.assertTrue(np.all(joint_action <= SIM_ACTION_HIGH))


if __name__ == "__main__":
    unittest.main(verbosity=2)
