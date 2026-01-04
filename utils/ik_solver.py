"""IK Solver wrapper for end-effector action space.

This module provides a clean interface for converting end-effector poses
to joint angles using MuJoCo's IK solver.
"""

from pathlib import Path
from typing import Optional
import numpy as np

from utils.constants import (
    SIM_ACTION_LOW,
    SIM_ACTION_HIGH,
    NUM_ARM_JOINTS,
    NUM_JOINTS,
    GRIPPER_IDX,
    DEFAULT_IK_TOLERANCE,
    DEFAULT_IK_MAX_ITER,
)
from utils.conversions import (
    quaternion_to_rotation_matrix,
    radians_to_normalized,
    gripper_normalized_to_radians,
    clip_joints_to_limits,
)


class IKSolver:
    """Wrapper for MuJoCo IK solver with tracking and conversion utilities.

    This class handles:
    - Lazy initialization of the underlying MuJoCo FK/IK solvers
    - Conversion from EE poses (xyz + quaternion) to joint angles
    - Tracking of IK success/failure statistics
    - Conversion to normalized action space for simulation
    """

    def __init__(
        self,
        scene_xml: Optional[str] = None,
        pos_tolerance: float = DEFAULT_IK_TOLERANCE,
        max_iterations: int = DEFAULT_IK_MAX_ITER,
    ):
        """Initialize IK solver.

        Args:
            scene_xml: Path to MuJoCo scene XML. If None, uses default.
            pos_tolerance: Position tolerance for IK success (meters)
            max_iterations: Max IK iterations
        """
        self.scene_xml = scene_xml
        self.pos_tolerance = pos_tolerance
        self.max_iterations = max_iterations

        # Lazy-loaded solvers
        self._fk = None
        self._ik = None

        # Statistics tracking
        self._failure_count = 0
        self._total_count = 0
        self._errors: list[float] = []

        # Last IK solution for warm-starting
        self._last_ik_joints: Optional[np.ndarray] = None

    def _ensure_initialized(self):
        """Lazily initialize FK/IK solvers."""
        if self._ik is not None:
            return

        # Import here to avoid circular dependencies and allow scripts/ to be optional
        import sys
        repo_root = Path(__file__).parent.parent
        sys.path.insert(0, str(repo_root / "scripts"))

        from test_fk_ik import MuJoCoFK, MuJoCoIK

        if self.scene_xml is None:
            self.scene_xml = str(repo_root / "scenes" / "so101_with_wrist_cam.xml")

        self._fk = MuJoCoFK(self.scene_xml)
        self._ik = MuJoCoIK(self._fk)
        print("IK solver initialized")

    def reset_stats(self):
        """Reset IK statistics for a new episode/evaluation."""
        self._failure_count = 0
        self._total_count = 0
        self._errors = []
        self._last_ik_joints = None

    def get_stats(self) -> dict:
        """Get current IK statistics.

        Returns:
            Dict with failure_count, total_count, failure_rate, avg_error_mm
        """
        rate = self._failure_count / max(1, self._total_count)
        avg_error = np.mean(self._errors) if self._errors else 0.0
        return {
            "failure_count": self._failure_count,
            "total_count": self._total_count,
            "failure_rate": rate,
            "avg_error_mm": avg_error,
        }

    def solve_ik(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        initial_angles: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, bool, float]:
        """Solve IK for target pose.

        Args:
            target_pos: Target position [x, y, z] in meters
            target_rot: Target rotation as 3x3 matrix
            initial_angles: Initial guess for joint angles (5-dim, radians)

        Returns:
            Tuple of (joint_angles_rad, success, error_meters)
        """
        self._ensure_initialized()

        if initial_angles is None:
            initial_angles = self._last_ik_joints if self._last_ik_joints is not None else np.zeros(NUM_ARM_JOINTS)

        ik_joints, success, error = self._ik.solve(
            target_pos=target_pos,
            target_rot=target_rot,
            initial_angles=initial_angles,
            max_iterations=self.max_iterations,
            pos_tolerance=self.pos_tolerance,
        )

        # Track statistics
        self._total_count += 1
        self._errors.append(error * 1000)  # Store in mm

        if not success:
            self._failure_count += 1
            if self._failure_count <= 3 or self._failure_count % 50 == 0:
                print(f"  [WARNING] IK failed ({self._failure_count}/{self._total_count}): "
                      f"target_pos={target_pos}, error={error*1000:.2f}mm")

        # Store for warm-starting next solve
        self._last_ik_joints = ik_joints.copy()

        return ik_joints, success, error

    def ee_to_joint_action(
        self,
        ee_action: np.ndarray,
        return_normalized: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Convert EE action to joint action.

        Args:
            ee_action: 8-dim array [x, y, z, qw, qx, qy, qz, gripper]
                - xyz: position in meters
                - quat: quaternion (scalar-first: w, x, y, z)
                - gripper: value in [0, 1] range
            return_normalized: If True, return normalized values for sim.
                              If False, return radians.

        Returns:
            Tuple of:
                - 6-dim joint action (normalized or radians depending on flag)
                - 5-dim IK joint angles in radians (for warm-starting next call)
                - IK success flag
        """
        # Extract EE pose components
        ee_pos = ee_action[:3]
        ee_quat = ee_action[3:7]
        gripper = ee_action[7]

        # Convert quaternion to rotation matrix
        ee_rot = quaternion_to_rotation_matrix(ee_quat)

        # Solve IK
        ik_joints, success, error = self.solve_ik(
            target_pos=ee_pos,
            target_rot=ee_rot,
        )

        # Convert gripper from [0, 1] to radians
        gripper_rad = gripper_normalized_to_radians(gripper)

        # Combine into full joint action
        joint_action_rad = np.concatenate([ik_joints, [gripper_rad]])
        joint_action_rad = clip_joints_to_limits(joint_action_rad)

        if return_normalized:
            joint_action = radians_to_normalized(joint_action_rad)
        else:
            joint_action = joint_action_rad

        return joint_action, ik_joints.copy(), success

    def forward_kinematics(self, joint_angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute forward kinematics.

        Args:
            joint_angles: 5-dim joint angles in radians (arm only)

        Returns:
            Tuple of (position, rotation_matrix)
        """
        self._ensure_initialized()
        return self._fk.forward(joint_angles)
