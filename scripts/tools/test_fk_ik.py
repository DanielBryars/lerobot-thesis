#!/usr/bin/env python
"""
Test Forward Kinematics (FK) and Inverse Kinematics (IK) for SO-100/SO-101 arm.

This script validates FK/IK using both MuJoCo (for simulation) and ikpy (for standalone IK).
Use this to verify the FK/IK pipeline before integrating into training.

Usage:
    python scripts/test_fk_ik.py                    # Run all tests
    python scripts/test_fk_ik.py --interactive      # Interactive mode with visualization
    python scripts/test_fk_ik.py --compare          # Compare MuJoCo vs ikpy FK
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add project paths
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

# Import shared utilities
from utils.conversions import rotation_matrix_to_quaternion

# MuJoCo imports
import mujoco
import mujoco.viewer

# Joint names (excluding gripper for FK/IK - gripper is just open/close)
ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
ALL_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class MuJoCoFK:
    """Forward Kinematics using MuJoCo simulation."""

    def __init__(self, scene_xml: str = None):
        if scene_xml is None:
            scene_xml = str(repo_root / "scenes" / "so101_with_wrist_cam.xml")

        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)

        # Get site ID for end-effector (gripperframe)
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

        # Get joint IDs
        self.joint_ids = {}
        for name in ALL_JOINT_NAMES:
            self.joint_ids[name] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

        print(f"MuJoCo FK initialized")
        print(f"  End-effector site: gripperframe (id={self.ee_site_id})")
        print(f"  Joints: {list(self.joint_ids.keys())}")

    def forward(self, joint_angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute FK: joint angles -> end-effector pose.

        Args:
            joint_angles: Array of 5 joint angles (radians) for arm joints
                         or 6 if including gripper

        Returns:
            position: End-effector position [x, y, z]
            rotation: End-effector rotation matrix (3x3)
        """
        # Reset to home position
        mujoco.mj_resetData(self.model, self.data)

        # Set joint angles (first 5 or 6 joints)
        for i, name in enumerate(ALL_JOINT_NAMES[:len(joint_angles)]):
            joint_id = self.joint_ids[name]
            # qpos index is the joint's qposadr
            qpos_idx = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_idx] = joint_angles[i]

        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)

        # Get end-effector position and rotation
        position = self.data.site_xpos[self.ee_site_id].copy()
        rotation = self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()

        return position, rotation

    def get_jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """Get the Jacobian matrix at current configuration."""
        # Set joint angles
        mujoco.mj_resetData(self.model, self.data)
        for i, name in enumerate(ALL_JOINT_NAMES[:len(joint_angles)]):
            joint_id = self.joint_ids[name]
            qpos_idx = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_idx] = joint_angles[i]

        mujoco.mj_forward(self.model, self.data)

        # Compute full Jacobian
        jacp = np.zeros((3, self.model.nv))  # Position Jacobian
        jacr = np.zeros((3, self.model.nv))  # Rotation Jacobian

        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)

        # Extract columns for arm joints only (using dof addresses)
        # The duplo freejoint takes DOFs 0-5, robot joints start after
        dof_indices = []
        for name in ARM_JOINT_NAMES:
            joint_id = self.joint_ids[name]
            dof_idx = self.model.jnt_dofadr[joint_id]
            dof_indices.append(dof_idx)

        jacp_arm = jacp[:, dof_indices]
        jacr_arm = jacr[:, dof_indices]

        return jacp_arm, jacr_arm


class MuJoCoIK:
    """Inverse Kinematics using MuJoCo's Jacobian-based IK."""

    def __init__(self, fk: MuJoCoFK):
        self.fk = fk
        self.model = fk.model
        self.data = fk.data

    def solve(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray = None,
        initial_angles: np.ndarray = None,
        max_iterations: int = 500,
        pos_tolerance: float = 1e-4,
        rot_tolerance: float = 1e-3,
        step_size: float = 1.0,
    ) -> tuple[np.ndarray, bool, float]:
        """
        Solve IK using damped least squares (Jacobian pseudoinverse).

        Args:
            target_pos: Target end-effector position [x, y, z]
            target_rot: Target rotation matrix (3x3), optional
            initial_angles: Starting joint angles, or None for current
            max_iterations: Maximum iterations
            pos_tolerance: Position error tolerance (meters)
            rot_tolerance: Rotation error tolerance (radians)
            step_size: Step size for updates

        Returns:
            joint_angles: Solution joint angles (5 values)
            success: Whether IK converged
            error: Final position error
        """
        if initial_angles is None:
            initial_angles = np.zeros(5)

        angles = initial_angles.copy().astype(np.float64)
        damping = 0.1  # Damping factor for stability

        for iteration in range(max_iterations):
            # Get current end-effector pose
            current_pos, current_rot = self.fk.forward(angles)

            # Position error
            pos_error = target_pos - current_pos
            pos_error_norm = np.linalg.norm(pos_error)

            # Check convergence
            if pos_error_norm < pos_tolerance:
                return angles, True, pos_error_norm

            # Get Jacobian
            jacp, jacr = self.fk.get_jacobian(angles)

            # Damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 dx
            if target_rot is not None:
                # Include rotation (6DOF)
                # Rotation error as angle-axis
                rot_error_mat = target_rot @ current_rot.T
                rot_error = rotation_matrix_to_axis_angle(rot_error_mat)

                J = np.vstack([jacp, jacr * 0.1])  # Scale rotation Jacobian down
                dx = np.concatenate([pos_error, rot_error * 0.1])  # Scale rotation error
            else:
                # Position only (3DOF)
                J = jacp
                dx = pos_error

            # Damped pseudoinverse
            JJT = J @ J.T
            JJT_damped = JJT + damping**2 * np.eye(JJT.shape[0])
            dq = J.T @ np.linalg.solve(JJT_damped, dx)

            # Update angles
            angles = angles + step_size * dq

            # Clip to joint limits (from MuJoCo model)
            joint_limits_low = np.array([-1.92, -1.75, -1.69, -1.66, -2.74])
            joint_limits_high = np.array([1.92, 1.75, 1.69, 1.66, 2.84])
            angles = np.clip(angles, joint_limits_low, joint_limits_high)

        return angles, False, pos_error_norm


def rotation_matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to axis-angle representation."""
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if angle < 1e-6:
        return np.zeros(3)

    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ]) / (2 * np.sin(angle))

    return axis * angle


def test_fk():
    """Test forward kinematics with various joint configurations."""
    print("\n" + "=" * 60)
    print("FORWARD KINEMATICS TEST")
    print("=" * 60)

    fk = MuJoCoFK()

    # Test configurations
    test_configs = [
        ("Home (all zeros)", np.zeros(5)),
        ("Shoulder pan +45°", np.array([np.pi/4, 0, 0, 0, 0])),
        ("Shoulder lift +45°", np.array([0, np.pi/4, 0, 0, 0])),
        ("Elbow flex -45°", np.array([0, 0, -np.pi/4, 0, 0])),
        ("Wrist flex +30°", np.array([0, 0, 0, np.pi/6, 0])),
        ("Wrist roll +90°", np.array([0, 0, 0, 0, np.pi/2])),
        ("Reach forward", np.array([0, 0.5, -0.5, 0, 0])),
        ("Reach left", np.array([np.pi/2, 0.3, -0.3, 0, 0])),
    ]

    print("\nJoint Configuration -> End-Effector Pose:")
    print("-" * 60)

    for name, angles in test_configs:
        pos, rot = fk.forward(angles)
        quat = rotation_matrix_to_quaternion(rot)

        print(f"\n{name}:")
        print(f"  Joints (deg): {np.degrees(angles).round(1)}")
        print(f"  Position:     [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] m")
        print(f"  Quaternion:   [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")

    return True


def test_ik():
    """Test inverse kinematics accuracy."""
    print("\n" + "=" * 60)
    print("INVERSE KINEMATICS TEST")
    print("=" * 60)

    fk = MuJoCoFK()
    ik = MuJoCoIK(fk)

    # Test: FK -> IK -> FK roundtrip
    test_configs = [
        ("Home", np.zeros(5)),
        ("Reach forward", np.array([0, 0.5, -0.5, 0, 0])),
        ("Reach left", np.array([np.pi/3, 0.3, -0.4, 0.2, 0])),
        ("Complex pose", np.array([0.5, 0.6, -0.8, 0.3, 0.7])),
    ]

    print("\nFK -> IK -> FK Roundtrip Test:")
    print("-" * 60)

    all_passed = True

    for name, original_angles in test_configs:
        # FK: Get target pose
        target_pos, target_rot = fk.forward(original_angles)

        # IK: Solve for angles (start from different initial guess)
        initial_guess = np.zeros(5)  # Start from home
        solved_angles, success, error = ik.solve(
            target_pos, target_rot, initial_angles=initial_guess
        )

        # Verify: FK with solved angles
        verify_pos, verify_rot = fk.forward(solved_angles)
        pos_error = np.linalg.norm(verify_pos - target_pos)

        status = "PASS" if success and pos_error < 0.001 else "FAIL"
        if status == "FAIL":
            all_passed = False

        print(f"\n{name}: {status}")
        print(f"  Original angles (deg): {np.degrees(original_angles).round(1)}")
        print(f"  Solved angles (deg):   {np.degrees(solved_angles).round(1)}")
        print(f"  Target pos:  {target_pos.round(4)}")
        print(f"  Verify pos:  {verify_pos.round(4)}")
        print(f"  Position error: {pos_error:.6f} m")
        print(f"  IK converged: {success}")

    return all_passed


def test_ik_random():
    """Test IK with random reachable targets."""
    print("\n" + "=" * 60)
    print("RANDOM IK TEST (100 samples)")
    print("=" * 60)

    fk = MuJoCoFK()
    ik = MuJoCoIK(fk)

    np.random.seed(42)
    n_tests = 100
    successes = 0
    errors = []

    for i in range(n_tests):
        # Generate random joint angles within limits
        random_angles = np.random.uniform(
            low=[-1.5, -1.5, -1.5, -1.5, -2.5],
            high=[1.5, 1.5, 1.5, 1.5, 2.5]
        )

        # Get target pose
        target_pos, target_rot = fk.forward(random_angles)

        # Solve IK from home position
        solved_angles, success, error = ik.solve(target_pos, target_rot)

        if success:
            successes += 1
            # Verify
            verify_pos, _ = fk.forward(solved_angles)
            pos_error = np.linalg.norm(verify_pos - target_pos)
            errors.append(pos_error)

    success_rate = successes / n_tests * 100
    mean_error = np.mean(errors) if errors else float('inf')
    max_error = np.max(errors) if errors else float('inf')

    print(f"\nResults:")
    print(f"  Success rate: {success_rate:.1f}% ({successes}/{n_tests})")
    print(f"  Mean position error: {mean_error:.6f} m")
    print(f"  Max position error:  {max_error:.6f} m")

    return success_rate > 90


def interactive_mode():
    """Interactive visualization with MuJoCo viewer."""
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Use MuJoCo viewer to manipulate joints and see FK results.")
    print("Press ESC to exit.")

    scene_xml = str(repo_root / "scenes" / "so101_with_wrist_cam.xml")
    model = mujoco.MjModel.from_xml_path(scene_xml)
    data = mujoco.MjData(model)

    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

    def print_ee_pose(model, data):
        """Callback to print end-effector pose."""
        pos = data.site_xpos[ee_site_id]
        rot = data.site_xmat[ee_site_id].reshape(3, 3)
        quat = rotation_matrix_to_quaternion(rot)

        # Get joint angles
        angles = data.qpos[:5].copy()

        print(f"\rJoints (deg): {np.degrees(angles).round(1)} | "
              f"EE pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] | "
              f"Quat: [{quat[0]:.2f}, {quat[1]:.2f}, {quat[2]:.2f}, {quat[3]:.2f}]", end="")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            print_ee_pose(model, data)
            viewer.sync()

    print("\n")


class IKPyFK:
    """Forward Kinematics using ikpy library (standalone, no MuJoCo needed)."""

    def __init__(self, urdf_path: str = None):
        try:
            import ikpy.chain
            self.ikpy = ikpy
        except ImportError:
            raise ImportError("ikpy not installed. Run: pip install ikpy")

        if urdf_path is None:
            urdf_path = str(repo_root / "assets" / "SO-ARM100" / "Simulation" / "SO100" / "so100.urdf")

        # Load chain from URDF - base to gripper (5 arm joints)
        self.chain = ikpy.chain.Chain.from_urdf_file(
            urdf_path,
            base_elements=["base"],
        )

        print(f"ikpy FK initialized from: {urdf_path}")
        print(f"  Chain links: {[link.name for link in self.chain.links]}")
        print(f"  Active joints: {sum(1 for link in self.chain.links if link.joint_type != 'fixed')}")

    def forward(self, joint_angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute FK: joint angles -> end-effector pose.

        Args:
            joint_angles: Array of 5 joint angles (radians) for arm joints

        Returns:
            position: End-effector position [x, y, z]
            rotation: End-effector rotation matrix (3x3)
        """
        # ikpy expects angles for all links including fixed ones
        # Build full angles array matching chain structure
        full_angles = np.zeros(len(self.chain.links))
        joint_idx = 0
        for i, link in enumerate(self.chain.links):
            if link.joint_type != 'fixed' and joint_idx < len(joint_angles):
                full_angles[i] = joint_angles[joint_idx]
                joint_idx += 1

        # Forward kinematics returns 4x4 transformation matrix
        transform = self.chain.forward_kinematics(full_angles)

        position = transform[:3, 3]
        rotation = transform[:3, :3]

        return position, rotation

    def inverse(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray = None,
        initial_angles: np.ndarray = None
    ) -> tuple[np.ndarray, bool]:
        """
        Solve IK using ikpy.

        Args:
            target_pos: Target end-effector position [x, y, z]
            target_rot: Target rotation matrix (3x3), optional
            initial_angles: Starting joint angles

        Returns:
            joint_angles: Solution joint angles (5 values)
            success: Whether IK found a solution
        """
        if initial_angles is None:
            initial_angles = np.zeros(5)

        # Build full initial angles
        full_initial = np.zeros(len(self.chain.links))
        joint_idx = 0
        for i, link in enumerate(self.chain.links):
            if link.joint_type != 'fixed' and joint_idx < len(initial_angles):
                full_initial[i] = initial_angles[joint_idx]
                joint_idx += 1

        # Solve IK
        if target_rot is not None:
            target_transform = np.eye(4)
            target_transform[:3, :3] = target_rot
            target_transform[:3, 3] = target_pos
            result = self.chain.inverse_kinematics_frame(
                target_transform,
                initial_position=full_initial,
            )
        else:
            result = self.chain.inverse_kinematics(
                target_pos,
                initial_position=full_initial,
            )

        # Extract active joint angles
        joint_angles = []
        for i, link in enumerate(self.chain.links):
            if link.joint_type != 'fixed':
                joint_angles.append(result[i])
        joint_angles = np.array(joint_angles[:5])  # First 5 arm joints

        # Verify solution
        verify_pos, _ = self.forward(joint_angles)
        error = np.linalg.norm(verify_pos - target_pos)
        success = error < 0.01  # 1cm tolerance

        return joint_angles, success


def test_ikpy():
    """Test ikpy FK/IK."""
    print("\n" + "=" * 60)
    print("IKPY FK/IK TEST")
    print("=" * 60)

    try:
        ikpy_fk = IKPyFK()
    except ImportError as e:
        print(f"Skipping ikpy test: {e}")
        return None
    except Exception as e:
        print(f"Error initializing ikpy: {e}")
        return None

    # Test FK
    print("\nikpy Forward Kinematics:")
    print("-" * 40)

    test_configs = [
        ("Home", np.zeros(5)),
        ("Shoulder pan +45°", np.array([np.pi/4, 0, 0, 0, 0])),
        ("Reach forward", np.array([0, 0.5, -0.5, 0, 0])),
    ]

    for name, angles in test_configs:
        try:
            pos, rot = ikpy_fk.forward(angles)
            print(f"  {name}: pos={pos.round(4)}")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")

    # Test IK roundtrip
    print("\nikpy IK Roundtrip:")
    print("-" * 40)

    all_passed = True
    for name, original_angles in test_configs:
        try:
            target_pos, target_rot = ikpy_fk.forward(original_angles)
            solved_angles, success = ikpy_fk.inverse(target_pos)
            verify_pos, _ = ikpy_fk.forward(solved_angles)
            error = np.linalg.norm(verify_pos - target_pos)

            status = "PASS" if success else "FAIL"
            if not success:
                all_passed = False
            print(f"  {name}: {status} (error={error:.4f}m)")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")
            all_passed = False

    return all_passed


def compare_mujoco_ikpy():
    """Compare MuJoCo and ikpy FK results."""
    print("\n" + "=" * 60)
    print("MUJOCO vs IKPY COMPARISON")
    print("=" * 60)

    mj_fk = MuJoCoFK()

    try:
        ikpy_fk = IKPyFK()
    except ImportError as e:
        print(f"Skipping comparison: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

    test_configs = [
        ("Home", np.zeros(5)),
        ("Shoulder pan +45°", np.array([np.pi/4, 0, 0, 0, 0])),
        ("Reach forward", np.array([0, 0.5, -0.5, 0, 0])),
    ]

    print("\nFK Position Comparison:")
    print("-" * 70)
    print(f"{'Config':<20} {'MuJoCo':<25} {'ikpy':<25} {'Diff (m)':<10}")
    print("-" * 70)

    for name, angles in test_configs:
        mj_pos, _ = mj_fk.forward(angles)
        ik_pos, _ = ikpy_fk.forward(angles)
        diff = np.linalg.norm(mj_pos - ik_pos)

        mj_str = f"[{mj_pos[0]:.3f}, {mj_pos[1]:.3f}, {mj_pos[2]:.3f}]"
        ik_str = f"[{ik_pos[0]:.3f}, {ik_pos[1]:.3f}, {ik_pos[2]:.3f}]"
        print(f"{name:<20} {mj_str:<25} {ik_str:<25} {diff:.4f}")

    print("\nNote: Differences are expected - MuJoCo uses SO101, ikpy uses SO100 URDF")

    return True


def main():
    parser = argparse.ArgumentParser(description="Test FK/IK for SO-100/SO-101 arm")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode with viewer")
    parser.add_argument("--fk-only", action="store_true", help="Only test FK")
    parser.add_argument("--ik-only", action="store_true", help="Only test IK")
    parser.add_argument("--ikpy", action="store_true", help="Test ikpy FK/IK")
    parser.add_argument("--compare", action="store_true", help="Compare MuJoCo vs ikpy")
    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
        return 0

    print("=" * 60)
    print("SO-100/SO-101 FK/IK TEST SUITE")
    print("=" * 60)

    results = {}

    if args.compare:
        results["MuJoCo vs ikpy"] = compare_mujoco_ikpy()
    elif args.ikpy:
        results["ikpy FK/IK"] = test_ikpy()
    else:
        if not args.ik_only:
            results["FK"] = test_fk()

        if not args.fk_only:
            results["IK Basic"] = test_ik()
            results["IK Random"] = test_ik_random()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for test_name, passed in results.items():
        if passed is None:
            status = "SKIPPED"
        else:
            status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")

    valid_results = [v for v in results.values() if v is not None]
    all_passed = all(valid_results) if valid_results else False
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
