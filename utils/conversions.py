"""Conversion functions for robot action spaces.

This module handles conversions between different representations:
- Radians (MuJoCo internal)
- Normalized [-100, 100] for joints, [0, 100] for gripper (simulation interface)
- Quaternions and rotation matrices (for EE poses)
"""

import numpy as np

from utils.constants import (
    SIM_ACTION_LOW,
    SIM_ACTION_HIGH,
    GRIPPER_IDX,
    NUM_JOINTS,
)


def radians_to_normalized(radians: np.ndarray) -> np.ndarray:
    """Convert joint radians to normalized values for simulation.

    The simulation interface expects normalized values:
    - Joints: maps [LOW, HIGH] -> [-100, 100]
    - Gripper: maps [LOW, HIGH] -> [0, 100]

    Args:
        radians: Array of joint angles in radians (6-dim)

    Returns:
        Array of normalized joint values (6-dim)
    """
    if len(radians) != NUM_JOINTS:
        raise ValueError(f"Expected {NUM_JOINTS} joints, got {len(radians)}")

    normalized = np.zeros(NUM_JOINTS, dtype=np.float32)
    for i in range(NUM_JOINTS):
        # Compute normalized position in [0, 1]
        t = (radians[i] - SIM_ACTION_LOW[i]) / (SIM_ACTION_HIGH[i] - SIM_ACTION_LOW[i])
        if i == GRIPPER_IDX:
            # Gripper: [0, 100]
            normalized[i] = t * 100.0
        else:
            # Joints: [-100, 100]
            normalized[i] = t * 200.0 - 100.0
    return normalized


def normalized_to_radians(normalized: np.ndarray) -> np.ndarray:
    """Convert normalized values to joint radians.

    Inverse of radians_to_normalized.

    Args:
        normalized: Array of normalized joint values (6-dim)
            - Joints in [-100, 100]
            - Gripper in [0, 100]

    Returns:
        Array of joint angles in radians (6-dim)
    """
    if len(normalized) != NUM_JOINTS:
        raise ValueError(f"Expected {NUM_JOINTS} joints, got {len(normalized)}")

    radians = np.zeros(NUM_JOINTS, dtype=np.float32)
    for i in range(NUM_JOINTS):
        if i == GRIPPER_IDX:
            # Gripper: [0, 100] -> [LOW, HIGH]
            t = normalized[i] / 100.0
        else:
            # Joints: [-100, 100] -> [LOW, HIGH]
            t = (normalized[i] + 100.0) / 200.0
        radians[i] = SIM_ACTION_LOW[i] + t * (SIM_ACTION_HIGH[i] - SIM_ACTION_LOW[i])
    return radians


def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [qw, qx, qy, qz] to 3x3 rotation matrix.

    Args:
        quat: Quaternion as [w, x, y, z] (scalar-first convention)

    Returns:
        3x3 rotation matrix
    """
    if len(quat) != 4:
        raise ValueError(f"Expected 4-element quaternion, got {len(quat)}")

    w, x, y, z = quat

    # Normalize quaternion
    n = np.sqrt(w*w + x*x + y*y + z*z)
    if n > 0:
        w, x, y, z = w/n, x/n, y/n, z/n

    # Build rotation matrix
    R = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    return R


def gripper_normalized_to_radians(gripper_norm: float) -> float:
    """Convert gripper value from [0, 1] to radians.

    Args:
        gripper_norm: Gripper value in [0, 1] range

    Returns:
        Gripper angle in radians
    """
    return SIM_ACTION_LOW[GRIPPER_IDX] + gripper_norm * (
        SIM_ACTION_HIGH[GRIPPER_IDX] - SIM_ACTION_LOW[GRIPPER_IDX]
    )


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [qw, qx, qy, qz].

    Uses Shepperd's method for numerical stability.

    Args:
        R: 3x3 rotation matrix

    Returns:
        Quaternion as [w, x, y, z] (scalar-first convention)
    """
    if R.shape != (3, 3):
        raise ValueError(f"Expected 3x3 rotation matrix, got {R.shape}")

    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    # Normalize
    quat = np.array([w, x, y, z])
    return quat / np.linalg.norm(quat)


def clip_joints_to_limits(joints: np.ndarray) -> np.ndarray:
    """Clip joint values to valid range.

    Args:
        joints: Array of joint angles in radians (6-dim)

    Returns:
        Clipped joint values
    """
    return np.clip(joints, SIM_ACTION_LOW, SIM_ACTION_HIGH)
