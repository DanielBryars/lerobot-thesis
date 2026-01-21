"""
MuJoCo visualization utilities for drawing paths and trajectories.

Classes:
    MujocoPathRenderer: Draws 3D paths/lines to MuJoCo scene
    TrajectoryTracker: Tracks a sequence of 3D positions over time
    FKSolver: Computes end-effector positions from joint angles
"""

import numpy as np
import mujoco
from dataclasses import dataclass, field
from typing import Optional


class MujocoPathRenderer:
    """Draws 3D paths and points to a MuJoCo scene.

    Works purely in world coordinates. Knows nothing about policies or physics.
    Just takes 3D points and draws them as geometry in the scene.
    """

    @staticmethod
    def draw_path(
        scene: mujoco.MjvScene,
        points: np.ndarray,
        color: tuple[float, float, float, float] = (0.2, 0.8, 0.2, 0.7),
        radius: float = 0.003,
        max_segments: int = 50,
        fade_alpha: bool = True,
    ) -> int:
        """Draw a path as connected line segments.

        Args:
            scene: MuJoCo scene to draw into
            points: Array of 3D points, shape (N, 3)
            color: RGBA color tuple
            radius: Line thickness
            max_segments: Max segments to draw (subsamples if needed)
            fade_alpha: If True, fade alpha along the path

        Returns:
            Number of segments drawn
        """
        if points is None or len(points) < 2:
            return 0

        # Subsample if too many points
        step = max(1, len(points) // max_segments)

        segments_drawn = 0
        for i in range(0, len(points) - step, step):
            p1 = points[i]
            p2 = points[min(i + step, len(points) - 1)]

            # Calculate alpha (fade along path if requested)
            if fade_alpha:
                progress = i / len(points)
                alpha = color[3] * (1.0 - progress * 0.5)
            else:
                alpha = color[3]

            segment_color = (color[0], color[1], color[2], alpha)

            if MujocoPathRenderer._add_line_segment(scene, p1, p2, segment_color, radius):
                segments_drawn += 1
            else:
                break  # Scene full

        return segments_drawn

    @staticmethod
    def draw_point(
        scene: mujoco.MjvScene,
        position: np.ndarray,
        color: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0),
        radius: float = 0.005,
    ) -> bool:
        """Draw a single point as a sphere.

        Args:
            scene: MuJoCo scene to draw into
            position: 3D position
            color: RGBA color tuple
            radius: Sphere radius

        Returns:
            True if drawn successfully, False if scene is full
        """
        if scene.ngeom >= scene.maxgeom:
            return False

        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, 0, 0], dtype=np.float64),
            position.astype(np.float64),
            np.eye(3, dtype=np.float64).flatten(),
            np.array(color, dtype=np.float64),
        )
        scene.ngeom += 1
        return True

    @staticmethod
    def _add_line_segment(
        scene: mujoco.MjvScene,
        p1: np.ndarray,
        p2: np.ndarray,
        color: tuple[float, float, float, float],
        radius: float,
    ) -> bool:
        """Add a single line segment (capsule) to scene.

        Returns False if scene is full.
        """
        if scene.ngeom >= scene.maxgeom:
            return False

        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64).flatten(),
            np.array(color, dtype=np.float64),
        )
        mujoco.mjv_connector(
            g,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            radius,
            p1.astype(np.float64),
            p2.astype(np.float64),
        )
        scene.ngeom += 1
        return True


@dataclass
class TrajectoryTracker:
    """Tracks a sequence of 3D positions over time.

    Generic tracker that can be used for actual robot path,
    predicted paths, ghost trails, etc.
    """

    max_length: int = 500
    points: list = field(default_factory=list)

    def record(self, position: np.ndarray) -> None:
        """Record a new position."""
        self.points.append(position.copy())
        if len(self.points) > self.max_length:
            self.points.pop(0)

    def get_points(self) -> np.ndarray:
        """Get all points as numpy array, shape (N, 3)."""
        if not self.points:
            return np.array([]).reshape(0, 3)
        return np.array(self.points)

    def clear(self) -> None:
        """Clear all recorded points."""
        self.points.clear()

    def __len__(self) -> int:
        return len(self.points)


class FKSolver:
    """Computes end-effector positions from joint angles using MuJoCo.

    Uses pure forward kinematics (mj_forward) - fast geometric calculation,
    no physics simulation.

    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data (uses simulation's data, saves/restores state)
        ee_site_name: Name of the end-effector site to track
    """

    def __init__(self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData, ee_site_name: str = "gripper_site"):
        self.mj_model = mj_model
        self.mj_data = mj_data  # Use simulation's data directly

        # Find end-effector site
        self.ee_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
        if self.ee_site_id == -1:
            for name in ["gripperframe", "ee_site", "end_effector", "gripper"]:
                self.ee_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, name)
                if self.ee_site_id != -1:
                    break

        if self.ee_site_id == -1:
            raise ValueError(f"Could not find end-effector site '{ee_site_name}'")

    def compute_ee_positions(self, joint_angles_sequence: np.ndarray) -> np.ndarray:
        """Compute EE positions for a sequence of joint angles.

        Args:
            joint_angles_sequence: Joint angles in DEGREES, shape (N, 6)

        Returns:
            EE positions, shape (N, 3)
        """
        # Save current state
        saved_qpos = self.mj_data.qpos.copy()
        saved_qvel = self.mj_data.qvel.copy()

        positions = []
        n = len(joint_angles_sequence)
        # Arm joints start at qpos[7] (after duplo's free joint at qpos[0:7])
        arm_joint_start = 7
        for i, joint_angles in enumerate(joint_angles_sequence):
            # Convert degrees to radians for MuJoCo
            self.mj_data.qpos[arm_joint_start:arm_joint_start+6] = np.radians(joint_angles[:6])
            mujoco.mj_forward(self.mj_model, self.mj_data)
            pos = self.mj_data.site_xpos[self.ee_site_id].copy()
            positions.append(pos)
            # Debug first and last iteration
            if i == 0 or i == n - 1:
                print(f"  FK[{i}]: joints={joint_angles[:3]}... -> site={pos}")

        # Restore state
        self.mj_data.qpos[:] = saved_qpos
        self.mj_data.qvel[:] = saved_qvel
        mujoco.mj_forward(self.mj_model, self.mj_data)

        return np.array(positions)
