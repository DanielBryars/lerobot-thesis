"""
Whisker visualization for action chunk predictions in MuJoCo.

Shows predicted future trajectories as lines emanating from the robot end-effector.
"""

import numpy as np
import mujoco
import torch

from utils.mujoco_viz import MujocoPathRenderer, TrajectoryTracker, FKSolver

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class WhiskerVisualizer:
    """Visualizes action chunk predictions as whiskers in MuJoCo.

    Uses utility classes from utils.mujoco_viz for clean separation of concerns.
    """

    def __init__(
        self,
        sim,  # SO100Sim instance
        whisker_color: tuple = (0.2, 0.8, 0.2, 0.7),  # RGBA - green
        ghost_color: tuple = (0.5, 0.5, 0.9, 0.3),  # RGBA - light blue
        actual_path_color: tuple = (1.0, 0.3, 0.1, 0.5),  # RGBA - orange
        whisker_radius: float = 0.003,
        ghost_radius: float = 0.002,
        max_ghost_trails: int = 12,
    ):
        self.sim = sim
        self.whisker_color = whisker_color
        self.ghost_color = ghost_color
        self.actual_path_color = actual_path_color
        self.whisker_radius = whisker_radius
        self.ghost_radius = ghost_radius
        self.max_ghost_trails = max_ghost_trails

        # Initialize components
        self.fk_solver = FKSolver(sim.mj_model, sim.mj_data, ee_site_name="gripperframe")
        self.actual_path_tracker = TrajectoryTracker(max_length=500)
        self.renderer = MujocoPathRenderer()

        # Get end-effector site ID (for recording actual position)
        self.ee_site_id = self.fk_solver.ee_site_id

        # Current whisker data
        self.whisker_points = None  # Shape: (N, 3) - predicted EE positions (the one we follow)
        self.alt_whisker_points = []  # List of alternative predictions (shown in grey)
        self.ghost_trails = []  # List of past whisker points arrays

        # Variance visualization color (grey)
        self.variance_color = (0.5, 0.5, 0.5, 0.4)  # RGBA - grey

        # Action chunk management
        self.current_action_chunk_normalized = None
        self.current_action_chunk_denorm = None
        self.chunk_step = 0

    def prepare_obs(self, obs: dict, device: str = "cuda") -> dict:
        """Convert sim observation to policy input format."""
        batch = {}

        # State
        state = np.array([obs[m + ".pos"] for m in MOTOR_NAMES], dtype=np.float32)
        batch["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(device)

        # Images
        for key, value in obs.items():
            if isinstance(value, np.ndarray) and value.ndim == 3:
                img = torch.from_numpy(value).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                batch[f"observation.images.{key}"] = img.to(device)

        return batch

    def update_whiskers_from_actions(self, action_chunk_denorm: np.ndarray, alt_chunks: list = None):
        """Update whisker visualization from action chunk.

        Args:
            action_chunk_denorm: Denormalized joint targets, shape (N, 6) - the one we follow
            alt_chunks: List of alternative action chunks (shown in grey for variance visualization)
        """
        # Save old whiskers as ghost trail
        if self.whisker_points is not None and len(self.whisker_points) > 1:
            self.ghost_trails.append(self.whisker_points.copy())
            if len(self.ghost_trails) > self.max_ghost_trails:
                self.ghost_trails.pop(0)

        # Compute predicted EE positions using FK for main whisker
        self.whisker_points = self.fk_solver.compute_ee_positions(action_chunk_denorm)

        # Compute alternative whiskers for variance visualization
        self.alt_whisker_points = []
        if alt_chunks:
            for alt_chunk in alt_chunks:
                alt_points = self.fk_solver.compute_ee_positions(alt_chunk)
                self.alt_whisker_points.append(alt_points)

    def record_actual_position(self):
        """Record current EE position. Call every simulation step."""
        pos = self.sim.mj_data.site_xpos[self.ee_site_id].copy()
        self.actual_path_tracker.record(pos)

    def clear_trails(self):
        """Clear all trails and whiskers (call on episode reset)."""
        self.ghost_trails.clear()
        self.alt_whisker_points.clear()
        self.actual_path_tracker.clear()
        self.whisker_points = None

    def draw_actual_path(self, scene: mujoco.MjvScene):
        """Draw the actual path taken (updates every step)."""
        actual_pts = self.actual_path_tracker.get_points()
        if len(actual_pts) >= 2:
            self.renderer.draw_path(scene, actual_pts, self.actual_path_color, self.ghost_radius, max_segments=60)

    def draw_whisker(self, scene: mujoco.MjvScene):
        """Draw current whisker prediction, alternative predictions, and ghost trails."""
        # Ghost trails (older = more faded)
        for i, ghost_pts in enumerate(self.ghost_trails):
            if ghost_pts is None or len(ghost_pts) < 2:
                continue
            age_factor = (i + 1) / max(len(self.ghost_trails), 1)
            alpha = self.ghost_color[3] * (age_factor ** 2)
            color = (self.ghost_color[0], self.ghost_color[1], self.ghost_color[2], alpha)
            self.renderer.draw_path(scene, ghost_pts, color, self.ghost_radius, max_segments=20)

        # Alternative whiskers (grey, showing variance)
        for alt_pts in self.alt_whisker_points:
            if alt_pts is not None and len(alt_pts) >= 2:
                self.renderer.draw_path(scene, alt_pts, self.variance_color, self.ghost_radius, max_segments=30)

        # Current whisker (the one we follow - green)
        if self.whisker_points is not None and len(self.whisker_points) >= 2:
            self.renderer.draw_path(scene, self.whisker_points, self.whisker_color, self.whisker_radius, max_segments=30)

    def add_whiskers_to_scene(self, scene: mujoco.MjvScene, show_ghosts: bool = True, show_actual_path: bool = True):
        """Draw all visualization elements to the MuJoCo scene."""
        if show_ghosts:
            self.draw_whisker(scene)
        if show_actual_path:
            self.draw_actual_path(scene)
