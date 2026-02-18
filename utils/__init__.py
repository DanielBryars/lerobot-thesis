"""Shared utilities for lerobot-thesis project."""

from utils.constants import MOTOR_NAMES, SIM_ACTION_LOW, SIM_ACTION_HIGH
from utils.conversions import (
    radians_to_normalized,
    normalized_to_radians,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    clip_joints_to_limits,
)
from utils.ik_solver import IKSolver
from utils.training import (
    CachedDataset,
    cycle,
    prepare_obs_for_policy,
    run_evaluation,
    get_action_space_info,
    get_camera_names,
    get_scene_metadata,
    create_output_dir,
    save_checkpoint,
    load_checkpoint,
)
from utils.vision import ObjectDetector, Detection
from utils.failure_analysis import (
    Outcome,
    EpisodeAnalysis,
    analyze_trajectory,
    compute_analysis_summary,
    format_analysis_report,
    get_failure_analysis_text,
)

__all__ = [
    # Constants
    "MOTOR_NAMES",
    "SIM_ACTION_LOW",
    "SIM_ACTION_HIGH",
    # Conversions
    "radians_to_normalized",
    "normalized_to_radians",
    "quaternion_to_rotation_matrix",
    "rotation_matrix_to_quaternion",
    "clip_joints_to_limits",
    # IK
    "IKSolver",
    # Training utilities
    "CachedDataset",
    "cycle",
    "prepare_obs_for_policy",
    "run_evaluation",
    "get_action_space_info",
    "get_camera_names",
    "get_scene_metadata",
    "create_output_dir",
    "save_checkpoint",
    "load_checkpoint",
    # Vision
    "ObjectDetector",
    "Detection",
    # Failure analysis
    "Outcome",
    "EpisodeAnalysis",
    "analyze_trajectory",
    "compute_analysis_summary",
    "format_analysis_report",
    "get_failure_analysis_text",
]
