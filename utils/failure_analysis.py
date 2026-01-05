"""Failure analysis utilities for robot learning evaluation.

This module provides tools for analyzing why episodes fail, categorizing
failures into types like:
- Never picked up: Robot failed to grasp the object
- Dropped during transport: Object was lifted but fell before reaching goal
- Missed goal: Object placed near but outside the target area
- Timeout: Max steps reached without clear outcome

This enables more nuanced evaluation beyond just success/failure rates.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional
import numpy as np


class Outcome(Enum):
    """Episode outcome categories."""
    SUCCESS = "success"
    NEVER_PICKED_UP = "never_picked_up"
    DROPPED_DURING_TRANSPORT = "dropped_during_transport"
    MISSED_GOAL = "missed_goal"
    TIMEOUT = "timeout"


@dataclass
class EpisodeAnalysis:
    """Detailed analysis of a single episode."""
    outcome: Outcome
    steps: int
    duration: float
    max_height: float
    was_lifted: bool
    was_dropped: bool
    final_distance_to_goal: float
    trajectory: list = field(default_factory=list)


# Default thresholds for pick-and-place task
DEFAULT_LIFT_THRESHOLD = 0.05  # 5cm above table = picked up
DEFAULT_DROP_THRESHOLD = 0.03  # If height drops below 3cm after being lifted = dropped
DEFAULT_NEAR_GOAL_THRESHOLD = 0.12  # Within 12cm of goal = "missed goal" vs "dropped during transport"


def analyze_trajectory(
    trajectory: List[np.ndarray],
    task_completed: bool,
    goal_position: np.ndarray,
    lift_threshold: float = DEFAULT_LIFT_THRESHOLD,
    drop_threshold: float = DEFAULT_DROP_THRESHOLD,
    near_goal_threshold: float = DEFAULT_NEAR_GOAL_THRESHOLD,
) -> Tuple[Outcome, dict]:
    """Analyze object trajectory to determine outcome category.

    Args:
        trajectory: List of (x, y, z) object positions over time
        task_completed: Whether the task was completed (e.g., object in bowl)
        goal_position: XY position of the goal (e.g., bowl center)
        lift_threshold: Height above which object is considered "lifted"
        drop_threshold: Height below which a lifted object is considered "dropped"
        near_goal_threshold: Distance within which a drop is "missed goal" vs "dropped transport"

    Returns:
        Tuple of (Outcome, metrics_dict)
    """
    if task_completed:
        return Outcome.SUCCESS, {}

    if not trajectory:
        return Outcome.TIMEOUT, {}

    # Extract z positions (heights)
    heights = [pos[2] for pos in trajectory]
    max_height = max(heights)

    # Check if object was ever lifted
    was_lifted = max_height > lift_threshold

    if not was_lifted:
        return Outcome.NEVER_PICKED_UP, {
            "max_height": max_height,
            "was_lifted": False,
        }

    # Find when max height occurred
    max_height_idx = heights.index(max_height)

    # Check if it was dropped (height decreased significantly after being lifted)
    final_height = heights[-1]
    was_dropped = final_height < drop_threshold and max_height_idx < len(heights) - 10

    # Check final XY position relative to goal
    final_pos = trajectory[-1]
    final_xy = np.array([final_pos[0], final_pos[1]])
    goal_xy = np.array([goal_position[0], goal_position[1]])
    distance_to_goal = np.linalg.norm(final_xy - goal_xy)

    metrics = {
        "max_height": max_height,
        "was_lifted": True,
        "was_dropped": was_dropped,
        "final_distance_to_goal": distance_to_goal,
    }

    if was_dropped:
        # Was it dropped near the goal or far away?
        if distance_to_goal < near_goal_threshold:
            return Outcome.MISSED_GOAL, metrics
        else:
            return Outcome.DROPPED_DURING_TRANSPORT, metrics

    # If we got here, it's a timeout without clear resolution
    return Outcome.TIMEOUT, metrics


def compute_analysis_summary(results: List[EpisodeAnalysis]) -> dict:
    """Compute summary statistics from a list of episode analyses.

    Args:
        results: List of EpisodeAnalysis objects

    Returns:
        Dictionary with summary statistics
    """
    if not results:
        return {}

    # Count outcomes
    outcome_counts = {}
    for outcome in Outcome:
        outcome_counts[outcome] = sum(1 for r in results if r.outcome == outcome)

    # Success rate
    success_rate = outcome_counts[Outcome.SUCCESS] / len(results)

    # Pick rate (how often did it successfully pick up the object)
    pick_rate = sum(1 for r in results if r.was_lifted) / len(results)

    # Drop rate (of those picked up, how many were dropped)
    picked_results = [r for r in results if r.was_lifted]
    drop_rate = 0.0
    if picked_results:
        drop_rate = sum(1 for r in picked_results if r.was_dropped) / len(picked_results)

    # Average steps for successes
    success_results = [r for r in results if r.outcome == Outcome.SUCCESS]
    avg_steps_success = 0.0
    avg_time_success = 0.0
    if success_results:
        avg_steps_success = np.mean([r.steps for r in success_results])
        avg_time_success = np.mean([r.duration for r in success_results])

    # Max height stats
    avg_max_height = np.mean([r.max_height for r in results])

    return {
        "outcome_counts": outcome_counts,
        "success_rate": success_rate,
        "pick_rate": pick_rate,
        "drop_rate": drop_rate,
        "avg_steps_success": avg_steps_success,
        "avg_time_success": avg_time_success,
        "avg_max_height": avg_max_height,
        "total_episodes": len(results),
    }


def format_analysis_report(summary: dict, model_path: str = None) -> str:
    """Format analysis summary as a human-readable report.

    Args:
        summary: Output from compute_analysis_summary()
        model_path: Optional model path to include in report

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("EVALUATION RESULTS")
    lines.append("=" * 60)

    if model_path:
        lines.append(f"Model: {model_path}")
    lines.append(f"Episodes: {summary.get('total_episodes', 0)}")
    lines.append("")

    lines.append("OUTCOME BREAKDOWN:")
    lines.append("-" * 40)

    outcome_counts = summary.get("outcome_counts", {})
    total = summary.get("total_episodes", 1)

    for outcome in Outcome:
        count = outcome_counts.get(outcome, 0)
        pct = 100 * count / total if total > 0 else 0
        bar = "#" * int(pct / 2)
        lines.append(f"  {outcome.value:30s} {count:3d} ({pct:5.1f}%) {bar}")

    lines.append("")
    lines.append("SUMMARY STATISTICS:")
    lines.append("-" * 40)
    lines.append(f"  Success rate:        {summary.get('success_rate', 0)*100:.1f}%")
    lines.append(f"  Pick success rate:   {summary.get('pick_rate', 0)*100:.1f}%")

    if summary.get("pick_rate", 0) > 0:
        lines.append(f"  Drop rate (of picks): {summary.get('drop_rate', 0)*100:.1f}%")

    if summary.get("avg_steps_success", 0) > 0:
        lines.append(f"  Avg steps (success): {summary.get('avg_steps_success', 0):.1f}")
        lines.append(f"  Avg time (success):  {summary.get('avg_time_success', 0):.2f}s")

    lines.append(f"  Avg max height:      {summary.get('avg_max_height', 0):.3f}m")
    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def get_failure_analysis_text(results: List[EpisodeAnalysis]) -> str:
    """Generate detailed failure analysis text.

    Args:
        results: List of EpisodeAnalysis objects

    Returns:
        Formatted failure analysis text
    """
    failures = [r for r in results if r.outcome != Outcome.SUCCESS]
    if not failures:
        return "No failures to analyze."

    lines = []
    lines.append("FAILURE ANALYSIS:")
    lines.append("-" * 40)

    never_picked = [r for r in failures if r.outcome == Outcome.NEVER_PICKED_UP]
    if never_picked:
        lines.append(f"  Never picked up ({len(never_picked)} episodes):")
        lines.append(f"    - Robot failed to grasp the object")
        lines.append(f"    - Avg max height: {np.mean([r.max_height for r in never_picked]):.3f}m")

    dropped = [r for r in failures if r.outcome == Outcome.DROPPED_DURING_TRANSPORT]
    if dropped:
        lines.append(f"  Dropped during transport ({len(dropped)} episodes):")
        lines.append(f"    - Picked up but dropped before reaching goal area")
        lines.append(f"    - Avg distance to goal: {np.mean([r.final_distance_to_goal for r in dropped]):.3f}m")

    missed = [r for r in failures if r.outcome == Outcome.MISSED_GOAL]
    if missed:
        lines.append(f"  Missed goal ({len(missed)} episodes):")
        lines.append(f"    - Got to goal area but placed outside")
        lines.append(f"    - Avg distance to goal center: {np.mean([r.final_distance_to_goal for r in missed]):.3f}m")

    timeouts = [r for r in failures if r.outcome == Outcome.TIMEOUT]
    if timeouts:
        lines.append(f"  Timeout ({len(timeouts)} episodes):")
        lines.append(f"    - Ran out of time without completing task")
        lines.append(f"    - Avg max height: {np.mean([r.max_height for r in timeouts]):.3f}m")

    return "\n".join(lines)
