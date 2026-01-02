#!/usr/bin/env python
"""
Evaluate ACT policy with detailed failure analysis.

Tracks duplo position throughout each episode to categorize outcomes:
- Success: Task completed (duplo in bowl)
- Never picked up: Duplo never lifted significantly
- Dropped during transport: Duplo was lifted but fell before reaching bowl
- Missed bowl: Duplo placed near but outside bowl
- Timeout: Max steps reached without clear outcome

Usage:
    python inference/evaluate_with_analysis.py outputs/train/act_20251231_114817/checkpoint_035000
    python inference/evaluate_with_analysis.py outputs/train/act_20251231_114817/final --episodes 50
"""

import argparse
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

import torch
from tqdm import tqdm

# Add project root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

from lerobot_robot_sim import SO100SimConfig, SO100Sim
import mujoco


class Outcome(Enum):
    SUCCESS = "success"
    NEVER_PICKED_UP = "never_picked_up"
    DROPPED_DURING_TRANSPORT = "dropped_during_transport"
    MISSED_BOWL = "missed_bowl"
    TIMEOUT = "timeout"


@dataclass
class EpisodeResult:
    outcome: Outcome
    steps: int
    duration: float
    max_height: float
    was_lifted: bool
    was_dropped: bool
    final_distance_to_bowl: float
    trajectory: list = field(default_factory=list)


MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Thresholds for detection
LIFT_THRESHOLD = 0.05  # 5cm above table = picked up
DROP_THRESHOLD = 0.03  # If height drops below 3cm after being lifted = dropped
BOWL_POSITION = np.array([0.217, -0.225])
BOWL_RADIUS = 0.06  # Bowl half-size
NEAR_BOWL_THRESHOLD = 0.12  # Within 12cm of bowl = "missed bowl" vs "dropped during transport"


def prepare_obs_for_policy(obs: dict, device: torch.device) -> dict:
    """Convert simulation observation to policy input format."""
    batch = {}

    state = []
    for motor in MOTOR_NAMES:
        key = f"{motor}.pos"
        state.append(obs.get(key, 0.0))
    batch["observation.state"] = torch.tensor([state], dtype=torch.float32, device=device)

    for key, value in obs.items():
        if isinstance(value, np.ndarray) and value.ndim == 3:
            img = torch.from_numpy(value).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            batch[f"observation.images.{key}"] = img.to(device)

    return batch


def get_duplo_position(sim_robot) -> np.ndarray:
    """Get current duplo position from simulation."""
    duplo_body_id = mujoco.mj_name2id(sim_robot.mj_model, mujoco.mjtObj.mjOBJ_BODY, "duplo")
    return sim_robot.mj_data.xpos[duplo_body_id].copy()


def analyze_episode(
    trajectory: list[np.ndarray],
    task_completed: bool,
    steps: int,
    max_steps: int
) -> Outcome:
    """Analyze trajectory to determine outcome category."""

    if task_completed:
        return Outcome.SUCCESS

    # Extract z positions (heights)
    heights = [pos[2] for pos in trajectory]
    max_height = max(heights)

    # Check if duplo was ever lifted
    was_lifted = max_height > LIFT_THRESHOLD

    if not was_lifted:
        return Outcome.NEVER_PICKED_UP

    # Find when max height occurred
    max_height_idx = heights.index(max_height)

    # Check if it was dropped (height decreased significantly after being lifted)
    final_height = heights[-1]
    was_dropped = final_height < DROP_THRESHOLD and max_height_idx < len(heights) - 10

    # Check final XY position relative to bowl
    final_pos = trajectory[-1]
    final_xy = np.array([final_pos[0], final_pos[1]])
    distance_to_bowl = np.linalg.norm(final_xy - BOWL_POSITION)

    if was_dropped:
        # Was it dropped near the bowl or far away?
        if distance_to_bowl < NEAR_BOWL_THRESHOLD:
            return Outcome.MISSED_BOWL
        else:
            return Outcome.DROPPED_DURING_TRANSPORT

    # If we got here, it's a timeout without clear resolution
    return Outcome.TIMEOUT


def run_episode(
    policy,
    preprocessor,
    postprocessor,
    sim_robot,
    device: torch.device,
    max_steps: int = 300,
    randomize: bool = True
) -> EpisodeResult:
    """Run a single episode and return detailed results."""

    policy.reset()
    sim_robot.reset_scene(randomize=randomize, pos_range=0.04, rot_range=np.pi)

    trajectory = []
    start_time = time.time()
    task_completed = False

    for step in range(max_steps):
        # Record duplo position
        duplo_pos = get_duplo_position(sim_robot)
        trajectory.append(duplo_pos)

        # Get observation and run policy
        obs = sim_robot.get_observation()
        batch = prepare_obs_for_policy(obs, device)
        batch = preprocessor(batch)

        with torch.no_grad():
            action = policy.select_action(batch)
        action = postprocessor(action)

        # Execute action
        action_np = action.cpu().numpy()
        if action_np.ndim > 1:
            action_np = action_np.flatten()[:6]
        action_dict = {f"{MOTOR_NAMES[i]}.pos": float(action_np[i]) for i in range(6)}
        sim_robot.send_action(action_dict)

        if sim_robot.is_task_complete():
            task_completed = True
            break

    duration = time.time() - start_time
    steps_taken = step + 1

    # Final position
    final_pos = get_duplo_position(sim_robot)
    trajectory.append(final_pos)

    # Analyze trajectory
    heights = [pos[2] for pos in trajectory]
    max_height = max(heights)
    was_lifted = max_height > LIFT_THRESHOLD
    was_dropped = was_lifted and heights[-1] < DROP_THRESHOLD

    final_xy = np.array([final_pos[0], final_pos[1]])
    final_distance_to_bowl = np.linalg.norm(final_xy - BOWL_POSITION)

    outcome = analyze_episode(trajectory, task_completed, steps_taken, max_steps)

    return EpisodeResult(
        outcome=outcome,
        steps=steps_taken,
        duration=duration,
        max_height=max_height,
        was_lifted=was_lifted,
        was_dropped=was_dropped,
        final_distance_to_bowl=final_distance_to_bowl,
        trajectory=trajectory
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate ACT policy with failure analysis")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=30, help="Number of episodes (default: 30)")
    parser.add_argument("--max_steps", type=int, default=300, help="Max steps per episode (default: 300)")
    parser.add_argument("--no_randomize", action="store_true", help="Disable randomization")
    parser.add_argument("--device", type=str, default="cuda", help="Device (default: cuda)")
    parser.add_argument("--verbose", action="store_true", help="Print per-episode details")

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    device = torch.device(args.device)

    print(f"Loading model from: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Episodes: {args.episodes}")
    print(f"Randomization: {'disabled' if args.no_randomize else 'enabled'}")
    print()

    # Load policy
    policy = ACTPolicy.from_pretrained(str(checkpoint_path))
    policy.eval()
    policy.to(device)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint_path),
    )

    # Initialize simulation
    config = SO100SimConfig(
        sim_cameras=["wrist_cam", "overhead_cam"],
        enable_vr=False,
        camera_width=640,
        camera_height=480,
    )
    sim_robot = SO100Sim(config)
    sim_robot.connect()

    # Run episodes
    results: list[EpisodeResult] = []

    print("Running evaluation...")
    for ep in tqdm(range(args.episodes), desc="Episodes"):
        result = run_episode(
            policy, preprocessor, postprocessor, sim_robot, device,
            max_steps=args.max_steps,
            randomize=not args.no_randomize
        )
        results.append(result)

        if args.verbose:
            print(f"  Episode {ep+1}: {result.outcome.value} "
                  f"(steps={result.steps}, max_h={result.max_height:.3f}, "
                  f"dist_to_bowl={result.final_distance_to_bowl:.3f})")

    sim_robot.disconnect()

    # Summarize results
    print()
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model: {checkpoint_path}")
    print(f"Episodes: {args.episodes}")
    print()

    # Count outcomes
    outcome_counts = {}
    for outcome in Outcome:
        outcome_counts[outcome] = sum(1 for r in results if r.outcome == outcome)

    print("OUTCOME BREAKDOWN:")
    print("-" * 40)
    for outcome, count in outcome_counts.items():
        pct = 100 * count / len(results)
        bar = "#" * int(pct / 2)
        print(f"  {outcome.value:30s} {count:3d} ({pct:5.1f}%) {bar}")

    print()
    print("SUMMARY STATISTICS:")
    print("-" * 40)

    # Success rate
    success_rate = outcome_counts[Outcome.SUCCESS] / len(results)
    print(f"  Success rate:        {success_rate*100:.1f}%")

    # Pick rate (how often did it successfully pick up the duplo)
    pick_rate = sum(1 for r in results if r.was_lifted) / len(results)
    print(f"  Pick success rate:   {pick_rate*100:.1f}%")

    # Drop rate (of those picked up, how many were dropped)
    picked_results = [r for r in results if r.was_lifted]
    if picked_results:
        drop_rate = sum(1 for r in picked_results if r.was_dropped) / len(picked_results)
        print(f"  Drop rate (of picks): {drop_rate*100:.1f}%")

    # Average steps for successes
    success_results = [r for r in results if r.outcome == Outcome.SUCCESS]
    if success_results:
        avg_steps = np.mean([r.steps for r in success_results])
        avg_time = np.mean([r.duration for r in success_results])
        print(f"  Avg steps (success): {avg_steps:.1f}")
        print(f"  Avg time (success):  {avg_time:.2f}s")

    # Max height stats
    avg_max_height = np.mean([r.max_height for r in results])
    print(f"  Avg max height:      {avg_max_height:.3f}m")

    print()
    print("=" * 60)

    # Failure analysis
    failures = [r for r in results if r.outcome != Outcome.SUCCESS]
    if failures:
        print()
        print("FAILURE ANALYSIS:")
        print("-" * 40)

        never_picked = [r for r in failures if r.outcome == Outcome.NEVER_PICKED_UP]
        if never_picked:
            print(f"  Never picked up ({len(never_picked)} episodes):")
            print(f"    - Robot failed to grasp the duplo")
            print(f"    - Avg max height: {np.mean([r.max_height for r in never_picked]):.3f}m")

        dropped = [r for r in failures if r.outcome == Outcome.DROPPED_DURING_TRANSPORT]
        if dropped:
            print(f"  Dropped during transport ({len(dropped)} episodes):")
            print(f"    - Picked up but dropped before reaching bowl area")
            print(f"    - Avg distance to bowl: {np.mean([r.final_distance_to_bowl for r in dropped]):.3f}m")

        missed = [r for r in failures if r.outcome == Outcome.MISSED_BOWL]
        if missed:
            print(f"  Missed bowl ({len(missed)} episodes):")
            print(f"    - Got to bowl area but placed outside")
            print(f"    - Avg distance to bowl center: {np.mean([r.final_distance_to_bowl for r in missed]):.3f}m")

        timeouts = [r for r in failures if r.outcome == Outcome.TIMEOUT]
        if timeouts:
            print(f"  Timeout ({len(timeouts)} episodes):")
            print(f"    - Ran out of time without completing task")
            print(f"    - Avg max height: {np.mean([r.max_height for r in timeouts]):.3f}m")

    print()


if __name__ == "__main__":
    main()
