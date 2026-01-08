#!/usr/bin/env python
"""
Evaluation script for Pi0/Pi0.5 models in SO-101 simulation.

This script evaluates a trained Pi0/Pi0.5 model on the pick-and-place task
using the SO-101 simulation environment.

Usage:
    # Evaluate a trained model
    python scripts/openpi/eval_pi0.py --checkpoint path/to/checkpoint \
        --episodes 50 --model pi0

    # With visualization
    python scripts/openpi/eval_pi0.py --checkpoint path/to/checkpoint \
        --episodes 20 --visualize

    # With MuJoCo viewer
    python scripts/openpi/eval_pi0.py --checkpoint path/to/checkpoint \
        --episodes 10 --mujoco-viewer
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# Add project root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.openpi.so101_policy import (
    SO101Config,
    SO101InputTransform,
    SO101OutputTransform,
    make_so101_policy_config,
    create_transforms,
)


def check_openpi_installed():
    """Check if openpi is installed."""
    try:
        import openpi
        return True
    except ImportError:
        return False


def load_model(model_type: str, checkpoint_path: Optional[str] = None):
    """Load Pi0/Pi0.5 model.

    Args:
        model_type: "pi0" or "pi0.5"
        checkpoint_path: Optional path to checkpoint

    Returns:
        Loaded model
    """
    if not check_openpi_installed():
        raise ImportError(
            "openpi not installed. Install with:\n"
            "  git clone https://github.com/Physical-Intelligence/openpi.git\n"
            "  cd openpi && pip install -e ."
        )

    from openpi.models import load_model as openpi_load_model

    model = openpi_load_model(model_type)

    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        # Load weights from checkpoint
        # This depends on the specific openpi API
        pass

    return model


class Pi0Policy:
    """Wrapper for Pi0/Pi0.5 policy for use with SO-101 sim."""

    def __init__(
        self,
        model,
        config: SO101Config,
        language_instruction: str = "Pick up the block and place it in the bowl",
    ):
        self.model = model
        self.config = config
        self.language = language_instruction

        # Create transforms
        self.input_transform, self.output_transform = create_transforms(config)

        # Internal state
        self.step_count = 0

    def reset(self):
        """Reset policy state between episodes."""
        self.step_count = 0
        # Reset any internal model state if needed

    def __call__(
        self,
        state: np.ndarray,
        images: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Get action from policy.

        Args:
            state: Robot state (joint positions + gripper)
            images: Dict of camera name -> image

        Returns:
            Robot action (joint targets + gripper)
        """
        # Transform observations
        obs = self.input_transform(state, images, self.language)

        # Get model prediction
        # This depends on the specific openpi API
        # action_normalized = self.model.predict(obs)
        action_normalized = np.zeros(7)  # Placeholder

        # Transform action
        action = self.output_transform(action_normalized)

        self.step_count += 1
        return action


def run_evaluation(
    policy: Pi0Policy,
    num_episodes: int = 50,
    max_steps: int = 300,
    randomize: bool = True,
    visualize: bool = False,
    mujoco_viewer: bool = False,
    verbose: bool = True,
) -> Tuple[float, float, float, Dict]:
    """Run evaluation episodes.

    Args:
        policy: Pi0 policy wrapper
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        randomize: Whether to randomize object positions
        visualize: Show camera feeds (OpenCV window)
        mujoco_viewer: Show MuJoCo 3D viewer
        verbose: Print progress

    Returns:
        Tuple of (success_rate, avg_steps, avg_time, failure_summary)
    """
    try:
        from utils.so100_sim import SO100Sim, SO100SimConfig
    except ImportError:
        print("Error: SO100Sim not available")
        sys.exit(1)

    # Create simulation
    sim_config = SO100SimConfig(
        camera_width=640,
        camera_height=480,
    )
    sim = SO100Sim(sim_config)

    # Statistics
    successes = 0
    total_steps = 0
    total_time = 0.0
    failures = {"NEVER_PICKED_UP": 0, "DROPPED": 0, "TIMEOUT": 0, "OTHER": 0}

    for ep in range(num_episodes):
        if verbose:
            print(f"\nEpisode {ep + 1}/{num_episodes}")

        # Reset
        sim.reset_scene(randomize=randomize)
        policy.reset()

        ep_start = time.time()
        success = False

        for step in range(max_steps):
            # Get observation
            state = sim.get_state()
            images = {
                "overhead_cam": sim.get_camera_image("overhead_cam"),
                "wrist_cam": sim.get_camera_image("wrist_cam"),
            }

            # Get action from policy
            action = policy(state, images)

            # Apply action
            sim.step(action)

            # Check success
            if sim.check_success():
                success = True
                break

            # Visualization
            if visualize:
                import cv2
                combined = np.hstack([images["overhead_cam"], images["wrist_cam"]])
                cv2.imshow("Cameras", combined[:, :, ::-1])  # RGB to BGR
                cv2.waitKey(1)

        ep_time = time.time() - ep_start

        if success:
            successes += 1
            total_steps += step + 1
            if verbose:
                print(f"  SUCCESS in {step + 1} steps ({ep_time:.2f}s)")
        else:
            # Analyze failure
            if step >= max_steps - 1:
                failures["TIMEOUT"] += 1
            else:
                failures["OTHER"] += 1
            if verbose:
                print(f"  FAILED after {step + 1} steps ({ep_time:.2f}s)")

        total_time += ep_time

    # Calculate metrics
    success_rate = successes / num_episodes
    avg_steps = total_steps / max(successes, 1)
    avg_time = total_time / num_episodes

    if verbose:
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Success Rate: {success_rate * 100:.1f}%")
        print(f"Avg Steps (success): {avg_steps:.1f}")
        print(f"Avg Time: {avg_time:.2f}s")
        print(f"Failures: {failures}")
        print("=" * 60)

    if visualize:
        cv2.destroyAllWindows()

    return success_rate, avg_steps, avg_time, failures


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Pi0/Pi0.5 model on SO-101 simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--model", type=str, default="pi0", choices=["pi0", "pi0.5"],
                        help="Model type (default: pi0)")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of evaluation episodes (default: 50)")
    parser.add_argument("--max-steps", type=int, default=300,
                        help="Max steps per episode (default: 300)")
    parser.add_argument("--language", type=str,
                        default="Pick up the block and place it in the bowl",
                        help="Language instruction")
    parser.add_argument("--no-randomize", action="store_true",
                        help="Don't randomize object positions")
    parser.add_argument("--visualize", action="store_true",
                        help="Show camera feeds during evaluation")
    parser.add_argument("--mujoco-viewer", action="store_true",
                        help="Show MuJoCo 3D viewer")

    args = parser.parse_args()

    print("=" * 60)
    print("Pi0/Pi0.5 Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes: {args.episodes}")
    print(f"Language: {args.language}")
    print("=" * 60)

    # Check if openpi is installed
    if not check_openpi_installed():
        print("\nError: openpi not installed.")
        print("This script requires the openpi package for model loading.")
        print("\nTo install:")
        print("  git clone https://github.com/Physical-Intelligence/openpi.git")
        print("  cd openpi && pip install -e .")
        print("\nAlternatively, for testing without openpi, use a placeholder policy.")
        sys.exit(1)

    # Load model
    model = load_model(args.model, args.checkpoint)

    # Create policy wrapper
    config = make_so101_policy_config(language=args.language)
    policy = Pi0Policy(model, config, args.language)

    # Run evaluation
    success_rate, avg_steps, avg_time, failures = run_evaluation(
        policy=policy,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        randomize=not args.no_randomize,
        visualize=args.visualize,
        mujoco_viewer=args.mujoco_viewer,
    )

    print(f"\nFinal Success Rate: {success_rate * 100:.1f}%")


if __name__ == "__main__":
    main()
