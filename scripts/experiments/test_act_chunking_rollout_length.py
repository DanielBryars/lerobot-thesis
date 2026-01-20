#!/usr/bin/env python3
"""
Test ACT policy with different n_action_steps (rollout length before re-prediction).

This experiment varies how often ACT re-predicts action chunks:
- n_action_steps=1: Re-predict every step (most reactive, like temporal ensemble)
- n_action_steps=100: Use full chunk before re-predicting (default, open-loop)

Uses the same run_evaluation infrastructure as the main eval.py to ensure consistency.

Usage:
    python scripts/experiments/test_act_chunking_rollout_length.py outputs/train/act_20260118_155135 --checkpoint checkpoint_045000 --episodes 10
    python scripts/experiments/test_act_chunking_rollout_length.py outputs/train/act_20260118_155135 --checkpoint checkpoint_045000 --episodes 5 --n-action-steps "1,5,10,50,100"
"""

import argparse
import json
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import torch

# Add project paths
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

from utils.training import run_evaluation

# Default n_action_steps values to test
DEFAULT_TEST_VALUES = [1, 2, 5, 10, 20, 50, 100]


def load_policy_and_processors(model_path: Path, device: torch.device):
    """Load ACT policy and processors."""
    print(f"Loading ACT from {model_path}...")
    policy = ACTPolicy.from_pretrained(str(model_path))
    policy.to(device)
    policy.eval()

    # Load processors
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(model_path)
    )

    chunk_size = policy.config.chunk_size
    original_n_action_steps = policy.config.n_action_steps
    print(f"  chunk_size={chunk_size}, original n_action_steps={original_n_action_steps}")

    return policy, preprocessor, postprocessor, chunk_size


def run_experiment(
    model_path: Path,
    n_action_steps_values: list,
    episodes_per_setting: int,
    device: torch.device,
    max_steps: int = 300,
) -> dict:
    """Run the full chunking experiment."""

    # Load policy once
    policy, preprocessor, postprocessor, chunk_size = load_policy_and_processors(model_path, device)
    original_n_action_steps = policy.config.n_action_steps

    # Filter out values larger than chunk_size
    valid_values = [v for v in n_action_steps_values if v <= chunk_size]
    if len(valid_values) < len(n_action_steps_values):
        skipped = [v for v in n_action_steps_values if v > chunk_size]
        print(f"Skipping n_action_steps > chunk_size ({chunk_size}): {skipped}")

    # Get action dim
    try:
        action_dim = policy.config.output_features['action'].shape[0]
    except:
        action_dim = 6

    results = {
        "model_path": str(model_path),
        "chunk_size": chunk_size,
        "original_n_action_steps": original_n_action_steps,
        "episodes_per_setting": episodes_per_setting,
        "max_steps": max_steps,
        "timestamp": datetime.now().isoformat(),
        "settings": {},
    }

    print(f"\n{'='*70}")
    print(f"ACT Chunking Rollout Length Experiment")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Chunk size: {chunk_size}")
    print(f"Episodes per setting: {episodes_per_setting}")
    print(f"Testing n_action_steps: {valid_values}")
    print(f"{'='*70}\n")

    for n_action_steps in valid_values:
        print(f"\n{'='*60}")
        print(f"Testing n_action_steps = {n_action_steps}")
        print(f"{'='*60}")

        # Modify config for this test
        policy.config.n_action_steps = n_action_steps
        # Reset the action queue with new maxlen
        policy._action_queue = deque([], maxlen=n_action_steps)

        # Run evaluation using the same infrastructure as eval.py
        eval_results = run_evaluation(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
            num_episodes=episodes_per_setting,
            randomize=True,
            action_dim=action_dim,
            depth_cameras=[],
            language_instruction=None,
            max_steps=max_steps,
            verbose=True,
            analyze_failures=True,
            visualize=False,
            mujoco_viewer=False,
        )

        success_rate, avg_steps, avg_time, ik_failure_rate, avg_ik_error, failure_summary = eval_results

        # Convert failure_summary keys to strings (Outcome enum -> str)
        failure_summary_serializable = {}
        if failure_summary:
            for k, v in failure_summary.items():
                key = str(k.name) if hasattr(k, 'name') else str(k)
                failure_summary_serializable[key] = v

        setting_result = {
            "n_action_steps": n_action_steps,
            "success_rate": success_rate * 100,  # Convert to percentage
            "avg_steps": avg_steps,
            "avg_time": avg_time,
            "failure_summary": failure_summary_serializable,
        }
        results["settings"][str(n_action_steps)] = setting_result

        print(f"\n  Results for n_action_steps={n_action_steps}:")
        print(f"    Success rate: {success_rate*100:.1f}%")
        print(f"    Avg steps: {avg_steps:.1f}")
        print(f"    Avg time: {avg_time:.2f}s")

    # Restore original config
    policy.config.n_action_steps = original_n_action_steps
    policy._action_queue = deque([], maxlen=original_n_action_steps)

    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'n_action_steps':>15} | {'Success Rate':>12} | {'Avg Steps':>10} | {'Avg Time':>10}")
    print(f"{'-'*15}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}")

    for n_steps in valid_values:
        r = results["settings"][str(n_steps)]
        print(f"{n_steps:>15} | {r['success_rate']:>11.1f}% | {r['avg_steps']:>10.1f} | {r['avg_time']:>9.2f}s")

    print(f"{'='*70}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Test ACT chunking rollout length strategies")
    parser.add_argument("path", type=str,
                        help="Local path to model directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_045000",
                        help="Checkpoint name (default: checkpoint_045000)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Episodes per n_action_steps setting")
    parser.add_argument("--max-steps", type=int, default=300,
                        help="Max steps per episode")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--n-action-steps", type=str, default=None,
                        help="Comma-separated n_action_steps values to test (default: 1,2,5,10,20,50,100)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    args = parser.parse_args()

    # Parse n_action_steps values
    if args.n_action_steps:
        n_action_steps_values = [int(x.strip()) for x in args.n_action_steps.split(",")]
    else:
        n_action_steps_values = DEFAULT_TEST_VALUES

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Build model path
    model_path = Path(args.path) / args.checkpoint
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    # Run experiment
    results = run_experiment(
        model_path=model_path,
        n_action_steps_values=n_action_steps_values,
        episodes_per_setting=args.episodes,
        device=device,
        max_steps=args.max_steps,
    )

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = REPO_ROOT / "outputs" / "experiments" / f"act_chunking_rollout_{timestamp}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
