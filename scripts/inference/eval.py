#!/usr/bin/env python
"""
Unified evaluation script for ACT and SmolVLA policies.

This script uses the same run_evaluation() function as the training scripts,
ensuring consistent evaluation between training checkpoints and standalone evaluation.

Usage:
    # Single checkpoint from HuggingFace
    python scripts/inference/eval.py danbhf/act_sim_pick_place_100k --checkpoint final --episodes 50

    # All checkpoints from HuggingFace
    python scripts/inference/eval.py danbhf/act_sim_pick_place_100k --all-checkpoints --episodes 20

    # Local path
    python scripts/inference/eval.py outputs/train/smolvla_20260106_014309 --local --checkpoint final

    # With camera feed visualization
    python scripts/inference/eval.py danbhf/act_sim_pick_place_100k --visualize --episodes 10

    # With MuJoCo 3D viewer
    python scripts/inference/eval.py danbhf/act_sim_pick_place_100k --mujoco-viewer --episodes 10

    # SmolVLA with custom language instruction
    python scripts/inference/eval.py danbhf/smolvla_sim_pick_place_200k --policy smolvla --language "Pick up the red block"
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from huggingface_hub import snapshot_download

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

from utils.training import run_evaluation


def find_checkpoints(model_dir: Path) -> list:
    """Find all checkpoint directories in a model directory."""
    checkpoints = []
    for d in model_dir.iterdir():
        if d.is_dir() and (d.name.startswith("checkpoint_") or d.name == "final"):
            if (d / "config.json").exists():
                checkpoints.append(d.name)
    # Sort by step number
    def sort_key(name):
        if name == "final":
            return float('inf')
        match = re.search(r'(\d+)', name)
        return int(match.group(1)) if match else 0
    return sorted(checkpoints, key=sort_key)


def load_policy_and_processors(model_path: Path, policy_type: str, device: torch.device, dataset: str = None):
    """Load policy and pre/post processors from a checkpoint."""
    # Load policy
    if policy_type == "act":
        policy = ACTPolicy.from_pretrained(str(model_path))
    else:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        policy = SmolVLAPolicy.from_pretrained(str(model_path))

    policy.eval()
    policy.to(device)

    # Try to load pre/post processors
    try:
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            pretrained_path=str(model_path)
        )
    except Exception as e:
        if dataset:
            from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
            dataset_metadata = LeRobotDatasetMetadata(dataset)
            preprocessor, postprocessor = make_pre_post_processors(
                policy.config,
                dataset_stats=dataset_metadata.stats
            )
        else:
            raise RuntimeError(
                f"Failed to load pre/post processors from {model_path}. "
                "Use --dataset to specify the training dataset."
            ) from e

    return policy, preprocessor, postprocessor


def evaluate_checkpoint(
    model_path: Path,
    policy_type: str,
    device: torch.device,
    episodes: int,
    language: str,
    dataset: str,
    visualize: bool,
    mujoco_viewer: bool,
    max_steps: int,
    temporal_ensemble_coeff: float = None,
    block_x: float = None,
    block_y: float = None,
) -> dict:
    """Evaluate a single checkpoint and return results."""
    policy, preprocessor, postprocessor = load_policy_and_processors(
        model_path, policy_type, device, dataset
    )

    # Detect depth cameras
    depth_cameras = []
    for key in policy.config.input_features.keys():
        if "_depth" in key:
            cam_name = key.replace("observation.images.", "")
            base_cam = cam_name.replace("_depth", "")
            if base_cam not in depth_cameras:
                depth_cameras.append(base_cam)

    # Get action dim
    try:
        action_dim = policy.config.output_features['action'].shape[0]
    except:
        action_dim = 6

    # Run evaluation
    lang = language if policy_type == "smolvla" else None
    results = run_evaluation(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=device,
        num_episodes=episodes,
        randomize=True,
        action_dim=action_dim,
        depth_cameras=depth_cameras,
        language_instruction=lang,
        max_steps=max_steps,
        verbose=True,
        analyze_failures=True,
        visualize=visualize,
        mujoco_viewer=mujoco_viewer,
        temporal_ensemble_coeff=temporal_ensemble_coeff,
        block_x=block_x,
        block_y=block_y,
    )

    success_rate, avg_steps, avg_time, ik_failure_rate, avg_ik_error, failure_summary = results
    return {
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "avg_time": avg_time,
        "ik_failure_rate": ik_failure_rate,
        "avg_ik_error": avg_ik_error,
        "failure_summary": failure_summary,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluation for ACT and SmolVLA policies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("path", type=str,
                        help="HuggingFace repo ID or local path (with --local)")
    parser.add_argument("--checkpoint", type=str, default="final",
                        help="Checkpoint name (default: final)")
    parser.add_argument("--all-checkpoints", action="store_true",
                        help="Evaluate all checkpoints and show comparison table")
    parser.add_argument("--local", action="store_true",
                        help="Treat path as local directory instead of HuggingFace repo")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of evaluation episodes (default: 50)")
    parser.add_argument("--policy", type=str, choices=["act", "smolvla"], default=None,
                        help="Policy type (auto-detected if not specified)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (default: cuda)")
    parser.add_argument("--language", type=str, default="Pick up the block and place it in the bowl",
                        help="Language instruction for SmolVLA models")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset repo ID to create processors from (if not saved)")
    parser.add_argument("--visualize", action="store_true",
                        help="Show camera feeds during evaluation (OpenCV window)")
    parser.add_argument("--mujoco-viewer", action="store_true",
                        help="Open MuJoCo 3D viewer window")
    parser.add_argument("--max-steps", type=int, default=300,
                        help="Max steps per episode (default: 300)")
    parser.add_argument("--ensemble", type=float, default=None, metavar="COEFF",
                        help="Enable temporal ensembling with given coefficient (e.g., 0.01). "
                             "Predicts every step and averages overlapping chunks.")
    parser.add_argument("--block-x", type=float, default=None,
                        help="X position for block center (default: scene XML default)")
    parser.add_argument("--block-y", type=float, default=None,
                        help="Y position for block center (default: scene XML default)")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get model directory
    if args.local:
        model_dir = Path(args.path)
        if not model_dir.exists():
            raise FileNotFoundError(f"Local path not found: {model_dir}")
        print(f"Using local path: {model_dir}")
    else:
        # Download from HuggingFace
        if args.all_checkpoints:
            print(f"Downloading all checkpoints from {args.path}...")
            model_dir = Path(snapshot_download(
                repo_id=args.path,
                local_dir=f"./eval_cache/{args.path.replace('/', '_')}"
            ))
        else:
            print(f"Downloading {args.path}/{args.checkpoint}...")
            model_dir = Path(snapshot_download(
                repo_id=args.path,
                allow_patterns=[f"{args.checkpoint}/*"],
                local_dir=f"./eval_cache/{args.path.replace('/', '_')}"
            ))
        print(f"Downloaded to: {model_dir}")

    # Determine checkpoints to evaluate
    if args.all_checkpoints:
        checkpoints = find_checkpoints(model_dir)
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {model_dir}")
        print(f"Found {len(checkpoints)} checkpoints: {checkpoints}")
    else:
        checkpoints = [args.checkpoint]

    # Auto-detect policy type from first checkpoint
    first_checkpoint_path = model_dir / checkpoints[0]
    config_path = first_checkpoint_path / "config.json"
    if args.policy is None:
        with open(config_path) as f:
            config = json.load(f)
        # Check both "type" field and "_class_name" for policy detection
        policy_type = config.get("type", "").lower()
        class_name = config.get("_class_name", "").lower()
        if "smolvla" in policy_type or "smolvla" in class_name:
            args.policy = "smolvla"
        else:
            args.policy = "act"
        print(f"Auto-detected policy type: {args.policy}")

    # Evaluate checkpoints
    all_results = {}

    for i, checkpoint_name in enumerate(checkpoints):
        model_path = model_dir / checkpoint_name
        print(f"\n{'='*60}")
        print(f"Evaluating checkpoint {i+1}/{len(checkpoints)}: {checkpoint_name}")
        print(f"{'='*60}")

        results = evaluate_checkpoint(
            model_path=model_path,
            policy_type=args.policy,
            device=device,
            episodes=args.episodes,
            language=args.language,
            dataset=args.dataset,
            visualize=args.visualize,
            mujoco_viewer=args.mujoco_viewer,
            max_steps=args.max_steps,
            temporal_ensemble_coeff=args.ensemble,
            block_x=args.block_x,
            block_y=args.block_y,
        )
        all_results[checkpoint_name] = results

        # Print single result
        print(f"\n  Success Rate: {results['success_rate']*100:.1f}%")
        print(f"  Avg Steps: {results['avg_steps']:.1f}")
        if results['ik_failure_rate'] is not None:
            print(f"  IK Failure Rate: {results['ik_failure_rate']*100:.2f}%")

    # Print summary table for batch evaluation
    if len(checkpoints) > 1:
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"{'Checkpoint':<25} {'Success':<12} {'Avg Steps':<12} {'IK Fail %':<12}")
        print("-"*80)

        best_checkpoint = None
        best_success = -1

        for checkpoint_name, results in all_results.items():
            ik_str = f"{results['ik_failure_rate']*100:.1f}%" if results['ik_failure_rate'] is not None else "N/A"
            print(f"{checkpoint_name:<25} {results['success_rate']*100:.1f}%{'':<6} {results['avg_steps']:<12.1f} {ik_str:<12}")

            if results['success_rate'] > best_success:
                best_success = results['success_rate']
                best_checkpoint = checkpoint_name

        print("-"*80)
        print(f"Best checkpoint: {best_checkpoint} ({best_success*100:.1f}% success)")
        print("="*80)
    else:
        # Single checkpoint - print detailed results
        checkpoint_name = checkpoints[0]
        results = all_results[checkpoint_name]

        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"  Model: {args.path}/{checkpoint_name}")
        print(f"  Episodes: {args.episodes}")
        print(f"  Success Rate: {results['success_rate']*100:.1f}%")
        print(f"  Avg Steps: {results['avg_steps']:.1f}")
        print(f"  Avg Time: {results['avg_time']:.2f}s")
        if results['ik_failure_rate'] is not None:
            print(f"  IK Failure Rate: {results['ik_failure_rate']*100:.2f}%")
            print(f"  Avg IK Error: {results['avg_ik_error']:.2f}mm")
        print("="*60)

        if results['failure_summary']:
            print("\nFailure Analysis:")
            for key, value in results['failure_summary'].items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
