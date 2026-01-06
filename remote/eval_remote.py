#!/usr/bin/env python
"""
Evaluate a trained policy from HuggingFace on simulation.

Usage:
    python remote/eval_remote.py danbhf/act_sim_pick_place_100k --checkpoint final --episodes 50
    python remote/eval_remote.py danbhf/act_sim_pick_place_100k --checkpoint checkpoint_050000 --episodes 30
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from huggingface_hub import hf_hub_download, snapshot_download

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

from utils.training import run_evaluation


def main():
    parser = argparse.ArgumentParser(description="Evaluate a policy from HuggingFace")
    parser.add_argument("repo_id", type=str, help="HuggingFace repo ID (e.g., danbhf/act_sim_pick_place_100k)")
    parser.add_argument("--checkpoint", type=str, default="final",
                        help="Checkpoint name (default: final)")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of evaluation episodes (default: 50)")
    parser.add_argument("--policy", type=str, choices=["act", "smolvla"], default=None,
                        help="Policy type (auto-detected if not specified)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (default: cuda)")
    parser.add_argument("--language", type=str, default="Pick up the block and place it in the bowl",
                        help="Language instruction for VLA models")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Download checkpoint
    checkpoint_path = args.checkpoint if args.checkpoint != "final" else "final"
    print(f"Downloading {args.repo_id}/{checkpoint_path}...")

    local_dir = snapshot_download(
        repo_id=args.repo_id,
        allow_patterns=[f"{checkpoint_path}/*"],
        local_dir=f"./eval_cache/{args.repo_id.replace('/', '_')}"
    )
    model_path = Path(local_dir) / checkpoint_path
    print(f"Model path: {model_path}")

    # Auto-detect policy type from config
    config_path = model_path / "config.json"
    if args.policy is None:
        import json
        with open(config_path) as f:
            config = json.load(f)
        if "smolvla" in config.get("_class_name", "").lower():
            args.policy = "smolvla"
        else:
            args.policy = "act"
        print(f"Auto-detected policy type: {args.policy}")

    # Load policy
    print(f"Loading {args.policy.upper()} policy...")
    if args.policy == "act":
        policy = ACTPolicy.from_pretrained(str(model_path))
    else:
        policy = SmolVLAPolicy.from_pretrained(str(model_path))

    policy.eval()
    policy.to(device)

    # Get preprocessor/postprocessor from checkpoint using the same factory as training
    try:
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            pretrained_path=str(model_path)
        )
        print("Loaded pre/post processors from checkpoint")
    except Exception as e:
        print(f"Warning: Could not load processors from checkpoint: {e}")
        print("Falling back to identity functions - actions may not be denormalized!")
        preprocessor = lambda x: x
        postprocessor = lambda x: x

    # Detect depth cameras (extract base camera name, not the _depth suffixed name)
    depth_cameras = []
    for key in policy.config.input_features.keys():
        if "_depth" in key:
            cam_name = key.replace("observation.images.", "")
            base_cam = cam_name.replace("_depth", "")  # e.g., "overhead_cam" not "overhead_cam_depth"
            if base_cam not in depth_cameras:
                depth_cameras.append(base_cam)

    if depth_cameras:
        print(f"Using depth cameras: {depth_cameras}")

    # Get action dim
    try:
        action_dim = policy.config.output_features['action'].shape[0]
    except:
        action_dim = 6

    print(f"\nRunning {args.episodes} evaluation episodes...")
    print("=" * 60)

    # Run evaluation
    language = args.language if args.policy == "smolvla" else None

    results = run_evaluation(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=device,
        num_episodes=args.episodes,
        randomize=True,
        action_dim=action_dim,
        depth_cameras=depth_cameras,
        language_instruction=language,
        max_steps=300,
        verbose=True,
        analyze_failures=True,
    )

    success_rate, avg_steps, avg_time, ik_failure_rate, avg_ik_error, failure_summary = results

    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Model: {args.repo_id}/{checkpoint_path}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Success Rate: {success_rate*100:.1f}%")
    print(f"  Avg Steps: {avg_steps:.1f}")
    print(f"  Avg Time: {avg_time:.2f}s")
    if ik_failure_rate is not None:
        print(f"  IK Failure Rate: {ik_failure_rate*100:.2f}%")
        print(f"  Avg IK Error: {avg_ik_error:.2f}mm")
    print("=" * 60)

    if failure_summary:
        print("\nFailure Analysis:")
        for key, value in failure_summary.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
