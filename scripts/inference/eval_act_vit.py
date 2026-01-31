#!/usr/bin/env python
"""
Evaluate ACT-ViT model on simulation.

Usage:
    python eval_act_vit.py outputs/act_vit_157ep/final --episodes 10
    python eval_act_vit.py outputs/act_vit_157ep --checkpoint checkpoint_025000 --episodes 20
"""

import argparse
from pathlib import Path
import sys
import json

import torch
import numpy as np

# Add project root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.factory import make_pre_post_processors

from models.act_vit import ACTViTPolicy
from utils.training import run_evaluation


def load_act_vit_model(model_path: Path, device: torch.device):
    """Load ACT-ViT model from checkpoint."""
    model_path = Path(model_path)

    # Load config
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config_dict = json.load(f)

    # Remove any extra fields that might cause issues
    extra_fields = ['use_peft', 'type']
    for field in extra_fields:
        config_dict.pop(field, None)

    # Convert input_features and output_features from dict to PolicyFeature objects
    from lerobot.configs.types import PolicyFeature, FeatureType
    if 'input_features' in config_dict:
        input_features = {}
        for key, value in config_dict['input_features'].items():
            if isinstance(value, dict):
                feat_type = FeatureType[value['type']] if isinstance(value['type'], str) else value['type']
                input_features[key] = PolicyFeature(type=feat_type, shape=tuple(value['shape']))
            else:
                input_features[key] = value
        config_dict['input_features'] = input_features

    if 'output_features' in config_dict:
        output_features = {}
        for key, value in config_dict['output_features'].items():
            if isinstance(value, dict):
                feat_type = FeatureType[value['type']] if isinstance(value['type'], str) else value['type']
                output_features[key] = PolicyFeature(type=feat_type, shape=tuple(value['shape']))
            else:
                output_features[key] = value
        config_dict['output_features'] = output_features

    config = ACTConfig(**config_dict)

    # Create policy
    policy = ACTViTPolicy(config)

    # Load weights
    model_weights_path = model_path / "model.safetensors"
    if model_weights_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(model_weights_path)
        policy.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Model weights not found: {model_weights_path}")

    policy.to(device)
    policy.eval()

    # Load preprocessor/postprocessor
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=model_path)

    return policy, preprocessor, postprocessor


def main():
    parser = argparse.ArgumentParser(description="Evaluate ACT-ViT model")
    parser.add_argument("model_path", type=str, help="Path to model directory or HuggingFace repo")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint subdirectory (e.g., checkpoint_025000)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--randomize", action="store_true", help="Randomize object position")
    parser.add_argument("--scene", type=str, default=None, help="Scene XML file (e.g., so101_with_confuser.xml)")
    parser.add_argument("--pickup_coords", action="store_true", help="Enable pickup coordinate injection")
    parser.add_argument("--block_x", type=float, default=None, help="Override block X position")
    parser.add_argument("--block_y", type=float, default=None, help="Override block Y position")

    args = parser.parse_args()

    device = torch.device(args.device)

    # Resolve model path
    model_path = Path(args.model_path)
    if args.checkpoint:
        model_path = model_path / args.checkpoint

    # Check if it's a HuggingFace repo
    if not model_path.exists() and "/" in args.model_path:
        from huggingface_hub import snapshot_download
        print(f"Downloading from HuggingFace: {args.model_path}")
        local_dir = Path(f"outputs/downloaded/{args.model_path.replace('/', '_')}")
        snapshot_download(args.model_path, local_dir=local_dir)
        model_path = local_dir
        if args.checkpoint:
            model_path = model_path / args.checkpoint

    print(f"Loading model from: {model_path}")
    policy, preprocessor, postprocessor = load_act_vit_model(model_path, device)

    print(f"Running {args.episodes} evaluation episodes...")

    # Run evaluation
    success_rate, avg_steps, avg_time, ik_failure_rate, avg_ik_error, failure_summary = run_evaluation(
        policy, preprocessor, postprocessor, device,
        num_episodes=args.episodes,
        randomize=args.randomize,
        scene=args.scene,
        pickup_coords=args.pickup_coords,
        block_x=args.block_x,
        block_y=args.block_y,
    )

    print()
    print("=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Success Rate: {success_rate * 100:.1f}%")
    print(f"Avg Steps (success): {avg_steps:.1f}")
    print(f"Avg Time (success): {avg_time:.2f}s")

    if failure_summary:
        print(f"Pick Rate: {failure_summary.get('pick_rate', 0) * 100:.1f}%")
        print(f"Drop Rate: {failure_summary.get('drop_rate', 0) * 100:.1f}%")
        print("Outcome counts:", failure_summary.get('outcome_counts', {}))


if __name__ == "__main__":
    main()
