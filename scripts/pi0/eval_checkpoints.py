#!/usr/bin/env python3
"""
Evaluate Pi0/ACT checkpoints in simulation without visualization.

This script is designed to run in headless environments (Docker, remote servers)
and can evaluate multiple checkpoints in sequence, generating a summary table.

Features:
- Auto-detects camera names from model config
- Supports both Pi0 and ACT policies
- Headless mode for Docker/remote evaluation
- Batch evaluation of all checkpoints in a directory
- JSON output for easy parsing

Usage:
    # Single model from HuggingFace
    python eval_checkpoints.py --model danbhf/pi0_so101_lerobot --episodes 10

    # Local checkpoint directory (evaluate all)
    python eval_checkpoints.py --model /app/outputs/pi0_run --local --all-checkpoints --episodes 10

    # Specific checkpoint
    python eval_checkpoints.py --model danbhf/pi0_so101_lerobot_20k --checkpoint checkpoint_010000 --episodes 20

    # Output results to JSON
    python eval_checkpoints.py --model danbhf/pi0_so101_lerobot --episodes 10 --output results.json
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project paths
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

# Motor names in order
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def find_checkpoints(model_dir: Path) -> list:
    """Find all checkpoint directories in a model directory."""
    checkpoints = []
    for d in model_dir.iterdir():
        if d.is_dir():
            # Check for Pi0 format (model.safetensors) or ACT format (config.json)
            if (d / "model.safetensors").exists() or (d / "config.json").exists():
                if d.name.startswith("checkpoint_") or d.name == "final":
                    checkpoints.append(d.name)

    # Also check root directory (for single checkpoint repos)
    if (model_dir / "model.safetensors").exists() or (model_dir / "config.json").exists():
        checkpoints.append(".")

    # Sort by step number
    def sort_key(name):
        if name == "." or name == "final":
            return float('inf')
        match = re.search(r'(\d+)', name)
        return int(match.group(1)) if match else 0

    return sorted(checkpoints, key=sort_key)


def detect_policy_type(model_path: Path) -> str:
    """Detect whether this is a Pi0 or ACT model."""
    config_path = model_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

        # Check various indicators
        policy_type = config.get("type", "").lower()
        class_name = config.get("_class_name", "").lower()

        if "pi0" in policy_type or "pi0" in class_name:
            return "pi0"
        elif "act" in policy_type or "act" in class_name:
            return "act"
        elif "smolvla" in policy_type or "smolvla" in class_name:
            return "smolvla"

    # Check for Pi0-specific files
    if (model_path / "model.safetensors").exists():
        # Large file size suggests Pi0 (7GB vs ~100MB for ACT)
        size_gb = (model_path / "model.safetensors").stat().st_size / (1024**3)
        if size_gb > 1:
            return "pi0"

    return "act"  # Default


def get_camera_config(model_path: Path) -> dict:
    """Extract camera configuration from model config."""
    config_path = model_path / "config.json"
    if not config_path.exists():
        return {"sim_cameras": ["overhead_cam", "wrist_cam"], "camera_mapping": {}}

    with open(config_path) as f:
        config = json.load(f)

    # Extract image input features
    input_features = config.get("input_features", {})

    camera_mapping = {}  # model_key -> sim_camera_name
    sim_cameras = set()

    for key in input_features:
        if key.startswith("observation.images."):
            cam_name = key.replace("observation.images.", "")

            # Map common model camera names to simulation camera names
            if cam_name in ["overhead_cam", "top", "base_0_rgb"]:
                camera_mapping[key] = "overhead_cam"
                sim_cameras.add("overhead_cam")
            elif cam_name in ["wrist_cam", "wrist", "left_wrist_0_rgb", "right_wrist_0_rgb"]:
                camera_mapping[key] = "wrist_cam"
                sim_cameras.add("wrist_cam")
            else:
                # Use as-is
                camera_mapping[key] = cam_name
                sim_cameras.add(cam_name)

    # Default if nothing found
    if not sim_cameras:
        sim_cameras = {"overhead_cam", "wrist_cam"}

    return {
        "sim_cameras": list(sim_cameras),
        "camera_mapping": camera_mapping,
        "input_features": input_features,
    }


def clean_pi0_config(model_path: Path):
    """Remove training-specific fields from config that cause loading errors."""
    config_path = model_path / "config.json"
    if not config_path.exists():
        return

    with open(config_path) as f:
        config = json.load(f)

    # Fields that may not be recognized by all LeRobot versions
    training_fields = [
        "use_peft", "freeze_vision_encoder", "train_expert_only",
        "push_to_hub", "repo_id", "private", "tags", "license",
    ]

    modified = False
    for field in training_fields:
        if field in config:
            del config[field]
            modified = True

    if modified:
        # Save cleaned config
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        print(f"  Cleaned config (removed training-specific fields)")


def load_pi0_policy(model_path: Path, device: str):
    """Load Pi0 policy from path."""
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy

    # Clean config to remove fields that may cause loading errors
    clean_pi0_config(model_path)

    policy = PI0Policy.from_pretrained(str(model_path))
    policy.to(device)
    policy.eval()
    return policy


def load_act_policy(model_path: Path, device: str):
    """Load ACT policy from path."""
    from lerobot.policies.act.modeling_act import ACTPolicy

    policy = ACTPolicy.from_pretrained(str(model_path))
    policy.to(device)
    policy.eval()
    return policy


def prepare_pi0_observation(obs: dict, camera_config: dict, device: str) -> dict:
    """Prepare observation dict for Pi0 policy."""
    import cv2

    policy_input = {}

    # State
    state = np.array([obs[m + ".pos"] for m in MOTOR_NAMES], dtype=np.float32)
    policy_input["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(device)

    # Images - map from simulation camera names to model input keys
    camera_mapping = camera_config.get("camera_mapping", {})
    input_features = camera_config.get("input_features", {})

    for model_key, sim_cam in camera_mapping.items():
        if model_key.startswith("observation.images."):
            img = obs.get(sim_cam)
            if img is None:
                # Try alternate names
                for alt in ["overhead_cam", "wrist_cam", "top", "wrist"]:
                    if alt in obs:
                        img = obs[alt]
                        break

            if img is not None:
                # Get expected size from input features
                feat = input_features.get(model_key, {})
                if hasattr(feat, 'shape') and len(feat.shape) == 3:
                    # shape is (C, H, W)
                    target_h, target_w = feat.shape[1], feat.shape[2]
                else:
                    # Default to 224x224 for Pi0
                    target_h, target_w = 224, 224

                # Resize and convert
                img_resized = cv2.resize(img, (target_w, target_h))
                img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                policy_input[model_key] = img_tensor.to(device)

    return policy_input


def evaluate_checkpoint(
    model_path: Path,
    policy_type: str,
    device: str,
    num_episodes: int,
    max_steps: int,
    visualize: bool = False,
) -> dict:
    """Evaluate a single checkpoint."""
    from lerobot_robot_sim import SO100Sim, SO100SimConfig

    print(f"  Loading {policy_type} policy from {model_path}...")

    # Get camera config
    camera_config = get_camera_config(model_path)
    sim_cameras = camera_config["sim_cameras"]

    # Load policy
    if policy_type == "pi0":
        policy = load_pi0_policy(model_path, device)
    else:
        policy = load_act_policy(model_path, device)

    print(f"  Policy loaded. Cameras: {sim_cameras}")

    # Create simulation
    scene_path = REPO_ROOT / "scenes" / "so101_with_wrist_cam.xml"
    sim_config = SO100SimConfig(
        scene_xml=str(scene_path),
        sim_cameras=sim_cameras,
        camera_width=640,
        camera_height=480,
    )
    sim = SO100Sim(sim_config)
    sim.connect()

    # Run evaluation
    successes = 0
    total_steps = 0
    all_times = []
    failure_modes = {"timeout": 0, "dropped": 0, "missed": 0}

    for ep in range(num_episodes):
        print(f"    Episode {ep + 1}/{num_episodes}...", end=" ", flush=True)

        sim.reset_scene(randomize=True, pos_range=0.04, rot_range=np.pi)

        if hasattr(policy, 'reset'):
            policy.reset()

        step_times = []
        ep_success = False

        for step in range(max_steps):
            t0 = time.perf_counter()

            obs = sim.get_observation()

            # Prepare observation based on policy type
            if policy_type == "pi0":
                policy_input = prepare_pi0_observation(obs, camera_config, device)
            else:
                # ACT uses preprocessor from training
                policy_input = prepare_act_observation(obs, camera_config, device)

            # Get action
            with torch.no_grad():
                action = policy.select_action(policy_input)

            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy().flatten()

            # Ensure we have the right number of actions
            action = action[:len(MOTOR_NAMES)]

            # Send action
            action_dict = {m + ".pos": float(action[i]) for i, m in enumerate(MOTOR_NAMES)}
            sim.send_action(action_dict)

            t1 = time.perf_counter()
            if step > 0:
                step_times.append(t1 - t0)

            # Render if visualizing
            if visualize:
                if not sim.render():
                    break

            # Check task completion
            if sim.is_task_complete():
                ep_success = True
                break

        if ep_success:
            successes += 1
            total_steps += step + 1
            print(f"SUCCESS at step {step + 1}")
        else:
            total_steps += max_steps
            failure_modes["timeout"] += 1
            print("TIMEOUT")

        if step_times:
            avg_time = np.mean(step_times) * 1000
            all_times.append(avg_time)

    sim.disconnect()

    # Calculate results
    success_rate = successes / num_episodes
    avg_steps = total_steps / num_episodes
    avg_time_ms = np.mean(all_times) if all_times else 0
    hz = 1000 / avg_time_ms if avg_time_ms > 0 else 0

    return {
        "success_rate": success_rate,
        "successes": successes,
        "episodes": num_episodes,
        "avg_steps": avg_steps,
        "avg_time_ms": avg_time_ms,
        "hz": hz,
        "failure_modes": failure_modes,
    }


def prepare_act_observation(obs: dict, camera_config: dict, device: str) -> dict:
    """Prepare observation dict for ACT policy."""
    policy_input = {}

    # State
    state = np.array([obs[m + ".pos"] for m in MOTOR_NAMES], dtype=np.float32)
    policy_input["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(device)

    # Images
    for cam_name in camera_config["sim_cameras"]:
        img = obs.get(cam_name)
        if img is not None:
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            policy_input[f"observation.images.{cam_name}"] = img_tensor.to(device)

    return policy_input


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Pi0/ACT checkpoints in simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace repo ID or local path (with --local)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Specific checkpoint name (default: evaluate root)")
    parser.add_argument("--all-checkpoints", action="store_true",
                        help="Evaluate all checkpoints in directory")
    parser.add_argument("--local", action="store_true",
                        help="Treat model as local path instead of HuggingFace repo")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes (default: 10)")
    parser.add_argument("--max-steps", type=int, default=300,
                        help="Max steps per episode (default: 300)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (default: cuda)")
    parser.add_argument("--visualize", action="store_true",
                        help="Show MuJoCo viewer (requires display)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    parser.add_argument("--policy-type", type=str, choices=["pi0", "act", "auto"],
                        default="auto", help="Policy type (default: auto-detect)")

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Checkpoint Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Episodes: {args.episodes}")
    print("=" * 60)

    # Get model directory
    if args.local:
        model_dir = Path(args.model)
        if not model_dir.exists():
            raise FileNotFoundError(f"Local path not found: {model_dir}")
    else:
        # Download from HuggingFace
        from huggingface_hub import snapshot_download

        cache_dir = Path("./eval_cache") / args.model.replace("/", "_")
        print(f"\nDownloading from HuggingFace to {cache_dir}...")
        model_dir = Path(snapshot_download(
            repo_id=args.model,
            local_dir=str(cache_dir),
        ))

    print(f"Model directory: {model_dir}")

    # Determine checkpoints to evaluate
    if args.all_checkpoints:
        checkpoints = find_checkpoints(model_dir)
        if not checkpoints:
            checkpoints = ["."]  # Just the root
        print(f"Found checkpoints: {checkpoints}")
    elif args.checkpoint:
        checkpoints = [args.checkpoint]
    else:
        checkpoints = ["."]  # Root directory

    # Detect policy type from first checkpoint
    first_path = model_dir / checkpoints[0] if checkpoints[0] != "." else model_dir
    if args.policy_type == "auto":
        policy_type = detect_policy_type(first_path)
        print(f"Auto-detected policy type: {policy_type}")
    else:
        policy_type = args.policy_type

    # Evaluate each checkpoint
    all_results = {}

    for checkpoint_name in checkpoints:
        if checkpoint_name == ".":
            model_path = model_dir
            display_name = "root"
        else:
            model_path = model_dir / checkpoint_name
            display_name = checkpoint_name

        if not model_path.exists():
            print(f"\nSkipping {display_name} - path not found")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {display_name}")
        print(f"{'='*60}")

        results = evaluate_checkpoint(
            model_path=model_path,
            policy_type=policy_type,
            device=device,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            visualize=args.visualize,
        )

        all_results[display_name] = results

        print(f"\n  Success: {results['successes']}/{results['episodes']} = {results['success_rate']*100:.1f}%")
        print(f"  Avg steps: {results['avg_steps']:.1f}")
        print(f"  Inference: {results['avg_time_ms']:.1f}ms = {results['hz']:.1f} Hz")

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'Checkpoint':<25} {'Success':<15} {'Avg Steps':<12} {'Hz':<10}")
    print("-" * 80)

    best_checkpoint = None
    best_success = -1

    for name, results in all_results.items():
        success_str = f"{results['successes']}/{results['episodes']} ({results['success_rate']*100:.0f}%)"
        print(f"{name:<25} {success_str:<15} {results['avg_steps']:<12.1f} {results['hz']:<10.1f}")

        if results['success_rate'] > best_success:
            best_success = results['success_rate']
            best_checkpoint = name

    print("-" * 80)
    if best_checkpoint:
        print(f"Best checkpoint: {best_checkpoint} ({best_success*100:.1f}% success)")
    print("=" * 80)

    # Save to JSON if requested
    if args.output:
        output_data = {
            "model": args.model,
            "policy_type": policy_type,
            "episodes": args.episodes,
            "results": all_results,
            "best_checkpoint": best_checkpoint,
            "best_success_rate": best_success,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
