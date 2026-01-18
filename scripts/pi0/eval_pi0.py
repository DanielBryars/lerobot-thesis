#!/usr/bin/env python3
"""
Pi0 Evaluation Script for SO-101 robot simulation.

Uses openpi JAX policy with lerobot_robot_sim for evaluation.
"""

import argparse
import os
import sys
import numpy as np

# Add lerobot-thesis to path (works in both Docker and WSL)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)  # For lerobot_compat
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

# Apply lerobot compatibility patch for 0.4.x (must be before openpi imports)
import lerobot_compat  # noqa: F401

# Motor names in order (must match simulation)
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Sim action space bounds (radians) - for converting normalized to radians
SIM_ACTION_LOW = np.array([-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.17453])
SIM_ACTION_HIGH = np.array([1.91986, 1.74533, 1.69, 1.65806, 2.74121, 1.74533])


def normalized_to_radians(normalized_dict):
    """Convert normalized motor positions back to radians."""
    radians = np.zeros(6, dtype=np.float32)
    for i, name in enumerate(MOTOR_NAMES):
        val = normalized_dict.get(name + ".pos", 0.0)
        if name == "gripper":
            t = val / 100.0
        else:
            t = (val + 100) / 200.0
        radians[i] = SIM_ACTION_LOW[i] + t * (SIM_ACTION_HIGH[i] - SIM_ACTION_LOW[i])
    return radians


def radians_to_normalized(radians):
    """Convert radians to normalized motor commands."""
    result = {}
    for i, name in enumerate(MOTOR_NAMES):
        if name == "gripper":
            t = (radians[i] - SIM_ACTION_LOW[i]) / (SIM_ACTION_HIGH[i] - SIM_ACTION_LOW[i])
            result[name + ".pos"] = float(t * 100)
        else:
            t = (radians[i] - SIM_ACTION_LOW[i]) / (SIM_ACTION_HIGH[i] - SIM_ACTION_LOW[i])
            result[name + ".pos"] = float(t * 200 - 100)
    return result


def download_checkpoint(model_id, checkpoint):
    """Download checkpoint from HuggingFace and return local path."""
    from huggingface_hub import snapshot_download

    cache_dir = os.path.expanduser("~/.cache/openpi_eval")
    safe_id = model_id.replace("/", "_")
    local_dir = cache_dir + "/" + safe_id

    if os.path.exists(local_dir) and len(os.listdir(local_dir)) > 0:
        # Check if params folder exists (indicates valid checkpoint)
        if os.path.exists(os.path.join(local_dir, "params")):
            print("Using cached checkpoint:", local_dir)
            return local_dir

    print("Downloading", model_id, "...")
    os.makedirs(local_dir, exist_ok=True)

    # Download everything - files may be at root or in checkpoint subfolder
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
    )

    # If checkpoint is in a subfolder, return that path
    ckpt_subdir = os.path.join(local_dir, checkpoint)
    if os.path.exists(ckpt_subdir) and os.path.exists(os.path.join(ckpt_subdir, "params")):
        print("Downloaded to:", ckpt_subdir)
        return ckpt_subdir

    print("Downloaded to:", local_dir)
    return local_dir


def main():
    parser = argparse.ArgumentParser(description="Evaluate Pi0 model in simulation")
    parser.add_argument("--model", type=str, default="danbhf/pi0_so101_20260110",
                        help="HuggingFace model ID")
    parser.add_argument("--checkpoint", type=str, default="19999",
                        help="Checkpoint name (folder name in repo)")
    parser.add_argument("--config", type=str, default="pi0_so101",
                        help="openpi config name")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=300,
                        help="Max steps per episode")
    parser.add_argument("--language", type=str,
                        default="Pick up the block and place it in the bowl",
                        help="Language instruction for policy")
    parser.add_argument("--visualize", action="store_true",
                        help="Show MuJoCo viewer")
    parser.add_argument("--record", type=str, default=None,
                        help="Record actions to file for playback")
    args = parser.parse_args()

    print("=" * 60)
    print("Pi0 Evaluation")
    print("=" * 60)
    print("Model:", args.model)
    print("Checkpoint:", args.checkpoint)
    print("Config:", args.config)
    print("Episodes:", args.episodes)
    print("=" * 60)

    # Download checkpoint
    checkpoint_path = download_checkpoint(args.model, args.checkpoint)

    # Load policy using openpi API
    print("Loading policy...")
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(
        config,
        checkpoint_path,
        default_prompt=args.language
    )
    print("Policy loaded!")

    # Create simulation
    from lerobot_robot_sim import SO100Sim, SO100SimConfig
    print("Registered: so100_sim, so101_sim")

    sim_config = SO100SimConfig(
        sim_cameras=["overhead_cam", "wrist_cam"],
        camera_width=640,
        camera_height=480,
    )

    print("Creating simulation...")
    sim = SO100Sim(sim_config)
    sim.connect()

    # Recording setup
    recording = None
    if args.record:
        recording = {"episodes": []}
        print("Recording to:", args.record)

    # Run evaluation
    print("\nRunning", args.episodes, "evaluation episodes...")
    successes = 0

    for ep in range(args.episodes):
        print("\n--- Episode", ep + 1, "/", args.episodes, "---")
        sim.reset_scene(randomize=True)

        # Record initial state
        episode_data = None
        if recording is not None:
            episode_data = {
                "initial_qpos": sim.mj_data.qpos.copy().tolist(),
                "actions": []
            }

        for step in range(args.max_steps):
            # Get observation
            obs = sim.get_observation()

            # Extract state: model was trained on normalized values, so use directly
            state = np.array([obs[m + ".pos"] for m in MOTOR_NAMES], dtype=np.float32)

            # Extract images
            images = {}
            for cam in ["overhead_cam", "wrist_cam"]:
                if cam in obs:
                    images[cam] = obs[cam]

            # Prepare policy input (openpi format)
            # Resize images to 224x224 for Pi0
            import cv2
            overhead_img = images.get("overhead_cam", images[list(images.keys())[0]])
            wrist_img = images.get("wrist_cam", overhead_img)  # Use overhead as fallback

            policy_input = {
                "observation/state": state,
                "observation/image": cv2.resize(overhead_img, (224, 224)),
                "observation/wrist_image": cv2.resize(wrist_img, (224, 224)),
            }

            # Get action from policy
            result = policy.infer(policy_input)
            action = result["actions"][0]  # First action from chunk

            # Model outputs normalized actions directly (trained on normalized data)
            action_dict = {m + ".pos": float(action[i]) for i, m in enumerate(MOTOR_NAMES)}

            # Record action
            if episode_data is not None:
                episode_data["actions"].append(action_dict)

            # Send action to simulation
            sim.send_action(action_dict)

            # Render if visualizing
            if args.visualize:
                if not sim.render():
                    print("  Viewer closed")
                    break

            # Check task completion
            if sim.is_task_complete():
                print("  SUCCESS at step", step + 1)
                successes += 1
                break

            if step % 50 == 0:
                print("  Step", step)
        else:
            print("  TIMEOUT after", args.max_steps, "steps")

        # Save episode to recording
        if episode_data is not None:
            recording["episodes"].append(episode_data)

    # Save recording
    if recording is not None and args.record:
        import json
        with open(args.record, 'w') as f:
            json.dump(recording, f)
        print("Saved recording to:", args.record)

    # Print results
    success_rate = successes / args.episodes * 100
    print("\n" + "=" * 60)
    print("Results:", successes, "/", args.episodes, "=", success_rate, "% success")
    print("=" * 60)

    sim.disconnect()


if __name__ == "__main__":
    main()
