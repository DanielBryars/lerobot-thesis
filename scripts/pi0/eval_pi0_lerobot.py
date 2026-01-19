#!/usr/bin/env python3
"""
Evaluate Pi0 model trained with LeRobot in simulation.

This script runs the PyTorch Pi0 model (not JAX) in the SO-100 simulation.

Usage:
    python scripts/pi0/eval_pi0_lerobot.py --model danbhf/pi0_so101 --episodes 10 --visualize
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

# Add src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from lerobot_robot_sim import SO100Sim, SO100SimConfig

# Motor names in order
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate Pi0 (LeRobot) in simulation")
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model repo ID or local path")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=300,
                        help="Max steps per episode")
    parser.add_argument("--visualize", action="store_true",
                        help="Show simulation viewer")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    args = parser.parse_args()

    print("=" * 60)
    print("Pi0 Evaluation (LeRobot PyTorch)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Load policy
    print("\nLoading policy...")
    from lerobot.common.policies.pi0.modeling_pi0 import Pi0Policy

    policy = Pi0Policy.from_pretrained(args.model)
    policy.to(args.device)
    policy.eval()
    print("Policy loaded!")

    # Get policy config for input format
    policy_config = policy.config

    # Create simulation
    print("\nCreating simulation...")
    sim_config = SO100SimConfig(
        sim_cameras=["overhead_cam", "wrist_cam"],
        camera_width=640,
        camera_height=480,
    )
    sim = SO100Sim(sim_config)
    sim.connect()

    # Run evaluation
    print(f"\nRunning {args.episodes} evaluation episodes...")
    successes = 0
    all_times = []

    for ep in range(args.episodes):
        print(f"\n--- Episode {ep + 1} / {args.episodes} ---")
        sim.reset_scene(randomize=True)

        # Timing stats
        times_obs = []
        times_prep = []
        times_infer = []
        times_action = []

        for step in range(args.max_steps):
            t0 = time.perf_counter()

            # Get observation
            obs = sim.get_observation()
            t1 = time.perf_counter()

            # Extract state
            state = np.array([obs[m + ".pos"] for m in MOTOR_NAMES], dtype=np.float32)

            # Extract and resize images
            import cv2
            overhead_img = obs.get("overhead_cam", obs[list(obs.keys())[0]])
            wrist_img = obs.get("wrist_cam", overhead_img)

            # Resize to expected input size (224x224 for Pi0)
            overhead_img = cv2.resize(overhead_img, (224, 224))
            wrist_img = cv2.resize(wrist_img, (224, 224))

            # Convert to torch tensors [B, C, H, W]
            overhead_tensor = torch.from_numpy(overhead_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            wrist_tensor = torch.from_numpy(wrist_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            state_tensor = torch.from_numpy(state).unsqueeze(0)

            # Prepare policy input
            policy_input = {
                "observation.images.top": overhead_tensor.to(args.device),
                "observation.images.wrist": wrist_tensor.to(args.device),
                "observation.state": state_tensor.to(args.device),
            }
            t2 = time.perf_counter()

            # Get action from policy
            with torch.no_grad():
                action = policy.select_action(policy_input)
            action = action.cpu().numpy().flatten()
            t3 = time.perf_counter()

            # Convert to action dict
            action_dict = {m + ".pos": float(action[i]) for i, m in enumerate(MOTOR_NAMES)}

            # Send action
            sim.send_action(action_dict)
            t4 = time.perf_counter()

            # Record times (skip first step - warmup)
            if step > 0:
                times_obs.append(t1 - t0)
                times_prep.append(t2 - t1)
                times_infer.append(t3 - t2)
                times_action.append(t4 - t3)

            # Render if visualizing
            if args.visualize:
                if not sim.render():
                    print("  Viewer closed")
                    break

            # Check task completion
            if sim.is_task_complete():
                print(f"  SUCCESS at step {step + 1}")
                successes += 1
                break

            if step % 50 == 0:
                print(f"  Step {step}")
        else:
            print(f"  TIMEOUT after {args.max_steps} steps")

        # Print timing stats
        if times_infer:
            avg_obs = np.mean(times_obs) * 1000
            avg_prep = np.mean(times_prep) * 1000
            avg_infer = np.mean(times_infer) * 1000
            avg_action = np.mean(times_action) * 1000
            total = avg_obs + avg_prep + avg_infer + avg_action
            hz = 1000 / total if total > 0 else 0
            print(f"\n  Timing (ms): obs={avg_obs:.1f}, prep={avg_prep:.1f}, infer={avg_infer:.1f}, action={avg_action:.1f}")
            print(f"  Total: {total:.1f}ms = {hz:.1f} Hz")
            all_times.append(total)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Success: {successes} / {args.episodes} = {100 * successes / args.episodes:.1f}%")
    if all_times:
        print(f"Avg inference: {np.mean(all_times):.1f}ms = {1000 / np.mean(all_times):.1f} Hz")


if __name__ == "__main__":
    main()
