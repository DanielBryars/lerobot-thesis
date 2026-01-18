#!/usr/bin/env python3
"""
Playback recorded Pi0 inference in MuJoCo viewer.

Usage:
    python scripts/pi0/playback_recording.py recording.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project paths
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot_robot_sim import SO100Sim, SO100SimConfig


def main():
    parser = argparse.ArgumentParser(description="Playback recorded inference")
    parser.add_argument("recording", type=str, help="Path to recording JSON file")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to play")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--loop", action="store_true", help="Loop playback")
    args = parser.parse_args()

    # Load recording
    print(f"Loading recording: {args.recording}")
    with open(args.recording) as f:
        recording = json.load(f)

    n_episodes = len(recording["episodes"])
    print(f"Found {n_episodes} episodes")

    if args.episode >= n_episodes:
        print(f"Episode {args.episode} not found (max: {n_episodes - 1})")
        return

    episode = recording["episodes"][args.episode]
    print(f"Playing episode {args.episode} with {len(episode['actions'])} actions")

    # Create simulation
    sim_config = SO100SimConfig(
        sim_cameras=["overhead_cam", "wrist_cam"],
        camera_width=640,
        camera_height=480,
    )
    sim = SO100Sim(sim_config)
    sim.connect()

    while True:
        # Set initial state from recording
        initial_qpos = episode["initial_qpos"]
        sim.mj_data.qpos[:len(initial_qpos)] = initial_qpos

        # Step to settle
        import mujoco
        for _ in range(10):
            mujoco.mj_step(sim.mj_model, sim.mj_data)

        print("Starting playback... (close viewer to exit)")

        # Play back actions
        for i, action_dict in enumerate(episode["actions"]):
            sim.send_action(action_dict)

            # Render
            if not sim.render():
                print("Viewer closed")
                sim.disconnect()
                return

            # Control playback speed
            time.sleep(0.02 / args.speed)  # ~50Hz base rate

            if i % 50 == 0:
                print(f"  Step {i}/{len(episode['actions'])}")

        print("Playback complete!")

        if not args.loop:
            break
        print("Looping...")

    # Keep viewer open
    print("Press Ctrl+C to exit...")
    try:
        while sim.render():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    sim.disconnect()


if __name__ == "__main__":
    main()
