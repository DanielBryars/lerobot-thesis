#!/usr/bin/env python3
"""
SO100 Leader-Follower Teleoperation Script for STS3250 motors.

This script allows you to control a follower robot arm using a leader arm.
The follower will mirror the movements of the leader in real-time.

Usage:
    python teleoperate_so100.py

Controls:
    - Move the leader arm physically, and the follower will mirror the movements
    - Press Ctrl+C to exit
"""

import time
import sys
import json
from pathlib import Path
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from SO100FollowerSTS3250 import SO100FollowerSTS3250, SO100FollowerSTS3250Config
from SO100LeaderSTS3250 import SO100LeaderSTS3250, SO100LeaderSTS3250Config


def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    config_file = Path(__file__).parent / config_path
    with open(config_file, 'r') as f:
        return json.load(f)


def main():
    print("=" * 70)
    print("SO100 STS3250 Leader-Follower Teleoperation")
    print("=" * 70)
    print()

    # Load configuration from config.json
    config = load_config()

    # Leader arm configuration
    leader_port = config["leader"]["port"]
    leader_id = config["leader"]["id"]

    # Follower arm configuration
    follower_port = config["follower"]["port"]
    follower_id = config["follower"]["id"]

    # Camera configuration for follower - empty for teleoperation (cameras only needed for recording)
    camera_config = {}

    print(f"Leader arm port: {leader_port}")
    print(f"Follower arm port: {follower_port}")
    print()

    # Configure and connect leader
    print(f"\nConnecting to leader arm at {leader_port}...")
    leader_cfg = SO100LeaderSTS3250Config(port=leader_port, id=leader_id)
    leader = SO100LeaderSTS3250(leader_cfg)

    try:
        leader.connect()
        print("[OK] Leader arm connected successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to connect to leader: {e}")
        return 1

    # Configure and connect follower
    print(f"\nConnecting to follower arm at {follower_port}...")
    follower_cfg = SO100FollowerSTS3250Config(port=follower_port, id=follower_id, cameras=camera_config)
    follower = SO100FollowerSTS3250(follower_cfg)

    try:
        follower.connect()
        print("[OK] Follower arm connected successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to connect to follower: {e}")
        leader.disconnect()
        return 1

    # Sync follower to leader initial position
    print("\nSyncing follower to leader initial position...")
    initial_action = leader.get_action()
    follower.send_action(initial_action)
    time.sleep(0.5)  # Give time for movement
    print("[OK] Initial sync complete")

    print("\n" + "=" * 70)
    print("Teleoperation Active!")
    print("=" * 70)
    print("\nMove the leader arm, and the follower will mirror your movements.")
    print("Press Ctrl+C to stop.")
    print()

    try:
        step = 0
        while True:
            # Read action from leader (current joint positions)
            action = leader.get_action()

            # Send action to follower
            follower.send_action(action)

            # Print status every 100 steps
            if step % 100 == 0:
                # Get joint names and print positions
                joint_names = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
                positions = [f"{name}: {action[name]:.2f}" for name in joint_names if name in action]
                print(f"Step {step}: {' | '.join(positions)}")

            step += 1

            # Small delay to avoid overwhelming the servos
            time.sleep(0.01)  # 100Hz update rate

    except KeyboardInterrupt:
        print("\n\nTeleoperation stopped by user.")
    except Exception as e:
        print(f"\n\n[ERROR] Error during teleoperation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        print("\nDisconnecting robots...")
        follower.disconnect()
        leader.disconnect()
        print("Done!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
