#!/usr/bin/env python3
"""
Diagnose motor wrap-around issues.

Displays raw encoder values and normalized values in real-time
to help debug wrap-around problems.

Usage:
    python diagnose_motor_wrap.py
"""

import sys
import time
import json
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "lerobot" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "recording"))

from SO100LeaderSTS3250 import SO100LeaderSTS3250, SO100LeaderSTS3250Config


def load_config():
    """Load config.json for leader arm port."""
    repo_root = Path(__file__).parent.parent.parent
    config_paths = [
        repo_root / "configs" / "config.json",
        Path("config.json"),
    ]
    for path in config_paths:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


def main():
    print("=" * 70)
    print("Motor Wrap-Around Diagnostics")
    print("=" * 70)
    print()

    # Load config
    config = load_config()
    if config and "leader" in config:
        port = config["leader"]["port"]
        leader_id = config["leader"]["id"]
    else:
        port = "COM8"
        leader_id = "leader_so100"

    print(f"Connecting to leader arm on {port}...")

    # Connect
    leader_config = SO100LeaderSTS3250Config(port=port, id=leader_id)
    leader = SO100LeaderSTS3250(leader_config)
    leader.connect()

    print("Connected!")
    print()

    # Show calibration
    print("Calibration ranges:")
    print("-" * 50)
    for motor, cal in leader.calibration.items():
        print(f"  {motor:15s}: min={cal.range_min:4d}, max={cal.range_max:4d}, range={cal.range_max - cal.range_min:4d}")
    print()
    print("Encoder range is 0-4095 (wraps at boundaries)")
    print()

    print("=" * 70)
    print("INSTRUCTIONS:")
    print("  1. Move joints slowly and watch the values")
    print("  2. When you see a 'jump', note which joint and direction")
    print("  3. Press Ctrl+C to exit")
    print("=" * 70)
    print()

    # Column headers
    motors = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    short_names = ["sh_pan", "sh_lft", "elb_fx", "wr_flx", "wr_rol", "grip"]

    print("Reading values... (updates every 100ms)")
    print()
    print("RAW = encoder ticks (0-4095), NORM = normalized (-100 to 100 or 0 to 100)")
    print()

    # Print header
    header = "| "
    for name in short_names:
        header += f"{name:^15s} | "
    print(header)
    print("|" + "-" * 17 * len(motors) + "|")

    prev_raw = {}
    jump_count = 0

    try:
        while True:
            # Read raw positions
            raw_positions = leader.bus.sync_read("Present_Position", normalize=False)

            # Read normalized (using our fixed get_action)
            norm_positions = leader.get_action()

            # Check for jumps
            jumps = []
            for motor in motors:
                if motor in prev_raw:
                    diff = abs(raw_positions[motor] - prev_raw[motor])
                    if diff > 500:  # Large jump
                        jumps.append((motor, prev_raw[motor], raw_positions[motor], diff))
                prev_raw[motor] = raw_positions[motor]

            # Print values
            line = "| "
            for motor in motors:
                raw = raw_positions.get(motor, 0)
                norm = norm_positions.get(f"{motor}.pos", 0)
                line += f"{raw:4d} ({norm:+6.1f}) | "
            print(f"\r{line}", end="", flush=True)

            # Print jumps on new line
            if jumps:
                jump_count += 1
                print()  # New line
                for motor, prev, curr, diff in jumps:
                    cal = leader.calibration[motor]
                    print(f"  !!! JUMP #{jump_count} on {motor}: {prev} -> {curr} (diff={diff}, cal_range={cal.range_min}-{cal.range_max})")
                print()

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nDone!")

    finally:
        leader.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
