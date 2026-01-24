#!/usr/bin/env python
"""Live monitoring of motor positions - raw and normalized values."""
import argparse
import time
import sys

from lerobot.motors import Motor, MotorNormMode, MotorCalibration
from lerobot.motors.feetech import FeetechMotorsBus


def create_bus(port: str):
    return FeetechMotorsBus(
        port=port,
        motors={
            "shoulder_pan": Motor(1, "sts3250", MotorNormMode.RANGE_M100_100),
            "shoulder_lift": Motor(2, "sts3250", MotorNormMode.RANGE_M100_100),
            "elbow_flex": Motor(3, "sts3250", MotorNormMode.RANGE_M100_100),
            "wrist_flex": Motor(4, "sts3250", MotorNormMode.RANGE_M100_100),
            "wrist_roll": Motor(5, "sts3250", MotorNormMode.RANGE_M100_100),
            "gripper": Motor(6, "sts3250", MotorNormMode.RANGE_0_100),
        },
    )


# Calibration data from the JSON files
CALIBRATION = {
    "leader": {
        "shoulder_pan": MotorCalibration(id=1, drive_mode=0, homing_offset=0, range_min=803, range_max=3306),
        "shoulder_lift": MotorCalibration(id=2, drive_mode=0, homing_offset=0, range_min=920, range_max=3195),
        "elbow_flex": MotorCalibration(id=3, drive_mode=0, homing_offset=0, range_min=939, range_max=3142),
        "wrist_flex": MotorCalibration(id=4, drive_mode=0, homing_offset=0, range_min=1015, range_max=3176),
        "wrist_roll": MotorCalibration(id=5, drive_mode=0, homing_offset=0, range_min=247, range_max=3888),
        "gripper": MotorCalibration(id=6, drive_mode=0, homing_offset=0, range_min=2046, range_max=3297),
    },
    "follower": {
        "shoulder_pan": MotorCalibration(id=1, drive_mode=0, homing_offset=0, range_min=794, range_max=3297),
        "shoulder_lift": MotorCalibration(id=2, drive_mode=0, homing_offset=0, range_min=942, range_max=3217),
        "elbow_flex": MotorCalibration(id=3, drive_mode=0, homing_offset=0, range_min=956, range_max=3159),
        "wrist_flex": MotorCalibration(id=4, drive_mode=0, homing_offset=0, range_min=885, range_max=3046),
        "wrist_roll": MotorCalibration(id=5, drive_mode=0, homing_offset=0, range_min=278, range_max=3919),
        "gripper": MotorCalibration(id=6, drive_mode=0, homing_offset=0, range_min=2047, range_max=3298),
    },
}

PORTS = {
    "leader": "COM8",
    "follower": "COM7",
}


def normalize(raw: int, cal: MotorCalibration, mode: MotorNormMode) -> float:
    """Manual normalization to show what lerobot would compute."""
    min_ = cal.range_min
    max_ = cal.range_max
    # Note: NOT clamping here so we can see out-of-range values
    if mode == MotorNormMode.RANGE_M100_100:
        return (((raw - min_) / (max_ - min_)) * 200) - 100
    elif mode == MotorNormMode.RANGE_0_100:
        return ((raw - min_) / (max_ - min_)) * 100
    return raw


def main():
    parser = argparse.ArgumentParser(description="Live motor position monitoring")
    parser.add_argument("arm", nargs="?", default="leader", choices=["leader", "follower"],
                        help="Which arm to monitor (default: leader)")
    parser.add_argument("--rate", type=float, default=10, help="Update rate in Hz (default: 10)")
    args = parser.parse_args()

    port = PORTS[args.arm]
    calibration = CALIBRATION[args.arm]

    print(f"Connecting to {args.arm} on {port}...")
    bus = create_bus(port)
    bus.connect()
    bus.disable_torque()

    # Set calibration so we can also read normalized values from lerobot
    bus.calibration = calibration

    print(f"\nMonitoring {args.arm} arm. Press Ctrl+C to exit.\n")

    modes = {
        "shoulder_pan": MotorNormMode.RANGE_M100_100,
        "shoulder_lift": MotorNormMode.RANGE_M100_100,
        "elbow_flex": MotorNormMode.RANGE_M100_100,
        "wrist_flex": MotorNormMode.RANGE_M100_100,
        "wrist_roll": MotorNormMode.RANGE_M100_100,
        "gripper": MotorNormMode.RANGE_0_100,
    }

    try:
        while True:
            # Clear screen and move cursor to top
            print("\033[H\033[J", end="")

            print(f"=== {args.arm.upper()} ARM LIVE ({port}) ===")
            print(f"{'Motor':<15} {'Raw':>6} {'Min':>6} {'Max':>6} {'Norm':>8} {'Status':<20}")
            print("-" * 70)

            for motor in bus.motors:
                cal = calibration[motor]
                mode = modes[motor]

                raw = bus.read("Present_Position", motor, normalize=False)
                norm = normalize(raw, cal, mode)

                # Status indicators
                status = ""
                if raw < cal.range_min:
                    status = f"BELOW MIN by {cal.range_min - raw}"
                elif raw > cal.range_max:
                    status = f"ABOVE MAX by {raw - cal.range_max}"
                elif raw < 100:
                    status = "!! WRAP DANGER (near 0)"
                elif raw > 3995:
                    status = "!! WRAP DANGER (near 4095)"

                # Highlight wrist_roll
                prefix = ">>> " if motor == "wrist_roll" else "    "

                print(f"{prefix}{motor:<11} {raw:>6} {cal.range_min:>6} {cal.range_max:>6} {norm:>+8.1f} {status:<20}")

            print("-" * 70)
            print("\nWrist roll details:")
            wrist_cal = calibration["wrist_roll"]
            wrist_raw = bus.read("Present_Position", "wrist_roll", normalize=False)
            print(f"  Raw: {wrist_raw}")
            print(f"  Range: {wrist_cal.range_min} - {wrist_cal.range_max}")
            print(f"  Distance to range_min: {wrist_raw - wrist_cal.range_min}")
            print(f"  Distance to range_max: {wrist_cal.range_max - wrist_raw}")
            print(f"  Distance to 0: {wrist_raw}")
            print(f"  Distance to 4095: {4095 - wrist_raw}")

            time.sleep(1.0 / args.rate)

    except KeyboardInterrupt:
        print("\n\nExiting...")
    finally:
        bus.disconnect()


if __name__ == "__main__":
    main()
