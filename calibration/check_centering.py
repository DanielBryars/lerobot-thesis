#!/usr/bin/env python
"""Check if motors are properly centered."""
from lerobot.motors import Motor, MotorNormMode
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


def check_arm(name: str, port: str):
    print(f"\n{'=' * 60}")
    print(f"{name} ({port})")
    print("=" * 60)

    print(f"\nHold the {name} arm at ZERO POSE (gripper vertical, moving finger up)")
    print("Press Enter when ready...")
    input()

    bus = create_bus(port)
    bus.connect()
    bus.disable_torque()

    # Note: STS3250 only has one offset register - "Homing_Offset" at address 31
    # This is the same as what Feetech Debug Tool calls "Position Offset Value"
    print(f"\n{'Motor':<15} {'ID':>4} {'Position':>10} {'Homing_Offset':>15} {'Dist from 2048':>15}")
    print("-" * 70)

    for motor in bus.motors:
        motor_id = bus.motors[motor].id
        pos = bus.read("Present_Position", motor, normalize=False)
        homing_offset = bus.read("Homing_Offset", motor, normalize=False)

        # Handle signed homing offset (sign-magnitude encoding, bit 11 is sign)
        if homing_offset > 2048:
            homing_signed = -(homing_offset - 2048)
        else:
            homing_signed = homing_offset

        dist = pos - 2048
        flag = " <-- NOT CENTERED!" if abs(dist) > 200 else ""
        print(f"{motor:<15} {motor_id:>4} {pos:>10} {homing_signed:>+15} {dist:>+15}{flag}")

    bus.disconnect()


print("=" * 70)
print("CENTERING CHECK")
print("=" * 70)
print("\nThis will check each arm one at a time.")

check_arm("Leader", "COM8")
check_arm("Follower", "COM7")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
print("\nIf any motor shows 'NOT CENTERED', run set_homing_offsets.py for that arm.")
