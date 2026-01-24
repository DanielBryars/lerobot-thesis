#!/usr/bin/env python
"""
Directly set homing offsets so motors read 2048 at zero pose.
"""
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


TARGET = 2048

print("=" * 70)
print("SET HOMING OFFSETS")
print("=" * 70)
print("\nThis will set EEPROM homing_offset so all motors read 2048 at zero pose.")
print("\nPosition BOTH arms at ZERO POSE first!")
print("Press Enter when ready...")
input()

for name, port in [("Leader", "COM8"), ("Follower", "COM7")]:
    print(f"\n{'=' * 50}")
    print(f"Processing {name} ({port})")
    print("=" * 50)

    bus = create_bus(port)
    bus.connect()
    bus.disable_torque()

    print(f"\n1. Current positions at zero pose:")
    positions = {}
    for motor in bus.motors:
        pos = bus.read("Present_Position", motor, normalize=False)
        positions[motor] = pos
        print(f"   {motor:<15}: {pos}")

    print(f"\n2. Calculating homing offsets (target = {TARGET}):")
    offsets = {}
    for motor, pos in positions.items():
        # homing_offset = current_position - target
        # After setting: new_reading = raw - homing_offset = raw - (raw - 2048) = 2048
        offset = pos - TARGET
        offsets[motor] = offset
        print(f"   {motor:<15}: {pos} - {TARGET} = {offset:+d}")

    print(f"\n3. Writing homing offsets to EEPROM...")
    for motor, offset in offsets.items():
        # The STS3250 uses signed 16-bit for homing offset
        # Write directly - lerobot handles the encoding
        try:
            bus.write("Homing_Offset", motor, offset, normalize=False)
            print(f"   {motor:<15}: wrote {offset:+d}")
        except Exception as e:
            print(f"   {motor:<15}: FAILED - {e}")

    print(f"\n4. Verifying (should all be ~{TARGET}):")
    all_ok = True
    for motor in bus.motors:
        new_pos = bus.read("Present_Position", motor, normalize=False)
        diff = new_pos - TARGET
        ok = "OK" if abs(diff) < 50 else "FAILED!"
        if abs(diff) >= 50:
            all_ok = False
        print(f"   {motor:<15}: {new_pos} (diff: {diff:+d}) {ok}")

    if all_ok:
        print(f"\n[OK] {name} centered successfully!")
    else:
        print(f"\n[!] {name} centering FAILED - check errors above")

    # Also verify by re-reading homing offset
    print(f"\n5. Re-reading homing offsets from EEPROM:")
    for motor in bus.motors:
        stored = bus.read("Homing_Offset", motor, normalize=False)
        expected = offsets[motor]
        # Handle signed conversion
        if stored > 32767:
            stored_signed = stored - 65536
        elif stored > 2048:
            stored_signed = stored - 4096  # For 12-bit signed
        else:
            stored_signed = stored
        match = "OK" if abs(stored_signed - expected) < 10 else f"MISMATCH (expected {expected})"
        print(f"   {motor:<15}: {stored} (signed: {stored_signed}) {match}")

    bus.disconnect()

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
print("\nNow run: python calibration/calibrate_from_zero.py --leader --follower")
