# SO100 Motor Calibration Guide

This document summarizes calibration options, past approaches, and the current setup for the SO100/SO101 robot arms with STS3250 motors.

## Quick Reference: Current Setup

| Component | Units | Calibration Method |
|-----------|-------|-------------------|
| Leader arm | Normalized (-100 to +100) | File-based JSON |
| Follower arm | Normalized (-100 to +100) | File-based JSON |
| Simulation | Radians internally | Converts from normalized |
| Dataset | Normalized (-100 to +100) | Stored as-is |

Calibration files: `~/.cache/huggingface/lerobot/calibration/`

---

## 1. Unit Options

### Option A: Normalized Values (Current Choice)
- **Range**: -100 to +100 for joints, 0 to +100 for gripper
- **Pros**: Abstract, hardware-agnostic, matches LeRobot convention
- **Cons**: Requires calibration to map to physical positions

```python
MotorNormMode.RANGE_M100_100  # Joints: -100 to +100
MotorNormMode.RANGE_0_100     # Gripper: 0 to +100
```

### Option B: Radians
- **Range**: Varies per joint (e.g., -1.92 to +1.92 for shoulder_pan)
- **Pros**: Standard robotics unit, direct physics meaning
- **Cons**: Requires knowing exact joint limits

```python
# Simulation action space bounds (radians)
SIM_ACTION_LOW  = [-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.17453]
SIM_ACTION_HIGH = [+1.91986, +1.74533, +1.69, +1.65806, +2.84121, +1.74533]
```

### Option C: Degrees
- **Pros**: Human-readable
- **Cons**: Less common in robotics software

### Option D: Raw Encoder Counts
- **Range**: 0 to 4095 (12-bit)
- **Pros**: Direct hardware values
- **Cons**: Not portable between robots, wrap-around issues

---

## 2. Calibration Methods

### Method A: File-Based JSON (Current Choice)

Stores calibration in JSON files rather than motor EEPROM.

**Location**: `~/.cache/huggingface/lerobot/calibration/`
- Leader: `teleoperators/so100_leader_sts3250/leader_so100.json`
- Follower: `robots/so100_follower_sts3250/follower_so100.json`

**Example calibration file**:
```json
{
    "shoulder_pan": {
        "id": 1,
        "drive_mode": 0,
        "homing_offset": 0,
        "range_min": 794,
        "range_max": 3297
    },
    "gripper": {
        "id": 6,
        "drive_mode": 0,
        "homing_offset": 0,
        "range_min": 2047,
        "range_max": 3298
    }
}
```

**Pros**:
- Version controllable
- Easy to modify and iterate
- Doesn't wear out EEPROM
- Portable between machines

**Cons**:
- Must ensure calibration file matches physical hardware
- Wrap-around not handled by firmware

### Method B: EEPROM-Based (Legacy, Still Supported)

Stores homing offset directly in motor EEPROM registers.

**Registers used**:
- `Homing_Offset`: Sign-magnitude encoded (bit 11 = sign)
- `Min_Position_Limit`: Raw encoder min
- `Max_Position_Limit`: Raw encoder max

**Pros**:
- Firmware handles offset automatically
- Calibration travels with the motor
- Wrap-around handled at firmware level

**Cons**:
- EEPROM has limited write cycles
- Requires unlock/lock sequence
- Harder to version control

---

## 3. The Wrap-Around Problem

### What Is It?

Motor encoders report 0-4095 (12-bit). When a joint crosses the 0/4095 boundary, values "wrap around":
- Moving from 4095 to 0 looks like a jump of -4095
- Linear interpolation breaks
- Calibration min/max calculations fail

### Solution: Center at 2048

The key insight from `lerobot-scratch/calibration/center_motors.py`:

> "Center motor readings by setting EEPROM homing_offset. This makes both arms read ~2048 at zero pose, avoiding wraparound issues."

**Strategy**:
1. Position arm at "zero pose" (all joints centered)
2. Read raw encoder value at each joint
3. Calculate offset: `homing_offset = raw_position - 2048`
4. Store offset in EEPROM

**Result**: Joint ranges stay in ~400-3700, never crossing 0/4095 boundary.

### Wrap-Around Detection Code

From `diagnose_motor_directions.py`:
```python
# Handle wraparound (if delta > 2048 or < -2048, it wrapped)
if leader_delta > 2048:
    leader_delta -= 4096
elif leader_delta < -2048:
    leader_delta += 4096
```

---

## 4. Past Work (lerobot-scratch)

### Calibration Scripts Created

| Script | Purpose |
|--------|---------|
| `center_motors.py` | Set EEPROM homing offset to center at 2048 |
| `calibrate_from_zero.py` | Zero-pose calibration, saves to JSON |
| `read_eeprom.py` | Inspect motor EEPROM values |
| `write_homing_offset.py` | Manually set EEPROM offset |
| `diagnose_motor_directions.py` | Detect inverted motors |
| `fix_gripper_calibration.py` | Gripper-specific tuning |

### Zero Pose Reference

The standard "zero pose" for calibration:
- All joints at 0 degrees (arm extended forward, horizontal)
- Gripper jaws VERTICAL (like gripping a horizontal bar)
- Gripper CLOSED

### Inverted Motor Handling

Some motors may be physically mounted inverted. Detection:
```python
# If range goes outside valid bounds, motor is inverted
if range_min < -500 or range_max > 4595:
    range_min, range_max = range_max, range_min
    drive_mode = 1  # Inverted
```

---

## 5. Current Pipeline (lerobot-thesis)

### Recording Flow
```
Physical Leader Arm (STS3250)
    ↓ SO100LeaderSTS3250.get_action()
    ↓ Calibration JSON applied
    ↓
Normalized Values (-100 to +100)
    ↓ Stored in LeRobot dataset
    ↓
Dataset (HuggingFace format)
```

### Simulation Flow
```
Normalized Values from Leader
    ↓ normalized_to_radians()
    ↓
Radians for MuJoCo Physics
    ↓ sim_robot.send_action()
    ↓
Simulated Robot Movement
```

### Playback Flow (Real Robot)
```
Dataset Normalized Values
    ↓ SO100FollowerSTS3250.send_action()
    ↓ Calibration JSON applied (same as leader)
    ↓
Physical Follower Movement
```

### Why This Works

Both leader and follower use:
- Same `MotorNormMode.RANGE_M100_100` for joints
- Same `MotorNormMode.RANGE_0_100` for gripper
- Calibration files with matching range_min/range_max

The normalized values are hardware-agnostic, so recording from leader and playing on follower produces matching movements.

---

## 6. Conversion Reference

### Normalized to Radians (for simulation)
```python
def normalized_to_radians(normalized: dict, use_degrees=False) -> np.ndarray:
    radians = np.zeros(6)
    for i, name in enumerate(MOTOR_NAMES):
        val = normalized.get(name, 0.0)
        if name == "gripper":
            t = val / 100.0  # 0-100 → 0-1
        else:
            t = (val + 100) / 200.0  # -100 to +100 → 0-1
        radians[i] = SIM_ACTION_LOW[i] + t * (SIM_ACTION_HIGH[i] - SIM_ACTION_LOW[i])
    return radians
```

### Raw Encoder to Normalized (inside LeRobot)
```python
# Handled by FeetechMotorsBus with calibration applied
# raw_position → apply homing_offset → map range_min/max → normalized
```

---

## 7. Troubleshooting

### Sim-to-Real Mismatch
1. Check calibration files exist for both leader and follower
2. Verify `range_min`/`range_max` values are similar
3. Check for inverted motors (`drive_mode: 1`)

### Wrap-Around Issues
1. Run `center_motors.py` to set EEPROM offsets
2. Verify all joints read ~2048 at zero pose
3. Check calibration ranges stay within 400-3700

### Gripper Issues
- Gripper uses different normalization (0-100 vs -100 to +100)
- May need separate calibration tuning

---

## 8. Files in This Repo

| File | Purpose |
|------|---------|
| `scripts/SO100LeaderSTS3250.py` | Leader arm class with file-based calibration |
| `scripts/SO100FollowerSTS3250.py` | Follower arm class with file-based calibration |
| `scripts/teleoperate_so100.py` | Leader→Follower teleoperation |
| `scripts/record_sim_vr_pickplace.py` | Record with leader→simulation |
| `scripts/playback_real_robot.py` | Playback dataset on real follower |
| `configs/config.json` | COM ports and arm IDs |

---

## References

- LeRobot calibration: `~/.cache/huggingface/lerobot/calibration/`
- lerobot-scratch calibration scripts: `E:/git/ai/lerobot-scratch/calibration/`
- lerobot-gym calibration check: `E:/git/ai/lerobot-gym/calibration_check.py`
