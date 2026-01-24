# SO100 Motor Calibration Guide

This document explains how calibration works for the SO100 robot arms with STS3250 Feetech motors.

## Overview

The SO100 uses Feetech STS3250 servo motors with 12-bit encoders (0-4095 ticks per revolution). Calibration involves two levels:

1. **EEPROM-level (Feetech firmware)**: Position Offset Value stored in motor
2. **Software-level (LeRobot)**: Range min/max stored in JSON calibration files

## The Encoder Wrap-Around Problem

### Symptom

When rotating the wrist_roll joint fully in one direction, the position value suddenly "jumps" from one extreme to the other:

```
wrist_roll: 3985 -> 98   (anticlockwise, jumped ~3887 ticks)
wrist_roll: 177 -> 4046  (clockwise, jumped ~3869 ticks)
```

### Cause

The encoder has a 0/4095 boundary. When the physical position crosses this boundary, the raw encoder value wraps around:

- Going anticlockwise past 0 → wraps to ~4095
- Going clockwise past 4095 → wraps to ~0

If the motor's working range spans this boundary (e.g., `range_min=247, range_max=3888`), normal operation will cause wrap-around.

### Solution: Homing Offset in EEPROM

The Feetech STS3250 motors have a **Homing_Offset** register (address 31, 2 bytes) in EEPROM that shifts the encoder's zero point. This is the same as what the Feetech Debug Tool calls "Position Offset Value". By setting an appropriate offset, you can move the 0/4095 boundary to a position the motor will never reach during normal operation.

**Note:** There is only ONE offset register in STS3250 motors. LeRobot calls it "Homing_Offset", Feetech Debug Tool calls it "Position Offset Value" - they are the same register at address 31.

**Before (Offset = 0):**
```
Encoder:     0 ----[247====working range====3888]---- 4095
                    ^                           ^
                    min                        max
Problem: Working range is near both boundaries!
```

**After (Offset = ~1737):**
```
Encoder:     0 --------[====working range====]-------- 4095
                       ^                      ^
                      new min               new max
The 0/4095 boundary is now far from the working range.
```

## Using the Feetech Debug Tool

### Reading Current Values

1. Open the Feetech Debug Tool (FD Software)
2. Connect to the motor (set correct COM port and baud rate)
3. Click "Read" to read current EEPROM values
4. Note the "Position Offset Value" field

### Setting Homing Offset

1. Move the motor to its **center position** (middle of working range)
2. Note the current "Present Position" value
3. Calculate offset: `offset = present_position - 2048`
   - This centers the working range around 2048 (middle of 0-4095)
4. Write the offset to "Homing_Offset" (called "Position Offset Value" in Feetech Debug Tool)
5. Click "Write" to save

**Note on sign encoding:** The STS3250 uses sign-magnitude encoding for Homing_Offset where bit 11 is the sign bit. When reading back, values > 2048 are negative (e.g., 2100 means -52).

### Verification

After setting the offset:
- Rotate the joint through its full range of motion
- The encoder value should stay well within 500-3500 range
- No wrap-around should occur at the extremes

## Annotated Screenshots

### Without Offset (Wrap-Around Problem)
![No Offset](NoOffsetAnnotated.jpg)

The encoder value wraps around when the joint reaches its limit.

### With Offset (Problem Fixed)
![With Offset](WithOffsetAnnotated.jpg)

The encoder value stays within a safe range throughout the joint's motion.

## LeRobot Calibration Files

LeRobot stores software-level calibration in JSON files at:
```
~/.cache/huggingface/lerobot/calibration/teleoperators/so100_leader_sts3250/leader_so100.json
~/.cache/huggingface/lerobot/calibration/robots/so100_follower_sts3250/follower_so100.json
```

### File Format

```json
{
    "motor_name": {
        "id": 1,                  // Motor ID on the bus
        "drive_mode": 0,          // Direction (0=normal, 1=reversed)
        "homing_offset": 0,       // Software offset (usually 0, use EEPROM instead)
        "range_min": 803,         // Minimum encoder value in working range
        "range_max": 3306         // Maximum encoder value in working range
    }
}
```

### How Normalization Works

LeRobot normalizes raw encoder values to a standard range:

- **RANGE_M100_100**: Maps `[range_min, range_max]` → `[-100, +100]`
- **RANGE_0_100**: Maps `[range_min, range_max]` → `[0, 100]`

The normalization formula:
```python
norm = ((raw - range_min) / (range_max - range_min)) * 200 - 100  # for RANGE_M100_100
```

### Re-calibrating After EEPROM Changes

If you change the Position Offset Value in EEPROM, you must re-run LeRobot calibration:

```bash
# Delete old calibration
rm -rf ~/.cache/huggingface/lerobot/calibration/teleoperators/so100_leader_sts3250/

# Run calibration
python -c "
from scripts.recording.SO100LeaderSTS3250 import SO100LeaderSTS3250, SO100LeaderSTS3250Config
leader = SO100LeaderSTS3250(SO100LeaderSTS3250Config(port='COM8', id='leader_so100'))
leader.connect(calibrate=True)
leader.disconnect()
"
```

## Motor Reference

| Joint | Motor ID | Typical Range | Notes |
|-------|----------|---------------|-------|
| shoulder_pan | 1 | 803-3306 | |
| shoulder_lift | 2 | 920-3195 | |
| elbow_flex | 3 | 939-3142 | |
| wrist_flex | 4 | 1015-3176 | |
| wrist_roll | 5 | 247-3888 | **Prone to wrap-around** - needs EEPROM offset |
| gripper | 6 | 2046-3297 | Normalized 0-100 (not -100 to +100) |

## Scripts in this Folder

### check_centering.py

Diagnostic script that reads current motor positions and EEPROM Homing_Offset values. Checks one arm at a time so you can hold each in the zero pose.

**Usage:**
```bash
python calibration/check_centering.py
```

**What it does:**
1. Prompts you to hold the Leader arm at zero pose, press Enter
2. Reads and displays motor data for Leader (COM8)
3. Prompts you to hold the Follower arm at zero pose, press Enter
4. Reads and displays motor data for Follower (COM7)
5. For each motor, displays:
   - `ID`: Motor ID on the bus
   - `Position`: Current raw encoder value (0-4095)
   - `Homing_Offset`: Value stored in motor EEPROM (decoded from sign-magnitude)
   - `Dist from 2048`: How far the current position is from center

**Example output:**
```
============================================================
Leader (COM8)
============================================================

Hold the Leader arm at ZERO POSE (gripper vertical, moving finger up)
Press Enter when ready...

Motor             ID   Position   Homing_Offset   Dist from 2048
----------------------------------------------------------------------
shoulder_pan       1       2048             +0              +0
shoulder_lift      2       2050             +2              +2
elbow_flex         3       2045             -3              -3
wrist_flex         4       2048             +0              +0
wrist_roll         5       2100            +52             +52
gripper            6       2048             +0              +0
```

**Interpreting results:**
- Motors with `Dist from 2048` close to 0 are well-centered
- Large distances (>200) indicate the motor needs re-centering
- The `Homing_Offset` shows what offset is currently applied in EEPROM

**When to use:**
- Before making any EEPROM changes (to record current state)
- After running `set_homing_offsets.py` to verify changes
- When diagnosing wrap-around issues

### set_homing_offsets.py

Sets EEPROM homing offsets so all motors read 2048 at the current physical position (zero pose).

**Usage:**
```bash
python calibration/set_homing_offsets.py
```

**What it does:**
1. Prompts you to position both arms at zero pose
2. For each arm (Leader COM8, Follower COM7):
   - Reads current position for each motor
   - Calculates offset needed: `offset = current_position - 2048`
   - Writes offset to motor EEPROM
   - Verifies the motor now reads ~2048
   - Re-reads EEPROM to confirm offset was stored

**Example output:**
```
Processing Leader (COM8)
==================================================

1. Current positions at zero pose:
   shoulder_pan   : 1850
   wrist_roll     : 3500
   ...

2. Calculating homing offsets (target = 2048):
   shoulder_pan   : 1850 - 2048 = -198
   wrist_roll     : 3500 - 2048 = +1452
   ...

3. Writing homing offsets to EEPROM...
   shoulder_pan   : wrote -198
   wrist_roll     : wrote +1452
   ...

4. Verifying (should all be ~2048):
   shoulder_pan   : 2048 (diff: +0) OK
   wrist_roll     : 2048 (diff: +0) OK
   ...
```

**Important:**
- Position BOTH arms at the exact same zero pose before running
- Zero pose = grippers vertical, moving finger above fixed finger
- After running, you must re-calibrate LeRobot (delete old JSON files)

**When to use:**
- Initial setup of new motors
- After physically repositioning a motor on the arm
- To fix wrap-around issues by centering the working range

## Troubleshooting

### "Jump" detected during operation

If you see sudden jumps in position values:
1. Run the diagnostic script: `python scripts/tools/diagnose_motor_wrap.py`
2. Identify which motor is wrapping (usually wrist_roll)
3. Use Feetech tool to set Position Offset Value
4. Re-run LeRobot calibration

### Calibration file not found

LeRobot looks for calibration files based on the teleoperator/robot `id` field in the config. Make sure the ID matches the folder name in the calibration cache.

### Recording still shows wrap-around

After changing EEPROM offset:
1. Power cycle the motor (offset takes effect on boot)
2. Delete and recreate the LeRobot calibration file
3. Verify with diagnostic script before recording

## Zero Pose Convention

Both leader and follower arms are calibrated with the same zero pose:

**Wrist roll zero position:** Grippers facing vertically, with the moving gripper finger **above** the fixed finger.

This convention must be consistent between leader and follower for teleoperation to work correctly.

## Does Changing EEPROM Offset Affect Existing Recordings?

**No, as long as you re-calibrate properly.**

Recorded data uses **normalized values** (-100 to +100), which represent positions in the **physical range of motion**, not raw encoder ticks.

**What happens when you change EEPROM offset:**

1. Raw encoder values for the same physical position change
2. You must re-calibrate LeRobot (move joints to their physical limits)
3. New `range_min`/`range_max` capture the same physical limits (just different raw values)
4. Normalized 0.0 = center of physical range (same physical position as before)
5. Normalized ±100 = physical limits (same as before)

**Example:**

| | Before Offset Change | After Offset Change |
|---|---|---|
| Physical position | Gripper horizontal | Gripper horizontal |
| Raw encoder | 1500 | 3200 |
| range_min | 247 | 2000 |
| range_max | 3888 | 5600 (wraps to ~1500) |
| **Normalized** | **+30.5** | **+30.5** |

The normalized value is the same because it represents the physical position, not the raw encoder value.

**The only way recordings would break:**
- If you don't re-calibrate after changing the offset
- If the physical range limits somehow change

## Choosing an EEPROM Offset Strategy

**Option A: Keep current zero pose, add offset to avoid wrap**
- Zero pose remains "grippers vertical"
- Set offset so the 0/4095 boundary is far from working range
- Current range 247-3888 spans near both boundaries
- Need offset of ~1800-2000 to center the range

**Option B: Change zero pose to horizontal**
- Zero pose becomes "grippers horizontal" (90° anticlockwise from current)
- This naturally centers the working range away from boundaries
- BUT: Inconsistent with how arms were originally calibrated

**Recommendation:** Option A - add an offset while keeping the same zero pose convention. This is less disruptive and maintains consistency with existing documentation/muscle memory.

## Wrist Roll Wrap-Around: Detailed Analysis

### The Fundamental Problem

The wrist_roll joint has ~320° of physical rotation, which maps to ~3641 encoder ticks. This is **89% of the full 4096-tick encoder range**. With such a large working range, it's impossible to position the range far from both the 0 and 4095 boundaries simultaneously.

**Current calibration:**
```
Leader wrist_roll:   range_min=247, range_max=3888 (span: 3641 ticks)
Follower wrist_roll: range_min=278, range_max=3919 (span: 3641 ticks)
```

```
Encoder:  0 ----[247========working range========3888]---- 4095
               ↑                                    ↑
          ~247 ticks                           ~207 ticks
          from 0                               from 4095
```

The working range is close to BOTH boundaries, leaving only ~200 ticks of margin on each side.

### Why Changing Zero Pose Doesn't Help

We considered changing the zero pose from vertical to horizontal (90° rotation) to shift the range away from boundaries:

**If rotating 90° anticlockwise (from user's perspective behind robot):**
- Old range: 247-3888
- New range: 1271-4912 → 4912 wraps to 816
- Result: Range becomes 816-1271, which **crosses the 0/4095 boundary**. Worse!

**If rotating 90° clockwise:**
- Old range: 247-3888
- New range: (247-1024)-(3888-1024) = -777-2864 → -777 wraps to 3319
- Result: Range becomes 2864-3319. Better, but...

**The problem:** Changing zero pose would invalidate all existing datasets and trained models because:
1. Datasets store normalized values (-100 to +100)
2. These were recorded with normalized 0 = vertical gripper
3. If we change to normalized 0 = horizontal gripper, all wrist_roll values shift by ~90°
4. Trained models would output commands that are 90° off

### Why Software Offset Doesn't Help

We investigated using the `homing_offset` field in the calibration JSON to apply a software offset for sim compatibility. However, **lerobot's normalization code doesn't use this field**:

```python
# From motors_bus.py _normalize() - only uses range_min, range_max, drive_mode
bounded_val = min(max_, max(min_, val))
norm = (((bounded_val - min_) / (max_ - min_)) * 200) - 100
```

The `homing_offset` field is only written to EEPROM, not applied as a software offset during normalization.

### Calibration File Locations

```
C:\Users\<user>\.cache\huggingface\lerobot\calibration\teleoperators\so100_leader_sts3250\leader_so100.json
C:\Users\<user>\.cache\huggingface\lerobot\calibration\robots\so100_follower_sts3250\follower_so100.json
```

### Normalization Mapping

| Normalized | Raw (Leader) | Physical Position |
|------------|--------------|-------------------|
| -100 | 247 | Clockwise limit |
| 0 | ~2067 | Vertical (zero pose) |
| +100 | 3888 | Anticlockwise limit |

The wrap-around occurs when going **anticlockwise past +100** (from user's perspective standing behind the robot). The encoder crosses from ~4095 to ~0, causing a sudden jump.

### Dataset Analysis

In a typical dataset, wrist_roll values are:
- `observation.state[4]`: -24 to +17 (follower position)
- `action[4]`: -2 to +99 (leader command)

Converting +99 normalized to raw: `((99+100)/200) * 3641 + 247 = 3871`, which is 17 ticks below range_max and 224 ticks from the wrap point. Close but safe.

### Practical Workaround: Careful Operation

Since fixing the wrap-around would invalidate existing data and models, the practical solution is:

**Avoid rotating the leader's wrist_roll to the anticlockwise extreme during teleoperation.**

The usable range is approximately **-100 to +95 normalized**, which provides plenty of range for typical pick-and-place tasks. Only extreme wrist rotations near the +100 limit risk wrap-around.

**Warning signs you're approaching the danger zone:**
- Normalized wrist_roll > +90
- Raw encoder > 3700
- Getting close to the anticlockwise physical stop

### Potential Software Fixes (Not Implemented)

If wrap-around becomes a frequent problem, these fixes could be added:

1. **Clamp at recording time:** Limit wrist_roll commands to [-95, +95] normalized
2. **Wrap detection:** If raw value suddenly drops below 247 (and you never go there legitimately), treat it as 4095
3. **Dead zone:** Add physical or software stops before the wrap point

### check_center_live.py

Live monitoring script to see raw and normalized values in real-time.

**Usage:**
```bash
python calibration/check_center_live.py          # defaults to leader
python calibration/check_center_live.py follower  # for follower
python calibration/check_center_live.py --rate 20 # faster update rate
```

**What it shows:**
- Raw encoder values for all motors
- Calibration range_min and range_max for reference
- Normalized values (without clamping, so you can see out-of-range)
- Status warnings when near danger zones
- Detailed wrist_roll distance calculations

Use this to monitor the wrist_roll position when testing the wrap-around behavior.
