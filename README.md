# LeRobot VR Simulation

VR-based teleoperation and training data recording for the SO100/SO101 robot arm using MuJoCo simulation and Meta Quest headsets.

## Overview

This repository provides tools for:
- **VR Teleoperation**: Control a simulated robot arm using a physical leader arm while viewing the scene in VR
- **Dataset Recording**: Record demonstration episodes in LeRobot v3.0 format with automatic task completion detection
- **Real Robot Playback**: Play back recorded episodes on a physical follower arm
- **HuggingFace Upload**: Upload recorded datasets to HuggingFace Hub for training

## Repository Structure

```
lerobot-thesis/
├── assets/
│   └── SO-ARM100/              # Robot meshes (git submodule)
├── calibration/
│   ├── leader_so100.json       # Leader arm calibration
│   └── follower_so100.json     # Follower arm calibration
├── configs/
│   └── config.json             # Serial port configuration
├── docs/
│   ├── CALIBRATION.md          # Detailed calibration guide
│   └── EXPERIMENTS.md          # Experiment log
├── scenes/
│   └── so101_with_wrist_cam.xml  # MuJoCo scene
├── recording/
│   ├── record_sim_vr_pickplace.py  # Record demos in VR
│   ├── playback_real_robot.py      # Play on real robot
│   ├── playback_sim_vr.py          # Play in simulation
│   ├── teleop_sim_vr.py            # VR teleoperation
│   ├── teleoperate_so100.py        # Leader-follower teleoperation
│   ├── SO100LeaderSTS3250.py       # Leader arm driver
│   ├── SO100FollowerSTS3250.py     # Follower arm driver
│   └── upload_dataset.py           # Upload to HuggingFace
├── training/
│   ├── train_act.py                # Full ACT training script
│   ├── train_act_simple.py         # Simple training wrapper
│   └── README.md                   # Training documentation
├── inference/
│   ├── run_act_sim.py              # Run trained policy in VR simulation
│   └── evaluate_with_analysis.py   # Evaluate with IK analysis
├── utils/                          # Shared utility modules
│   ├── constants.py                # Joint limits, action bounds
│   └── conversions.py              # Coordinate conversions (FK/IK helpers)
├── scripts/
│   ├── test_fk_ik.py               # FK/IK module tests
│   ├── teleop_ee_sim.py            # End-effector teleoperation
│   └── merge_datasets.py           # Merge multiple datasets
├── tests/
│   └── test_conversions.py         # Unit tests for conversions
├── src/
│   └── lerobot_robot_sim/      # LeRobot plugin for simulation
└── vendor/
    └── scservo_sdk/            # Feetech servo SDK
```

## Prerequisites

- Windows 10/11 (VR support is Windows-only)
- Meta Quest 2/3 headset with Quest Link or Air Link
- Physical SO100 leader arm (for teleoperation)
- Python 3.10+

## Installation

1. **Clone with submodules**:
   ```bash
   git clone --recursive https://github.com/yourusername/lerobot-thesis.git
   cd lerobot-thesis
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the simulation plugin**:
   ```bash
   pip install -e src/lerobot_robot_sim --no-deps
   ```

5. **Copy calibration files**:
   ```bash
   # Windows (PowerShell)
   mkdir -Force "$env:USERPROFILE\.cache\huggingface\lerobot\calibration\teleoperators\so100_leader_sts3250"
   mkdir -Force "$env:USERPROFILE\.cache\huggingface\lerobot\calibration\robots\so100_follower_sts3250"
   copy calibration\leader_so100.json "$env:USERPROFILE\.cache\huggingface\lerobot\calibration\teleoperators\so100_leader_sts3250\"
   copy calibration\follower_so100.json "$env:USERPROFILE\.cache\huggingface\lerobot\calibration\robots\so100_follower_sts3250\"
   ```

6. **Configure serial ports** (edit `configs/config.json`):
   ```json
   {
     "leader": {"port": "COM8", "id": "leader_so100"},
     "follower": {"port": "COM7", "id": "follower_so100"}
   }
   ```

## Usage

### Recording Demonstrations (VR)

Record pick-and-place demonstrations for training:

```bash
python recording/record_sim_vr_pickplace.py --num_episodes 20
```

**Options**:
- `--num_episodes N`: Number of episodes to record (default: 10)
- `--fps N`: Recording frame rate (default: 30)
- `--task "description"`: Task description for dataset
- `--pos_range N`: Position randomization in cm (default: 4)
- `--rot_range N`: Rotation randomization in degrees (default: 180)
- `--no-randomize`: Disable object randomization
- `--no-upload`: Don't upload to HuggingFace after recording

**Controls During Recording**:
- `ENTER` - Start recording / Save episode
- `D` - Discard current episode
- `R` - Reset scene (when not recording)
- `Q` - Quit
- `SPACEBAR` - Recenter VR view (keyboard fallback)

**VR Controller Controls**:
- Left Thumbstick: Move forward/back, left/right
- Right Thumbstick: Move up/down, rotate view
- X Button (left): Recenter robot in front of you

Episode auto-completes when the Duplo block lands in the bowl.

### Playback on Real Robot

Play back recorded episodes on the physical follower arm:

```bash
python recording/playback_real_robot.py danbhf/sim_pick_place_20251229_144730
```

**Options**:
- `--episode N`: Play specific episode only
- `--loop`: Loop playback continuously
- `--local`: Force load from local path

**Controls During Playback**:
- `ENTER/SPACE` - Start / Pause
- `R` - Replay current episode
- `N` - Next episode
- `Q` - Quit

### Playback in Simulation (VR)

Preview recorded episodes in VR simulation:

```bash
python recording/playback_sim_vr.py danbhf/sim_pick_place_20251229_144730
```

### Leader-Follower Teleoperation

Direct teleoperation without simulation (leader controls follower):

```bash
python recording/teleoperate_so100.py
```

### VR Teleoperation (Test Mode)

Test VR setup without recording:

```bash
python recording/teleop_sim_vr.py          # With leader arm
python recording/teleop_sim_vr.py --test   # VR only, no arm
```

### Upload Dataset

Manually upload a recorded dataset:

```bash
python recording/upload_dataset.py datasets/20251229_101340 danbhf/my_dataset_name
```

### Training a Policy

Train an ACT (Action Chunking Transformer) policy on your recorded data:

```bash
# Basic training (50k steps)
python training/train_act.py danbhf/sim_pick_place_20251229_101340

# Training with simulation evaluation after each checkpoint
python training/train_act.py danbhf/sim_pick_place_20251229_101340 \
    --steps 50000 --eval_episodes 10 --eval_randomize

# Quick test run
python training/train_act.py danbhf/sim_pick_place_20251229_101340 \
    --steps 5000 --batch_size 4 --save_freq 1000 --eval_episodes 5
```

**Training Options:**
- `--steps N`: Training steps (default: 50000)
- `--batch_size N`: Batch size (default: 8)
- `--lr N`: Learning rate (default: 1e-5)
- `--save_freq N`: Checkpoint save frequency (default: 5000)
- `--eval_episodes N`: Run N simulation episodes after each checkpoint (default: 0 = disabled)
- `--eval_randomize`: Randomize object position during evaluation
- `--no_wandb`: Disable WandB logging

See [training/README.md](training/README.md) for more options.

### Evaluating a Trained Policy

Run your trained ACT policy in the VR simulation:

```bash
# Run with final model (VR mode)
python inference/run_act_sim.py outputs/train/act_20251229_111846/final

# Without VR (uses MuJoCo viewer) - faster for batch evaluation
python inference/run_act_sim.py outputs/train/act_20251229_111846/final --no_vr

# Run 20 episodes with WandB logging
python inference/run_act_sim.py outputs/train/act_20251229_111846/final \
    --episodes 20 --no_vr

# Compare checkpoints (disable WandB for quick tests)
python inference/run_act_sim.py outputs/train/act_20251229_111846/checkpoint_025000 \
    --episodes 10 --no_vr --no_wandb
```

**Options:**
- `--episodes N`: Number of evaluation episodes (default: 5)
- `--fps N`: Simulation frame rate (default: 30)
- `--max_steps N`: Maximum steps per episode (default: 300)
- `--no_vr`: Use MuJoCo viewer instead of VR headset
- `--no_randomize`: Disable object randomization
- `--pos_range N`: Position randomization in cm (default: 4)
- `--rot_range N`: Rotation randomization in degrees (default: 180)
- `--no_wandb`: Disable WandB logging

**Output:** Reports success rate, average steps, and average time per episode. Logs to WandB for historical comparison.

## End-Effector Action Space

This repository supports training in two action spaces:

### Joint Action Space (Default)
- 6-dimensional: `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`
- Normalized values: -100 to +100 (joints), 0 to +100 (gripper)
- Direct motor commands

### End-Effector Action Space
- 8-dimensional: `[x, y, z, qw, qx, qy, qz, gripper]`
- Position in meters, quaternion orientation, normalized gripper
- Requires FK/IK conversion at inference time

**Converting a Dataset:**
```bash
python recording/convert_to_ee_actions.py danbhf/source_dataset danbhf/output_dataset_ee
```

**Training with EE Actions:**
```bash
python training/train_act.py danbhf/my_dataset_ee --steps 50000 --eval_episodes 30
```

The `utils/` module provides shared conversion functions used across all scripts:
- `radians_to_normalized()` / `normalized_to_radians()` - Convert between radian and normalized action space
- `quaternion_to_rotation_matrix()` / `rotation_matrix_to_quaternion()` - Quaternion ↔ rotation matrix
- `clip_joints_to_limits()` - Enforce joint limits

## Scene Description

The simulation includes:
- **SO101 Robot Arm**: 6-DOF arm with gripper
- **Wrist Camera**: 640x480 camera mounted on the gripper
- **Overhead Camera**: Fixed camera with top-down view
- **Table**: Wooden table surface
- **Duplo Block**: Red building block (pick-and-place target)
- **Bowl**: Blue bowl (placement target)

## Calibration

See [docs/CALIBRATION.md](docs/CALIBRATION.md) for detailed calibration information.

Calibration files for both arms are included in `calibration/` and must be copied to the LeRobot cache location before first use (see Installation step 5).

## Troubleshooting

### VR Not Working
- Ensure Quest Link or Air Link is connected
- Check that SteamVR is not running (conflicts with OpenXR)
- The Meta Quest runtime should be set as the default OpenXR runtime
- Use SPACEBAR in console to recenter if controllers don't respond

### Leader Arm Not Responding
- Check the COM port in `configs/config.json`
- Ensure the arm is powered and connected via USB
- Try: `python -c "from lerobot.motors.feetech import FeetechMotorsBus; print('OK')"`

### Upload Failures
- Ensure you're logged into HuggingFace: `huggingface-cli login`
- Check your internet connection
- Verify the repository name format: `username/dataset_name`

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [LeRobot](https://github.com/huggingface/lerobot) - Robot learning framework
- [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) - Robot arm design
- [MuJoCo](https://mujoco.org/) - Physics simulation
- [PyOpenXR](https://github.com/cmbruns/pyopenxr) - OpenXR bindings for Python
