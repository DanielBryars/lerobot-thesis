# LeRobot VR Simulation

VR-based teleoperation and training data recording for the SO100/SO101 robot arm using MuJoCo simulation and Meta Quest headsets.

## Overview

This repository provides tools for:
- **VR Teleoperation**: Control a simulated robot arm using a physical leader arm while viewing the scene in VR
- **Dataset Recording**: Record demonstration episodes in LeRobot v3.0 format with automatic task completion detection
- **HuggingFace Upload**: Upload recorded datasets to HuggingFace Hub for training

## Repository Structure

```
lerobot-thesis/
├── assets/
│   └── SO-ARM100/          # Robot meshes (git submodule)
├── configs/
│   └── config.json         # Serial port configuration
├── scenes/
│   └── so101_with_wrist_cam.xml  # MuJoCo scene with robot, table, objects
├── scripts/
│   ├── teleop_sim_vr.py    # VR teleoperation (standalone)
│   ├── vr_openxr_viewer.py # VR renderer with full controls
│   ├── record_sim_vr_pickplace.py  # Record pick-and-place demos
│   └── upload_dataset.py   # Upload datasets to HuggingFace
├── src/
│   └── lerobot_robot_sim/  # LeRobot plugin for simulation
└── vendor/
    └── scservo_sdk/        # Feetech servo SDK (for motor control)
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

5. **Configure serial port** (edit `configs/config.json`):
   ```json
   {
     "leader": {"port": "COM8"},
     "follower": {"port": "COM7"}
   }
   ```

## Usage

### VR Teleoperation (View Only)

Test the VR setup without a physical arm:

```bash
cd scripts
python teleop_sim_vr.py --test
```

### VR Teleoperation with Leader Arm

Control the simulation with your physical leader arm:

```bash
cd scripts
python teleop_sim_vr.py
```

**VR Controls**:
- Left Thumbstick: Move forward/back (Y), left/right (X)
- Right Thumbstick: Move up/down (Y), rotate view (X)
- X Button (left controller): Recenter robot in front of you

### Recording Demonstrations

Record pick-and-place demonstrations for training:

```bash
cd scripts
python record_sim_vr_pickplace.py --num_episodes 10
```

**Options**:
- `--num_episodes N`: Number of episodes to record (default: 10)
- `--fps N`: Recording frame rate (default: 30)
- `--task "description"`: Task description for dataset
- `--pos_range N`: Position randomization in cm (default: 2)
- `--rot_range N`: Rotation randomization in degrees (default: 180)
- `--no-randomize`: Disable object randomization
- `--no-upload`: Don't upload to HuggingFace after recording

**During Recording**:
- Press `q` to stop and save the current episode
- Press `d` to discard the current episode
- Episode auto-completes when the Duplo block lands in the bowl

### Uploading Datasets

Manually upload a recorded dataset:

```bash
cd scripts
python upload_dataset.py path/to/dataset repo_id
```

## Scene Description

The simulation includes:
- **SO101 Robot Arm**: 6-DOF arm with gripper
- **Wrist Camera**: 640x480 camera mounted on the gripper
- **Overhead Camera**: Fixed camera with top-down view
- **Table**: Wooden table surface
- **Duplo Block**: Red building block (pick-and-place target)
- **Bowl**: Blue bowl (placement target)

Task completion is detected when the Duplo block enters the bowl.

## Calibration

The leader arm requires calibration. Calibration files are stored in:
```
~/.cache/huggingface/lerobot/calibration/teleoperators/so100_leader_sts3250/
```

## Troubleshooting

### VR Not Working
- Ensure Quest Link or Air Link is connected
- Check that SteamVR is not running (conflicts with OpenXR)
- The Meta Quest runtime should be set as the default OpenXR runtime

### Leader Arm Not Responding
- Check the COM port in `configs/config.json`
- Ensure the arm is powered and connected via USB
- Try running `python -c "from lerobot.motors.feetech import FeetechMotorsBus; print('OK')"`

### Upload Failures
- Ensure you're logged into HuggingFace: `huggingface-cli login`
- Check your internet connection
- Verify the repository name format: `username/dataset_name`

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [LeRobot](https://github.com/huggingface/lerobot) - Robot learning framework
- [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) - Robot arm design and simulation models
- [MuJoCo](https://mujoco.org/) - Physics simulation
- [PyOpenXR](https://github.com/cmbruns/pyopenxr) - OpenXR bindings for Python
