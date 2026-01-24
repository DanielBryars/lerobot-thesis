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
│   ├── so101_with_wrist_cam.xml  # Standard scene (52° overhead FOV)
│   └── so101_rgbd.xml            # RGBD scene with D435 specs (58° FOV)
├── scripts/recording/
│   ├── record_sim_vr_pickplace.py  # Record demos in VR
│   ├── rerecord_dataset.py         # Re-record dataset with new scene/cameras
│   ├── sim_teleop_viewer.py        # Interactive sim viewer with depth
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
python scripts/recording/record_sim_vr_pickplace.py --task "Pick up the Duplo block and place it in the bowl" --num_episodes 20
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
python scripts/recording/playback_real_robot.py danbhf/sim_pick_place_20251229_144730
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
python scripts/recording/playback_sim_vr.py danbhf/sim_pick_place_20251229_144730
```

### Leader-Follower Teleoperation

Direct teleoperation without simulation (leader controls follower):

```bash
python scripts/recording/teleoperate_so100.py
```

### VR Teleoperation (Test Mode)

Test VR setup without recording:

```bash
python scripts/recording/teleop_sim_vr.py          # With leader arm
python scripts/recording/teleop_sim_vr.py --test   # VR only, no arm
```

### Simulation Viewer

Interactive viewer for testing scenes and visualizing depth:

```bash
python scripts/recording/sim_teleop_viewer.py                              # Default RGBD scene
python scripts/recording/sim_teleop_viewer.py --scene scenes/so101_with_wrist_cam.xml  # Custom scene
```

**Controls:**
- `W/S/A/D` - Move arm forward/back/left/right
- `Q/E` - Move arm up/down
- `1-6` - Select joint to control
- `+/-` - Adjust selected joint
- `Z` - Toggle depth view (shows colorized depth from overhead camera)
- `ESC` - Quit

The viewer displays external camera, wrist camera, overhead camera, and optionally depth visualization.

### Object Positioning Tool

Find coordinates for custom block positions before recording:

```bash
python scripts/tools/position_objects.py
```

**Controls:**
- `W/S` - Move block forward/backward (X axis, 1cm steps)
- `A/D` - Move block left/right (Y axis, 1cm steps)
- `SHIFT + WASD` - Fine movement (2mm steps)
- `P` - Print block position with copy-paste command
- `R` - Reset scene
- `ESC` - Quit

**Output example:**
```
>>> BLOCK POSITION: x=0.2500, y=0.1500
    Height: z=0.0200, Rotation: 45.0 deg
    Command: --block-x 0.250 --block-y 0.150
```

Use the printed `--block-x` and `--block-y` values with `record_sim_vr_pickplace.py`.

**Note:** Moving the block far from its initial position may cause rendering artifacts (disappearing floor, visual glitches). These are cosmetic only - the coordinates are still accurate.

### Re-recording Datasets

Re-record an existing dataset with a new scene or camera setup while preserving the exact same movements:

```bash
# Re-record with depth enabled
python scripts/recording/rerecord_dataset.py danbhf/source_dataset --depth

# Specify output name
python scripts/recording/rerecord_dataset.py danbhf/source_dataset --depth --output my_rgbd_dataset

# Use custom scene
python scripts/recording/rerecord_dataset.py danbhf/source_dataset --scene scenes/custom.xml
```

**Options:**
- `--depth`: Enable depth recording for overhead camera
- `--output NAME`: Output dataset name (default: auto-generated with timestamp)
- `--scene PATH`: Custom scene XML path
- `--fps N`: Override FPS (default: use source dataset FPS)
- `--root DIR`: Output root directory (default: ./datasets)

**Requirements:**
- Source dataset must have `action_joints` field (joint-space actions)
- Preserves both `action_joints` and `action` (EE) fields from source

This is useful for:
- Adding depth data to existing RGB datasets
- Testing different camera angles with identical movements
- Comparing training with different observation setups

### Upload Dataset

Manually upload a recorded dataset:

```bash
python scripts/recording/upload_dataset.py datasets/20251229_101340 danbhf/my_dataset_name
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
python scripts/recording/convert_to_ee_actions.py danbhf/source_dataset danbhf/output_dataset_ee
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
- **Overhead Camera**: Fixed camera with top-down view (supports RGBD)
- **Table**: Wooden table surface
- **Duplo Block**: White building block (pick-and-place target)
- **Bowl**: Cream bowl (placement target)

### Available Scenes
- `scenes/so101_with_wrist_cam.xml` - Standard scene with 52° overhead FOV
- `scenes/so101_rgbd.xml` - RGBD scene with Intel RealSense D435 specs (58° vertical FOV)

### Depth Rendering

MuJoCo can render depth from **any camera** - it's a render mode, not a scene-specific feature. The RGBD scene just uses the D435's correct field of view.

```python
# Enable depth rendering on any MuJoCo renderer
renderer.enable_depth_rendering()
depth = renderer.render()  # Returns [H, W] float32 in meters
renderer.disable_depth_rendering()
```

**Testing depth with the teleop viewer:**
```bash
python scripts/recording/sim_teleop_viewer.py                          # Default scene
python scripts/recording/sim_teleop_viewer.py --scene scenes/so101_rgbd.xml  # D435 FOV
# Press Z to toggle depth view
```

**Recording with depth:**
```bash
python scripts/recording/record_sim_vr_pickplace.py --depth --num_episodes 40
```

**Training with depth:**
```bash
# RGB + Depth
python training/train_act.py dataset --cameras wrist_cam,overhead_cam,overhead_cam_depth

# RGB only (baseline)
python training/train_act.py dataset --cameras wrist_cam,overhead_cam
```


python scripts/tools/visualize_temporal_ensemble_interactive.py outputs/train/act_20260118_155135 --checkpoint checkpoint_045000

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

Here are the key command lines from your experiments:
                                                                                                                                       Spatial Visualization (with circles/spheres)                                                                                       
  # Scatter visualization from CSV (spheres in MuJoCo)
  python scripts/experiments/eval_spatial_generalization.py outputs/experiments/spatial_eval_combined.csv --scatter

  # With custom sphere size and transparency
  python scripts/experiments/eval_spatial_generalization.py outputs/experiments/spatial_eval_combined.csv --scatter --sphere-radius
  0.008 --sphere-alpha 0.4

  # Visualize from JSON heatmap
  python scripts/experiments/eval_spatial_generalization.py outputs/experiments/spatial_eval_20260121_160613.json --visualize

  Other Recently Used Commands

  # Temporal ensemble visualization (live with whiskers)
  python scripts/tools/visualize_temporal_ensemble_live.py outputs/train/act_20260118_155135 --checkpoint checkpoint_045000

  # Interactive frame-by-frame viewer
  python scripts/tools/visualize_temporal_ensemble_interactive.py outputs/train/act_20260118_155135 --checkpoint checkpoint_045000

  # Evaluation with temporal ensembling
  python scripts/inference/eval.py outputs/train/act_20260118_155135 --local --checkpoint checkpoint_045000 --episodes 50 --ensemble
  0.01

  # Standard evaluation
  python scripts/inference/eval.py outputs/train/act_20260118_155135 --local --checkpoint checkpoint_045000 --episodes 50

  # Data scaling experiment (currently running)
  python scripts/experiments/data_scaling_experiment.py --resume-from 2

  # Brady Bunch tiled video (ffmpeg)
  c:/ffmpeg/bin/ffmpeg -framerate 30 -i overview/frame_%05d.png -i side_view/frame_%05d.png ... -filter_complex
  "[0:v]scale=640:360..." temporal_ensemble_6angles.mp4

  Available CSV Files for Visualization

  outputs/experiments/spatial_eval_combined.csv  # 2630 episodes, all grids combined
  outputs/experiments/spatial_eval_fine_grid.csv # Fine 7x7 grid


## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [LeRobot](https://github.com/huggingface/lerobot) - Robot learning framework
- [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) - Robot arm design
- [MuJoCo](https://mujoco.org/) - Physics simulation
- [PyOpenXR](https://github.com/cmbruns/pyopenxr) - OpenXR bindings for Python
