# Real Robot Recording

Scripts for teleoperating and recording datasets with physical SO100 robot arms using STS3250 motors.

## Setup

### 1. Configure Hardware

Edit `config.json` with your COM ports and camera indices:

```json
{
  "leader": {
    "port": "COM8",
    "id": "leader_so100"
  },
  "follower": {
    "port": "COM7",
    "id": "follower_so100"
  },
  "cameras": {
    "base_0_rgb": {
      "index_or_path": 2,
      "width": 640,
      "height": 480,
      "fps": 30
    },
    "left_wrist_0_rgb": {
      "index_or_path": 0,
      "width": 640,
      "height": 480,
      "fps": 30
    }
  }
}
```

### 2. Calibrate Arms

First time setup - calibrate both arms:

```bash
# Calibrate leader (teleoperator)
lerobot-calibrate --teleop.type=so100_leader_sts3250 --teleop.port=COM8 --teleop.id=leader_so100

# Calibrate follower (robot)
lerobot-calibrate --robot.type=so100_follower_sts3250 --robot.port=COM7 --robot.id=follower_so100
```

Calibration files are saved to `~/.cache/huggingface/lerobot/calibration/`

### 3. Test Teleoperation

```bash
cd scripts/real_robot
python teleoperate_so100.py
```

Move the leader arm - the follower should mirror your movements.

## Recording Datasets

### Quick Start

```bash
cd scripts/real_robot
python record_dataset.py \
    --repo-id danbhf/real_pick_place \
    --task "Pick up the block and place it in the bowl" \
    --num-episodes 20
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--repo-id` | (required) | HuggingFace dataset ID |
| `--task` | (required) | Task description |
| `--num-episodes` | 50 | Number of episodes to record |
| `--fps` | 30 | Recording FPS |
| `--root` | data | Local data directory |
| `--no-push` | false | Don't push to HuggingFace |
| `--display` | false | Show camera feeds |

### Recording Controls

- **Space** - Start/stop recording episode
- **Enter** - Save episode and move to next
- **Backspace** - Discard current episode
- **Escape** - Exit recording

## Files

| File | Description |
|------|-------------|
| `SO100LeaderSTS3250.py` | Leader arm class with STS3250 motors |
| `SO100FollowerSTS3250.py` | Follower robot class with STS3250 motors |
| `config.json` | Hardware configuration (COM ports, cameras) |
| `teleoperate_so100.py` | Test teleoperation script |
| `record_dataset.py` | Dataset recording script |

## Camera Names for Pi0

The camera names in `config.json` are configured for Pi0 compatibility:
- `base_0_rgb` - Overhead/base camera (maps to Pi0's base image)
- `left_wrist_0_rgb` - Wrist camera (maps to Pi0's left wrist image)

## Troubleshooting

### COM port not found
- Check Device Manager for correct COM port numbers
- Ensure USB cables are connected

### Camera not working
- Check camera indices with `python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"`
- On Windows, DirectShow backend is used automatically

### Calibration issues
- Delete calibration files and recalibrate:
  ```bash
  rm ~/.cache/huggingface/lerobot/calibration/teleoperators/leader_so100.json
  rm ~/.cache/huggingface/lerobot/calibration/robots/follower_so100.json
  ```
