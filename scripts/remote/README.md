# Remote Training Setup

Run training on remote GPU instances (vast.ai, RunPod, or any Docker-capable cloud).

## Quick Start

### 1. Build Docker Image

```bash
# From repository root
docker build -t aerdanielbryars101/lerobot-training:latest -f remote/Dockerfile .

# Push to Docker Hub
docker login
docker push aerdanielbryars101/lerobot-training:latest
```

### 2. Set Environment Variables

```bash
export HF_TOKEN="your_huggingface_token"
export WANDB_API_KEY="your_wandb_key"
```

### 3. Run Training

**Local Docker (for testing):**
```bash
docker run --gpus all \
    -e HF_TOKEN=$HF_TOKEN \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    aerdanielbryars101/lerobot-training:latest \
    --dataset danbhf/sim_pick_place_40ep_rgbd_ee \
    --cameras wrist_cam,overhead_cam,overhead_cam_depth \
    --steps 50000 --batch_size 16
```

**vast.ai:**
```bash
# Install vastai CLI
pip install vastai
vastai set api-key YOUR_API_KEY

# Search for instances
python remote/launch_vast.py search --gpu RTX4090 --max_price 0.50

# Launch training
python remote/launch_vast.py launch \
    --dataset danbhf/sim_pick_place_40ep_rgbd_ee \
    --cameras wrist_cam,overhead_cam,overhead_cam_depth \
    --steps 100000 --batch_size 16 \
    --gpu RTX4090 --max_price 0.50

# Monitor instances
python remote/launch_vast.py list
```

---

## Files

| File | Description |
|------|-------------|
| `Dockerfile` | Docker image with CUDA, MuJoCo, and headless rendering |
| `train_remote.py` | Training wrapper with HF/WandB auth handling |
| `launch_vast.py` | Helper script for vast.ai instance management |
| `requirements.txt` | Python dependencies for Docker image |

---

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | (required) | HuggingFace dataset ID |
| `--cameras` | all | Comma-separated camera names |
| `--steps` | 50000 | Training steps |
| `--batch_size` | 16 | Batch size |
| `--lr` | 1e-5 | Learning rate |
| `--save_freq` | 5000 | Checkpoint save frequency |
| `--eval_episodes` | 30 | Episodes per checkpoint evaluation |
| `--eval_randomize` | true | Randomize object positions in eval |
| `--use_joint_actions` | false | Use joint action space (6-dim) |
| `--cache_dataset` | false | Cache dataset in memory |
| `--no_wandb` | false | Disable WandB logging |
| `--wandb_project` | lerobot-thesis | WandB project name |
| `--run_name` | auto | WandB run name |
| `--upload_repo` | none | HF repo for final model upload |

---

## Example Experiments

### RGB+Depth Extended Training (100k steps)
```bash
python remote/launch_vast.py launch \
    --dataset danbhf/sim_pick_place_40ep_rgbd_ee \
    --cameras wrist_cam,overhead_cam,overhead_cam_depth \
    --steps 100000 --batch_size 16 \
    --run_name "rgbd_100k" \
    --gpu RTX4090
```

### Joint Action Space with Depth
```bash
python remote/launch_vast.py launch \
    --dataset danbhf/sim_pick_place_40ep_rgbd_joint \
    --cameras wrist_cam,overhead_cam,overhead_cam_depth \
    --steps 50000 --batch_size 16 \
    --use_joint_actions \
    --run_name "joint_rgbd_50k" \
    --gpu RTX4090
```

### RGB-Only Joint Action Space
```bash
python remote/launch_vast.py launch \
    --dataset danbhf/sim_pick_place_merged_40ep \
    --cameras wrist_cam,overhead_cam \
    --steps 50000 --batch_size 16 \
    --use_joint_actions \
    --run_name "joint_rgb_50k" \
    --gpu RTX4090
```

---

## vast.ai Tips

### Finding Good Instances
```bash
# RTX 4090 - Best performance/price for training
python remote/launch_vast.py search --gpu RTX4090 --max_price 0.50

# A100 - For larger batch sizes
python remote/launch_vast.py search --gpu A100 --max_price 1.50

# Any GPU under $0.30/hr
python remote/launch_vast.py search --max_price 0.30
```

### Monitoring
- **WandB**: All training metrics are logged to your WandB project
- **vast.ai Console**: Monitor instance status at https://vast.ai/console
- **CLI**: `python remote/launch_vast.py list`

### Cost Estimates
| GPU | Price/hr | 50k steps | 100k steps |
|-----|----------|-----------|------------|
| RTX 4090 | ~$0.40 | ~$1.20 | ~$2.40 |
| RTX 3090 | ~$0.25 | ~$1.00 | ~$2.00 |
| A100 40GB | ~$1.20 | ~$2.40 | ~$4.80 |

*Times estimated at ~3 hours for 50k steps with batch_size=16*

---

## Manual vast.ai Setup (Alternative)

If you prefer to use the vast.ai web interface:

1. Go to https://vast.ai/console/create
2. Select a GPU instance (RTX 4090 recommended)
3. Set Docker image: `aerdanielbryars101/lerobot-training:latest`
4. Set environment variables:
   - `HF_TOKEN`: Your HuggingFace token
   - `WANDB_API_KEY`: Your WandB key
5. Set Docker command:
   ```
   --dataset danbhf/sim_pick_place_40ep_rgbd_ee --cameras wrist_cam,overhead_cam,overhead_cam_depth --steps 50000 --batch_size 16
   ```
6. Click "RENT" and wait for the instance to start

---

## Troubleshooting

### MuJoCo rendering fails
The Docker image uses OSMesa for headless rendering. If you see OpenGL errors:
- Make sure `MUJOCO_GL=osmesa` is set
- Check that OSMesa packages are installed

### Out of memory
- Reduce batch size: `--batch_size 8`
- Use a GPU with more VRAM (A100 40GB)

### Dataset download fails
- Check your HF_TOKEN is valid
- Verify dataset exists: `huggingface-cli dataset-info DATASET_NAME`

### WandB not logging
- Check WANDB_API_KEY is set
- Verify with: `wandb login --relogin`
