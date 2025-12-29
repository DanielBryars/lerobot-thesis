# Training Scripts

Scripts for training imitation learning policies on recorded datasets.

## Quick Start

Train an ACT policy on your recorded dataset:

```bash
python training/train_act.py danbhf/sim_pick_place_20251229_101340
```

## Scripts

### train_act.py

Train an ACT (Action Chunking Transformer) policy.

```bash
# Basic training (50k steps)
python training/train_act.py danbhf/sim_pick_place_20251229_101340

# Quick test (5k steps, smaller batch)
python training/train_act.py danbhf/sim_pick_place_20251229_101340 --steps 5000 --batch_size 4

# Full training with custom settings
python training/train_act.py danbhf/sim_pick_place_20251229_101340 \
    --steps 100000 \
    --batch_size 8 \
    --lr 1e-5 \
    --chunk_size 100
```

**Arguments:**
- `dataset`: HuggingFace dataset repo ID (required)
- `--steps`: Training steps (default: 50000)
- `--batch_size`: Batch size (default: 8)
- `--lr`: Learning rate (default: 1e-5)
- `--chunk_size`: ACT chunk size - actions per prediction (default: 100)
- `--output_dir`: Output directory (default: outputs/train/act_TIMESTAMP)
- `--log_freq`: Log frequency (default: 100)
- `--save_freq`: Checkpoint save frequency (default: 5000)
- `--device`: Device - cuda or cpu (default: cuda)
- `--num_workers`: Dataloader workers (default: 4)

## Output

Training outputs are saved to `outputs/train/act_<timestamp>/`:

```
outputs/train/act_20251229_120000/
├── checkpoint_005000/     # Checkpoint at step 5000
├── checkpoint_010000/     # Checkpoint at step 10000
├── ...
└── final/                 # Final trained model
    ├── config.json
    ├── model.safetensors
    └── ...
```

## ACT Policy

ACT (Action Chunking with Transformers) predicts a sequence of future actions given:
- Current robot state (joint positions)
- Camera images (wrist_cam, overhead_cam)

Key hyperparameters:
- `chunk_size`: Number of future actions to predict (default: 100)
- `batch_size`: Training batch size (reduce if OOM)
- `lr`: Learning rate (default: 1e-5)

## Hardware Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM recommended
- **RAM**: 16GB+ system RAM
- **Storage**: ~1GB per checkpoint

If you get OOM errors, try:
1. Reduce `--batch_size` (e.g., 4 or 2)
2. Reduce `--chunk_size` (e.g., 50)
3. Use `--num_workers 0`

## Next Steps

After training, you can:
1. Evaluate the policy in simulation using `recording/playback_sim_vr.py`
2. Deploy to real robot (coming soon)
