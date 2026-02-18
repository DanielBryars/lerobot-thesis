# Experiment 15: Pickup-Only Model Training & Evaluation

## Motivation

Previous experiments (12-14) showed that the PICK_UP subtask is **100% position-invariant** when the robot naturally approaches the block (Exp 13b). However, the full-task model uses subtask conditioning (one-hot + pickup coordinates) and chunk_size=100, meaning it trains on action chunks that span multiple subtask phases.

**Question**: Can we train a simpler, dedicated pickup-only model that:
1. Doesn't need subtask conditioning (it only does one thing: pick up)
2. Uses shorter action chunks matched to the subtask length (~18-40 frames)
3. Could serve as a modular building block in a hierarchical pipeline

## Dataset Creation

### Tool: `scripts/tools/split_subtask_dataset.py`

Splits the full pick-and-place dataset into per-subtask episodes. Each PICK_UP segment from the original 220-episode dataset becomes its own episode in the new dataset.

Source dataset: `danbhf/sim_pick_place_2pos_220ep_v2` (220 episodes, 2 positions + gap-filling)

### Dataset A: Padded (20 frames before/after)

```bash
python scripts/tools/split_subtask_dataset.py danbhf/sim_pick_place_2pos_220ep_v2 \
    -o datasets/sim_pick_place_220ep_pickup_padded --subtask PICK_UP \
    --pad-before 20 --pad-after 20
```

- **Total frames**: 9,766
- **Episodes**: ~220 PICK_UP segments
- **Segment length**: ~18 steps (PICK_UP) + 20 before + 20 after = ~58 frames avg
- **Padding rationale**: 20 frames before captures the end of MOVE_TO_SOURCE approach; 20 after captures start of MOVE_TO_DEST. This gives the model context for approaching and releasing.

### Dataset B: Full Approach (100 frames before, 20 after)

```bash
python scripts/tools/split_subtask_dataset.py danbhf/sim_pick_place_2pos_220ep_v2 \
    -o datasets/sim_pick_place_220ep_pickup_full_approach --subtask PICK_UP \
    --pad-before 100 --pad-after 20
```

- **Total frames**: 13,213
- **Episodes**: ~220 PICK_UP segments
- **Segment length**: ~18 steps (PICK_UP) + up to 100 before + 20 after = ~60-120 frames
- **Padding rationale**: 100 frames before captures the full MOVE_TO_SOURCE approach, so the model learns to approach AND pick up. This avoids the cold-start problem at the expense of a larger task.

---

## Exp 15a: Pickup-Only, 20-frame Padding (chunk=80)

### Training Config

| Parameter | Value |
|-----------|-------|
| Model | `outputs/train/act_vit_20260217_111337` |
| Dataset | `danbhf/sim_pick_place_220ep_pickup_padded` |
| Total frames | 9,766 |
| chunk_size | 80 |
| Vision backbone | ViT-B/16 |
| Cameras | wrist_cam, overhead_cam |
| fix_state | Yes |
| Subtask conditioning | No |
| Pickup coords | No |
| Blinkering | No |

### Evaluation from Home Position

**Training positions (20 episodes each)**:

| Position | Success | Approached | Avg Height |
|----------|---------|------------|------------|
| (0.213, 0.254) | 0% (0/20) | 0/20 | 0.010m |
| (0.213, -0.047) | 0% (0/20) | 20/20 | 0.010m |

Position 2 gets 100% approach rate but 0% success -- the model moves toward the block but can't pick it up from home. Position 1 doesn't even approach.

**5x5 spatial grid (5 episodes each)**:

```
     Y\X 0.10 0.16 0.22 0.29 0.35
0.38      .    .    .    .    .
0.30      .    .    .    .    .
0.23      .    .    .    .    .
0.15      .    .    .  100    .
0.08      .    .    .    .    .
```

Overall: **4%** (1/25 positions at 100%: (0.2875, 0.1550)). This single success is interesting -- it's near where some training episodes' approach phases end.

### Evaluation Pre-Positioned (IK teleport above block)

IK solver places arm at z=0.08m above block position, gripper open. Positions with IK error >30mm are skipped.

**Training positions (20 episodes each)**:

| Position | Success | Approached | IK Error | Notes |
|----------|---------|------------|----------|-------|
| (0.213, 0.254) | 0% (0/20) | 20/20 | 28.6mm | Arm positioned, approached block, but couldn't grasp |
| (0.213, -0.047) | 0% (0/20) | 0/20 | 35.3mm | IK error >30mm, skipped |

Position 1 is telling: with pre-positioning the arm gets close (20/20 approach vs 0/20 from home), but still can't pick up. The IK joint configuration is out-of-distribution compared to training data.

**5x5 spatial grid (5 episodes each)**:

```
     Y\X 0.10 0.16 0.22 0.29 0.35
0.38      .    .    .    .    .
0.30      .    .    .    .    .
0.23      .    .    .  100    .
0.15      .    .    .    .    .
0.08      .    .    .    .    .
```

Overall: **4%** (1/25 at 100%: (0.287, 0.230)). Same position that succeeded from home. Pre-positioning dramatically improved approach rates (60% vs ~16% from home) but didn't unlock new successful positions.

```
Approach Rate Grid (pre-positioned):
     Y\X 0.10 0.16 0.22 0.29 0.35
0.38      .  100  100  100    .
0.30      .    .  100  100  100
0.23      .    .  100  100  100
0.15      .    .  100  100  100
0.08      .    .  100  100  100
```

15/25 positions have 100% approach rate when pre-positioned, but only 1 actually picks up.

---

## Exp 15b: Pickup-Only, Full Approach (chunk=100)

### Training Config

| Parameter | Value |
|-----------|-------|
| Model | `outputs/train/act_vit_20260217_130449` |
| Dataset | `danbhf/sim_pick_place_220ep_pickup_full_approach` |
| Total frames | 13,213 |
| chunk_size | 100 |
| Vision backbone | ViT-B/16 |
| Cameras | wrist_cam, overhead_cam |
| fix_state | Yes |
| Subtask conditioning | No |
| Pickup coords | No |
| Blinkering | No |

### Evaluation from Home Position

**Training positions (20 episodes each)**:

| Position | Success | Approached | Avg Height |
|----------|---------|------------|------------|
| (0.213, 0.254) | 0% (0/20) | 0/20 | 0.010m |
| (0.213, -0.047) | 0% (0/20) | 0/20 | 0.010m |

0% everywhere, 0 approaches. The model doesn't even move toward the block.

**5x5 spatial grid (5 episodes each)**:

```
     Y\X 0.10 0.16 0.22 0.29 0.35
0.38      .    .    .    .    .
0.30      .    .    .    .    .
0.23      .    .    .    .    .
0.15      .    .    .    .    .
0.08      .    .    .    .    .
```

Overall: **0%** everywhere. Some positions at X>=0.225 show approach rates (the model starts moving) but heights remain at 0.010m -- no grasping behavior. The full-approach model seems worse than the padded model.

### Evaluation Pre-Positioned (IK teleport above block)

**Training positions (20 episodes each)**:

| Position | Success | Approached | IK Error | Notes |
|----------|---------|------------|----------|-------|
| (0.213, 0.254) | 0% (0/20) | 20/20 | 28.6mm | Same as Model A: arm near block but no grasping |
| (0.213, -0.047) | 0% (0/20) | 0/20 | 35.3mm | IK error >30mm, skipped |

**5x5 spatial grid (5 episodes each)**:

```
     Y\X 0.10 0.16 0.22 0.29 0.35
0.38      .    .    .    .    .
0.30      .    .    .    .    .
0.23      .    .    .    .    .
0.15      .    .    .    .    .
0.08      .    .    .    .  100
```

Overall: **4%** (1/25 at 100%: (0.350, 0.080)). Different success position than Model A. This position had the lowest IK error (7.2mm) -- essentially the only position where IK gave a good solution.

```
Approach Rate Grid (pre-positioned):
     Y\X 0.10 0.16 0.22 0.29 0.35
0.38      .  100  100  100    .
0.30      .    .  100  100  100
0.23      .    .  100  100  100
0.15      .    .  100  100  100
0.08      .    .  100  100  100
```

Same approach pattern as Model A (IK solutions are model-independent), but success at a different position.

---

## Analysis & Key Findings

### Why 0% from Home?

Both models were trained on data that starts **near the block** (padded PICK_UP segments), not from the home position. The training data distribution is:

- **Model A (20-frame pad)**: Data starts ~20 frames before PICK_UP transition, when the arm is already approaching the block. The home pose is completely out-of-distribution.
- **Model B (100-frame pad)**: Data starts ~100 frames before PICK_UP, which covers the full MOVE_TO_SOURCE. But the action chunks are 100 steps, and the model predicts one chunk then runs open-loop. Without subtask conditioning to signal "approach first", the model produces a blend of approach + pickup actions that doesn't work.

### Pre-Positioning: IK Joint Configs are Out-of-Distribution

Pre-positioning dramatically improved **approach rates** (0-16% from home -> 60% pre-positioned) but did NOT improve **pickup success** (4% for both models in both conditions). The IK solver places the arm near the block, but:

1. **IK joint configurations differ from natural approaches** (confirmed in Exp 13a with 59-degree mismatches). The 5-DOF arm reaches the same EE position via very different joint angles.
2. **With fix_state, the model sees these OOD joint angles** as `observation.state` and produces poor actions.
3. **High IK errors**: Most positions have 17-50mm IK error, meaning the arm isn't even at the intended position. Only (0.350, 0.080) achieves <10mm error.

### Model A vs Model B

| | Model A (padded, chunk=80) | Model B (full approach, chunk=100) |
|---|---|---|
| From home: training pos | 0% (1 approach) | 0% (0 approaches) |
| From home: grid | 4% (1/25 at (0.287, 0.155)) | 0% |
| Pre-positioned: training pos | 0% (20/20 approach) | 0% (20/20 approach) |
| Pre-positioned: grid | 4% (1/25 at (0.287, 0.230)) | 4% (1/25 at (0.350, 0.080)) |

Neither model is significantly better. Both learned fragile grasping that only works at specific positions where the starting state happens to match training data. Model B's only success is at the position with lowest IK error (7.2mm).

### The Fundamental Problem

Without subtask conditioning, the pickup-only model doesn't know whether it should be:
1. Approaching the block (MOVE_TO_SOURCE behavior)
2. Grasping the block (PICK_UP behavior)

The padded data includes both phases, and the model averages them into an incoherent trajectory. Pre-positioning bypasses the approach problem but exposes the IK distribution mismatch (same issue as Exp 13a).

### Why Subtask-Conditioned Models Work Better

The Exp 12/13 subtask-conditioned model achieves **100% pickup at every reachable position** because:
1. **Subtask one-hot** tells the model which behavior to execute
2. **Natural approach** (MOVE_TO_SOURCE) produces in-distribution starting states for PICK_UP
3. **Blinkering** forces position-invariant wrist camera features during PICK_UP
4. **Selective coordinates** mask position during PICK_UP, preventing spatial overfitting

The pickup-only model lacks all four of these mechanisms.

---

## Comparison with Previous Experiments

| Experiment | Model Type | Pickup (Training Pos) | Spatial Pickup | Approach Rate |
|------------|-----------|----------------------|----------------|---------------|
| Exp 12 (full task) | Subtask-conditioned | 80-85% | 100% at reachable | N/A (full task) |
| Exp 14b (completion head) | Subtask + auxiliary | 95% | 4/25 | N/A |
| **15a from home** | No conditioning | **0%** | **4% (1/25)** | 16% |
| **15a pre-positioned** | No conditioning | **0%** | **4% (1/25)** | 60% |
| **15b from home** | No conditioning | **0%** | **0%** | ~16% |
| **15b pre-positioned** | No conditioning | **0%** | **4% (1/25)** | 60% |

**Conclusion**: Subtask conditioning is essential. Without it, dedicated pickup-only models fail even when pre-positioned near the block. The IK-based pre-positioning creates out-of-distribution starting states that the model can't handle. The full-task subtask-conditioned approach (natural approach -> PICK_UP) remains superior.

---

## Data Files & Scripts

### Models
| Model | Path | Dataset |
|-------|------|---------|
| 15a (padded, chunk=80) | `outputs/train/act_vit_20260217_111337` | `danbhf/sim_pick_place_220ep_pickup_padded` |
| 15b (full approach, chunk=100) | `outputs/train/act_vit_20260217_130449` | `danbhf/sim_pick_place_220ep_pickup_full_approach` |

### CSV Results
| File | Description |
|------|-------------|
| `outputs/experiments/pickup_model_training_20260217_124716.csv` | 15a: Training positions, from home |
| `outputs/experiments/pickup_model_grid_20260217_124744.csv` | 15a: 5x5 grid, from home |
| `outputs/experiments/pickup_model_training_prepos_20260217_193335.csv` | 15a: Training positions, pre-positioned |
| `outputs/experiments/pickup_model_grid_prepos_20260217_193353.csv` | 15a: 5x5 grid, pre-positioned |
| `outputs/experiments/pickup_model_training_20260217_143404.csv` | 15b: Training positions, from home |
| `outputs/experiments/pickup_model_grid_20260217_143442.csv` | 15b: 5x5 grid, from home |
| `outputs/experiments/pickup_model_training_prepos_20260217_193440.csv` | 15b: Training positions, pre-positioned |
| `outputs/experiments/pickup_model_grid_prepos_20260217_193458.csv` | 15b: 5x5 grid, pre-positioned |

### Scripts
| Script | Purpose |
|--------|---------|
| `scripts/tools/split_subtask_dataset.py` | Dataset splitting tool |
| `scripts/experiments/eval_pickup_model_spatial.py` | Evaluation (with `--pre-position` for IK teleport) |

---

# Experiment 16: TinyVLA Baseline

## Motivation

Baseline TinyVLA (Qwen2.5-VL-3B VLA model) against ACT-ViT for pickup task.
VLA treats robot control as text generation: camera images + instruction -> discretized actions.

## Setup
- **Model**: Qwen2.5-VL-3B-Instruct, fine-tuned via SFT
- **Dataset**: `danbhf/sim_pick_place_220ep_pickup_tight` (same as Exp 15b)
- **Horizon**: 8 (predicts 8 future actions per inference)
- **Action encoding**: Discretized to 0-1000 integers, space-separated
- **Training**: 32 epochs, batch_size=4, lr=4e-5, gradient checkpointing, SDPA attention
- **No proprioception**: Vision-only (unlike ACT which uses joint state)

### Comparison Table
| | ACT-ViT (Exp 15b) | TinyVLA |
|--|---------|---------|
| Parameters | ~80M | ~3.75B |
| Uses proprioception | Yes (joint state) | No (vision-only) |
| Prediction horizon | 50 | 8 |
| Action representation | Continuous (normalized) | Discretized (0-1000) |
| Inference speed | ~5ms | ~200-500ms |

### Scripts
| Script | Purpose |
|--------|---------|
| `scripts/training/train_tinyvla.py` | TinyVLA training (SFT on Qwen2.5-VL-3B) |
| `scripts/inference/eval_tinyvla.py` | Evaluation wrapper with TinyVLAPolicy |

### Datasets
| Dataset | Notes |
|---------|-------|
| `danbhf/sim_pick_place_220ep_pickup_tight` | Training data (same as Exp 15b) |
| `danbhf/sim_pick_place_20260218_192430` | Recorded pickup demo - NOT USABLE: block not in camera view at start |

## Results

*Training in progress on RunPod RTX 5090...*
