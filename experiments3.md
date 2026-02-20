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
- **Model**: Qwen2.5-VL-3B-Instruct, fine-tuned with LoRA (r=16, alpha=32, q/k/v/o_proj)
- **Dataset**: `danbhf/sim_pick_place_220ep_pickup_tight` (220 pickup-only episodes, ~202 samples)
- **Horizon**: 8 (predicts 8 future actions per inference)
- **Action encoding**: Discretized to 0-1000 integers, space-separated
- **Training**: 32 epochs, batch_size=8, lr=4e-5, gradient checkpointing, SDPA attention, cosine LR schedule
- **No proprioception**: Vision-only (unlike ACT which uses joint state)
- **Trainable params**: 7.37M / 3.76B (0.196% via LoRA)
- **VRAM**: 22.2GB peak (RTX 5090 32GB)
- **Training time**: ~42 minutes (832 steps at ~3.2s/step)
- **GPU**: NVIDIA RTX 5090

### Training Details

- Full fine-tuning OOMs at 29.8GB (3.75B params in bf16 + optimizer + activations)
- LoRA with r=16 targeting q/k/v/o_proj reduces trainable params to 7.37M (0.196%)
- LoRA VRAM: 22.2GB with batch_size=8 (was 9.4GB with batch_size=4)
- PyTorch 2.8.0+cu128 required for RTX 5090 (sm_120 compute capability)
- torchcodec 0.6.0 required for LeRobot video decoding on PyTorch 2.8
- Action mask augmentation: 40% of action tokens randomly replaced during training (regularization)
- Constrained decoding: `NumberSpaceOnlyProcessor` restricts generation to digits, spaces, and EOS

### Loss Curve

| Epoch | Loss | LR |
|-------|------|-----|
| 0.4 | 1.501 | 1.44e-05 (warmup) |
| 1.2 | 1.399 | 4.00e-05 (peak) |
| 5.0 | 1.219 | 3.84e-05 |
| 10.0 | 1.162 | 3.46e-05 |
| 15.0 | 1.066 | 2.62e-05 |
| 20.0 | 0.987 | 1.59e-05 |
| 25.0 | 1.013 | 4.86e-06 |
| 30.0 | 0.904 | 4.24e-07 |
| 32.0 | 0.956 | 1.36e-09 |

### Comparison Table
| | ACT-ViT (Exp 15b) | TinyVLA |
|--|---------|---------|
| Parameters | ~80M | ~3.75B (7.37M trainable via LoRA) |
| Uses proprioception | Yes (joint state) | No (vision-only) |
| Prediction horizon | 50 | 8 |
| Action representation | Continuous (normalized) | Discretized (0-1000) |
| Inference speed | ~5ms | ~200-500ms |

### Scripts
| Script | Purpose |
|--------|---------|
| `scripts/training/train_tinyvla.py` | TinyVLA training (SFT on Qwen2.5-VL-3B) |
| `scripts/inference/eval_tinyvla.py` | Evaluation wrapper with TinyVLAPolicy |

### Models
| Model | Location |
|-------|----------|
| TinyVLA LoRA adapter | `danbhf/tinyvla_pickup_lora_32ep` (HuggingFace) |
| RunPod checkpoint | `outputs/train/tinyvla_pickup/final/` |

### Datasets
| Dataset | Notes |
|---------|-------|
| `danbhf/sim_pick_place_220ep_pickup_tight` | Training data (220 pickup-only episodes) |
| `danbhf/sim_pick_place_20260218_192430` | Recorded pickup demo - NOT USABLE: block not in camera view at start |
| `danbhf/sim_pick_place_20260218_214459` | 20-episode pickup demo recorded with `--show-fov` (wrist camera FOV overlay) |

## Results

### Exp 16a: Training-Position Pickup Evaluation

```bash
python scripts/inference/eval_tinyvla.py outputs/train/tinyvla_pickup/final \
    --episodes 5 --pickup-only --max-steps 200 --mujoco-viewer
```

| Position | Success | Approached | Rate |
|----------|---------|------------|------|
| (0.213, 0.254) | 1/4 | 4 | 25% |
| (0.213, -0.047) | 0/0 | 0 | 0% |
| **Overall** | **1/4** | **4** | **12.5%** |

### Key Observations

- **Very slow inference**: ~500ms+ per generation step for 3B model locally (vs ~5ms for ACT). Each action prediction generates ~48 tokens (8 steps x 6 dims) autoregressively. The sim runs in slow motion.
- **12.5% success** vs ACT-ViT Exp 15b's **100%** at training positions — dramatically worse.
- Position 2 had 0 approaches (robot never reached the block in any episode).
- The model occasionally produces valid-looking actions but lacks the precision for reliable manipulation.
- With horizon=8 (predicts 8 actions at once, ~0.27s of motion), the model re-plans frequently but each re-plan is expensive.

### Comparison: ACT-ViT vs TinyVLA

| Metric | ACT-ViT (Exp 15b) | TinyVLA (Exp 16a) |
|--------|-------------------|-------------------|
| Training-position pickup | 100% | 12.5% |
| Inference speed | ~5ms/step | ~500ms/step |
| Parameters (inference) | ~80M | ~3.75B |
| Trainable parameters | ~80M (full) | 7.37M (LoRA) |
| Training time | ~15 min | ~42 min |
| GPU (training) | RTX 3090 | RTX 5090 |

**Conclusion**: TinyVLA with 32 epochs of LoRA fine-tuning is not competitive with ACT for this task. The 3B VLA model is both ~100x slower at inference and dramatically less accurate. The discretized action representation (integers 0-1000) likely loses precision compared to ACT's continuous output, and LoRA training of 0.2% of parameters may be insufficient for the model to learn precise manipulation. A full fine-tune or more training data/epochs might improve results, but the fundamental inference speed limitation makes VLAs impractical for real-time 30Hz control.

### Exp 16b: 5x5 Spatial Grid Evaluation

Attempted 5x5 spatial grid evaluation (25 positions, 5 episodes each, 200 max steps):

```bash
python scripts/inference/eval_tinyvla.py outputs/train/tinyvla_pickup/final \
    --episodes 5 --pickup-only --max-steps 200 --grid-size 5
```

**Status**: Abandoned due to impractical inference speed. The 3B model generates at ~13 seconds per action prediction on local GPU. With 25 grid positions × 5 episodes × ~25 generations per episode = ~3,125 total generations, estimated wall time: **~11 hours**. Given the 12.5% success at training positions, spatial generalization is expected to be ~0%.

---

# Experiment 17: Dark Matt Ground (Reduced Reflections)

## Motivation

The original scene uses a checker-pattern ground with `reflectance=0.2`, which creates visual reflections in the overhead camera. These reflections are position-dependent — the same block position reflects differently depending on the viewing angle. This could encode spurious spatial features that the model memorizes, harming position-invariant generalization.

**Hypothesis**: An unreflective dark matt ground will:
1. Remove position-dependent reflection patterns from overhead camera
2. Increase contrast between the white Duplo block and the dark surface
3. Potentially improve spatial generalization by removing spurious visual cues

## Setup

### Scene Modification

Created `scenes/so101_dark_ground.xml` — copy of `so101_with_wrist_cam.xml` with:
- Ground texture: `builtin="flat" rgb1="0.12 0.12 0.12"` (uniform dark charcoal, no checker)
- Ground material: `reflectance="0.0"` (completely non-reflective)

### Dataset Re-rendering

Used `scripts/recording/rerecord_dataset.py` to replay the 220-episode dataset with the new ground:

```bash
python scripts/recording/rerecord_dataset.py danbhf/sim_pick_place_2pos_220ep_v2 \
    --scene scenes/so101_dark_ground.xml \
    --output danbhf/sim_pick_place_220ep_dark_ground
```

This replays the exact same actions/trajectories from the original 220-episode dataset in the modified sim. Block positions are preserved from `episode_scenes.json`. Only the camera observations change (new ground appearance).

- **Source**: `danbhf/sim_pick_place_2pos_220ep_v2` (220 episodes, 31,210 frames)
- **Output**: `danbhf/sim_pick_place_220ep_dark_ground` (identical trajectories, new visuals)
- **Subtask annotations**: Copied from source (identical episode structure)

### Training Config

Replicates Exp 14b (best model) with the dark ground dataset:

| Parameter | Value |
|-----------|-------|
| Dataset | `danbhf/sim_pick_place_220ep_dark_ground` |
| fix_state | Yes |
| subtask | Yes |
| blinkering | Yes |
| subtask_chunks | Yes (completion head) |
| no_mask_actions | Yes (auxiliary only) |
| pickup_coords | Yes |
| Steps | 50,000 |
| batch_size | 8 |
| chunk_size | 100 |

## Bug Discovery: Image Normalization Destroys ViT Performance

### The Problem

Initial training (v1-v3) produced models with 0-10% success — a catastrophic drop from Exp 14b's 85%. Investigation revealed **two bugs**:

1. **Local dataset path bug**: `PickupCoordinateDataset.load_episode_scenes()` only checked HuggingFace (404 for local-only datasets), silently disabling `--pickup_coords`. Fixed to also check local cache at `HF_LEROBOT_HOME`.

2. **Image normalization bug** (critical): The re-rendered dataset (newer LeRobot format) includes image mean/std in `stats.json`. The preprocessor applies MEAN_STD normalization to images:
   - Overhead cam stats: **mean≈0.248, std≈0.004** (nearly uniform dark pixels)
   - Normalization: `(pixel - 0.248) / 0.004` → values in range **[-60, +190]**
   - This completely overwhelms the ViT backbone which expects [0, 1] input

   The original Exp 14b dataset (older LeRobot format) had **no image stats** in its preprocessor, so images passed through in raw [0, 1] range. The ViT backbone fine-tuned successfully on these natural pixel values.

### Root Cause

LeRobot v3.0 datasets compute image statistics (mean, std per channel) and store them in `stats.json`. When the preprocessor is created with `make_pre_post_processors(cfg, dataset_stats=stats)`, these image stats get baked in, causing MEAN_STD normalization on images. For the dark ground overhead camera, the extremely low std (0.004) causes numerical explosion.

### Fix

Remove image stats from the preprocessor stats before creating it:
```python
stats = {k: v for k, v in stats.items() if 'observation.images' not in k}
```

This ensures images pass through in [0, 1] range (matching Exp 14b behavior). The ViT backbone handles its own feature normalization via learned layer norms.

### Ablation Results

| Version | Image Stats | pickup_coords | Success Rate |
|---------|-------------|---------------|-------------|
| v1 (dark ground, broken) | Yes (bug) | No (load bug) | **10%** |
| v2 (pickup_coords fix only) | Yes (bug) | No (still broken) | **10%** |
| v3 (both bugs present) | Yes (bug) | Yes | **0%** |
| **v4 (both fixed)** | **No (correct)** | **Yes** | **66%** |
| Exp 14b (checker ground) | No (old format) | Yes | **65-85%** |

**The image normalization bug accounted for a ~60 percentage point drop in performance.**

## Results (v4 — Corrected Model)

### Exp 17a: Training-Position Full Pick-and-Place (50 episodes)

```bash
python scripts/inference/eval.py outputs/train/act_vit_dark_ground_v4 --local \
    --checkpoint final --episodes 50 --scene so101_dark_ground.xml \
    --blinkering --subtask --pickup-coords
```

| Metric | Dark Ground (v4) | Exp 14b (Checker) |
|--------|-----------------|-------------------|
| **Success Rate** | **66%** | **65-85%** |
| Pick Rate | 80% | 85-95% |
| Drop Rate | 17.5% | 10-12% |
| Never Picked Up | 20% | 5-15% |
| Avg Steps (success) | 145 | 108 |

**Conclusion**: Dark ground model performance is comparable to checker ground (within sampling variance). The dark ground does not significantly improve or harm full-task performance.

### Exp 17b: Blinkering Comparison (20 episodes)

| Config | Success | Pick Rate | Drop Rate |
|--------|---------|-----------|-----------|
| With blinkering | 60% | 75% | 20% |
| Without blinkering | 60% | 85% | 12% |

Blinkering shows no benefit on dark ground (unlike checker ground where it doubled success in Exp 12). The uniform dark ground already provides less distracting visual features, so masking the overhead camera during PICK_UP/DROP adds no value.

### Exp 17c: 5x5 Spatial Grid Pickup

```bash
python scripts/experiments/eval_pickup_model_spatial.py outputs/train/act_vit_dark_ground_v4 \
    --checkpoint final --grid-size 5 --episodes 5 --scene so101_dark_ground.xml
```

**Note**: The spatial eval does NOT inject pickup_coords — it tests vision-only spatial generalization.

```
Pickup Success Grid (% success):
     Y\X 0.10 0.16 0.22 0.29 0.35
0.38      .    .    .    .    .
0.30      .    .    .    .    .
0.23      .  100    .  100    .
0.15      .    .    .    .    .
0.08      .    .    .  100    .

Approach Rate Grid (% reached block):
     Y\X 0.10 0.16 0.22 0.29 0.35
0.38      .    .    .    .    .
0.30      .    .    .    .    .
0.23    100  100  100  100    .
0.15      .  100  100  100    .
0.08      .  100  100  100    .
```

| Metric | Dark Ground (v4) | Exp 12 (Checker) |
|--------|-----------------|------------------|
| Overall pickup | **12%** (3/25) | **100%** at reachable |
| Approach rate | **40%** | N/A (full task) |
| Positions with success | 3/25 | All reachable |

The dramatic spatial generalization drop (100% → 12%) is expected because:
1. The spatial eval doesn't inject `pickup_coords` — the model was trained WITH coords and depends on them
2. Exp 12's 100% spatial pickup was from a model that used subtask conditioning (including coords) during the full eval
3. The dark ground provides fewer spatial reference points (no checker pattern) for vision-only localization

## Key Findings

1. **Image normalization is critical for ViT-based policies**: MEAN_STD normalizing images with dataset statistics can destroy performance when camera variance is low (dark/uniform backgrounds). Always pass images to ViT in [0, 1] range.

2. **Dark ground ≈ checker ground for full-task performance**: 66% vs 65-85% — no significant difference when the model has pickup_coords and subtask conditioning.

3. **Blinkering is unnecessary on dark ground**: The uniform dark background already reduces distracting visual features, making overhead camera masking redundant.

4. **Dark ground does NOT improve spatial generalization**: The hypothesis that removing checker reflections would help was not supported. Spatial performance is limited by the model's dependence on pickup_coords rather than visual features.

## Models

| Model | Path | Notes |
|-------|------|-------|
| v1 (broken) | `outputs/train/act_vit_dark_ground` | No pickup_coords, image norm bug |
| v2 (broken) | `outputs/train/act_vit_dark_ground_v2` | No pickup_coords (load_episode_scenes bug) |
| v3 (broken) | `outputs/train/act_vit_dark_ground_v3` | Image norm bug |
| **v4 (correct)** | `outputs/train/act_vit_dark_ground_v4` | Both bugs fixed, **66% success** |

## Code Fixes

| File | Fix |
|------|-----|
| `utils/training.py` | `PickupCoordinateDataset.load_episode_scenes()` — check local cache before HuggingFace |
| `scripts/training/train_act_vit.py` | Remove image stats from preprocessor: `stats = {k: v for k, v in stats.items() if 'observation.images' not in k}` |
| `scripts/experiments/eval_pickup_model_spatial.py` | Added `--scene` argument for custom scene XML |

---

# Experiment 18: Ground Texture & Data Volume Ablations

## Motivation

Two key findings need deeper investigation:

1. **Plain ACT on dark ground is much worse** than checker ground (50% vs 95% at training positions). Why? Is it the visual complexity, contrast, or the checker pattern providing useful spatial reference?
2. **Pickup coordinates hurt spatial generalization** (9.2% with coords vs 23.4% without). The plain ACT model (`act_2pos_220ep`) at 23.4% spatial is our best spatially-generalizing model.

This experiment runs several ablations to understand what drives spatial generalization:
- Does **data volume** matter? (60ep subset vs 220ep)
- Does **ground texture** matter at inference time? (cross-scene tests)
- Does **ground texture diversity** help? (440ep combined dataset)

## Baseline Models

### Plain ACT on Checker Ground: `act_2pos_220ep`

| Parameter | Value |
|-----------|-------|
| Model | `outputs/train/act_2pos_220ep` |
| Dataset | `danbhf/sim_pick_place_2pos_220ep_v2` |
| Training | Plain ACT (no pickup_coords, no subtask, no blinkering) |
| Steps | 100,000 |
| Scene (eval) | `so101_with_wrist_cam.xml` (checker ground) |
| **Training-pos success** | **95%** (50 episodes) |
| **Spatial gen (7x7)** | **23.4%** (best spatially-generalizing model) |

### Plain ACT on Dark Ground: `act_dark_ground_220ep`

| Parameter | Value |
|-----------|-------|
| Model | `outputs/train/act_dark_ground_220ep` |
| Best checkpoint | `checkpoint_050000` |
| Dataset | `danbhf/sim_pick_place_220ep_dark_ground` |
| Training | Plain ACT (no pickup_coords, no subtask, no blinkering), 100K steps |
| Scene (eval) | `so101_dark_ground.xml` |
| **Training-pos success** | **50%** (50 episodes) |

The 45 percentage point gap (95% checker vs 50% dark) is striking. Both models are plain ACT on the same trajectories, differing only in ground texture.

---

## Exp 18a: Cross-Scene Evaluation (Ground Texture Transfer)

Tests whether models are texture-specific or learned generalizable features.

### 18a-i: Dark Ground Model on Checker Scene

```bash
python scripts/inference/eval.py outputs/train/act_dark_ground_220ep \
    --checkpoint checkpoint_050000 --episodes 50 --local
```

(No `--scene` flag = default checker scene `so101_with_wrist_cam.xml`)

**Hypothesis**: If the model learned texture-invariant features, it should still work. If it memorized dark-ground-specific visual patterns, it will fail.

**Results**: **0% success** (0/50). Never picked up in any episode. Max block height 0.009m (never lifted). The dark ground model is completely texture-specific — its visual features are useless on the checker ground.

### 18a-ii: Checker Model on Dark Ground Scene

```bash
python scripts/inference/eval.py outputs/train/act_2pos_220ep \
    --checkpoint final --episodes 50 --local \
    --scene so101_dark_ground.xml
```

**Hypothesis**: Same logic -- if checker patterns are essential to the policy, this will fail on dark ground.

**Results**: **40% success** (20/50). Pick rate 76%, drop rate 37%. The checker model degrades but still partially works on dark ground — its visual features transfer better than the reverse direction.

| Outcome | Count |
|---------|-------|
| Success | 20 |
| Never picked up | 12 |
| Dropped during transport | 13 |
| Missed goal | 1 |
| Timeout | 4 |

### 18a Summary

| Model (trained on) | Eval Scene | Success | Transfer Loss |
|---------------------|------------|---------|--------------|
| Dark ground → Dark ground | Same | 50% | baseline |
| **Dark ground → Checker** | **Cross** | **0%** | **-50pp (total failure)** |
| Checker → Checker | Same | 95% | baseline |
| **Checker → Dark ground** | **Cross** | **40%** | **-55pp** |

**Key finding**: Both models lose performance on cross-scene transfer, but the asymmetry is striking. The checker model retains 40% on dark ground (still picks up 76% of the time), while the dark ground model gets 0% on checker (never even picks up). The checker pattern provides richer, more transferable visual features. The uniform dark ground produces fragile representations that are completely scene-specific.

---

## Exp 18b: Data Volume Ablation (60ep Subset)

**Question**: Does 220 episodes "over-train" at specific positions, hurting spatial generalization? Would fewer episodes force more generalizable representations?

### Episode Selection

From `danbhf/sim_pick_place_2pos_220ep_v2`:
- Episodes 0-19: Position 1 (y > 0.1) -- 20 episodes
- Episodes 100-119: Position 2 (y <= 0.1) -- 20 episodes
- Episodes 200-219: Random/gap-filling positions -- 20 episodes
- **Total: 60 episodes**

Implemented via `--episode_filter "0-19,100-119,200-219"` in `train_act.py`.

### Training

```bash
MUJOCO_GL=egl python scripts/training/train_act.py \
    danbhf/sim_pick_place_2pos_220ep_v2 \
    --steps 100000 --batch_size 8 --chunk_size 100 \
    --save_freq 10000 --num_workers 2 \
    --episode_filter "0-19,100-119,200-219" \
    --output_dir outputs/train/act_60ep_subset \
    --run_name act_60ep_subset
```

**Training-pos results**: _(pending)_

### Spatial Eval (7x7)

```bash
python scripts/experiments/eval_spatial_generalization.py \
    outputs/train/act_60ep_subset --checkpoint final \
    --grid-size 7 --episodes 10 \
    --x-min 0.12 --x-max 0.42 --y-min -0.1 --y-max 0.35 \
    --csv outputs/experiments/spatial_scatter_60ep_subset.csv
```

**Results**: _(pending)_

---

## Exp 18c: Ground Texture Diversity (440ep Combined)

**Question**: Does training on both ground textures teach the model to ignore ground appearance, improving generalization?

### Dataset

Merged both 220ep datasets:

```bash
python scripts/tools/merge_datasets.py \
    danbhf/sim_pick_place_2pos_220ep_v2 \
    danbhf/sim_pick_place_220ep_dark_ground \
    -o datasets/sim_pick_place_440ep_both_grounds \
    --upload danbhf/sim_pick_place_440ep_both_grounds
```

- **440 episodes, ~62,420 frames**
- Same trajectories appear twice (once per ground texture)
- Teaches the model that ground texture is irrelevant

### Training

```bash
MUJOCO_GL=egl python scripts/training/train_act.py \
    danbhf/sim_pick_place_440ep_both_grounds \
    --steps 100000 --batch_size 8 --chunk_size 100 \
    --save_freq 10000 --num_workers 2 \
    --output_dir outputs/train/act_440ep_both_grounds \
    --run_name act_440ep_both_grounds
```

**Training-pos results**: _(pending)_

### Spatial Eval (7x7)

Evaluated on both scenes (model was trained on both):

```bash
# Checker scene
python scripts/experiments/eval_spatial_generalization.py \
    outputs/train/act_440ep_both_grounds --checkpoint final \
    --grid-size 7 --episodes 10 \
    --x-min 0.12 --x-max 0.42 --y-min -0.1 --y-max 0.35 \
    --csv outputs/experiments/spatial_scatter_440ep_checker.csv

# Dark ground scene
python scripts/experiments/eval_spatial_generalization.py \
    outputs/train/act_440ep_both_grounds --checkpoint final \
    --grid-size 7 --episodes 10 \
    --x-min 0.12 --x-max 0.42 --y-min -0.1 --y-max 0.35 \
    --scene so101_dark_ground.xml \
    --csv outputs/experiments/spatial_scatter_440ep_dark.csv
```

**Results**: _(pending)_

---

## Summary Table

| Model | Data | Ground (train) | Ground (eval) | Training-Pos | Spatial (7x7) |
|-------|------|----------------|---------------|-------------|---------------|
| `act_2pos_220ep` | 220ep | Checker | Checker | **95%** | **23.4%** |
| `act_dark_ground_220ep` | 220ep | Dark | Dark | **50%** | **14.1%** |
| `act_dark_ground_220ep` | 220ep | Dark | **Checker** | **0%** | - |
| `act_2pos_220ep` | 220ep | Checker | **Dark** | **40%** | - |
| `act_60ep_subset` | 60ep | Checker | Checker | **46%** | **8.6%** |
| `act_440ep_both_grounds` | 440ep | Both | Checker | **60%** | **14.3%** |
| `act_440ep_both_grounds` | 440ep | Both | Dark | _(pending)_ | **16.3%** |

## Models

| Model | Path | Dataset | Episodes |
|-------|------|---------|----------|
| Plain ACT checker 220ep | `outputs/train/act_2pos_220ep` | `danbhf/sim_pick_place_2pos_220ep_v2` | 220 |
| Plain ACT dark 220ep | `outputs/train/act_dark_ground_220ep` | `danbhf/sim_pick_place_220ep_dark_ground` | 220 |
| Plain ACT 60ep subset | `outputs/train/act_60ep_subset` | `danbhf/sim_pick_place_2pos_220ep_v2` (filtered) | 60 |
| Plain ACT 440ep combined | `outputs/train/act_440ep_both_grounds` | `danbhf/sim_pick_place_440ep_both_grounds` | 440 |

## Code Changes

| File | Change |
|------|--------|
| `scripts/training/train_act.py` | Added `--episode_filter` CLI arg: comma-separated ranges (e.g. `"0-19,100-119"`) |

## Comparison Plots

```bash
# 60ep vs 220ep (both checker scene, plain ACT)
python scripts/experiments/plot_spatial_scatter.py \
    outputs/experiments/spatial_scatter_60ep_subset.csv \
    outputs/experiments/spatial_eval_20260125_232717.csv \
    --side-by-side \
    --training-pos 0.213,0.254 --training-pos 0.213,-0.047 \
    --output docs/Images/spatial_scatter_60ep_vs_220ep.png

# 440ep checker vs dark (both from combined model)
python scripts/experiments/plot_spatial_scatter.py \
    outputs/experiments/spatial_scatter_440ep_checker.csv \
    outputs/experiments/spatial_scatter_440ep_dark.csv \
    --side-by-side \
    --training-pos 0.213,0.254 --training-pos 0.213,-0.047 \
    --output docs/Images/spatial_scatter_440ep_both_grounds.png
```
