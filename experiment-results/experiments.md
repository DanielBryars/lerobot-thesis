# Experiments Log

## 2026-01-21: ACT Spatial Generalization Evaluation

**Model**: `outputs/train/act_20260118_155135/checkpoint_045000`
**Task**: Pick up the block and place it in the bowl
**Training position**: (0.217, 0.225)

### Experiment Setup

Evaluated ACT policy at a 5x5 grid of block positions across the workspace:
- X range: 0.15 to 0.35 (forward from robot)
- Y range: -0.15 to 0.30 (side to side)
- 20 episodes per position = 500 total episodes
- Position sorted by distance from training position (center-out evaluation)

### Results Summary

**Overall Success Rate: 12.6%**

| Distance from Training | Success Rate | Notes |
|------------------------|--------------|-------|
| d < 0.05m | 60-70% | Near training distribution |
| d = 0.07-0.10m | 10-35% | Some generalization |
| d > 0.11m | 0-5% | Near complete failure |

### Key Findings

1. **ACT has very limited spatial generalization**
   - Performance degrades sharply beyond ~5cm from training position
   - Near-zero success beyond ~10cm displacement

2. **Directional asymmetry**
   - Same distance, different directions yield different results
   - e.g., d=0.077: 15% at (0.15, 0.19) vs 35% at (0.20, 0.30)

3. **False positives near goal**
   - Positions near y=-0.15 (near bowl) showed "1 step" successes
   - Block randomly landed in/near bowl due to position noise, not policy skill

### Raw Data by Position (sorted by distance)

| Distance | Position | Success Rate |
|----------|----------|--------------|
| 0.041 | (0.20, 0.19) | 60% |
| 0.050 | (0.25, 0.19) | 70% |
| 0.077 | (0.15, 0.19) | 15% |
| 0.077 | (0.20, 0.30) | 35% |
| 0.082 | (0.25, 0.30) | 35% |
| 0.091 | (0.30, 0.19) | 10% |
| 0.101 | (0.15, 0.30) | 20% |
| 0.112 | (0.30, 0.30) | 0% |
| 0.138 | (0.35, 0.19) | 0% |
| 0.151+ | (various) | 0-10% |

### Files

- CSV: `outputs/experiments/spatial_eval_20260121_154657.csv`
- JSON: `outputs/experiments/spatial_eval_20260121_155655.json`

---

## 2026-01-21: ACT Fine-Grained Spatial Evaluation

**Model**: `outputs/train/act_20260118_155135/checkpoint_045000`
**Grid**: 7x7 centered on training position (0.217, 0.225)
- X range: 0.17 to 0.27 (±5cm)
- Y range: 0.15 to 0.30 (±7.5cm)
- 10 episodes per position = 490 total episodes

### Results Summary

**Overall Success Rate: 51.6%**

| Distance | Success Rate |
|----------|--------------|
| d < 0.02m | 90-100% |
| d = 0.02-0.04m | 70-100% |
| d = 0.04-0.06m | 60-90% |
| d = 0.06-0.08m | 10-50% (varies by direction) |
| d > 0.08m | 0-30% |

### Key Finding: Directional Asymmetry

At the same distance (~7.5cm), success varies dramatically by direction:

| Direction | Position | Distance | Success |
|-----------|----------|----------|---------|
| +Y (away from bowl) | (0.20, 0.30) | 0.076 | 50% |
| -Y (toward bowl) | (0.20, 0.15) | 0.076 | 0% |
| +Y | (0.22, 0.30) | 0.075 | 20% |
| -Y | (0.22, 0.15) | 0.075 | 0% |

**Interpretation**: The robot learned to move blocks in the -Y direction (toward bowl at y=-0.225).
- Blocks at y > training: Robot just moves further in learned direction → some success
- Blocks at y < training: Robot overshoots or has wrong geometry → failure

### Files

- CSV: `outputs/experiments/spatial_eval_fine_grid.csv`
- JSON: `outputs/experiments/spatial_eval_20260121_160613.json`

### Heatmap Visualizations

- Wide grid (5x5): `outputs/experiments/spatial_heatmap_wide.png`
- Fine grid (7x7): `outputs/experiments/spatial_heatmap_fine.png`
- Success vs Distance: `outputs/experiments/spatial_success_vs_distance.png`

### Combined Analysis (990 episodes - initial)

| Distance (m) | Success Rate | N Episodes |
|--------------|--------------|------------|
| 0.01 | 97% | 30 |
| 0.03 | 79% | 100 |
| 0.05 | 64% | 200 |
| 0.07 | 25% | 160 |
| 0.09 | 18% | 120 |
| 0.11+ | 0-10% | 200+ |

**Key Finding: 50% success threshold crossed at approximately 7cm from training position.**

---

## 2026-01-21: Extended Spatial Evaluation (2630 episodes)

**Model**: `outputs/train/act_20260118_155135/checkpoint_045000`

Expanded spatial evaluation to fill coverage gaps and extend testing range to 30cm from training position.

### Additional Evaluation Runs

| Region | Grid | Episodes | Success Rate |
|--------|------|----------|--------------|
| Right side (x: 0.27-0.35) | 5×5 | 250 | 12% |
| Lower area (y: 0.05-0.15) | 5×5 | 250 | 10% |
| Left side (x: 0.12-0.18) | 5×5 | 250 | 8% |
| Intermediate lower (y: 0.10-0.16) | 4×4 | 160 | 15% |
| y=-0.05 to 0.05 band | 5×5 | 250 | 0% |
| Far left (x: 0.05-0.15) | 4×4 | 160 | 5% |
| Far right (x: 0.35-0.42) | 4×4 | 160 | 0% |
| Far right-bottom corner | 4×4 | 160 | 0.6% |

### Final Results Summary

**Total Episodes: 2630**
**Overall Success Rate: 17.2%** (453/2630)

| Distance (cm) | Episodes | Interpretation |
|---------------|----------|----------------|
| 0-5 | ~400 | Core training region |
| 5-10 | ~750 | Good generalization zone |
| 10-15 | ~450 | Degraded performance |
| 15-20 | ~350 | Near-complete failure |
| 20-30 | ~500 | Complete failure |

### Coverage Distribution

```
Distance  Episodes  Coverage
0-5cm     ~400      ████████████████
5-10cm    ~750      ██████████████████████████████
10-15cm   ~450      ██████████████████
15-20cm   ~350      ██████████████
20-30cm   ~500      ████████████████████
```

### Visualization

MuJoCo scatter visualization with:
- Green/red flat circles for success/failure at each test position
- Blue distance rings at 5cm, 10cm, 15cm, 20cm from training position
- Overlapping translucent circles show test density

**Files**:
- Combined CSV: `outputs/experiments/spatial_eval_combined.csv`
- Scatter plot: `outputs/experiments/spatial_scatter_1900.png`

### Implications for Thesis

This demonstrates a critical limitation of behavior cloning approaches like ACT:
- Training data was collected at a single position with small noise
- Policy memorizes the exact visual-motor mapping for that position
- No ability to interpolate or extrapolate to novel positions
- Suggests need for either: (a) diverse training data, or (b) policy architecture with better spatial reasoning

---

## 2026-01-21: Pi0 Flow Matching Visualization

**Model**: `danbhf/pi0_so101_lerobot`
**Task**: Pick up the block and place it in the bowl
**Result**: SUCCESS at step 155

### Flow Matching Denoising Analysis

Visualized the 10-step flow matching denoising process from noise (t=1.0) to final actions (t=0.0).

**Screenshot**: `/d/git/leeds-msc-ai-fp/LeedsMsc/CameraShapshots/Pi0 Whisker Convergence.png`

#### Convergence Data (normalized action space)

| Step | t    | mean  | std  | min   | max  |
|------|------|-------|------|-------|------|
| 0    | 1.00 | -0.09 | 1.00 | -3.13 | 3.18 |
| 5    | 0.50 | -0.07 | 0.59 | -1.78 | 1.47 |
| 10   | 0.00 | -0.06 | 0.58 | -1.10 | 0.75 |

#### Key Observations

1. **Flow matching converges gradually in normalized space**
   - std decreases: 1.0 (noise) → 0.5-0.7 (final)
   - Range tightens: [-3, 3] → [-1, 1.5]

2. **Visual chaos explained**
   - Small differences in normalized space → large angle differences after denormalization
   - FK amplifies these into scattered 3D positions
   - The "crystallization" IS happening, just hard to see visually

3. **Model actively steers toward targets**
   - In chunk 4 (step 150), mean shifts from 0.05 → 0.49 during denoising
   - The velocity field is pushing toward the correct action, not just reducing variance

#### How Flow Matching Works

```
Start: x₁ ~ N(0, I)           # Pure noise at t=1.0
For step in range(10):
    v_t = model(x_t, t)       # Predict velocity
    x_{t-dt} = x_t + dt * v_t # Move along velocity field
End: x₀ = actions             # Final actions at t=0.0
```

Each step moves the sample along the learned velocity field. Unlike diffusion which gradually removes noise, flow matching learns direct paths from noise to data.

---

## 2026-01-21: Pi0 Whisker Visualization (First Success)

**Model**: `danbhf/pi0_so101_lerobot`
**Task**: Pick up the block and place it in the bowl
**Result**: SUCCESS at step 145

First successful Pi0 run with FK-based whisker visualization. Confirmed Pi0 works with:
- chunk_size=50, n_action_steps=50 (open-loop within each chunk)
- 224x224 images from overhead_cam and wrist_cam
- ~315ms per chunk prediction (10 internal denoising steps)
- ~5-7ms per step execution

---

## 2026-01-20: ACT Chunking Rollout Length Experiment

**Model**: `outputs/train/act_20260118_155135/checkpoint_045000`
**Results**: See `act_chunking_rollout_20260120_113808.json`

Tested different `n_action_steps` values (rollout length before re-prediction).

### ACT VAE Stochasticity Finding

During variance visualization experiments, discovered:
- ACT is **deterministic** in eval mode (uses zero latent vector)
- To get stochastic sampling, must patch `torch.zeros` to return `torch.randn` for latent-sized tensors
- Even with stochastic sampling, variance is tiny (~0.1mm in FK space)
- The VAE latent has minimal influence on outputs - ACT is very confident

---

## 2026-01-21: ACT Temporal Ensembling Analysis

**Model**: `outputs/train/act_20260118_155135/checkpoint_045000`

Investigated temporal ensembling as described in the ACT paper (Algorithm 2).

### Current Model Configuration

```
chunk_size: 100
n_action_steps: 100
temporal_ensemble_coeff: null (DISABLED)
```

The model runs **open-loop**: predict 100 actions, execute all 100, then re-predict. No ensembling.

### What is Temporal Ensembling?

With ensembling enabled, the policy predicts every step, creating overlapping chunks:

```
Step 0:  Predict [a₀, a₁, ..., a₉₉]  →  Execute a₀
Step 1:  Predict [b₀, b₁, ..., b₉₉]  →  Execute weighted_avg(a₁, b₀)
Step 2:  Predict [c₀, c₁, ..., c₉₉]  →  Execute weighted_avg(a₂, b₁, c₀)
```

Weights: `wᵢ = exp(-coeff * i)` where older predictions get more weight (coeff=0.01 default).

### Visualization Experiment

Simulated temporal ensembling by predicting every step and computing weighted averages:

| Metric | Observation |
|--------|-------------|
| Ensemble size | Grows from 1 → chunk_size over time |
| Smoothing effect | Ensembled trajectory significantly smoother than raw |
| Raw vs Ensembled diff | Grows as more chunks contribute |

### Stochastic VAE vs Temporal Ensembling

| Aspect | Stochastic VAE | Temporal Ensembling |
|--------|---------------|---------------------|
| What it does | Sample different latents → different predictions | Average predictions from consecutive steps |
| Purpose | Explore action uncertainty | Smooth out prediction jitter |
| Requires | Multiple forward passes at same timestep | Re-predicting every step |
| Finding | ~0.1mm variance (VAE has minimal effect) | Significantly smooths raw predictions |

### To Enable Temporal Ensembling

```python
temporal_ensemble_coeff = 0.01  # Original ACT paper value
n_action_steps = 1  # Must predict every step
```

### Files

- Visualization: `outputs/experiments/temporal_ensemble_viz.png`
- Script: `scripts/tools/visualize_temporal_ensemble.py`

### Live Visualization

Created a MuJoCo live visualization showing temporal ensembling in action:

```bash
python scripts/tools/visualize_temporal_ensemble_live.py outputs/train/act_20260118_155135 --checkpoint checkpoint_045000
```

**What you see:**
- **Grey whiskers** = Individual chunk predictions (overlapping futures from consecutive predictions)
- **Green whisker** = Ensembled trajectory (weighted average of all grey predictions)

**Key observation:** As the episode progresses, more chunks contribute to the ensemble:
- Step 10: ~10 chunks contributing
- Step 50: ~50 chunks contributing
- Step 122 (success): 100 chunks contributing

The visualization makes it intuitive why ensembling works - multiple "opinions" about the future, predicted from slightly different robot states, converge into a stable consensus trajectory.

**Script:** `scripts/tools/visualize_temporal_ensemble_live.py`

### Multi-Angle Recording with Whiskers

Extended the live visualization to record from 5 camera angles simultaneously, with whisker visualizations included in all angles.

**Camera Presets:**
| # | Name | View Description |
|---|------|------------------|
| 1 | Overview | Standard 3/4 view (azimuth=135, elevation=-25) |
| 2 | Side View | Profile view (azimuth=90, elevation=-15) |
| 3 | Front View | Head-on view (azimuth=180, elevation=-20) |
| 4 | Top Down | Bird's eye view (azimuth=90, elevation=-89) |
| 5 | Close Up | Detailed gripper view (azimuth=120, elevation=-10) |
| 6 | Tracking | **Follows gripper** with smooth EMA tracking (distance=0.35) |

### Tracking Camera (Camera 6)

Added a tracking camera that follows the gripper movement with smooth interpolation:

**Implementation:**
- Uses exponential moving average (EMA) for smooth tracking
- Smoothing factor: 0.15 (lower = smoother, higher = more responsive)
- Camera maintains fixed azimuth/elevation/distance while lookat tracks gripper
- Gripper position obtained from `site_xpos[ee_site_id]` each frame

```python
class SmoothTracker:
    def __init__(self, smoothing: float = 0.15):
        self.smoothing = smoothing
        self.current_lookat = None

    def update(self, target_pos: np.ndarray) -> np.ndarray:
        if self.current_lookat is None:
            self.current_lookat = target_pos.copy()
        else:
            # Exponential moving average for smooth tracking
            self.current_lookat = (1 - self.smoothing) * self.current_lookat + self.smoothing * target_pos
        return self.current_lookat.copy()
```

**Recording Output (with tracking):**
- 121 frames × 6 angles = 726 total frames
- Resolution: 1280×720 per angle
- Location: `outputs/recordings/temporal_ensemble_20260122_002233/`

**Brady Bunch Tiled Videos:**

*5-angle version (3×2 with black cell):*
- File: `outputs/recordings/temporal_ensemble_tiled.mp4`
- Resolution: 1920×720

*6-angle version (3×2 full grid):*
- File: `outputs/recordings/temporal_ensemble_6angles.mp4`
- Resolution: 1920×720
- Layout:
  ```
  ┌──────────┬──────────┬──────────┐
  │ Overview │Side View │Front View│
  ├──────────┼──────────┼──────────┤
  │ Top Down │ Close Up │ Tracking │
  └──────────┴──────────┴──────────┘
  ```

```bash
# Command to create 6-angle tiled video
ffmpeg -framerate 30 \
  -i overview/frame_%05d.png -i side_view/frame_%05d.png -i front_view/frame_%05d.png \
  -i top_down/frame_%05d.png -i close_up/frame_%05d.png -i tracking/frame_%05d.png \
  -filter_complex "[0:v]scale=640:360[v0];[1:v]scale=640:360[v1];[2:v]scale=640:360[v2];
    [3:v]scale=640:360[v3];[4:v]scale=640:360[v4];[5:v]scale=640:360[v5];
    [v0][v1][v2]hstack=inputs=3[top];[v3][v4][v5]hstack=inputs=3[bottom];
    [top][bottom]vstack=inputs=2[out]" \
  -map "[out]" -c:v libx264 temporal_ensemble_6angles.mp4
```

### Interactive Frame-by-Frame Viewer

Created an interactive viewer for stepping through temporal ensembling frame-by-frame:

```bash
python scripts/tools/visualize_temporal_ensemble_interactive.py outputs/train/act_20260118_155135 --checkpoint checkpoint_045000
```

**Controls:**
| Key | Action |
|-----|--------|
| SPACE/RIGHT | Step forward one frame |
| LEFT | Step backward (rewind) |
| P | Play/Pause continuous playback |
| 1-5 | Switch camera viewpoints |
| R | Toggle recording |
| ESC/Q | Quit |

**Features:**
- Frame-by-frame stepping with full state caching
- Rewind capability (restores simulation state from cache)
- Same whisker visualization as live version
- Optional recording of stepped-through frames

**Script:** `scripts/tools/visualize_temporal_ensemble_interactive.py`

---

## 2026-01-22: Temporal Ensembling Evaluation

**Model**: `outputs/train/act_20260118_155135/checkpoint_045000`
**Task**: Pick up the block and place it in the bowl
**Episodes**: 50 per condition

### Experiment: Does Temporal Ensembling Improve Success Rate?

Added `--ensemble` option to eval.py to enable temporal ensembling during inference.

**Command:**
```bash
# Without ensemble (baseline)
python scripts/inference/eval.py outputs/train/act_20260118_155135 --local --checkpoint checkpoint_045000 --episodes 50

# With ensemble
python scripts/inference/eval.py outputs/train/act_20260118_155135 --local --checkpoint checkpoint_045000 --episodes 50 --ensemble 0.01
```

### Results

| Condition | Success Rate | Dropped | Never Picked | Avg Steps |
|-----------|--------------|---------|--------------|-----------|
| **No Ensemble** | 82% | 7 | 1 | 165 |
| **Ensemble (0.01)** | **90%** | **1** | 4 | 154 |

### Key Finding: +8% Success Rate with Ensembling

**Why it works:**
- Temporal ensembling dramatically reduces **drops during transport** (7 → 1)
- The smoothing effect prevents jerky movements that could dislodge the block
- Ensemble averages multiple "opinions" about the trajectory, reducing noise

**Trade-off:**
- Slightly more "never picked up" failures (1 → 4), possibly due to slower initial approach
- Longer inference time (0.76s → 3.54s avg) since we predict every step instead of every 100 steps

### Implementation

Temporal ensembling predicts a full chunk every step and averages overlapping predictions:

```python
# weights decay exponentially with age
weights = exp(-coeff * age)  # coeff=0.01

# Ensembled action = weighted average of all chunk predictions for current step
action = sum(predictions * weights) / sum(weights)
```

**Files modified:**
- `utils/training.py` - Added `temporal_ensemble_coeff` parameter to `run_evaluation()`
- `scripts/inference/eval.py` - Added `--ensemble` CLI option

---

## 2026-01-22: Data Scaling Experiment (In Progress)

**Objective**: Evaluate the effect of training data quantity on ACT policy success rate.

### Experiment Design

Train ACT models with varying amounts of training data and evaluate ALL checkpoints for each, creating a results matrix.

**Episode Counts**: 1, 2, 5, 10, 20, 40, 60, 80, 100, 120, 140, 157
**Dataset**: `danbhf/sim_pick_place_157ep` (22,534 total frames)
**Training Steps**: 45,000 per model (same as original)
**Checkpoint Frequency**: Every 5,000 steps (9 checkpoints + final = 10 per model)
**Evaluation Episodes**: 20 per checkpoint

### Pre-computed Episode Boundaries

```python
EPISODE_FRAME_COUNTS = {
    1: 163, 2: 290, 5: 648, 10: 1385, 20: 2776,
    40: 6559, 60: 9625, 80: 12421, 100: 15116,
    120: 17727, 140: 20336, 157: 22534
}
```

### Script

```bash
# Full experiment
python scripts/experiments/data_scaling_experiment.py

# Test mode (quick validation)
python scripts/experiments/data_scaling_experiment.py --test

# Resume from specific episode count
python scripts/experiments/data_scaling_experiment.py --resume-from 40

# Evaluation only (on existing checkpoints)
python scripts/experiments/data_scaling_experiment.py --eval-only
```

### Expected Outputs

- Results JSON: `outputs/experiments/data_scaling/results.json`
- Summary CSV: `outputs/experiments/data_scaling/summary.csv`
- Model checkpoints: `outputs/experiments/data_scaling/ep_XXX/checkpoint_YYYYYY/`

### Results Matrix Format

```
episodes  checkpoint_005000  checkpoint_010000  ...  final
1         0.XX               0.XX              ...  0.XX
2         0.XX               0.XX              ...  0.XX
...
157       0.XX               0.XX              ...  0.XX
```

### Estimated Runtime

- ~3.5 hours per model (45,000 steps + evaluation)
- 12 models × 3.5 hours = ~42 hours total

### Status: RUNNING

Started: 2026-01-22 ~01:55
Current progress: Training ep_001 (1 episode)

---

## 2026-01-31: A-B-A Validation - ACT-ViT on Confuser Dataset (A2)

**Objective**: Test if ViT backbone resolves confuser block issues seen with ResNet.

### Training

**Model**: ACT-ViT (ViT-B/16 backbone)
**Dataset**: `danbhf/sim_pick_place_2pos_220ep_confuser` (220 episodes, 2 block positions with confuser)
**Training**: 50,000 steps, batch_size=64, lr=1e-5, chunk_size=50
**Duration**: ~9.5 hours
**Final Loss**: 0.053 (best: 0.052)
**Output**: `checkpoints/act_a2_vit/`

### Evaluation Results

| Metric | Value |
|--------|-------|
| Success Rate | **72%** |
| Pick Rate | 96% |
| Drop Rate | 20.8% |
| Avg Steps (success) | 140 |

**Failure Breakdown**:
- Never picked up: 2/50 (4%)
- Dropped during transport: 10/50 (20%)
- Timeout: 2/50 (4%)

### Comparison with ResNet

| Backbone | Cameras | Success Rate | Pick Rate | Drop Rate | Never Picked |
|----------|---------|--------------|-----------|-----------|--------------|
| ResNet-18 | 2 (wrist+overhead) | 50% | 70% | 20% | 30% |
| **ViT-B/16** | **1 (wrist only)** | **72%** | **96%** | 20.8% | **4%** |

**Key Finding**: ViT backbone achieves +22% higher success rate with fewer cameras!

### Key Observations

1. **ViT achieves 72% on confuser dataset** - need to compare with ResNet baseline
2. **High pick rate (96%)** - the model successfully identifies and picks up blocks
3. **Main failure mode is dropping (20.8%)** - transport phase needs improvement

---

## 2026-02-01: Smaller ViT Experiment (ViT-B/32)

**Status**: COMPLETED

**Objective**: Test if smaller patch size (fewer patches) maintains performance while being faster.

| Model | Patches/Image | Hidden Dim | Params |
|-------|---------------|------------|--------|
| ViT-B/16 | 196 (14×14) | 768 | 126M |
| ViT-B/32 | 49 (7×7) | 768 | 128M |

### Training

**Model**: ACT-ViT (ViT-B/32 backbone)
**Dataset**: `danbhf/sim_pick_place_2pos_220ep_confuser` (220 episodes)
**Training**: 50,000 steps, batch_size=64, lr=1e-5, chunk_size=50
**Duration**: 9h 47m (587 minutes)
**Final Loss**: 0.061 (best: 0.061)
**Output**: `checkpoints/act_a2_vit_b32/`
**WandB**: https://wandb.ai/bryars-bryars/lerobot-thesis/runs/jhv8w3jt

### Evaluation Results

| Metric | Value |
|--------|-------|
| Success Rate | **60%** |
| Pick Rate | 80% |
| Drop Rate | 20% |
| Avg Steps (success) | 177 |

**Failure Breakdown**:
- Never picked up: 10/50 (20%)
- Dropped during transport: 7/50 (14%)
- Missed goal: 1/50 (2%)
- Timeout: 2/50 (4%)

### Comparison with ViT-B/16 and ResNet

| Backbone | Cameras | Patches | Success Rate | Pick Rate | Never Picked |
|----------|---------|---------|--------------|-----------|--------------|
| ResNet-18 | 2 (wrist+overhead) | N/A | 50% | 70% | 30% |
| **ViT-B/16** | **1 (wrist only)** | **196** | **72%** | **96%** | **4%** |
| ViT-B/32 | 1 (wrist only) | 49 | 60% | 80% | 20% |

### Key Findings

1. **ViT-B/32 performs worse than ViT-B/16** (60% vs 72%)
2. **Spatial resolution matters**: Coarser patches (32×32 vs 16×16) reduce the model's ability to precisely locate and grasp the block
3. **Pick rate dropped significantly**: 80% vs 96% - the model struggles more with initial block detection/grasping with fewer patches
4. **Training loss similar**: Both reached ~0.06 final loss, but the lower spatial resolution hurts generalization
5. **Speed difference negligible**: ViT-B/32 doesn't provide meaningful speed benefits given the performance degradation

**Recommendation**: Use ViT-B/16 over ViT-B/32 for manipulation tasks where spatial precision matters.

---

## 2026-02-01: Frozen Backbone Experiment (ViT-B/16)

**Status**: COMPLETED

**Objective**: Test if pretrained ViT features (ImageNet) are sufficient for manipulation, or if fine-tuning is necessary.

**Training**:
- Output: `checkpoints/act_a2_vit_frozen/`
- WandB: https://wandb.ai/bryars-bryars/lerobot-thesis/runs/jdcsjjwq
- Total parameters: 126.4M
- Trainable parameters: 40.6M (32%)
- Frozen backbone: 85.8M (68%)

**Hypothesis**: Frozen ImageNet features may lack domain-specific visual patterns for robotics, requiring fine-tuning for optimal performance.

### Results

| Metric | Value |
|--------|-------|
| Success Rate | **74%** |
| Pick Rate | 84% |
| Drop Rate | 9.5% |
| Avg Steps (success) | 171 |

**Key Finding**: Frozen backbone **outperforms** unfrozen (74% vs 72%) with 68% fewer trainable parameters!

This suggests:
1. Pre-trained ImageNet features transfer well to robotic manipulation
2. Full fine-tuning may cause slight overfitting on this dataset size
3. Frozen backbone is more parameter-efficient and trains faster

---

## 2026-02-02: Two-Camera ViT Bug Discovery

**Status**: BUG FOUND AND FIXED

### Problem

All previous ViT experiments used only **wrist_cam** (1 camera). When testing with both cameras:

| Cameras | Success Rate |
|---------|-------------|
| 1 (wrist only) | 72% |
| 2 (wrist + overhead) | **36%** ← Much worse! |

### Root Cause

In `models/act_vit.py`, both cameras shared the **same positional embeddings**:

```python
# BUG: Both cameras get identical position embeddings
patch_pos = self.encoder_vit_pos_embed.weight.unsqueeze(1)  # Same for all cameras
```

The model couldn't distinguish which patches came from which camera. Wrist and overhead patches were spatially concatenated but had identical positional encodings - the model was confused.

### Fix

Added **learnable camera embeddings** to distinguish patches from different cameras:

```python
# Each camera gets a unique embedding added to all its patches
self.encoder_camera_embed = nn.Embedding(num_cameras, config.dim_model)

# In forward pass:
camera_embed = self.encoder_camera_embed.weight[cam_idx]
patch_pos = patch_pos + camera_embed  # Position + Camera identity
```

Now each patch gets: `patch_features + position_embedding + camera_embedding`

This separates:
1. **Where** in the image (positional) - shared across cameras
2. **Which** camera (camera embedding) - one learnable vector per camera

### Re-training Results (Camera Embeddings Only)

Re-trained 2-camera model from checkpoint 15000 with camera embeddings fix, keeping chunk_size=100.

**Result: 36% success** — identical to the broken model. Camera embeddings alone did NOT help.

This revealed the real confounding variable: **chunk_size**, not cameras.

---

## 2026-02-03: Chunk Size vs Camera Count — Isolating the Variable

**Status**: COMPLETED

**Objective**: Determine whether 2-camera degradation (72% → 36%) was caused by multiple cameras or by the larger chunk_size used in 2-camera experiments.

### Background

All previous 2-camera experiments used chunk_size=100 (default), while the successful 1-camera model used chunk_size=50. These two variables changed simultaneously, making it impossible to attribute the performance drop.

### Dataset

**All models trained on**: `danbhf/sim_pick_place_2pos_220ep_confuser`
- 220 episodes, 2 block positions with confuser block present
- 31,210 total frames
- Cameras available: wrist_cam (640×480), overhead_cam (640×480)

### Experiment Design

| Model | Backbone | Cameras | Chunk Size | Camera Embeddings |
|-------|----------|---------|------------|-------------------|
| A (baseline) | ViT-B/16 | 1 (wrist) | 50 | N/A |
| B (original 2-cam) | ViT-B/16 | 2 (wrist+overhead) | 100 | No |
| C (camera fix) | ViT-B/16 | 2 (wrist+overhead) | 100 | Yes |
| **D (chunk fix)** | **ViT-B/16** | **2 (wrist+overhead)** | **50** | **Yes** |

All models: 50,000 steps, lr=1e-5, batch_size=8 (2-cam) or 64 (1-cam).

### Training

**Model D**: `checkpoints/act_a2_vit_2cam_chunk50/`
- Training steps: 50,000
- Best loss: 0.1000

### Results

| Model | Cameras | Chunk Size | Success Rate | Pick Rate | Drop Rate | Never Picked |
|-------|---------|------------|--------------|-----------|-----------|--------------|
| A (1-cam baseline) | 1 (wrist) | 50 | **72%** | 96% | 20.8% | 4% |
| B (2-cam, big chunk) | 2 | 100 | 36% | 58% | — | — |
| C (2-cam + cam embed) | 2 | 100 | 36% | 58% | 27.6% | 42% |
| **D (2-cam, small chunk)** | **2** | **50** | **58%** | **92%** | **32.6%** | **8%** |

**Failure breakdown for Model D (50 episodes)**:
- Never picked up: 4 (8%)
- Dropped during transport: 13 (26%)
- Missed goal: 2 (4%)
- Timeout: 2 (4%)

### Key Findings

1. **Chunk size was the primary variable** — reducing chunk_size from 100 to 50 improved success from 36% to 58% (+22 percentage points)
2. **Camera embeddings had no measurable effect** — Models B and C both scored 36% with chunk_size=100
3. **Pick rate recovered** — 2-cam chunk_50 achieves 92% pick rate (comparable to 1-cam's 96%), vs 58% with chunk_100
4. **Drop rate is the remaining gap** — 32.6% drop rate vs 20.8% for 1-cam, accounting for most of the 14% success gap
5. **Larger chunk_size hurts more with more cameras** — more visual tokens + longer prediction horizon may exceed model capacity

### Comparison with ResNet Baseline

| Backbone | Cameras | Chunk Size | Success Rate | Pick Rate |
|----------|---------|------------|--------------|-----------|
| ResNet-18 | 2 (wrist+overhead) | 100 | 50% | 70% |
| ViT-B/16 | 2 (wrist+overhead) | 100 | 36% | 58% |
| ViT-B/16 | 2 (wrist+overhead) | 50 | **58%** | **92%** |
| ViT-B/16 | 1 (wrist only) | 50 | **72%** | **96%** |

ResNet-18 with 2 cameras and chunk_size=100 actually outperformed ViT-B/16 under the same settings (50% vs 36%). This suggests ResNet's compact representation may be more robust to larger chunk sizes, while ViT's higher-dimensional patch tokens require shorter prediction horizons to train effectively.

### Implications

- **Chunk size is a critical hyperparameter** that interacts with model capacity and input complexity
- **Adding cameras is not free** — more visual information requires careful tuning of other hyperparameters
- **Fair comparisons require controlling chunk_size** — previous 1-cam vs 2-cam comparisons were confounded
- Future work: test ResNet 2-cam with chunk_size=50 for a fully controlled comparison

---

## 2026-02-03: Pickup Coordinates Conditioning

**Status**: COMPLETED

**Objective**: Test whether providing explicit block XY coordinates as input improves policy performance beyond what the vision backbone alone provides.

### Dataset

**Dataset**: `danbhf/sim_pick_place_2pos_220ep_confuser` (220 episodes, 2 block positions with confuser)

### Training

**Model**: ACT-ViT (ViT-B/16) with `--pickup_coords` enabled
- 1 camera (wrist_cam), chunk_size=50
- Block XY coordinates from `episode_scenes.json` fed as additional conditioning
- X bounds: (0.1, 0.38), Y bounds: (-0.28, 0.27)
- 50,000 steps, batch_size=8, lr=1e-5
- Best loss: 0.1053
- Output: `checkpoints/act_a2_vit_coords/`
- WandB: https://wandb.ai/bryars-bryars/lerobot-thesis/runs/lsvn7ccg

### Results

| Metric | Value |
|--------|-------|
| Success Rate | **76%** |
| Pick Rate | 90% |
| Drop Rate | 15.6% |
| Avg Steps (success) | 167 |
| Avg Max Height | 0.249 |

**Failure Breakdown**:
- Never picked up: 5/50 (10%)
- Dropped during transport: 6/50 (12%)
- Missed goal: 1/50 (2%)
- Timeout: 0/50 (0%)

### Comparison — All ViT-B/16 Models on Confuser Dataset

| Model | Cameras | Chunk Size | Coords | Success | Pick Rate | Drop Rate |
|-------|---------|------------|--------|---------|-----------|-----------|
| ViT baseline | 1 (wrist) | 50 | No | 72% | 96% | 20.8% |
| ViT frozen | 1 (wrist) | 50 | No | 74% | 84% | 9.5% |
| **ViT + coords** | **1 (wrist)** | **50** | **Yes** | **76%** | **90%** | **15.6%** |
| ViT 2-cam | 2 | 50 | No | 58% | 92% | 32.6% |
| ViT 2-cam | 2 | 100 | No | 36% | 58% | 27.6% |
| ResNet-18 | 2 | 100 | No | 50% | 70% | 20% |

### Key Findings

1. **Coords provide a modest +4% improvement** over the unfrozen ViT baseline (76% vs 72%)
2. **Drop rate improved significantly**: 15.6% vs 20.8% — explicit coordinates help the transport phase
3. **Avg max height is much higher** (0.249 vs 0.173 for 2-cam) — model lifts block more confidently
4. **Zero timeouts** — model is more decisive with explicit spatial information
5. **Pick rate slightly lower** (90% vs 96%) — possible interference between visual and coordinate signals during reaching

### Interpretation

Providing block coordinates gives the model a small but real advantage. The improvement is modest because the ViT backbone already extracts spatial information from wrist camera images fairly effectively. The main benefit is in the transport phase (lower drop rate), suggesting coordinates help the model maintain a more stable grasp trajectory when it knows exactly where the block started.

---

## 2026-02-03: Temporal Ensembling on 2-Camera Model

**Status**: COMPLETED

**Objective**: Test if temporal ensembling (which improved 1-cam from 82% to 90%) helps the 2-camera model.

### Results

| Condition | Success | Pick Rate | Drop Rate | Timeouts |
|-----------|---------|-----------|-----------|----------|
| 2-cam chunk_50 (no ensemble) | **58%** | **92%** | 32.6% | 2 |
| 2-cam chunk_50 + ensemble 0.01 | 50% | 78% | **15.4%** | 8 |

### Key Finding

Temporal ensembling **hurt** the 2-camera model (-8%). While it halved the drop rate (good), it made the approach too cautious — more timeouts and missed pickups. The smoothing that helps 1-camera models appears to hurt with 2 cameras, likely because the overhead view introduces more prediction variance that the ensemble averages into a sluggish trajectory.

---

## Notes / Future Work

### GPU Utilization Optimization

**TODO**: Investigate increasing batch size or other hyperparameters to better utilize GPU capacity during training.

Current observations:
- ACT-ViT training runs at ~1-2 it/s with batch_size=64, which may indicate room for improvement
- Speed fluctuates significantly (1.1-2.3 it/s) likely due to data loading patterns
- Consider: larger batch size, data prefetching, disk caching, or mixed precision training
