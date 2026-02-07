# Experiments Part 2

## Summary of Previous Experiments (from experiments.md)

Research focused on **training ACT (Action Chunking Transformer) policies for a pick-and-place task** with a simulated SO-101 robot arm, specifically tackling the **"confuser" disambiguation problem** - where an identical-looking distractor block confuses the policy.

### Key Problem
When you place a second identical block (the "confuser") in the scene, basic ACT models trained without it fail dramatically (~30% success → from 90%+ baseline).

### Major Experiments

**Experiment 5: Coordinate Conditioning**
- Added pickup target coordinates (x,y) as input to tell the model which block to grab
- **5a-5b**: Works great! Model trained WITHOUT confuser achieves **100% success** on confuser scene when given coordinates
- **5c-5d**: Training WITH confuser data causes **complete failure (0%)** despite longer training (200k steps) and lower loss - classic **mode collapse** where model outputs average trajectory between the two block positions

**Experiment 6: Vision Transformer Backbone**
- Hypothesis: ResNet's Global Average Pooling loses spatial info needed for coordinate conditioning
- Replaced ResNet18 with ViT-B/16 (86M → 126M parameters)

### Breakthrough Finding

| Model | Backbone | Dataset | Steps | Loss | Success |
|-------|----------|---------|-------|------|---------|
| ACT | ResNet18 | Mixed confuser | 200k | 0.056 | **0%** |
| ACT-ViT | ViT-B/16 | Mixed confuser | 50k | 0.121 | **100%** |

**ViT avoids mode collapse** where ResNet fails completely - even with 4x more training and half the loss.

### Current Limitation
Testing shows the model doesn't truly generalize to **novel positions** - it learns to distinguish "which of the known training positions" rather than arbitrary (x,y) targeting. This suggests training data needs blocks at many random positions for true spatial grounding.

---

## Experiment 7: Subtask Decomposition

### Motivation

Break the monolithic pick-and-place policy into subtask-conditioned segments. Each subtask gets:
1. A **subtask identifying token** (behavioral mode)
2. Optionally a **coordinate token** (spatial target)

### Proposed Subtask Breakdown

| Subtask | Token | Coord | Description |
|---------|-------|-------|-------------|
| `move_to_source` | 0 | source (x,y) | Approach the target block |
| `pick_up` | 1 | source (x,y) | Close gripper, grasp block |
| `move_to_dest` | 2 | dest (x,y) | Transport to drop location |
| `drop` | 3 | dest (x,y) | Open gripper, release block |

### Why This Could Help

1. **Clearer conditioning** - Subtask token provides strong behavioral mode switching. "Move to source" vs "pick up" are very different motion patterns - the model doesn't have to infer which mode from visual context alone.

2. **Better coordinate usage** - Each subtask has semantically meaningful coords:
   - `move_to_source` + source_coord
   - `pick_up` + source_coord (for fine alignment)
   - `move_to_dest` + dest_coord
   - `drop` + dest_coord

3. **Potential for generalization** - `move_to(coord)` could become a reusable skill that works for any target, not just memorized positions.

### Key Challenges

1. **Segmentation** - How to label training data with subtask boundaries?
2. **Inference transitions** - Who decides when to switch subtasks at runtime?
3. **Architecture** - How to inject the subtask token into the model?

### Segmentation Approaches

#### Approach A: Gripper State Heuristic

Use gripper position/velocity to detect phase transitions:

```
gripper_open → gripper_closing = end of move_to_source
gripper_closed → holding        = end of pick_up
holding → gripper_opening       = end of move_to_dest
gripper_open + near_dest        = end of drop
```

**Pros**: Simple, deterministic, uses data already in episodes
**Cons**: Gripper timing may not perfectly align with semantic phases

#### Approach B: Proximity + Gripper Combined

Combine spatial proximity with gripper state:

```python
# Pseudocode
if gripper_open and dist_to_block > threshold:
    subtask = MOVE_TO_SOURCE
elif gripper_open and dist_to_block < threshold:
    subtask = PICK_UP
elif gripper_closed and dist_to_bowl > threshold:
    subtask = MOVE_TO_DEST
else:
    subtask = DROP
```

**Pros**: More semantically accurate boundaries
**Cons**: Requires block/bowl positions (available in episode_scenes.json)

#### Approach C: Velocity/Motion Segmentation

Detect phase changes from motion patterns:
- High velocity movement → transport phases
- Low velocity + gripper motion → manipulation phases
- Zero velocity moments → potential phase boundaries

**Pros**: Task-agnostic, could generalize to other tasks
**Cons**: Noisier, harder to tune thresholds

#### Approach D: Fixed Proportional Splits

Assume each episode follows similar timing:
- First 25%: move_to_source
- 25-40%: pick_up
- 40-75%: move_to_dest
- 75-100%: drop

**Pros**: Dead simple, no per-frame analysis needed
**Cons**: Inaccurate for variable-length phases, probably too crude

### Recommended Approach: B (Proximity + Gripper)

We have all the data needed:
- Gripper state: in `observation.state` (last dimension)
- Block position: in `episode_scenes.json`
- Bowl position: fixed in scene XML
- End effector position: can compute via FK or use wrist cam position

This gives clean semantic boundaries that match human intuition of the subtasks.

### Implementation Plan: Subtask Annotation Script

**Data Available:**
- `observation.state`: 6 dims - [5 arm joints, gripper] (normalized units)
- `episode_scenes.json`: duplo (x,y,z) and bowl (x,y,z) per episode
- Bowl position: fixed at (0.217, -0.225, 0.0)

**Need to compute:**
- End-effector position via Forward Kinematics (FK) from joint angles
- Distance to block and bowl
- Gripper state threshold (open vs closed)

**Proposed Heuristics:**

```python
GRIPPER_OPEN_THRESHOLD = 30.0    # Above this = open (normalized units)
GRIPPER_CLOSED_THRESHOLD = 15.0  # Below this = closed
NEAR_BLOCK_DIST = 0.08           # 8cm = "near" the block
NEAR_BOWL_DIST = 0.10            # 10cm = "near" the bowl

def get_subtask(gripper_val, dist_to_block, dist_to_bowl, holding_block):
    if not holding_block:
        if dist_to_block > NEAR_BLOCK_DIST:
            return MOVE_TO_SOURCE  # Approaching block
        else:
            return PICK_UP  # Near block, attempting grasp
    else:
        if dist_to_bowl > NEAR_BOWL_DIST:
            return MOVE_TO_DEST  # Transporting to bowl
        else:
            return DROP  # Near bowl, releasing
```

### Final Approach: Forward-Only Distance State Machine

Episodes always follow the same sequence, so we use a simple forward-only state machine based purely on EE-to-target distances:

```
MOVE_TO_SOURCE ──[near block]──► PICK_UP ──[far from block]──► MOVE_TO_DEST ──[near bowl]──► DROP
     0                              1                              2                           3
```

**Transitions (always in order, never backwards):**
1. Start in `MOVE_TO_SOURCE` (state 0)
2. When `dist_to_block < NEAR_THRESHOLD` → switch to `PICK_UP` (state 1)
3. When `dist_to_block > FAR_THRESHOLD` → switch to `MOVE_TO_DEST` (state 2) - block has been lifted
4. When `dist_to_bowl < NEAR_THRESHOLD` → switch to `DROP` (state 3)
5. Stay in `DROP` until episode ends

**Only requires:**
- FK solver for EE position (from joint angles)
- Block position (from episode_scenes.json)
- Bowl position (from episode_scenes.json, usually fixed at 0.217, -0.225)
- Two distance thresholds: NEAR (~5-8cm) and FAR (~10-15cm)

**No need for:**
- Gripper state tracking
- "Holding" detection
- Backwards transitions

**Script:** `scripts/tools/annotate_subtasks.py`

### Annotation Results (2026-02-05)

**Dataset:** `sim_pick_place_157ep` (157 episodes, 22,534 frames)

**Subtask Distribution:**
| Subtask | Frames | Percentage |
|---------|--------|------------|
| MOVE_TO_SOURCE | 7,957 | 35.3% |
| PICK_UP | 6,242 | 27.7% |
| MOVE_TO_DEST | 4,378 | 19.4% |
| DROP | 3,957 | 17.6% |

**Key Finding:** Actions (not observation.state) contain the correct joint target positions for FK computation. The observation.state in this dataset appears to be recorded at lower frequency or has recording issues.

**Threshold Tuning:**
- NEAR_THRESHOLD = 0.06m (6cm) - works well, typical pickup gets within 1-2.5cm
- FAR_THRESHOLD = 0.12m (12cm) - detects lift after grasp

**Edge Cases:**
- Episode 22: Never reaches block (min dist 7.9cm > 6cm threshold) - likely a failed episode
- Most episodes transition cleanly through all 4 subtasks

### Next Steps: Training with Subtask Conditioning

#### Step 1: Dataset Wrapper Modifications
- Load `subtask_annotations.json` alongside parquet data
- Add `observation.subtask` field to each batch (one-hot or integer)
- Optionally add phase-appropriate coordinates:
  - Subtasks 0,1 (MOVE_TO_SOURCE, PICK_UP): use source block coords
  - Subtasks 2,3 (MOVE_TO_DEST, DROP): use bowl/destination coords

#### Step 2: Model Architecture Changes
- Add subtask embedding layer (4 subtasks → embedding dim)
- Inject subtask token into transformer encoder alongside:
  - Robot state token
  - Coordinate token (if using)
  - Image patch tokens

#### Step 3: Inference Strategy
Two options for runtime subtask transitions:

**Option A: External State Machine**
- Same FK-based state machine used for annotation
- Compute EE position each step, determine subtask
- Simple, deterministic, matches training distribution

**Option B: Learned Transitions**
- Model predicts next subtask alongside actions
- More flexible but requires additional training signal
- Could learn to handle edge cases better

**Recommended:** Start with Option A (external state machine) for simplicity.

#### Step 4: Experiments to Run
1. **Baseline**: ACT-ViT with subtask token only (no coords) on 157ep
2. **+ Coords**: Add phase-appropriate coordinates
3. **Confuser test**: Train on mixed dataset, test disambiguation
4. **Generalization**: Test if subtask decomposition helps with novel positions

---

### Implementation Complete (2026-02-05)

#### Files Modified/Created

1. **`utils/training.py`** - Added `SubtaskDataset` class:
   - Loads `subtask_annotations.json`
   - Adds subtask one-hot (4 dims) to `observation.environment_state`
   - Concatenates with pickup coords if both enabled (2+4=6 dims)

2. **`scripts/training/train_act_vit.py`** - Added `--subtask` flag:
   - Loads subtask annotations (local first, then HuggingFace)
   - Configures combined environment_state feature size
   - Merges normalization stats properly

3. **`scripts/tools/annotate_subtasks.py`** - Creates subtask annotations:
   - Uses FK to compute EE position from actions
   - Forward-only state machine based on distance thresholds
   - Saves to `meta/subtask_annotations.json`

#### Usage

```bash
# Annotate a dataset
python scripts/tools/annotate_subtasks.py --dataset datasets/sim_pick_place_157ep --visualize

# Train with subtask only
python scripts/training/train_act_vit.py danbhf/sim_pick_place_157ep --subtask

# Train with subtask + pickup coords
python scripts/training/train_act_vit.py danbhf/sim_pick_place_157ep --subtask --pickup_coords
```

#### Combined Feature Dimensions
| Configuration | `observation.environment_state` dims |
|--------------|-------------------------------------|
| pickup_coords only | 2 (x, y normalized) |
| subtask only | 4 (one-hot) |
| both | 6 (coords + one-hot) |

#### Key Finding
The `action` column (not `observation.state`) contains correct joint targets for FK computation. Actions are in normalized [-100, 100] format.

---

### Experiment 7a: Subtask Conditioning Baseline (2026-02-05)

**Purpose**: Test if subtask token conditioning helps the model learn distinct behaviors for each phase.

**Dataset**: `danbhf/sim_pick_place_157ep`
**Model**: ACT-ViT with subtask conditioning (4-dim one-hot in environment_state)
**Training**: 50k steps

**WandB Run**: https://wandb.ai/bryars-bryars/lerobot-thesis/runs/msohwyz2

**Configuration**:
- Subtask: enabled (4-dim one-hot)
- Pickup coords: disabled
- Total params: 126M
- environment_state dim: 4

**Status**: ✅ Complete

**Results**:
- Best loss: **0.0554** (comparable to baseline 0.054-0.061)
- Training time: ~2.75 hours
- All checkpoints saved (5k-50k + final)

**Evaluation Results (50 episodes)**:
- Success Rate: **84.0%**
- Pick Rate: 96.0%
- Avg Steps (success): 133.4
- Avg Time (success): 0.55s

**Failure Breakdown**:
| Failure Type | Count |
|-------------|-------|
| Never picked up | 2 |
| Dropped during transport | 3 |
| Missed goal | 1 |
| Timeout | 2 |

**Analysis**:
This model uses **subtask-only conditioning** (no pickup coordinates). The 84% success rate demonstrates that the subtask token provides useful behavioral mode information, and the model can successfully complete tasks using visual feedback alone. This serves as a baseline for testing whether adding coordinate conditioning improves performance.

**Key Insight**: The subtask state machine at inference time correctly transitions through the 4 phases (MOVE_TO_SOURCE → PICK_UP → MOVE_TO_DEST → DROP) based on FK distance calculations.

---

### Experiment 7b: Subtask + Pickup Coords Combined (2026-02-05)

**Purpose**: Test if adding pickup coordinates alongside subtask conditioning improves performance.

**Dataset**: `danbhf/sim_pick_place_157ep`
**Model**: ACT-ViT with subtask + pickup coords (6-dim environment_state)
**Training**: 50k steps

**WandB Run**: https://wandb.ai/bryars-bryars/lerobot-thesis/runs/os2boiqa

**Configuration**:
- Subtask: enabled (4-dim one-hot)
- Pickup coords: enabled (2-dim normalized x,y)
- Total environment_state: 6 dims
- Output dir: `outputs/train/act_vit_subtask_coords_157ep`

**Status**: ✅ Complete

**Training Results**:
- Best loss: **0.0556** (comparable to 7a's 0.0554)
- Training time: 132 minutes

**Evaluation Results (50 episodes each)**:

| Configuration | Success Rate | Pick Rate | Notes |
|--------------|--------------|-----------|-------|
| Full coords (all phases) | 86% | 96% | Coords provided throughout |
| **Selective coords** | **90%** | **100%** | Coords zeroed during PICK_UP/DROP |

**Failure Breakdown (Selective Coords)**:
- Dropped during transport: 4
- Missed goal: 1
- Never picked up: 0 (100% pick rate!)

**Key Finding**:
Selective coordinate masking outperforms full conditioning! The model achieves:
- Higher success rate (90% vs 86%)
- Perfect pick rate (100% vs 96%)

This validates the hypothesis: **During fine manipulation (PICK_UP and DROP), visual feedback is more reliable than coordinate input.** The robot is close enough to see the target clearly, and the coordinate may introduce noise or create a distribution mismatch (trained with coords, now receiving zeros).

**Comparison Across All Experiments**:

| Model | Configuration | Success Rate |
|-------|---------------|--------------|
| ACT-ViT (7a) | Subtask only | 84% |
| ACT-ViT (7b) | Subtask + full coords | 86% |
| **ACT-ViT (7b)** | **Subtask + selective coords** | **90%** |

**Insight**: The subtask token provides behavioral mode switching, while coordinates are most useful during navigation (MOVE_TO_SOURCE, MOVE_TO_DEST) and should be masked during manipulation (PICK_UP, DROP) where camera feedback dominates.

---

### Implementation: Selective Coords

Added `--selective-coords` flag to `scripts/inference/eval.py`:

```bash
# Run with selective coords (zeros during PICK_UP and DROP)
python scripts/inference/eval.py model_path --subtask --pickup-coords --selective-coords
```

**Code change** in `utils/training.py`:
```python
# Inside run_evaluation step loop
if selective_coords and subtask_state in (1, 3):  # PICK_UP or DROP
    zeroed_coords = torch.zeros_like(pickup_coord_tensor)
    batch["observation.environment_state"] = torch.cat([zeroed_coords, subtask_tensor], dim=1)
```

---

### Experiment 8: Position Invariance Testing (2026-02-05)

**Purpose**: Test how well the coordinate-conditioned model generalizes to novel block positions outside the training distribution.

**Training Data Distribution**:
- X range: 0.177 to 0.257 (~8cm)
- Y range: 0.185 to 0.265 (~8cm)
- 155 unique (x,y) positions across 157 episodes

**Model Tested**: ACT-ViT (7b) with selective coords

**Results (10 episodes each)**:

| Position | Distribution | Success Rate |
|----------|--------------|--------------|
| (0.22, 0.22) | In-dist center | **100%** |
| (0.18, 0.19) | In-dist edge | 90% |
| (0.30, 0.22) | OOD (+3cm X) | 70% |
| (0.22, 0.30) | OOD (+4cm Y) | 50% |
| (0.15, 0.22) | OOD (-3cm X) | 40% |
| (0.15, 0.30) | Diagonal OOD | **10%** |

**Key Findings**:
1. **Perfect in-distribution**: 100% at training center
2. **Rapid degradation OOD**: Performance drops sharply outside training range
3. **Asymmetric sensitivity**: -X direction (closer to robot) degrades faster than +X
4. **Coordinate conditioning ≠ spatial grounding**: The model learns "which known position" rather than true arbitrary (x,y) targeting

**Implication**: To achieve position invariance, training data needs blocks at many random positions across the workspace, not just a small region.

---

### Experiment 9: Relative/Delta Actions (Proposed)

**Motivation**: Current models predict **absolute joint targets**. This may limit generalization because "move to 45°" is position-specific, while "move +5°" is transferable.

**Current Action Space**:
```
action = [target_joint_1, target_joint_2, ..., target_joint_6]  # Absolute positions
```

**Proposed Delta Action Space**:
```
action = [delta_joint_1, delta_joint_2, ..., delta_joint_6]  # Relative changes
next_state = current_state + action
```

**Potential Benefits**:
1. **Better generalization** - "move right" means the same thing regardless of starting pose
2. **More transferable skills** - A "reach forward" motion pattern could work from any position
3. **Smaller action magnitudes** - Deltas are typically small, easier to predict accurately

**Challenges**:
1. **Dataset modification** - Need to compute `delta = action[t] - state[t-1]` for each timestep
2. **Accumulation errors** - Small prediction errors accumulate over time
3. **Clipping needed** - Large deltas could cause unsafe motions
4. **Training changes** - May need different normalization, loss weighting

**Implementation Plan**:

1. **Create DeltaActionDataset wrapper**:
   ```python
   class DeltaActionDataset(torch.utils.data.Dataset):
       def __getitem__(self, idx):
           item = self.dataset[idx]
           # Compute delta: action - current_state
           delta = item['action'] - item['observation.state'][:6]
           item['action'] = delta
           return item
   ```

2. **Modify inference**:
   ```python
   # Instead of: target = policy_output
   # Use: target = current_state + policy_output
   delta = postprocessor(policy.predict_action(batch))
   target = current_joints + delta
   ```

3. **Training**: Use same ACT-ViT architecture, just with delta targets

**Hypothesis**: Delta actions + subtask conditioning could improve position invariance by learning relative motions for each phase rather than absolute trajectories.

---

### Experiment 9a: Delta vs Absolute Action Analysis (2026-02-05)

**Purpose**: Analyze the statistical properties of absolute vs delta action representations to understand potential benefits for learning.

**Analysis Script**: `scripts/tools/analyze_delta_actions.py`

#### Frame-to-Frame Delta Analysis

Frame-to-frame deltas represent `action[t] - action[t-1]` - the change in joint targets between consecutive timesteps. This is what a delta-action policy would actually predict.

**Key Finding: 99.7% Variance Reduction**

| Metric | Absolute Actions | Frame Deltas | Ratio |
|--------|-----------------|--------------|-------|
| Mean variance (per joint) | 1086.15 | 2.92 | **0.0027** |
| Average magnitude | 36.93 | 0.98 | 0.026 |
| Average L2 norm | 107.42 | 3.30 | 0.031 |

**Per-Joint Frame Delta Statistics (normalized units [-100, 100]):**

| Joint | Mean | Std | Min | Max | |Mean| |
|-------|------|-----|-----|-----|-------|
| 1 | 0.33 | 1.81 | -7.75 | 21.17 | 1.05 |
| 2 | -0.03 | 1.96 | -10.90 | 9.14 | 1.27 |
| 3 | -0.11 | 1.69 | -11.44 | 11.98 | 1.00 |
| 4 | 0.20 | 2.15 | -11.48 | 16.01 | 1.30 |
| 5 | 0.03 | 0.80 | -5.22 | 6.37 | 0.44 |
| 6 | 0.06 | 1.51 | -8.87 | 8.07 | 0.79 |

**Frame Deltas by Subtask:**

| Subtask | Mean |delta| | L2 Norm | Frames | Notes |
|---------|-------------|---------|--------|-------|
| MOVE_TO_SOURCE | 1.01 | 3.22 | 7,802 | Navigation to block |
| PICK_UP | 0.72 | 2.56 | 6,240 | Fine manipulation (smallest) |
| MOVE_TO_DEST | 1.36 | 4.58 | 4,378 | Fast transport (largest) |
| DROP | 0.89 | 3.24 | 3,957 | Release motion |

#### Analysis: Why Delta Actions Might Help

1. **Dramatically lower variance**: Frame deltas have ~0.3% of the variance of absolute actions. This means:
   - More consistent training signal
   - Easier for the model to learn accurate predictions
   - Less memorization of specific positions required

2. **Small, bounded magnitudes**: Typical deltas are ±2-3 units (vs ±100 for absolute). Benefits:
   - Natural action clipping (large deltas = unsafe)
   - Smoother trajectories
   - Easier to detect anomalous predictions

3. **Position-invariant representation**: "Move +1.0 units" means the same thing regardless of:
   - Starting position
   - Block location
   - Robot configuration

4. **Subtask-appropriate scaling**:
   - PICK_UP has smallest deltas (precise, careful movements)
   - MOVE_TO_DEST has largest deltas (fast transport)
   - This natural scaling could help the model learn phase-appropriate behaviors

#### Potential Challenges

1. **Error accumulation**: Small prediction errors compound over time
   - Mitigation: Periodic re-grounding from visual feedback
   - Mitigation: Subtask conditioning helps model know "when to stop"

2. **Normalization**: Need appropriate scaling since deltas are ~30x smaller
   - Use separate normalization stats for delta dataset
   - Or scale up deltas during training

3. **Chunk prediction**: ACT predicts 100-step chunks
   - For deltas: each step in chunk is relative to previous
   - Cumsum needed to get absolute positions for execution
   - Or: predict absolute chunk, just train on delta loss

#### Comparison Summary Table

| Aspect | Absolute Actions | Delta Actions |
|--------|-----------------|---------------|
| What it predicts | "Go to position X" | "Move by delta d" |
| Variance | High (1086) | Very low (2.9) |
| Position-specific | Yes | No |
| Error propagation | Single step | Accumulates |
| Generalization potential | Low | High |
| Training signal consistency | Variable | Consistent |

#### Next Steps for Experiment 9

1. **Create DeltaActionDataset wrapper** that transforms actions to frame-to-frame deltas
2. **Modify inference loop** to accumulate deltas: `target = current + delta`
3. **Train ACT-ViT** with delta actions + subtask conditioning
4. **Compare position generalization** between absolute and delta models

---

### Experiment 9b: Delta Actions Training (2026-02-05)

**Purpose**: Train with delta/relative actions instead of absolute joint targets to test position invariance hypothesis.

**Dataset**: `danbhf/sim_pick_place_157ep`
**Model**: ACT-ViT with delta actions + subtask + pickup coords
**Training**: 50k steps

**WandB Run**: https://wandb.ai/bryars-bryars/lerobot-thesis/runs/hup0a2tx

**Configuration**:
- Delta actions: enabled
- Subtask: enabled (4-dim one-hot)
- Pickup coords: enabled (2-dim normalized x,y)
- environment_state: 6 dims (coords + subtask)
- Output dir: `outputs/train/act_vit_delta_subtask_coords_157ep`

**Delta Action Stats (computed from parquet)**:
```
Mean: [ 0.005, -0.038, -0.016,  0.053, -0.019, -0.005]
Std:  [ 5.05,   4.51,   4.45,   4.80,   1.42,   2.16]
```

Compare to absolute action stats:
```
Mean: [-9.20, -2.60, 13.98, 30.92, 58.24, 26.90]
Std:  [35.82, 33.73, 36.61, 44.58, 19.46, 19.72]
```

**Key Observations**:
- Delta mean is ~0 (as expected for relative changes)
- Delta std is ~7x smaller than absolute std
- This confirms delta actions are much more consistent

**Status**: 🔄 Training in progress...

**Implementation Details**:

1. **DeltaActionDataset wrapper** (`utils/training.py`):
   - Converts action chunks to frame-to-frame deltas
   - delta[0] = action[0] - state[:action_dim]
   - delta[i] = action[i] - action[i-1] for i > 0

2. **Inference conversion** (`utils/training.py` in `run_evaluation`):
   ```python
   if delta_actions:
       current_state = batch["observation.state"][:NUM_JOINTS]
       joint_action = current_state + joint_action  # delta to absolute
   ```

3. **Stats computation** (fast path from parquet):
   - Avoids slow video decoding by reading actions directly from parquet
   - Computes consecutive frame deltas

**Training Results**:
- Best loss: **0.0885** (not directly comparable to absolute due to different normalization)
- Training time: 133 minutes

**Evaluation Results**:
- Success Rate: **0%** (10 episodes)
- Pick Rate: 10%
- Max height reached: 4.2cm (vs ~22cm for successful picks)

**Root Cause Analysis**:

The delta action approach **fails** when combined with ACT's action chunking. Key issues:

1. **First-frame delta explosion**: The model outputs a very large first delta (e.g., -22 units) trying to "correct" from the initial state to a canonical trajectory start. This suggests the model learned to output large initial corrections rather than smooth incremental movements.

2. **Chunk-delta mismatch**: In a delta chunk [d0, d1, d2, ...]:
   - d0 is relative to initial state
   - d1 is relative to the PREDICTED target after d0
   - d2 is relative to predicted target after d1

   But during execution, if the robot doesn't reach target[0] exactly, d1 is applied from the wrong position. Errors compound across the chunk.

3. **Position controller latency**: The robot uses position control with rate limits. It doesn't instantly reach target positions. Delta actions assume perfect execution of each step.

**Debug Trace (first 3 steps)**:
```
step 0: delta = [-22.0, -0.5,  7.0, -42.4, 68.8, -5.2]  # HUGE first delta
step 1: delta = [-0.76, -2.6,  0.8,  1.4, -0.3,  0.8]  # reasonable
step 2: delta = [-0.91, -2.6,  1.0,  1.7, -0.2,  1.1]  # reasonable
```

**Lessons Learned**:

1. Delta actions don't naturally fit with action chunking architectures like ACT
2. The variance reduction (99.7%) in delta space doesn't translate to better learning when chunks are predicted holistically
3. Need a different approach for position-invariant action representations

**Potential Solutions (Future Work)**:

1. **Single-step prediction**: Use a per-step policy without chunking
2. **Hybrid approach**: Use absolute for chunk, delta for fine-tuning
3. **Chunk conversion**: Convert delta chunk to absolute at prediction time, execute absolute
4. **Shorter chunks**: Use much smaller chunk_size (e.g., 10) with delta actions
5. **Velocity actions**: Predict joint velocities instead of position deltas

---

### Summary: Experiment 7-9 Results

| Experiment | Configuration | Success Rate | Notes |
|------------|---------------|--------------|-------|
| 7a | Subtask only | 84% | Baseline with subtask conditioning |
| 7b | Subtask + coords (full) | 86% | Slight improvement |
| 7b | Subtask + selective coords | **90%** | Best - zero coords during PICK_UP/DROP |
| 8 | Position invariance test | 100% → 10% | Rapid OOD degradation |
| 9b | Delta actions + subtask + coords | **0%** | Failed - chunking incompatibility |

**Key Findings**:

1. **Selective coordinate masking** (90%) outperforms full conditioning (86%)
2. **Position generalization** is limited - needs diverse training positions
3. **Delta actions** don't work well with action chunking architectures

