# Experiments Log

## TODO: Follow-up Required

### ACT Model - 100k steps (2026-01-05)
- **Model**: `danbhf/act_sim_pick_place_100k`
- **Dataset**: `danbhf/sim_pick_place_40ep_rgbd_ee`
- **Training**: 100k steps, ~510 minutes on vast.ai
- **Best loss**: 0.0307
- **WandB**: https://wandb.ai/bryars-bryars/lerobot-thesis/runs/c3jay0gt
- **Status**: Training complete, uploaded to HF
- **TODO**: Run evaluation in simulation to measure success rate

### SmolVLA Model - 200k steps (2026-01-06) [FAILED - 0% SUCCESS]
- **Model**: `danbhf/smolvla_sim_pick_place_200k`
- **Dataset**: `danbhf/sim_pick_place_40ep_rgbd_ee` (RGBD + EE space)
- **Training**: 200k steps on H100
- **Evaluation**: 0% success across ALL checkpoints (10k through 200k)
- **Status**: Complete failure - needs investigation

**Possible causes (unknown which):**
1. SmolVLA training didn't work well, inference code is fine
2. Training worked, inference code is broken
3. Both training and inference are broken

**Complicating factors:**
- Used EE action space (8-dim) with IK conversion - known to have ~28% IK failure rate
- Used RGBD cameras - FOV mismatch issues between scenes (58° vs 52°)
- Models from HuggingFace may differ from original source
- No baseline comparison (SmolVLA hasn't been validated on this task before)

**Next steps:**
- Simplify: Train SmolVLA with joint space (6-dim) + RGB only (no depth)
- Use `danbhf/sim_pick_place_merged_40ep` dataset (correct FOV, joint actions)
- Compare against known-working ACT joint-space baseline (73.3% success)

## Engineering Issues Can Dominate Results

**Key Lesson**: When evaluation results don't match expectations, the problem is usually a bug in the evaluation code, not the model. We spent significant time debugging eval code issues before getting meaningful results.

### Bug Synopsis (2026-01-06)

| Bug | Symptom | Fix |
|-----|---------|-----|
| `IKSolver(verbose=True)` | TypeError - unexpected kwarg | Remove verbose parameter |
| `SO100SimConfig` params | TypeError - fps, randomize_position, randomize_rotation don't exist | Remove params, use `reset_scene(randomize=True)` |
| `sim.reset()` | AttributeError - no reset() method | Use `reset_scene()` instead |
| Scene XML path | FileNotFoundError | Use absolute path `/app/scenes/so101_rgbd.xml` |
| Camera mismatch | KeyError - missing observation.images.X | Detect cameras from policy.config.input_features |
| Postprocessor format | ValueError - expects dict not tensor | Wrap action: `{"action": action}`, unwrap after |
| Processor loading | Actions not denormalized (434mm IK error) | Load from checkpoint with DataProcessorPipeline.from_pretrained() |
| depth_cameras naming | KeyError - overhead_cam_depth not found | Pass base name "overhead_cam", not full "overhead_cam_depth" |
| is_depth check | RGB treated as depth (3.3% success) | Only check `"_depth" in key`, not `key in depth_cameras` |

### Evaluation Progress

| Model | Expected | Remote (Before) | Remote (After Fix) | Local | Notes |
|-------|----------|-----------------|-------------------|-------|-------|
| act_joint_space_90pct | 90% | 23.3% | **73.3%** | 70% | FIXED! |
| act_rgbd_ee_100pct (EE+depth) | 100% | 3.3% | ~3% | 40-60% | EE models underperform |

**Remote eval now matches local!** The joint-space model achieves ~73% on both local and remote.

### Local Testing Results (2026-01-06)

Tested checkpoints locally with train_act.py's run_evaluation (known working code):

**Joint-space model** (`act_20260104_001039/final`): **70% success** ✓

**RGBD+EE models** show overfitting after ~20k steps:
| Run | checkpoint_005000 | checkpoint_010000 | checkpoint_015000 | checkpoint_020000 | checkpoint_025000+ |
|-----|-------------------|-------------------|-------------------|-------------------|-------------------|
| act_20260104_165201 | 20% | 10% | 30% | **40%** | 0% (overfits) |
| act_20260104_221957 | - | 0% | - | **60%** | 40% → 30% |

**Key finding**: The "100pct" in model names refers to using 100% of training data, NOT 100% success rate. Best RGBD+EE achieves ~60% locally.

### Investigation Notes (2026-01-06)

**Image normalization is the key issue:**
- The DataProcessorPipeline normalizes images with MEAN_STD using very small std values (~0.002)
- This causes normalized pixel values to range from -245 to +312 (should be ~-2 to +2)
- Bypassing image normalization (state-only) makes it worse (0% success)
- The model was likely TRAINED with these extreme normalized values

**Key observations from debug session:**
```
Before preprocessing:
  overhead_cam: min/max: 0.000/1.000
  overhead_cam_depth: min/max: 0.173/0.298

After DataProcessorPipeline:
  overhead_cam: min/max: -245.543/312.462
  overhead_cam_depth: min/max: -227.162/10.484
```

**Hypothesis:** The LeRobot dataloader during training may have processed images differently than the simulation outputs. Need to investigate:
1. How images are stored/loaded in the LeRobot dataset
2. Whether there's a transform applied during training that we're missing
3. Whether the original eval used a different preprocessing pipeline

### Key Differences Found (train_act.py vs utils/training.py)

1. **Processor type**: train_act.py uses `make_pre_post_processors(cfg, dataset_stats=stats)` which returns `PolicyProcessorPipeline`, but eval_remote.py was using `DataProcessorPipeline.from_pretrained()` directly
2. **Postprocessor call**: train_act.py calls `postprocessor(action)` with tensor directly, not wrapped in dict
3. **SO100SimConfig**: train_act.py specifies `camera_width=640, camera_height=480` and uses default scene_xml (not explicit path)

**Fixes applied (not yet tested):**
- Updated `eval_remote.py` to use `make_pre_post_processors(policy.config, pretrained_path=...)`
- Updated `utils/training.py` to call `postprocessor(action)` directly (not wrapped)
- Updated `utils/training.py` SO100SimConfig to match train_act.py (added camera_width/height, removed scene_xml)

### ROOT CAUSE FOUND (2026-01-06)

**Missing `policy.reset()` call!** The ACT policy uses action chunking and has internal state that must be reset between episodes.

**Fixes applied:**
1. Added `policy.reset()` in utils/training.py before each episode
2. Changed postprocessor call to `postprocessor(action)` (tensor directly, not wrapped in dict)
3. Use `make_pre_post_processors(policy.config, pretrained_path=...)` instead of `DataProcessorPipeline.from_pretrained()`
4. Removed explicit `scene_xml` path, added `camera_width=640, camera_height=480`

**Verified locally:**
- train_act.py: 63% success
- utils/training.py (fixed): 57% success
- Both now comparable!

## Evaluation Results (2026-01-07)

### Joint-Space RGB Model
**Model**: `danbhf/act_joint_space_90pct`
```
Success Rate: 73.3%
Avg Steps: 182.3
Avg Time: 0.65s

Failure Analysis:
  SUCCESS: 22
  NEVER_PICKED_UP: 2
  DROPPED_DURING_TRANSPORT: 6
  pick_rate: 93.3%
  drop_rate: 21.4%
  avg_steps_success: 139.5
  avg_max_height: 0.205m
```

**Observation**: 6/30 episodes dropped the block during transport. Grip could be tighter.

### EE-Space RGB Model
**Model**: `danbhf/act_ee_space_90pct`
```
Success Rate: 63.3%
Avg Steps: 183.8
Avg Time: 1.57s
IK Failure Rate: 28.14%
Avg IK Error: 8.02mm

Failure Analysis:
  SUCCESS: 19
  NEVER_PICKED_UP: 7
  DROPPED_DURING_TRANSPORT: 2
  MISSED_GOAL: 1
  TIMEOUT: 1
  pick_rate: 76.7%
  drop_rate: 13.0%
  avg_steps_success: 116.6
  avg_max_height: 0.155m
```

**Observation**: High IK failure rate (28%) causes many "never_picked_up" failures - the policy requests positions the IK solver can't reach.

### Comparison: Joint-Space vs EE-Space (RGB only)

| Metric | Joint-Space | EE-Space | Winner |
|--------|-------------|----------|--------|
| Success Rate | **73.3%** | 63.3% | Joint |
| Pick Rate | **93.3%** | 76.7% | Joint |
| Drop Rate | 21.4% | **13.0%** | EE |
| IK Failures | 0% | 28.14% | Joint |
| Avg Steps (success) | 139.5 | **116.6** | EE |

**Conclusion**: Joint-space is more reliable due to avoiding IK failures. EE-space drops less when it does pick up, but fails to pick up more often.

### TODO: Post-Training Grip Modification

**Idea**: Modify trained models to grip the lego block tighter to reduce drops during transport.

Possible approaches:
1. **Action space bias**: Add constant offset to gripper action during inference
2. **Gripper action clipping**: Clip gripper values to ensure minimum grip force
3. **Post-hoc fine-tuning**: Fine-tune on subset of successful grasps with tighter grip
4. **Residual policy**: Train small residual network to adjust grip based on height/velocity

Need to investigate which approach is most practical without full retraining.

---

## Known Issues

### Evaluation Code
- `utils/training.py` `run_evaluation()` has API mismatches:
  - `IKSolver()` doesn't accept `verbose` parameter - FIXED
  - `SO100SimConfig` doesn't have `fps`, `randomize_position`, `randomize_rotation` - FIXED
  - Camera detection from policy config - FIXED
  - Depth camera naming - FIXED
  - RGB vs depth detection - FIXED
- Still investigating why results don't match original experiments
