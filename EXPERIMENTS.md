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

### SmolVLA Model - Joint Space (2026-01-08) [FAILED - 0% SUCCESS]
- **Model**: `danbhf/smolvla_so101_200k`
- **Dataset**: `danbhf/sim_pick_place_merged_40ep` (RGB + joint space)
- **Training**: 200k steps on H100
- **Evaluation** (2026-01-10): 0% success across ALL checkpoints (2k through 200k)
- **Status**: Complete failure - same as EE-space version

**Evaluation Results:**
| Checkpoint | Success Rate |
|------------|-------------|
| checkpoint_002000 | 0.0% |
| checkpoint_004000 | 0.0% |
| checkpoint_006000 | 0.0% |
| checkpoint_008000 | 0.0% |
| checkpoint_010000 | 0.0% |
| checkpoint_012000 | 0.0% |
| checkpoint_014000 | 0.0% |
| checkpoint_016000 | 0.0% |
| checkpoint_018000 | 0.0% |
| checkpoint_020000 | 0.0% |
| final | 0.0% |

**Failure mode**: All episodes result in "never_picked_up" - the robot doesn't attempt to grasp.

**Baseline comparison**: ACT joint-space model achieves 80% success on same task, confirming eval infrastructure works.

**Conclusion**: SmolVLA training is not working for this task. Issue is NOT the action space (tried both EE and joint), NOT the dataset format. Likely a fundamental issue with SmolVLA training or inference code.

### Pi0 Model - 40k Training Attempt (2026-01-11) [FAILED - DISK FULL x2]
- **Attempted**: 40k steps to see if longer training helps
- **Failed at**: Step 10000 with "No space left on device"
- **Root cause**: vast.ai instance had NO usable large disk - only 50GB overlay

**What went wrong:**
- `df -h` showed 2TB disks, but they were read-only nvidia driver mounts
- `/workspace` was on the same 50GB overlay as `/`, not a separate disk
- Symlink trick didn't help because there was no actual large disk to symlink to

**vast.ai Disk Verification Checklist (DO THIS BEFORE TRAINING):**
```bash
# 1. Check total disk space
df -h

# 2. CRITICAL: Verify /workspace (or /data) is on a SEPARATE large disk
df -h /workspace
# BAD: Shows "overlay" filesystem → it's on the small Docker disk
# GOOD: Shows "/dev/sda1" or similar → it's a real separate disk

# 3. If /workspace is on overlay, look for actual large disk:
mount | grep -v "tmpfs\|proc\|sys\|nvidia\|loop"
# Look for a large ext4 mount you can write to

# 4. Test write access to the large disk:
touch /path/to/large/disk/test && rm /path/to/large/disk/test

# 5. Estimate storage needed:
#    - Pi0 checkpoints: ~13GB each (train_state + params + assets)
#    - 40k steps with save_interval=5000 → 8 checkpoints → ~100GB needed
#    - 40k steps with save_interval=10000 → 4 checkpoints → ~50GB needed
```

**When renting vast.ai instance:**
- Look for "Disk Space" in the listing - ensure it's >100GB
- Prefer instances that explicitly show disk mounted at `/workspace` or `/data`
- After connecting, ALWAYS run the verification checklist above

**Training command (once disk is verified):**
```bash
cd /app/openpi

# Symlink checkpoints to large disk (replace /data with actual mount point)
rm -rf checkpoints && ln -s /data/checkpoints checkpoints
mkdir -p /data/checkpoints

# Run training
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_so101 \
    --exp-name=pi0_so101_40k \
    --num-train-steps=40000 \
    --save-interval=10000 \
    --overwrite
```

### ACT Model - 157 Episodes (2026-01-18) [SUCCESS - 100% at 45k]
- **Dataset**: `danbhf/sim_pick_place_157ep` (157 episodes, joint space)
- **Training**: 50k steps, ~2 hours locally on RTX 5090
- **Best checkpoint**: `checkpoint_045000` - **100% success** (10/10 episodes)
- **Final checkpoint**: `checkpoint_050000` - 80% success (dropped during transport)
- **Best loss**: 0.0633
- **WandB**: https://wandb.ai/bryars-bryars/lerobot-thesis/runs/17akuj2p

**Checkpoint performance:**
| Checkpoint | Success Rate | Notes |
|------------|-------------|-------|
| 040000 | 60% | 3 dropped, 1 missed |
| 045000 | **100%** | Perfect run! |
| 050000 | 80% | 2 dropped (slight overfit) |

**Key finding**: 157 episodes (4x more data) improved ACT from 73% → 100% success.
The model peaks around 45k steps, slight overfitting after that.

---

### Pi0 Model - LeRobot PyTorch (2026-01-18/19) [SUCCESS - TRAINED]
- **Docker image**: `aerdanielbryars101/lerobot-pi0:latest`
- **Dataset**: `danbhf/sim_pick_place_157ep` (157 episodes, joint space)
- **Training**: vast.ai H100, ~1.3s/step with gradient checkpointing

**Models trained:**

| Model | Steps | Final Loss | Training Time | HuggingFace |
|-------|-------|------------|---------------|-------------|
| Pi0 5K | 5,000 | 0.032 | ~1.5 hours | `danbhf/pi0_so101_lerobot` |
| Pi0 20K | 20,000 | ~0.025 | ~7 hours | `danbhf/pi0_so101_lerobot_20k` |

**Training metrics (20K run):**
- Loss: Started ~0.25, dropped to ~0.05 by 5K, settled ~0.025 by 20K
- grad_norm: Stable around 1.2-1.4 throughout
- Learning rate: Decayed from 2.5e-05 to 2.5e-06
- Loss curve was noisy after initial drop (normal for small dataset fine-tuning)

**Key observations:**
- 20K model achieved lower loss than 5K (0.025 vs 0.032)
- Training stable throughout, no exploding gradients
- Model size: 7.01GB (safetensors format)
- TODO: Run inference evaluation to compare 5K vs 20K success rates

**Training command:**
```bash
DATASET=danbhf/sim_pick_place_157ep STEPS=20000 BATCH_SIZE=32 \
    JOB_NAME=pi0_so101_20k REPO_ID=danbhf/pi0_so101_lerobot_20k bash /app/train.sh
```

**Inference test (2026-01-19) - SUCCESS (loads, but 0% eval):**
Both models load and produce valid actions on H100, but simulation evaluation shows 0% success:

| Model | Action Range | Inference Speed |
|-------|-------------|-----------------|
| 5K (`danbhf/pi0_so101_lerobot`) | -0.87 to 0.92 | 824 Hz (1.2ms) |
| 20K (`danbhf/pi0_so101_lerobot_20k`) | -1.36 to 1.40 | 802 Hz (1.2ms) |

- 20K model has larger action range - may indicate more decisive actions
- Both models extremely fast on H100 (~800 Hz)
- Tied weights warning (`embed_tokens.weight`) is safe to ignore
**Simulation evaluation (2026-01-19) - 0% SUCCESS:**
- Both 5K and 20K models fail all episodes in simulation
- Models load correctly and produce actions at 800+ Hz
- Actions appear to be in reasonable range but robot does not complete task

**Whisker visualization diagnosis (2026-01-19):**
- Used `visualize_whiskers_pi0.py` to see what model is predicting
- **Result**: Pi0 predicts chaotic, random movements near starting position
- Whiskers show tiny perturbations, NOT purposeful trajectories toward block
- Compare to ACT which shows smooth arcs toward the target
- **Conclusion**: Pi0 has NOT learned the task - outputs are essentially noise
- See screenshot: `CameraShapshots/Pi0 Fail does not move.png`

**Root cause hypothesis:**
1. **Insufficient data**: 157 episodes may not be enough for Pi0's 3B parameters
2. **Missing preprocessing**: Pi0 may need specific image normalization we're not applying
3. **Action space mismatch**: Pi0 base expects different action format than our joint positions
4. **Training issue**: The remote training may have had issues we didn't catch

---

### Whisker Visualization for Policy Debugging (2026-01-19)

**Problem:** Pi0 models produce valid-looking actions but achieve 0% success in simulation.
Need to understand what the model is actually predicting at each timestep.

**Idea:** Visualize predicted action trajectories as "whiskers" - faint lines emanating from
the robot end-effector showing the predicted future path at each timestep.

**Why this is useful:**
1. **Policy uncertainty made visible** - See where the policy is confident vs hesitant
2. **Action-chunk structure observable** - See chunk smoothness, mode collapse, multi-modal futures
3. **Sim-to-real diagnostics** - Wide whiskers = brittle policy, narrow bundles = robust
4. **Compare policy heads** - ACT vs Pi0 diffusion vs Pi0 flow matching

**Implementation:** `scripts/tools/visualize_whiskers.py`
- Forward-simulate action chunks in a copy of MuJoCo state
- Render predicted EE positions as faded capsule segments
- Works with ACT (known working) first, then extend to Pi0

**Academic framing:** "3D uncertainty-aware policy rollout visualization for VLA models"
- Reveals temporal confidence structure of learned policies
- Highlights multi-modal futures under partial observability
- Provides qualitative safety and robustness diagnostics

**Status:** Working for ACT, extended to Pi0

**Scripts:**
- `scripts/tools/visualize_whiskers_act.py` - ACT policy visualization
- `scripts/tools/visualize_whiskers_pi0.py` - Pi0 policy visualization

**Features:**
- Green whiskers: Current prediction (full action chunk forward-simulated)
- Blue ghost trails: Past predictions (fading with age)
- Orange line: Actual path robot took
- Interactive controls: SPACE pause, LEFT/RIGHT step through history
- History playback: Rewind to see exact state at each timestep

**Key findings from ACT visualization:**
- ACT predicts smooth, purposeful trajectories toward the target
- Predictions align well with actual path taken
- Model constantly re-predicts (visual servoing with learned intent)
- Only first action is executed, rest shows "plan"

**Key findings from Pi0 visualization (2026-01-19):**
- Pi0 whiskers show chaotic, random movements near starting position
- Model is NOT predicting purposeful trajectories toward block
- Predictions are essentially noise - small random perturbations
- This explains 0% success rate: model hasn't learned the task at all

**Windows Pi0 loading fix:**
- Must set `PYTHONIOENCODING=utf-8` for model weights to load
- LeRobot Pi0 code prints checkmark character (✓) that Windows console can't encode
- Without this, model returns random weights and silently fails

**Progress made (2026-01-18):**
- Switched from JAX/openpi to LeRobot PyTorch (JAX crashed RTX 5090)
- Fixed Windows path bug in LeRobot (backslash in HF repo IDs)
- Fixed missing image stats in dataset
- Confirmed tied weights warning is safe to ignore
- Local training works with batch_size=2, gradient_checkpointing on 32GB VRAM

---

### Pi0 Model - JAX/openpi SO-101 with 157 Episodes (2026-01-18) [ABANDONED]
- **Model**: `danbhf/pi0_so101_pick_place_157`
- **Dataset**: `danbhf/sim_pick_place_157ep_pi0` (157 episodes, normalized gripper [0-1])
- **Training**: 5k steps on H100 using openpi JAX implementation
- **Docker**: `aerdanielbryars101/openpi-training:latest`
- **Checkpoint**: 4999 (final)
- **Size**: ~43GB
- **Status**: Training complete, uploaded to HF, awaiting evaluation

**Key improvements over previous attempt:**
1. **More data**: 157 episodes vs 40 episodes (nearly 4x more)
2. **Normalized gripper**: Converted gripper from [0-97] to [0-1] range for Pi0 compatibility
3. **Delta actions enabled**: Using `use_delta_actions=True` in config
4. **Fixed action padding**: Actions padded from 6→32 dims (matching Pi0 base weights)

**Dataset conversion:**
- Created `scripts/tools/convert_dataset_pi0.py` to normalize gripper values
- Original: `danbhf/sim_pick_place_157ep` → Converted: `danbhf/sim_pick_place_157ep_pi0`

**Training command:**
```bash
cd /app/openpi
uv run scripts/compute_norm_stats.py --config-name=pi0_so101
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_so101 \
    --exp-name=so101_pick_place_157 \
    --num-train-steps=5000 \
    --batch-size=16 \
    --save-interval=1000 \
    --overwrite
```

**Local inference setup (2026-01-18):**
- Windows: openpi has complex JAX dependencies that conflict with existing packages
- Solution: Use WSL2 with GPU passthrough (RTX 5090 works with CUDA 13.1)
- WSL setup:
  ```bash
  # Clone openpi and sync deps
  git clone https://github.com/Physical-Intelligence/openpi.git ~/openpi
  cd ~/openpi
  uv sync

  # Patch for SO-101 config
  uv run python /mnt/e/git/ai/lerobot-thesis/scripts/pi0/patch_openpi_so101.py \
      --config-path ~/openpi/src/openpi/training/config.py

  # CRITICAL: openpi bundles old lerobot 0.1.0 - upgrade for v3 dataset support
  uv pip install "lerobot>=0.4.0" --upgrade

  # Run eval
  PYTHONPATH="/mnt/e/git/ai/lerobot-thesis/src:$PYTHONPATH" uv run python \
      /mnt/e/git/ai/lerobot-thesis/scripts/pi0/eval_pi0.py \
      --model danbhf/pi0_so101_pick_place_157 --checkpoint 4999 --episodes 5 --visualize
  ```

**Known issues:**
- openpi pins lerobot==0.1.0 which lacks `lerobot.cameras` module and v3 dataset support
- Must upgrade to lerobot>=0.4.0 for both training (Docker) and inference (WSL)
- Docker image patches this in Dockerfile
- `lerobot_robot_sim` import path changed: `lerobot.cameras` → `lerobot.cameras.configs`
- `lerobot.common.datasets` doesn't exist in 0.4.x - use `lerobot_compat.py` shim for inference

**RTX 5090 + JAX - ABANDONED (2026-01-18):**
- First run with JAX/cuDNN autotuning caused full system crash requiring power cycle
- Disabling autotuning (`XLA_FLAGS="--xla_gpu_autotune_level=0"`) prevents crash but too slow (3.3 Hz)
- Second crash during inference even with autotuning enabled
- **Conclusion**: JAX + RTX 5090 + WSL2 is unstable. Switching to LeRobot PyTorch implementation.

**Pivot to LeRobot PyTorch Pi0:**
- JAX scripts moved to `scripts/pi0_jax/` (archived)
- Using LeRobot PyTorch implementation instead: https://github.com/huggingface/lerobot
- Base models: `lerobot/pi0_base`, `lerobot/pi05_base`
- Will retrain from scratch rather than convert JAX weights
- PyTorch works natively on Windows with RTX 5090

**LeRobot Pi0 Setup (Local Windows):**
```bash
# Install Pi0 dependencies
pip install "lerobot[pi]@git+https://github.com/huggingface/lerobot.git"

# Reinstall PyTorch nightly for RTX 5090 support (lerobot overwrites it)
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall

# Train Pi0
lerobot-train --dataset.repo_id=danbhf/sim_pick_place_157ep --policy.type=pi0 \
    --policy.pretrained_path=lerobot/pi0_base --policy.repo_id=danbhf/pi0_so101_test \
    --output_dir=./outputs/pi0_test --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 --policy.device=cuda --steps=5000 --batch_size=16
```

**LeRobot Pi0 Docker (vast.ai):**
```bash
# Build and push
docker build -t aerdanielbryars101/lerobot-pi0:latest -f scripts/pi0/Dockerfile .
docker push aerdanielbryars101/lerobot-pi0:latest

# On vast.ai instance:
huggingface-cli login
DATASET=danbhf/sim_pick_place_157ep STEPS=5000 BATCH_SIZE=32 \
    JOB_NAME=pi0_so101_5k REPO_ID=danbhf/pi0_so101_lerobot bash /app/train.sh
```

**Key differences from JAX/openpi:**
- PyTorch instead of JAX - native Windows + CUDA support
- Uses `lerobot.scripts.train` instead of openpi's train.py
- Model format is PyTorch .safetensors, not JAX params
- Should run at 20+ Hz on RTX 5090

**Windows path bug fix (https://github.com/huggingface/lerobot/issues/2552):**
LeRobot converts `/` to `\` in HuggingFace repo IDs on Windows, breaking downloads.
Patch `venv/Lib/site-packages/lerobot/policies/factory.py` in 3 places:
```python
# Line ~244 (after "if pretrained_path:")
pretrained_path = str(pretrained_path).replace("\\", "/")

# Line ~491 (in make_policy, before kwargs assignment)
kwargs["pretrained_name_or_path"] = str(cfg.pretrained_path).replace("\\", "/")

# Line ~505 (in PEFT section)
peft_pretrained_path = str(cfg.pretrained_path).replace("\\", "/")
```

**Missing embed_tokens.weight warning - SAFE TO IGNORE:**
When loading Pi0 base, you'll see:
```
Warning: Could not remap state dict keys: Missing key(s) in state_dict:
"model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
```
This is a **tied weight** - PaliGemma has `tie_word_embeddings: True`, meaning `embed_tokens`
and `lm_head` share the same tensor. Only `lm_head.weight` is saved in checkpoint, and
at runtime they point to the same memory. No weights are actually missing.

**Dataset stats fix:**
LeRobot v3 datasets may be missing image stats in `meta/stats.json`. Fix with:
```bash
python scripts/tools/fix_dataset_stats.py danbhf/sim_pick_place_157ep
```

**IMPORTANT: Version tag caching issue:**
LeRobot downloads datasets by version tag (e.g., `v3.0`). If you fix stats.json on HuggingFace
but the tag still points to the old commit, LeRobot will download the old version!

Fix: Delete and recreate the tag after fixing stats:
```bash
python -c "from huggingface_hub import HfApi; api = HfApi(); api.delete_tag('danbhf/sim_pick_place_157ep', tag='v3.0', repo_type='dataset')"
python -c "from huggingface_hub import HfApi; api = HfApi(); api.create_tag('danbhf/sim_pick_place_157ep', tag='v3.0', repo_type='dataset')"
```

Then on vast.ai, clear LeRobot's cache:
```bash
rm -rf /root/.cache/huggingface/lerobot/danbhf/sim_pick_place_157ep
```

**Torchcodec ffmpeg error:**
If you see `Could not load libtorchcodec` errors, set pyav backend:
```bash
export VIDEO_BACKEND=pyav
```
Or add to Dockerfile: `ENV VIDEO_BACKEND=pyav`

**Camera configuration approach:**
- Pi0 base expects 3 cameras (`base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`)
- Our dataset has 2 cameras (`overhead_cam`, `wrist_cam`)
- Solution: Configure 2-camera variant and load weights with `strict=False`
- Camera views are "just a list" with shared encoder, so camera count is flexible
- Missing per-camera embeddings/projections get randomly initialized during fine-tune
- This is fine for fine-tuning - we're re-learning the adapter that maps 2 views into shared latent space

**Experiments to try:**
1. 2-camera config, load base weights non-strict, fine-tune all
2. 2-camera config, freeze vision backbone first few k steps, then unfreeze
3. If mismatch issues: duplicate one camera view to satisfy 3-camera interface

**CRITICAL: uv run reverts lerobot version!**
- `uv run` re-syncs dependencies from pyproject.toml before running
- This REVERTS any `uv pip install` upgrades back to the pinned version
- Solution: Run the fix script OR manually edit openpi's pyproject.toml:
  ```bash
  cd ~/openpi
  bash /mnt/e/git/ai/lerobot-thesis/scripts/pi0/fix_openpi_lerobot.sh
  ```

**Manual fix (if script not available):**
```bash
cd ~/openpi

# 1. Change lerobot pin to >=0.4.0
sed -i 's/"lerobot",/"lerobot>=0.4.0",/' pyproject.toml

# 2. Remove lerobot git source (conflicts with PyPI)
sed -i '/^lerobot = { git/d' pyproject.toml

# 3. Remove numpy<2.0.0 constraint (needed for newer lerobot)
sed -i 's/"numpy>=1.22.4,<2.0.0"/"numpy>=1.22.4"/' pyproject.toml
sed -i 's/"numpy>=1.22.4,<2.0.0"/"numpy>=1.22.4"/' packages/openpi-client/pyproject.toml

# 4. Remove rlds dependency group (conflicts with numpy>=2)
sed -i '/^rlds = \[/,/^\]/d' pyproject.toml
sed -i '/^dlimp = { git/d' pyproject.toml

# 5. Re-sync dependencies
uv sync
```

After running, verify: `uv run python -c "import lerobot; print(lerobot.__version__)"` → should show 0.4.x

- This is the same fix applied in the Docker image (see `scripts/pi0/Dockerfile`)

---

### Pi0 Model - SO-101 (2026-01-10) [FAILED - 0% SUCCESS]
- **Model**: `danbhf/pi0_so101_20260110`
- **Dataset**: `danbhf/sim_pick_place_merged_40ep` (RGB + joint space)
- **Training**: 20k steps on H100 using openpi JAX implementation
- **Docker**: `aerdanielbryars101/openpi-training:latest`
- **Evaluation** (2026-01-10): 0% success (0/10 episodes)
- **Status**: Complete failure - same pattern as SmolVLA

**Failure mode**: Robot moves in correct general direction but **gripper never closes**:
- Gripper values stay at 15-30 (open), should go to ~0 to grasp
- Arm rotates anti-clockwise (towards block) but never picks up

**Debug findings (2026-01-10):**
1. **Fixed normalization bug**: Original eval converted actions incorrectly (values -2455 to +3367). Fixed to use normalized values directly (range -70 to +60).
2. **Gripper stays open**: Both checkpoint 5000 and 19999 show same behavior - arm moves but gripper never closes
3. **Model learned partial behavior**: Robot moves towards block area but grasping action not learned

**Research on Pi0 training requirements:**
- Typical successful fine-tuning uses **50-100+ episodes** (we only have 40)
- Pi0 uses **delta actions** by default (relative to first state in chunk)
- Early checkpoints (5k steps) sometimes perform better than late ones due to overfitting

**Baseline comparison**: ACT joint-space model achieves 80% success on same task with same 40 episodes.

**Gripper analysis:**
The gripper is encoded differently from other joints:
- Other joints: [-100, 100] normalized range
- Gripper: [0, 100] range where 0=closed, 100=open

Training data gripper stats:
- Range: 0 (closed) to 76 (open)
- Mean: 30.2, Std: 20.4

Model output gripper: stays at 15-30 (near the mean!)
- The model learned to output the **average** gripper position
- It did NOT learn the temporal sequence of when to close the gripper
- This is classic "mean regression" behavior from insufficient data

**Checkpoint comparison:**
| Checkpoint | Arm Movement | Gripper | Result |
|------------|--------------|---------|--------|
| 5000 | None | Open (~18) | 0/1 timeout |
| 10000 | Not recorded | - | - |
| 15000 | Not recorded | - | - |
| 19999 | Anti-clockwise towards block | Open (15-30) | 0/1 timeout |

Recordings: `pi0_recording_5000.json`, `pi0_recording_fixed.json` (19999)

**Would longer training help?**
Unlikely to help significantly because:
1. Model already converged to outputting mean values
2. More steps would just reinforce this behavior
3. Need more diverse training data showing gripper close/open transitions

**Conclusion**: VLA models (Pi0, SmolVLA) may need more training data than ACT for this task. The model learned some movement behavior but not the grasping action. With only 40 episodes, the model learns to output average values rather than learning conditional behavior.


### SmolVLA Training Performance Notes (2026-01-08)

**Expected training times (from HuggingFace/LeRobot docs):**
- A100: ~4-5 hours for 20k steps (default fine-tune recipe)
- H100: ~2-3.5 hours for 20k steps (best case, compute-bound)

**Observed on H100 80GB:**
- ~2.25s/iteration with batch_size 128
- GPU utilization bounces 0% → 90-100% → 0% (classic data loading bottleneck)
- 100k steps ≈ 62 hours at this rate

**Known issue:** SmolVLA training often becomes dataloader/IO-bound rather than compute-bound.
- LeRobot GitHub issue shows H100 example: update step ~0.3s but data fetch spikes to ~6s
- This makes the H100's extra compute power irrelevant

**Attempted mitigations:**
- `--cache_dataset`: Loads all data into RAM (~45GB), but sets num_workers=0 (single-threaded batch prep)
- Without cache: default num_workers=4, still slow
- Increasing num_workers to 16 may help

**Root cause:** The bottleneck is likely CPU-side batch preparation (image preprocessing, tokenization) not raw data loading.

**Solution found:** Use more DataLoader workers! With 256 CPU cores available:
- 4 workers (default): ~4 it/s
- 64 workers: ~20 it/s (but unstable, drops to ~3 it/s)
- 32 workers: testing...
- 16 workers: testing...

### Pi0.5 Training Notes

**Important advice (from friend with experience):**
> "For Pi0.5, use directly the implementation of Physical Intelligence in JAX. The one in LeRobot was repeatedly reported as not working (which is the case for a lot of HF models btw)"

**Action:** Use the original JAX implementation from Physical Intelligence, NOT the LeRobot/HuggingFace port.

### Openpi Integration (2026-01-08)

Created integration for Physical Intelligence's openpi (Pi0/Pi0.5) framework:

**Files created:**
- `scripts/pi0/so101_policy.py` - SO-101 robot input/output transforms
- `scripts/pi0/convert_lerobot_to_openpi.py` - LeRobot to RLDS format converter
- `scripts/pi0/so101_config.py` - Training configs for SO-101 robot
- `scripts/pi0/Dockerfile` - Docker image for vast.ai training
- `tests/test_openpi_converter.py` - Unit tests (15 tests passing)

**Docker image:** `aerdanielbryars101/openpi-training:latest`

**Key learnings from Docker setup:**
1. openpi requires Python 3.11+ (original had 3.10)
2. pip hangs on openpi's complex dependencies - use `uv` instead
3. Folder naming collision: our `scripts/openpi` shadowed the real `openpi` package
4. Renamed to `scripts/pi0` to avoid namespace conflict
5. openpi imports `pytest` in non-test code (gemma_pytorch.py) - must install pytest

**Output format (RLDS compatible):**
- `observation/state`: Joint positions [N, 6] float32
- `observation/image`: Main camera [N, H, W, 3] uint8
- `observation/wrist_image`: Wrist camera [N, H, W, 3] uint8 (optional)
- `action`: Robot actions [N, 6] float32
- `language_instruction`: Task description string
- `is_first/is_last/is_terminal`: Episode boundaries

### Pi0 Training Test (2026-01-08)

**Successfully ran test training on vast.ai H100:**

```bash
# 1. Compute normalization stats (required)
cd /app/openpi && uv run scripts/compute_norm_stats.py --config-name=pi0_aloha_sim

# 2. Run training
cd /app/openpi && XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    uv run scripts/train.py pi0_aloha_sim \
    --exp-name=test_run \
    --num-train-steps=100 \
    --overwrite
```

**Results (100 test steps):**
- Initial loss: 0.3614
- grad_norm: 4.8452
- Training speed: ~2.33s/step
- Memory warning: "Can't reduce memory use below 21.81GiB by rematerialization"
- Checkpoint saved successfully to `/app/openpi/checkpoints/pi0_aloha_sim/test_run/99`
- WandB logging worked: https://wandb.ai/bryars-bryars/openpi

**Key insight - Normalization stats:**
- openpi requires pre-computed mean/std for all features
- Run `compute_norm_stats.py` before training
- Stats normalize inputs to ~[-1, 1] range for stable training
- Model output is denormalized back to real units at inference

**TODO:**
- Create proper SO-101 config (not using aloha_sim hack)
- Compute norm stats for `danbhf/sim_pick_place_merged_40ep`
- Run full training run on SO-101 dataset
- Compare Pi0 vs SmolVLA on same task

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

---

## System Notes

### RTX 5090 Driver (2026-01-19)
- **Current driver**: 591.44 (December 4th, 2025)
- **Available update**: 591.74 (January 5th, 2026)
- **Issue**: System crashes/reboots during training (happened twice during ACT training)
- **Plan**: If another crash occurs, upgrade to 591.74
- **Note**: PyTorch nightly cu128 required for RTX 5090 support

---

## Real Robot Datasets

| Dataset | Task | Episodes | Date | Notes |
|---------|------|----------|------|-------|
| `danbhf/real_pick_place_0001` | Pick up the block and place it in the bowl | - | 2026-01-17 | First real robot recording |

**Recording setup (2026-01-17):**
- Scripts in `scripts/real_robot/`
- Hardware config in `scripts/real_robot/config.json` (COM7/COM8, cameras 0/2)
- Recording controls: Right arrow = save episode, Left arrow = discard, Escape = stop
- Camera names match Pi0: `base_0_rgb`, `left_wrist_0_rgb`

## Vast.ai Storage Notes

**Disk quota issues:**
- Default 50GB is NOT enough for Pi0 training
- Updated `find_and_rent.py` default to 150GB
- For 20k steps with save_interval=5000: 4 checkpoints × ~40GB = ~160GB needed
- Use `--disk 300` for safety

**Persistent storage (added 2026-01-18):**
- `find_and_rent.py` now supports persistent volumes that survive instance termination
- `--list-storage` - show existing volumes
- `--create-storage 500` - create 500GB persistent volume
- `--storage <id>` - attach existing volume (mounted at `/storage`)
- Symlink checkpoints: `ln -s /storage/checkpoints /app/openpi/checkpoints`

## Side Projects

### HIL-SERL Exploration (2026-01-11)
- **Location**: `E:\git\ai\lerobot-training-fun`
- **Reference**: PDF tutorial from `docs/2510.12403v1.pdf` (pages 23-33)
- **Goal**: Learn HIL-SERL (Human-in-the-Loop Sample Efficient Robot RL) workflow

**What is HIL-SERL?**
- Real-world RL training that achieves 99%+ success in 1-2 hours
- Combines SAC + RLPD + human interventions during training
- Uses a **learned reward classifier** instead of hand-crafted rewards

**Key insight - Reward Classifier:**
- Binary classifier: "Does this camera frame look like task success?"
- Trained on labeled success/failure frames from demonstrations
- Replaces manual reward engineering with visual success detection
- Could be reusable as building block for other training pipelines

**Files structure:**
```
lerobot-training-fun/
├── reference/ch3/          # Original tutorial code (read-only)
├── hilserl/                # Working copies to modify
│   ├── 01_reward_classifier.py
│   ├── 02_actor.py
│   ├── 03_learner.py
│   └── 04_hil_serl.py
└── requirements.txt
```

**Progress:**
- [x] Read tutorial from PDF
- [x] Set up folder structure
- [x] Created requirements.txt
- [ ] Train reward classifier on example dataset
- [ ] Train reward classifier on own dataset
- [ ] Run full HIL-SERL training loop

**Simulation limitations:**
- Gripper friction is unrealistically high in the current MuJoCo setup
- Some grips succeed in simulation that would likely fail on real hardware
- This may cause sim-to-real transfer issues - policies might learn "lazy" grips that don't work physically
- TODO: Tune friction parameters or add grip quality filtering to training data

**Data augmentation strategy:**
The main goal is to generate more training data for VLA models (Pi0, SmolVLA) which seem to need more than 40 episodes:
1. Use trained ACT policy (73% success) to run episodes automatically in simulation
2. Train a visual success classifier (HIL-SERL style) to filter the generated episodes
3. Keep only successful episodes to augment the training dataset
4. Retrain VLA models on the larger dataset

This is a form of "policy distillation" - using a smaller, task-specific model (ACT) to generate data for larger generalist models.

**Potential applications:**
- Learned reward classifier as success detector for other RL/IL experiments
- Visual "task done" detector for auto-stopping episodes
- Data quality filtering for auto-generated episodes
- Scalable data collection without manual teleoperation
