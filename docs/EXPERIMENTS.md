# Experiments Log

This document tracks experiments conducted with the LeRobot simulation and real robot setup.

---

## 2025-12-29

### Experiment 1: Simple Pick and Place (VR Simulation)

**Dataset:** `danbhf/sim_pick_place_20251229_101340`

**Setup:**
- 20 episodes recorded
- Task: Pick up the Duplo block and place it in the bowl
- Recording method: Physical SO100 leader arm controlling MuJoCo simulation via VR
- FPS: 30
- Randomization: ±4cm position, ±180° rotation on Duplo block

**Hardware:**
- Leader arm: SO100 with STS3250 motors on COM8
- VR headset: Meta Quest (via Quest Link)
- Calibration: File-based JSON (`~/.cache/huggingface/lerobot/calibration/`)

**Notes:**
- First successful 20-episode recording session
- VR controller focus issues encountered (workaround: spacebar to recenter view)
- Sim-to-real transfer previously validated on earlier dataset

**Next Steps:**
- [x] Train ACT policy on this dataset
- Test policy in simulation
- Deploy to real follower robot

---

### Experiment 2: ACT Policy Training

**Dataset:** `danbhf/sim_pick_place_20251229_101340` (20 episodes)

**Training Command:**
```bash
python training/train_act.py danbhf/sim_pick_place_20251229_101340 --steps 5000 --batch_size 4
```

**Status:** Training successfully running on CUDA

**Issues Resolved:**

1. **PyTorch CUDA support:** RTX 5090 (Blackwell architecture) requires PyTorch nightly builds with CUDA 12.8+ support.
   ```bash
   pip uninstall torch torchvision
   pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
   ```

2. **Tensor dimension mismatch:** ACT model expects `observation.state` to be 2D `[batch, state_dim]`, but LeRobot dataset returns 3D `[batch, n_obs_steps, state_dim]`. Fixed by squeezing the temporal dimension when `n_obs_steps=1`.

**Training Progress (5000 steps test run):**
```
Step    100/5000 | Loss: 10.9634
Step    500/5000 | Loss: 3.0496
Step   1000/5000 | Loss: 2.3347
Step   1500/5000 | Loss: 1.8772
Step   1800/5000 | Loss: 1.7787
```

**Full Training (50000 steps):**
- Model: `outputs/train/act_20251229_111846/final`
- Checkpoints saved every 5000 steps
- WandB logging enabled
- Final model size: ~200MB

**Notes:**
- Loss decreasing steadily - model is learning
- Using both wrist_cam and overhead_cam as input
- Chunk size: 100 (predicts 100 future actions)

---

### Experiment 2b: Additional Recording Session (Lincoln)

**Dataset:** `danbhf/sim_pick_place_20251229_144730`

**Setup:**
- 20 episodes recorded by Lincoln
- Task: Pick up the Duplo block and place it in the bowl
- Same recording setup as Experiment 1
- FPS: 30
- Randomization: ±4cm position, ±180° rotation on Duplo block

**Notes:**
- Second recording session to expand training data
- Combined with Experiment 1 dataset = 40 total episodes

---

### Experiment 3: ACT Policy Evaluation (VR Simulation)

**Model:** `outputs/train/act_20251229_111846/final` (50k steps)

**Evaluation Command:**
```bash
python inference/run_act_sim.py outputs/train/act_20251229_111846/final --no_vr
```

**Setup:**
- Same simulation environment used for recording
- Same randomization: ±4cm position, ±180° rotation
- Max 300 steps per episode (10 seconds at 30fps)
- FPS: 30

**Results (5 episodes):**
```
Episode 1: Task completed at step 92
Episode 2: Task completed at step 120
Episode 3: Task completed at step 110
Episode 4: Task completed at step 102
Episode 5: Timed out after 300 steps

Success rate: 4/5 (80.0%)
Average steps: 144.8
```

**Notes:**
- Policy successfully learned pick-and-place from 20 demonstration episodes
- 80% success rate with full randomization is a strong result
- Failed episode likely due to challenging initial object placement
- Training dataset: `danbhf/sim_pick_place_20251229_101340`

**Next Steps:**
- [ ] Train on combined dataset (40 episodes) for improved robustness
- [ ] Test with different randomization ranges
- [ ] Deploy to real follower robot
- [ ] Try Lincoln's dataset: `danbhf/sim_pick_place_20251229_144730`

---

### Experiment 4: Training with In-Loop Evaluation

**Dataset:** `danbhf/sim_pick_place_20251229_101340` (20 episodes)

**Training Command:**
```bash
python training/train_act.py danbhf/sim_pick_place_20251229_101340 \
    --steps 50000 --eval_episodes 10 --eval_randomize
```

**New Features Added:**
- Simulation evaluation runs after each checkpoint (every 5000 steps)
- Tracks success rate, average steps, and average time per episode
- Logs `eval/success_rate`, `eval/avg_steps`, `eval/avg_time` to WandB
- Enables learning curve visualization (success rate vs training step)

**Inference Improvements:**
- Added timing per episode
- Added optional WandB logging for evaluation runs
- Logs per-episode metrics and summary statistics

**Status:** Training in progress

**Expected Outcome:**
- WandB dashboard showing both loss curves and success rate curves
- Ability to identify optimal checkpoint (not necessarily the final one)
- Better understanding of how many training steps are needed

---
