# MSc Thesis Experiments Summary
## Evaluating Spatial Generalization in Robot Learning Policies

**Student**: [Name]
**Date**: 21 January 2026
**Supervisor Summary**: Progress report on imitation learning experiments

---

## 1. Research Overview

This work investigates the **spatial generalization capabilities** of imitation learning policies for robotic manipulation. The core question: *How well do learned policies transfer to object positions not seen during training?*

**Task**: Pick-and-place manipulation — a robot arm must pick up a block and place it in a bowl.

**Models Evaluated**:
- **ACT (Action Chunking with Transformers)**: A behavior cloning approach that predicts sequences of actions from visual observations
- **Pi0**: A Vision-Language-Action (VLA) model using flow matching for action generation

**Simulation Environment**: MuJoCo physics simulation with SO-100 robot arm, dual cameras (overhead + wrist-mounted)

---

## 2. Key Experiment: ACT Spatial Generalization

### 2.1 Methodology

The ACT policy was trained on demonstrations collected with the block at position (0.217, 0.225) with ±2cm noise. To evaluate generalization, I systematically tested the policy at **990 different block positions** across the robot's workspace:

- **Wide grid**: 5×5 positions spanning x:[0.15, 0.35], y:[-0.15, 0.30] — 500 episodes
- **Fine grid**: 7×7 positions centered on training area — 490 episodes
- **20 episodes per position** with random block orientation

### 2.2 Results

| Distance from Training | Success Rate | Interpretation |
|------------------------|--------------|----------------|
| < 1 cm | **97%** | Within training noise |
| 3 cm | **79%** | Good generalization |
| 5 cm | **64%** | Moderate degradation |
| 7 cm | **25%** | Severe degradation |
| > 10 cm | **0-10%** | Near-complete failure |

**Critical Finding**: The **50% success threshold occurs at approximately 7cm** from the training position. Beyond 10cm, the policy fails almost entirely.

### 2.3 Directional Asymmetry

An unexpected finding: at equal distances, success rates vary dramatically by direction:

| Direction | Distance | Success Rate |
|-----------|----------|--------------|
| Toward bowl (−Y) | 7.5 cm | **0%** |
| Away from bowl (+Y) | 7.5 cm | **50%** |

**Interpretation**: The policy learned to move blocks toward the bowl (−Y direction). When blocks start closer to the bowl than training, the robot overshoots. When blocks start farther, the robot simply moves further in the learned direction.

---

## 3. Pi0 Flow Matching Analysis

### 3.1 Visualization of Action Generation

I developed tools to visualize Pi0's internal **flow matching** process — the 10-step denoising that transforms random noise into coherent actions.

**Key Observations**:
- Standard deviation decreases: 1.0 (pure noise) → 0.5-0.7 (final actions)
- The velocity field actively "steers" toward target actions, not just reducing variance
- Convergence appears chaotic visually due to forward kinematics amplification, but is mathematically smooth

### 3.2 Comparison with ACT

| Aspect | ACT | Pi0 |
|--------|-----|-----|
| Action generation | Deterministic (VAE latent ≈ 0) | Stochastic (flow matching) |
| Chunk size | 100 steps | 50 steps |
| Inference time | ~50ms | ~315ms (10 denoising steps) |
| Architecture | Transformer encoder-decoder | VLM + flow matching head |

---

## 4. Technical Contributions

### 4.1 Visualization Tools Developed

1. **Whisker Visualization**: Real-time display of predicted action trajectories using forward kinematics
2. **Flow Convergence Viewer**: Visualization of Pi0's 10-step denoising process
3. **Spatial Heatmap Generator**: Matplotlib and MuJoCo-based success rate visualization
4. **Scatter Plot Visualization**: Individual episode outcomes with distance rings

### 4.2 Evaluation Infrastructure

- Automated spatial evaluation framework with CSV logging
- Center-out evaluation ordering (test near training first, then expand)
- Combined analysis across 990+ episodes

---

## 5. Implications for Thesis

### 5.1 Behavior Cloning Limitations

The ACT results demonstrate a fundamental limitation of behavior cloning:
- **Memorization over generalization**: The policy learns a mapping specific to training positions
- **No spatial reasoning**: Cannot interpolate or extrapolate to novel configurations
- **Brittle to distribution shift**: Small positional changes cause catastrophic failure

### 5.2 Potential Solutions (Future Work)

1. **Diverse training data**: Collect demonstrations across many positions
2. **Data augmentation**: Synthetically vary object positions during training
3. **VLA models**: Test whether language-conditioned models (like Pi0) generalize better
4. **Explicit spatial encoding**: Add object position to observation space

---

## 6. Files and Artifacts

| File | Description |
|------|-------------|
| `spatial_eval_combined.csv` | 990 episode results with positions and outcomes |
| `spatial_heatmap_*.png` | Grid-based success rate visualizations |
| `spatial_scatter.png` | Individual episode scatter plot |
| `spatial_success_vs_distance.png` | Success rate decay curve |
| `experiments.md` | Detailed experiment log |

---

## 7. Next Steps

1. **Fill coverage gaps**: Running additional evaluations to complete spatial map
2. **Pi0 spatial evaluation**: Compare VLA generalization to ACT
3. **Analyze failure modes**: Categorize why the policy fails at different positions
4. **Training data diversity experiment**: Retrain ACT with varied block positions

---

*Generated: 21 January 2026*
