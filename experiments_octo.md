# Experiment: Fine-tune Octo-Small on SO-101 Sim Pick-and-Place

## Background

Octo is a generalist robot policy pretrained on 800K episodes from Open X-Embodiment.
We evaluate Octo-Small (27M params) as an alternative to ACT-ViT (our current best model).

### Why Octo?
- **Pretrained on diverse manipulation**: 800K episodes, 25+ robot embodiments
- **Proven fine-tuning recipe**: Paper demonstrates fine-tuning with ~100 demos in <5 hours on a single A5000
- **Berkeley Pick-Up precedent**: Joint position control fine-tuning explicitly demonstrated
- **Small model**: 27M params (comparable to our ACT-ViT) — fast inference

### Key Paper Insights (Octo: An Open-Source Generalist Robot Policy, 2024)
- Fine-tuning recipe: ~100 demos, 50K steps, single A5000 GPU
- Berkeley Pick-Up used **joint position control** — direct match to our setup
- Diffusion head outperforms MSE and discretized heads (Table II)
- Wrist camera challenging: "Often finetuning results were stronger with only 3rd person camera"
- Proprio inputs "seemed generally worse" during pretraining (causal confusion)
- Full model fine-tuning recommended over freezing subsets
- Hyperparams: LR 3e-4, warmup 2000, inverse sqrt decay, weight decay 0.1, grad clip 1.0

### Why Delta Joint Actions?
- Octo pretrained on delta end-effector control — delta actions are natural
- Delta representation has ~99.7% lower variance than absolute (from our DeltaActionDataset analysis)
- May improve spatial generalization (position-independent control)
- Note: Our Exp 6 showed delta+ACT chunking = 0%, but Octo uses shorter action horizons (4-step default)

## Dataset

**Base**: `danbhf/sim_pick_place_2pos_220ep_v2` (220 episodes, 2 positions)
**Delta version**: `danbhf/sim_pick_place_2pos_220ep_v2_delta`

Conversion: For each episode:
- `delta[0] = action[0] - state[0]` (first action relative to current state)
- `delta[t] = action[t] - action[t-1]` for t > 0 (subsequent relative to previous)
- Gripper (dim 5) stays absolute

## Model Configuration

- **Base**: Octo-Small (27M params, pretrained `hf://rail-berkeley/octo-small-1.5`)
- **Action head**: Diffusion head, re-initialized for 6-dim delta joint actions
- **Cameras**: overhead_cam (256x256) + wrist_cam (128x128)
- **Proprio**: 6-dim joint state (LowdimObsTokenizer, 256 bins)
- **Language**: Fixed "Pick up the block and place it in the bowl" (t5-base tokenizer)
- **Observation history**: T=2 frames (current + 1 previous)
- **Action horizon**: 4 steps (Octo default) — also test longer horizons

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Learning rate | 3e-4 |
| Warmup steps | 2000 |
| LR schedule | Inverse square root decay |
| Weight decay | 0.1 |
| Gradient clip | 1.0 |
| Batch size | 64 |
| Total steps | 50,000 |
| Optimizer | AdamW |
| Fine-tuning | Full model (not frozen) |

## Variants to Test

| ID | Cameras | Proprio | Notes |
|----|---------|---------|-------|
| A | overhead only (256x256) | No | Paper suggests 3rd-person only often best |
| B | overhead + wrist (256+128) | No | Full camera setup |
| C | overhead only (256x256) | Yes | With proprio input |
| D | overhead + wrist (256+128) | Yes | Full setup with proprio |

## Evaluation

- 50 episodes at training positions (same as ACT-ViT baseline)
- Compare to ACT-ViT Exp 14b: **85% success**
- Also test spatial generalization on 5x5 grid

## Results

*(To be filled after training)*

| Variant | Training Pos Success | Notes |
|---------|---------------------|-------|
| A | | |
| B | | |
| C | | |
| D | | |
