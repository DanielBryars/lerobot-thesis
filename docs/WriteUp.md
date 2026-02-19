# Can spatial generalisation of low-cost behaviour-cloned manipulation policies be significantly improved through (a) architectural changes, (b) data diversity, and (c) hierarchical decomposition — without increasing model scale?

## Background

### LLMs and the AI Renaissance

From the "Attention is All You Need" breakthrough (Vaswani et al., 2017), transformer-based architectures have spawned research into the emergent behaviour of Large Language Models. Along with Chain-of-Thought prompting (Wei et al., 2022), LLMs are now used in a variety of tasks which are transforming society in many areas. In addition to the transformer, two other ingredients are required to derive the latent knowledge and reasoning gains we have seen: lots of data and lots of compute (Kaplan et al., 2020). The volume of AI-related scientific publications has grown exponentially, with submissions to conferences such as NeurIPS and ICML roughly doubling every two to three years (Sevilla et al., 2022).

The increased interest and investment in AI — global private AI investment exceeded $90 billion in 2024 (Stanford HAI, 2024) — has led to the significant availability of GPU compute, both cloud-based (e.g. vast.ai, Lambda Labs, RunPod) and in consumer hardware (e.g. NVIDIA RTX 5090 with 32GB VRAM). Open models, through platforms such as HuggingFace and Kaggle, have improved the availability of foundation models (pretrained models) which can be finetuned. These websites also provide a nexus for datasets and a community to drive progress. Experiment tracking tooling such as Weights & Biases, and advances in frameworks such as PyTorch (Paszke et al., 2019) help democratise access to AI research and development, providing access to independent researchers as well as universities and large companies.

### Combining Foundation Models for Robotics

Combining vision models and language models into a VLM (such as SigLIP (Zhai et al., 2023) and Gemma, see PaliGemma (Beyer et al., 2024)) allows interesting problems to be solved, such as image captioning and visual question answering (Zhou, Liu and Gao, 2024).

Recently there has been interest in combining Vision, Language and "Action Heads" — so-called VLAs (Vision-Language-Action models) to perform generalised robotic control - introduced by RT-2 (Brohan et al., 2023), with examples such as Octo (Team et al., 2024), OpenVLA (Kim et al., 2024), and Pi0 (Black et al., 2024). The theory is that the vision model provides the features to recognise objects in the world, the LLM provides a semantic link between those objects and language, and through training with large numbers of robot demonstration episodes we can build a model which generalises over robotic tasks. Essentially: leverage the existing investment and bring the advances in LLMs and vision models into the realm of robots. 

### Problem with Current VLAs for Experimentation under constrained resources

OpenVLA (Kim et al., 2024), Pi0, and Pi0.5 (Physical Intelligence, 2025) are still quite large models for a self-funded individual to experiment with. For example, a Pi0 checkpoint is 13GB and training requires at least 48GB of VRAM. Inference, even on a 32GB RTX 5090, runs at best at 4Hz. Zero-shot performance is weak, so some finetuning is required to get any results. I tried these models and found training time and resource requirements unwieldy.

Not everything is open source. Datasets are generally proprietary — for example, the sub-task annotations used to train Pi0 and Pi0.5 — because collecting robot data is expensive (Mandlekar et al., 2018).

### Low-Cost Approaches

The paper "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (Zhao et al., 2023) provides a way to use Behavioural Cloning (Pomerleau, 1991) to teach a model to perform a task such as tying shoelaces or picking up a block. ACT (Action Chunking with Transformers) learns trajectories, predicting the next n chunks from the current observational state, it handles multi-modal distributions where the expert shows multiple possible paths to the same outcome.  

ACT is small, cheap to train, and inference is very fast, allowing rapid simulation-based experimentation.

My tests with ACT show that it learns a task very well with just an hour of training, but does not generalise to different positions — it essentially learns a trajectory based on visual feedback. Within the training distribution the policy works well, but after training on a task such as picking up a block, it cannot pick up the block at a different location (see Results).

## Hirearchical Approaches for General Purpose Robotics

"Walking is a different skill to tying shoelaces." 

Historically hierarchical methods such as gated modular polices - mixture of experts (Nowlan, 1990; Jacobs et al., 1991), or task and motion planning (TAML). Have been used to compose smaller models orchestrated by a high level planner. Given the gains and availabliltiy of large langage models, it should be feasible to use a large capable model to break a long horizon task down into sub tasks that are shorter and exectuted on smaller models. This would lever the capability and availability of very large LLM models in the cloud where latency is not a factor, and combine it with high frequency local control on constrained hardware and to train on hardware more easily available to the "home gamer".

Hoi Fai Yu and Abdulrahman Altahhan (2025) used hirearchical approach to learn an optimal policy between pushing and grabbing a target. They bootstrap two seperate low level models using BC to competence and then introduce a third controller model and use RL to learn the optimal strategy.  

## Proposal

Use a small model to learn position-invariant subtasks. Use a larger, more capable LLM to orchestrate those tasks — effectively hoisting the "L" from VLA and handling language-conditioned planning separately.

## Setup

### Robots

This project uses LeRobot by HuggingFace (Cadene et al., 2024). LeRobot is designed to make it more accessible to experiment with robotics, comprising a collection of models based on PyTorch (for example, they rewrote the Pi0/Pi0.5 models from JAX to PyTorch), a dataset format and viewer, and an open-source design for a 3D-printed robot (SO-100/SO-101).

#### Recorded Dataset Viewer
![Huggingface Dataset Viewer](images/HuggingFaceDatasetViewer.png)

- Wrist Camera  640x480x3 (RGB) FOV: 71°
- Overhead Camera (Nuroum V11), 640x480x3 (RGB), FOV: 52°

I have two 6-DOF SO-100 robot arms. These use Feetech servo motors with counts from 0–4095 and an offset which handles wrap-around. The arm was calibrated in joint-angle space, mapping counts to degrees, with the gripper mapped 0–100 for closed and open. 

#### SO100 Robot in Leader Follower Configuration
![Robots](images/DansRobotSetup.jpeg)


#### Calibration
The URDF model from the SO-100 robot repository was used to normalise the data to [-1, 1] for training and denormalise to joint angles at inference. Normalisation is important to maximise dynamic range during training and ensure stable gradient flow (Ioffe & Szegedy, 2015). This sounds simple but took a surprisingly long time to get right. Some datasets are recorded in radians (e.g. the Open X-Embodiment format (Open X-Embodiment Collaboration, 2024)) while others use delta actions (e.g. Pi0) instead of absolute positions.

![Wraparound error no offse](images/NoOffsetAnnotated.jpg)
![Firmware Wraparound fix](images/WithOffsetAnnotated.jpg)


A simple teleoperation script was used to test the calibration of the robots, configuring one as a follower and one as a leader to ensure they were in sync.

A simulation based on the SO-100 MJCF model was built in MuJoCo (Todorov et al., 2012), and a teleoperation script was built with the real leader SO-100 set up to drive the simulation — this proved the calibration. Friction coefficients were configured between the gripper and the block through trial and error.


## Recording Setup

The simulation environment was extended using MuJoCo for physics and OpenXR to view the simulation environment through a Meta Quest 3 headset for training. The human expert would move the real leader arm to generate demonstration episodes.

 Demonstrations are recorded using a physical SO-100 leader arm to control either a real follower arm or a simulated one. For real-robot recording, teleoperate_so100.py streams joint positions   from the leader to the follower in real-time. For simulation-based recording — the method used for all training datasets in this project — record_sim_vr_pickplace.py reads the leader arm and   drives the MuJoCo SO-101 simulation, while the scene is rendered stereoscopically to a Meta Quest 3 headset via OpenXR, giving the operator a 3D view of the workspace. The operator performs
  pick-and-place demonstrations, saving/discarding episodes interactively. For gap-filling at specific block positions, record_targeted_positions.py automates the block placement. Completed
  datasets are uploaded to HuggingFace Hub using upload_dataset.py.

  REAL-TO-REAL PIPELINE
  =====================

    ┌─────────────┐    joint positions    ┌──────────────┐
    │ SO-100 Leader├─────────────────────►│ SO-100 Follower│
    │ (physical)  │   teleoperate_so100  │  (physical)   │
    └─────────────┘        .py           └───────────────┘


  REAL-TO-SIM PIPELINE (used for all training data)
  ==================================================

    ┌─────────────┐   joint positions   ┌──────────────────┐   stereo   ┌──────────────┐
    │ SO-100 Leader├───────────────────►│  MuJoCo SO-101    ├──────────►│ Meta Quest 3 │
    │ (physical)  │                     │  Simulation       │  OpenXR   │  (3D view)   │
    └─────────────┘                     │                   │           └──────────────┘
                                        │  record_sim_vr_   │
                                        │  pickplace.py     │
                                        │                   │
                                        │  ┌─────────────┐  │
                                        │  │ wrist_cam   │  │
                                        │  │ overhead_cam│  │  save/discard
                                        │  │ (640x480)   │  │  per episode
                                        │  └─────────────┘  │      │
                                        └───────────────────┘      │
                                                                   ▼
                                                          ┌─────────────────┐
                                                          │  Local Dataset  │
                                                          │  (LeRobot v3.0) │
                                                          └────────┬────────┘
                                                                   │ upload_dataset.py
                                                                   ▼
                                                          ┌─────────────────┐
                                                          │  HuggingFace Hub│
                                                          │  danbhf/*       │
                                                          └─────────────────┘


Datasets recorded:

| Dataset                                   | Episodes | Description |
|-------------------------------------------|----------|-------------|
| sim_pick_place_20251229_101340             | 20       | First recording session, single position. ACT baseline (80% at 50k steps). |
| sim_pick_place_20251229_144730             | 20       | Second recording session (Lincoln). Less used. |
| sim_pick_place_merged_40ep                 | 40       | Merged from the two 20ep sessions. Joint space, RGB only. SmolVLA still 0%. |
| sim_pick_place_40ep_rgbd_ee                | 40       | RGBD + end-effector action space. SmolVLA training — 0% success. |
| sim_pick_place_157ep                       | 157      | Primary single-position dataset (~0.22, 0.22). 22.5k frames, 2 cameras. Best ACT-ViT result: 90% (subtask + selective coords). Spatial gen: 18.4% on 7×7 grid. |
| sim_pick_place_157ep_pi0                   | 157      | Converted from 157ep for Pi0/openpi (normalised gripper). Abandoned — poor results. |
| sim_pick_place_2pos_220ep_v2               | 220      | 100ep at pos1 + 100ep at pos2 + 20 gap-filling at ~20 random positions. Key dataset for Exp 12/14. Pickup is 100% position-invariant. Best model (Exp 14b): 85% with auxiliary completion head. |
| sim_pick_place_2pos_220ep_confuser         | 220      | Same as 220ep but with distractor blocks for coordinate conditioning. |
| sim_pick_place_2pos_220ep_confuser_rand    | 220      | Randomised confuser variant. |
| sim_pick_place_220ep_confuser_5x           | 220      | 5× augmented confuser dataset. |
| sim_pick_place_220ep_confuser_mixed_5x     | 220      | Mixed confuser with 5× augmentation. |



*Note:  as per the ACT paper demand was used for training*

## Training Pipeline

Training is run via train_act.py (ResNet backbone) or train_act_vit.py (ViT backbone), which load a LeRobot dataset from HuggingFace Hub, optionally caching and resizing images in memory or on disk. Every --save_freq steps (default 5000), a checkpoint is saved and, if (optionally set with --eval_episodes), the policy is evaluated in-the-loop by spinning up a MuJoCo simulation and running N episodes. Each episode runs the policy open-loop (chunk_size=100 actions per observation), and success is detected by is_task_complete() which checks whether the duplo block's XY position is within the 12cm x 12cm bowl bounds and its Z height is below 5cm (i.e. resting, not held). For failed episodes, analyze_trajectory() tracks the block's position over time and classifies the failure mode: 

- NEVER_PICKED_UP (block never exceeded 3cm height)
- DROPPED_DURING_TRANSPORT (lifted then fell far from goal)
- MISSED_GOAL (dropped near but outside the bowl)
- TIMEOUT (max 300 steps reached) 
  
These metrics — success rate, pick rate, drop rate, and per-outcome counts — are logged to Weights & Biases alongside the training loss.

![wandb](images/pi0_so101_lerobot_training_wandb.png)



  TRAINING + EVALUATION PIPELINE
  ===============================

    ┌─────────────────┐
    │  HuggingFace Hub│
    │  danbhf/*       │
    └────────┬────────┘
             │ download
             ▼
    ┌─────────────────────────────────────────────────────────┐
    │  train_act.py / train_act_vit.py                        │
    │                                                         │
    │  ┌──────────────┐    ┌────────────────┐                 │
    │  │ LeRobot       │    │ CachedDataset / │                │
    │  │ Dataset       ├───►│ DiskCachedDataset│                │
    │  │ (HF v3.0)    │    │ (optional resize)│                │
    │  └──────────────┘    └───────┬────────┘                 │
    │                              │ batches                   │
    │                              ▼                           │
    │                     ┌────────────────┐                   │
    │                     │   ACT Policy    │                   │
    │                     │ ResNet18 / ViT  │◄── normalize     │
    │                     │ Transformer Enc │    (mean/std)     │
    │                     │ Transformer Dec │                   │
    │                     │ + optional      │                   │
    │                     │   completion    │                   │
    │                     │   head          │                   │
    │                     └───────┬────────┘                   │
    │                             │ L1 loss (+KL +completion)  │
    │                             ▼                            │
    │                     ┌────────────────┐                   │
    │                     │   Optimizer     │                   │
    │                     │   (AdamW)       │                   │
    │                     └───────┬────────┘                   │
    │                             │                            │
    │          every save_freq steps (default 5000)            │
    │                             │                            │
    │                ┌────────────┴────────────┐               │
    │                ▼                         ▼               │
    │   ┌──────────────────┐     ┌──────────────────────┐     │
    │   │  save_checkpoint  │     │  run_evaluation()     │     │
    │   │  checkpoint_NNNNN/│     │  (N episodes in sim)  │     │
    │   │  ├─ model.safetens│     │                       │     │
    │   │  ├─ optimizer.pt  │     │  ┌─────────────────┐  │     │
    │   │  ├─ training_meta │     │  │ MuJoCo SO-101   │  │     │
    │   │  │   data.json    │     │  │ Simulation      │  │     │
    │   │  └─ preprocessor/ │     │  └────────┬────────┘  │     │
    │   └──────────────────┘     │           │ per step   │     │
    │                             │           ▼            │     │
    │                             │  ┌─────────────────┐   │     │
    │                             │  │ is_task_complete │   │     │
    │                             │  │ duplo XY in bowl │   │     │
    │                             │  │ AND Z < 5cm      │   │     │
    │                             │  └────────┬────────┘   │     │
    │                             │      fail │            │     │
    │                             │           ▼            │     │
    │                             │  ┌─────────────────┐   │     │
    │                             │  │analyze_trajectory│   │     │
    │                             │  │ (failure_analysis│   │     │
    │                             │  │  .py)            │   │     │
    │                             │  │                  │   │     │
    │                             │  │ Track block Z:   │   │     │
    │                             │  │  max_h < 3cm?    │   │     │
    │                             │  │  → NEVER_PICKED  │   │     │
    │                             │  │  lifted→fell?    │   │     │
    │                             │  │  → DROPPED_      │   │     │
    │                             │  │    TRANSPORT     │   │     │
    │                             │  │  near bowl?      │   │     │
    │                             │  │  → MISSED_GOAL   │   │     │
    │                             │  │  300 steps?      │   │     │
    │                             │  │  → TIMEOUT       │   │     │
    │                             │  └────────┬────────┘   │     │
    │                             │           │            │     │
    │                             └───────────┼────────────┘     │
    │                                         │                  │
    └─────────────────────────────────────────┼──────────────────┘
                                              │
                                ┌─────────────▼──────────────┐
                                │     Weights & Biases        │
                                │                             │
                                │  train/loss, train/kl_loss  │
                                │  eval/success_rate          │
                                │  eval/pick_rate             │
                                │  eval/drop_rate             │
                                │  eval/outcome_never_picked  │
                                │  eval/outcome_dropped_*     │
                                │  eval/outcome_missed_goal   │
                                │  eval/outcome_timeout       │
                                └─────────────────────────────┘



## Inference

Inference scripts were written for the simulated environment to start the robot in the zero pose, pick up the block and drop it in the bowl. After every 

## ACT (RESNET FLAVOUR)

```
                    TRAINING                              INFERENCE

    Actions ──────────┐
    (what robot did)  │
                      ▼
                 ┌─────────┐
                 │   VAE   │
    Robot State ─►│ Encoder │──► z (latent)           z = zeros
                 └─────────┘         │                     │
                                     ▼                     ▼
    Robot State ───────────────► ┌─────────┐         ┌─────────┐
    Pickup Coords ─────────────► │ Encoder │         │ Encoder │
    Images ──► ResNet ─────────► │         │         │         │
                                 └────┬────┘         └────┬────┘
                                      │                   │
                                      ▼                   ▼
                                 ┌─────────┐         ┌─────────┐
                                 │ Decoder │         │ Decoder │
                                 └────┬────┘         └────┬────┘
                                      │                   │
                                      ▼                   ▼
                              Predicted Actions    Predicted Actions
```

## Part 1: The Inputs

When training, each batch contains:

```python
batch = {
    'observation.state': tensor (B, 6),      # Robot joint positions
    'observation.environment_state': tensor (B, 2),  # Pickup coords (x, y)
    'observation.images.wrist_cam': tensor (B, 3, 480, 640),  # RGB image
    'observation.images.overhead_cam': tensor (B, 3, 480, 640),
    'action': tensor (B, 100, 6),            # 100 future actions (chunk_size)
}
```

## Part 2: ResNet (The Vision Backbone)

### Is ResNet Frozen?

**NO!** The backbone has its own learning rate (`optimizer_lr_backbone`), typically lower than the main LR. So ResNet IS being trained, just more slowly.

```python
def get_optim_params(self) -> dict:
    return [
        {"params": [p for n, p in self.named_parameters()
                    if not n.startswith("model.backbone")]},  # Main LR
        {"params": [p for n, p in self.named_parameters()
                    if n.startswith("model.backbone")],
         "lr": self.config.optimizer_lr_backbone},  # Separate backbone LR
    ]
```

### What ResNet Outputs

```python
# ResNet processes each camera image
self.backbone = IntermediateLayerGetter(backbone_model,
                                         return_layers={"layer4": "feature_map"})

# For a 480x640 image, layer4 outputs a 15x20 feature map with 512 channels
# (downsampled by 32x)
cam_features = self.backbone(img)["feature_map"]  # Shape: (B, 512, 15, 20)

# Project to model dimension and flatten spatially
cam_features = self.encoder_img_feat_input_proj(cam_features)  # (B, 256, 15, 20)
cam_features = rearrange(cam_features, "b c h w -> (h w) b c")  # (300, B, 256)
```

**Key point**: ResNet outputs a **spatial feature map** (15×20 = 300 spatial positions), NOT a single CLS token. Each position corresponds to a region of the image. These 300 tokens all go into the transformer.

## Part 3: What is z? (The VAE Latent)

The "z" is a **style variable** that captures the variability in how expert demonstrator performs the same task.

### Why z Exists

Given the same starting state, there are many valid ways to pick up a block. You might approach from the left or right, go faster or slower, etc. The VAE learns to encode this "style" into z. This enables the model to learn different trajectories (a multi-modal dataset) without collapsing to an average.

### During Training

```python
# VAE encoder sees: [CLS token, robot_state, action_1, action_2, ..., action_100]
# It processes the GROUND TRUTH actions

cls_token_out = self.vae_encoder(vae_encoder_input)  # CLS summarizes the style

# Project to mean and variance of a Gaussian
latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
mu = latent_pdf_params[:, :latent_dim]           # Mean
log_sigma_x2 = latent_pdf_params[:, latent_dim:] # Log variance

# Sample z using reparameterization trick (allows gradients to flow)
z = mu + exp(log_sigma_x2 / 2) * random_noise
```

### During Inference

```python
# No actions to encode, so z = zeros
z = torch.zeros([batch_size, latent_dim])
```

The model learns to work with z=0 because the KL loss pushes the distribution toward standard normal (mean=0, std=1).

## Part 4: The Main Encoder-Decoder

### Encoder

The encoder receives a sequence of tokens:

```
Token sequence: [z, robot_state, pickup_coords, img1_pos1, img1_pos2, ..., img1_pos300, img2_pos1, ...]
                 ▲       ▲            ▲              ▲
                 │       │            │              └── 300 spatial features per camera
                 │       │            └── Your (x,y) coordinates
                 │       └── 6 joint angles
                 └── Latent style variable (32-dim typically)
```

For 2 cameras: 1 + 1 + 1 + 300 + 300 = **603 tokens** go into the encoder.

Each token is 256-dimensional. The encoder is a standard transformer that lets all tokens attend to each other.

### Decoder

```python
# Decoder input: 100 learned query embeddings (one per output timestep)
decoder_in = torch.zeros((chunk_size, batch_size, dim_model))

# Cross-attention: queries attend to encoder output
decoder_out = self.decoder(decoder_in, encoder_out)

# Project to actions
actions = self.action_head(decoder_out)  # (B, 100, 6)
```

The decoder uses **cross-attention** to look at the encoder's output and generate 100 action predictions.

## Part 5: The Loss Function

```python
def forward(self, batch):
    actions_hat, (mu, log_sigma_x2) = self.model(batch)

    # L1 loss: predicted actions vs ground truth
    l1_loss = F.l1_loss(batch['action'], actions_hat)

    # KL divergence: push z distribution toward N(0,1)
    kl_loss = (-0.5 * (1 + log_sigma_x2 - mu² - exp(log_sigma_x2))).mean()

    loss = l1_loss + kl_weight * kl_loss
```

- **L1 loss**: "Did you predict the right actions?"
- **KL loss**: "Is your z distribution close to standard normal?"

## Part 6: How Gradients Flow

When `loss.backward()` runs:

```
loss = l1_loss + kl_loss
         │          │
         ▼          ▼
    action_head  vae_encoder_latent_output_proj
         │          │
         ▼          ▼
    decoder      vae_encoder
         │
         ▼
    encoder ◄────── encoder_latent_input_proj (connects z to encoder)
     │  │  │
     │  │  └── encoder_img_feat_input_proj → ResNet backbone
     │  └── encoder_robot_state_input_proj
     └── encoder_env_state_input_proj (PICKUP COORDS)
```

# Experiments

## Model Selection and Evaluation

Before committing to **ACT** as the primary policy architecture, four **Vision–Language–Action (VLA)** models were evaluated on the pick-and-place task:

- **OpenVLA** (7B parameters)
- **SmolVLA** (450M parameters)
- **Pi0** (3B parameters, via both JAX/openpi and LeRobot PyTorch)
- **ACT** (ResNet-18 backbone, ~4M parameters)

OpenVLA was tested first on the real robot using **zero-shot inference**. Using `bridge_orig` normalisation, the model produced only small delta actions that failed to move the arm meaningfully. A finetuning pipeline (`openvla_finetune.py`) was then implemented using **LoRA (rank 32)** on a **5 Hz–downsampled 20-episode dataset**, with **4-bit quantisation** to fit on a single GPU. The finetuned model predicted constant actions.

Further investigation revealed **four fundamental implementation errors**, documented in `OPENVLA_ISSUES_FOUND.md`:

1. Action tokens were never included in the training sequence (the model was trained to predict the token following `"Out:"` with no action ground truth)
2. Min/max normalisation was used instead of the official **q01/q99** scheme
3. Quantisation was enabled contrary to official guidance
4. The `ActionTokenizer` returned bin indices rather than decoded token strings

A corrected pipeline (`openvla_finetune_fixed.py`) was built with proper sequence construction, label masking, and q01/q99 normalisation. Models were trained for **2.5k** and **5k steps**. The fixed model produced some arm movement on the real robot but never achieved task completion.

**SmolVLA**, pretrained on **487 LeRobot community datasets** (including SO-100), was finetuned on both an **RGBD end-effector dataset** and a simplified **RGB joint-space variant** using the LeRobot training stack. It achieved **0% success** across all checkpoints in both configurations, producing plausible-looking but ineffective motions.

**Pi0** was evaluated twice. The JAX/openpi implementation trained on **40 episodes** caused the gripper to regress to its mean value: the robot moved but never completed the task. A second attempt using LeRobot’s PyTorch implementation on **157 episodes** initially yielded **0% success** due to a missing **action denormalisation postprocessor**. After fixing this issue, the model produced its first successes but remained unreliable.

By contrast, **ACT** reached **80% success** with just **20 episodes** and **50k training steps**, completing training in **under 90 minutes on a single GPU**. This stark contrast — a **4M-parameter model outperforming 7B- and 450M-parameter VLAs on limited data** — motivated the decision to focus all subsequent experiments on ACT.

---

### VLA Evaluation Summary

| ID  | Date           | Model (Params)        | Dataset              | What was tested                                  | Result                                   | Takeaway |
|-----|----------------|-----------------------|----------------------|--------------------------------------------------|-------------------------------------------|----------|
| V01 | 2025-11-24     | OpenVLA (7B)          | 20ep, 30 Hz          | Zero-shot inference on real robot (bridge_orig) | Tiny delta actions, no meaningful motion  | Zero-shot on unseen robot/task not viable |
| V02 | 2025-11-25–27  | OpenVLA (7B)          | 20ep, 5 Hz           | LoRA finetuning (rank 32, 4-bit quant, 2.5k)    | Constant actions, 0% success              | 4 critical implementation bugs identified |
| V03 | 2025-11-28–29  | OpenVLA (7B)          | 20ep, 5 Hz           | Fixed finetuning (q01/q99, no quant, 2.5k+5k)   | Some movement, no task completion         | 20 demos insufficient even after fixes    |
| V04 | 2026-01-08–10  | SmolVLA (450M)        | 40ep RGBD+EE; RGB+J  | Finetuning pretrained compact VLA              | 0% success across all checkpoints         | SmolVLA recipe not viable for this setup  |
| V05 | 2026-01-10     | Pi0 (3B, JAX/openpi)  | 40ep                 | Finetuning flow-matching VLA                   | 0%; gripper regresses to mean             | Gripper control fails at low data         |
| V06 | 2026-01-19–21  | Pi0 (3B, LeRobot PT)  | 157ep                | Re-attempt with working training stack         | First successes after denorm fix          | Pre/post-processing is critical           |
| C01 | 2026-01-06–07  | ACT-ResNet (4M)       | 40ep, joint vs EE    | Baselines; action space comparison             | Joint 93.3% peak; EE 90% peak             | ACT effective with minimal data           |
| C02 | 2026-01-18     | ACT-ResNet (4M)       | 157ep                | Data scaling at fixed position                 | Best checkpoint 100% (10/10)              | ACT scales reliably with more data        |

# Debugging Tools

To understand what the policy is "thinking", I built a whisker visualisation tool (utils/whisker_visualizer.py, utils/mujoco_viz.py) that renders predicted future trajectories directly into the MuJoCo 3D scene. Since ACT predicts a chunk of 100 future joint-angle targets, the tool takes each predicted joint configuration, runs forward kinematics via mj_forward() to compute the resulting gripper Cartesian position, and draws the sequence as a coloured 3D path — a "whisker" — emanating from the current end-effector. The current prediction is shown as a green whisker, while previous predictions are retained as fading blue ghost trails (up to 12), making it visually apparent how the policy's plan evolves over time. An orange trail records the gripper's actual executed path for comparison. 

![Predicted VS actual](images/ACT_Rollout_weirdness.png)

![Whisker](images/ACT_showing_actual_whiskers_used.png)

For temporal ensembling experiments (visualize_temporal_ensemble_live.py), grey whiskers show the individual chunk predictions from consecutive time steps while the green whisker shows the exponentially-weighted ensemble average — making it intuitive why ensembling produces smoother trajectories: multiple overlapping "opinions" about the future, predicted from slightly different robot states, converge into a stable consensus. 

![Ensemble](images/Ensemble.png)

The same tool was adapted for Pi0 (visualize_whiskers_pi0.py), where it visualised the flow-matching denoising process: the 10 iterative refinement steps from pure noise to final actions, revealing how small differences in normalised action space get amplified by FK into visually scattered 3D positions that progressively "crystallise" into a coherent trajectory. Multi-angle recordings from 5 simultaneous camera viewpoints with whisker overlays were composited into tiled videos for analysis and presentation.

![Robots](images/Pi0WhiskerConvergence.png)
![Robots](images/Pi0_Fail_does_not_move.png)

## End Effector Space Experiments

I experimented with training ACT in end-effector (EE) action space, where the policy predicts 8-dimensional actions — XYZ position (3), quaternion orientation (4), and gripper (1) — instead of the default 6-dimensional normalised joint angles. The idea being that end-effector space would be easier to learn in (cf. Goodfellow et al., 2016, §5.3 on the effect of coordinate representations on learning), and could potentially transfer to other robot topologies without retraining.

The existing joint-space dataset (sim_pick_place_merged_40ep, 40 episodes) was converted offline using convert_to_ee_actions.py, which runs MuJoCo forward kinematics on each frame's joint angles to compute the gripper site's Cartesian pose, with Shepperd's method for rotation matrix to quaternion conversion. At inference, a damped least-squares Jacobian IK solver (utils/ik_solver.py) converts the predicted EE pose back to joint angles. The first EE training run (Exp 14) matched the joint-space baseline at 90% peak success but converged more slowly (peak at 50k vs 15k steps) and suffered a 12-16% IK failure rate, indicating the network was predicting unreachable poses. Investigation revealed a quaternion double-cover bug: q and -q represent the same rotation, but arbitrary sign choices created artificial discontinuities between consecutive frames that the network struggled to learn. After enforcing quaternion continuity by flipping signs when dot(q_prev, q_curr) < 0 (Exp 15), EE-space training improved significantly — 93.3% peak success, 0% IK failures, and faster convergence (76.7% at just 5k steps). A validation experiment (Exp 16) trained on the preserved action_joints field of the same EE dataset to confirm data integrity, achieving 73.3% — lower than EE's 93.3%, likely because the EE representation encodes task-relevant spatial information more directly and at a more uniform scale (~0.1-0.4m vs ~-100 to +100 degrees). Despite these promising results, EE inference was 7x slower due to iterative IK solving, and the approach was ultimately set aside in favour of joint-space training on larger datasets where the simpler pipeline meant I could run more experiments.

![Quaternion Wrap Around Bug](images/WrapAroundErrorInRecording.png)

## RGB vs RGBD: 

## Depth Input and Confuser Scene Design

### RGBD Experiments

To test whether depth information improves manipulation performance, I created a dedicated **RGBD scene** (`scenes/so101_rgbd.xml`) modelled on an **Intel RealSense D435** depth camera. The scene uses the D435’s **58° vertical field of view**, replacing the standard Nuroum V11’s 52°.

A **40-episode dataset** was re-recorded by replaying identical joint trajectories using `rerecord_dataset.py`, producing the dataset **`sim_pick_place_40ep_rgbd_ee`**. This dataset contains three image streams:

- Wrist RGB  
- Overhead RGB  
- Overhead depth  

Depth images were rendered using **MuJoCo’s built-in depth buffer** and stored as **3-channel grayscale uint8** (0–255 mapped to 0–2 m) for compatibility with the LeRobot data pipeline.

Training **ACT** in **end-effector (EE) action space** with **RGB+Depth** inputs (Exp 18) achieved a **peak success rate of 100% at 15–25k steps**, compared to **93.3%** for the RGB-only baseline (Exp 15). However, training was **highly unstable**, collapsing to **33% by 50k steps** with large variance between checkpoints.

A control experiment (Exp 19), intended as an RGB-only baseline on the same dataset, initially appeared to exhibit the same instability. Investigation revealed a **substring-matching bug** in the camera filter (`"overhead_cam"` matching `"overhead_cam_depth"`), which silently included the depth channel and made the experiment equivalent to Exp 18. After fixing the filter to require an **exact match**, the corrected RGB-only baseline confirmed that depth **marginally improves peak performance** but **significantly reduces training stability** on small datasets. The most likely cause is accelerated overfitting due to the additional input channel.

Neverthe less No clear advantage from depth in the configurations tested (but I'm going to revisit this).

### Confuser and Coordinate-Conditioning Scenes

In parallel, several **scene variants** were created for the coordinate-conditioning and confuser experiments. The base scene (`so101_with_wrist_cam.xml`) contains the **SO-101 arm**, a single **white Duplo block**, and a **red bowl**.

The confuser scene (`so101_with_confuser.xml`) adds an **identical white Duplo block** at a fixed distractor position (**x = 0.25, y = 0.05**). This forces the policy to disambiguate between the target and the distractor, motivating the inclusion of **explicit (x, y) block coordinates** as an additional observation input to the model.

## Spatial Generalisation and Task Decomposition

To quantify how well a behaviour-cloned **ACT** policy generalises beyond its training distribution, I conducted systematic spatial evaluations totalling **2,630+ episodes**. The evaluated model (checkpoint **45k**), trained on **157 episodes** at a single block position (~(0.22, 0.22)), was first tested across increasingly fine spatial grids.

### Single-Position Spatial Generalisation

The model was evaluated on a **5×5 grid** spanning the full workspace  
(**X:** 0.15–0.35, **Y:** −0.15–0.30), with **20 episodes per position** (**500 total**).  
This yielded an overall success rate of **12.6%**.

A finer **7×7 grid**, centred on the training position  
(±5 cm in X, ±7.5 cm in Y; **10 episodes per position**, **490 total**), achieved **51.6%** success.  
This reveals that the **50% success threshold occurs at approximately 7 cm displacement**.

Performance degrades sharply with distance:

- **60–70%** within **5 cm**
- **10–35%** at **7–10 cm**
- **Near-zero** beyond **11 cm**

These results confirm that the policy memorises a **narrow visual–motor mapping**, rather than learning a spatially general skill.

A strong **directional asymmetry** was also observed. At equal displacements (~7.5 cm):

- **+Y offsets** (away from the bowl) retained **20–50%** success
- **−Y offsets** (towards the bowl) dropped to **0%**

This occurs because the policy only learned to transport blocks in a single direction during training.

### Two-Position Dataset and Subtask Analysis

These findings motivated a second round of experiments using a more diverse **220-episode dataset**, comprising two distinct block positions plus **20 gap-filling episodes** at random locations. With blinkering and state-fix improvements (Exp 12), full-task spatial evaluation on a **5×5 grid** yielded **9.6% overall success**, with successes clustering near the two training positions.

A critical insight emerged when evaluating **subtasks in isolation** (Exp 13b, natural approach method):

- **PICK_UP:** **100% success at all 9 reachable positions** on the 5×5 grid (5/5 episodes each) — suggesting strong position invariance for grasping, though only testable where the robot could naturally navigate (9 of 25 grid positions)
- **MOVE_TO_SOURCE:** **16 of 25** grid positions unreachable due to approach trajectory constraints
- **DROP:** **35% failure rate** even at the trained bowl position

However, this result did not hold universally across model variants: the best overall model (Exp 14b, 85% full-task success with auxiliary completion head) regressed to only **4 of 25** positions at 100% pickup, suggesting a **trade-off between training-position performance and spatial generalization**.

This decomposition shows that **pickup can generalise well** within the robot's reachable workspace, while **navigation and placement do not**. These results directly informed the thesis proposal: use a **high-level planner** for navigation and task sequencing, while relying on the low-level **ACT policy** for manipulation primitives that exhibit strong (though not unlimited) position invariance.

---

## Related Experiments

| ID  | Date           | Experiment                          | Question                                 | Setup                         | Result                                                      | Takeaway |
|-----|----------------|-------------------------------------|------------------------------------------|-------------------------------|-------------------------------------------------------------|----------|
| C06 | 2026-01-20     | ACT rollout chunking                | Does shorter execution horizon help?     | 10 eps per setting            | n_action_steps 1–5 → 0%; sweet spot ~20 → 100%              | Optimal replan frequency exists; too-frequent replanning destabilises |
| C07 | 2026-01-21–22  | ACT spatial generalisation (1-pos)  | What is the generalisation radius?       | 5×5 + 7×7 grids (990–2630 eps)| ~50% threshold at ~7 cm; far OOD mostly 0–10%               | BC policies memorise workspace region; need data diversity or conditioning |
| C08 | 2026-01-22     | Temporal ensembling (ACT)           | Does ensembling reduce drops?            | 50 eps per condition          | 82% → 90%; drops 7 → 1; never-pick increased                | Ensembling smooths transport but can hurt pickup; trade-off |



![Robots](images/spatial_scatter_1900.png)

![Robots](images/spatial_eval_2pos_combined_plot.png)

![Robots](images/spatial_success_vs_distance.png)

![Robots](images/TwoPosPos1.png)
![Robots](images/TwoPosPos2.png)

# Attempting to acheive Spacial Invariance

## Vision Transformer Backbone

The default ACT architecture uses a **ResNet-18** backbone, which processes each camera image into a spatial feature map (e.g. 15×20 = 300 tokens at 640×480). I replaced this with a **ViT-B/16** (Vision Transformer, patch size 16) to test whether a transformer-based vision backbone — with its global self-attention over image patches — would improve spatial generalisation compared to ResNet's local convolutional receptive fields.

### Single-Camera Results

The initial comparison used the 220-episode confuser dataset (`sim_pick_place_2pos_220ep_confuser`) with a single wrist camera:

| Backbone | Cameras | Patches/Image | Success | Pick Rate | Drop Rate |
|----------|---------|---------------|---------|-----------|-----------|
| ResNet-18 | 2 (wrist + overhead) | 300 | 50% | 70% | 20% |
| **ViT-B/16** | **1 (wrist only)** | **196** | **72%** | **96%** | **20.8%** |
| ViT-B/32 | 1 (wrist only) | 49 | 60% | 80% | 20% |

ViT-B/16 outperformed the two-camera ResNet baseline by +22% using only the wrist camera. The pick rate jumped from 70% to 96%, suggesting ViT's patch-level attention is better at localising the block for grasping. ViT-B/32 (coarser 32×32 patches, producing only 49 tokens) dropped to 60%, confirming that **spatial resolution of the patch grid matters** for precise manipulation.

### Frozen vs Fine-Tuned Backbone

I also tested whether ImageNet-pretrained ViT features are sufficient without fine-tuning:

| Backbone | Trainable Params | Success | Pick Rate | Drop Rate |
|----------|-----------------|---------|-----------|-----------|
| ViT-B/16 (fine-tuned) | 126M (100%) | 72% | 96% | 20.8% |
| ViT-B/16 (frozen) | 41M (32%) | 74% | 84% | 9.5% |

Freezing the backbone produced comparable success (74% vs 72%) with a substantially lower drop rate (9.5% vs 20.8%), suggesting that fine-tuning may cause slight overfitting on this dataset size. The frozen model trains faster and is more parameter-efficient, though its lower pick rate (84% vs 96%) indicates the pretrained features are slightly less precise for initial block localisation.

### Two-Camera Bug and Chunk Size Interaction

Adding the overhead camera to ViT initially **halved** performance (72% → 36%). Investigation revealed a positional embedding bug: both cameras shared identical positional encodings, so the model could not distinguish which patches came from which camera. After adding per-camera learned embeddings, two-camera ViT recovered to 58% — but only after also reducing `chunk_size` from 100 to 50. The interaction between visual complexity and prediction horizon is significant: ViT's higher-dimensional patch tokens require shorter chunks to train effectively.

| Config | Cameras | Chunk Size | Camera Embeds | Success | Pick Rate |
|--------|---------|------------|---------------|---------|-----------|
| ViT 1-cam | 1 (wrist) | 50 | N/A | 72% | 96% |
| ViT 2-cam (buggy) | 2 | 100 | No | 36% | 58% |
| ViT 2-cam (fixed) | 2 | 50 | Yes | 58% | 92% |
| ResNet 2-cam | 2 | 100 | N/A | 50% | 70% |

### Coordinate Conditioning

Finally, I tested providing explicit block (x, y) coordinates as an additional input to the ViT model:

| Config | Coords | Success | Pick Rate | Drop Rate |
|--------|--------|---------|-----------|-----------|
| ViT baseline | No | 72% | 96% | 20.8% |
| ViT + coords | Yes | 76% | 90% | 15.6% |

Coordinates gave a modest +4% success improvement, with the main benefit in the transport phase (drop rate 15.6% vs 20.8%). The slight pick rate decrease (90% vs 96%) suggests minor interference between visual and coordinate signals during reaching.

### Summary

The ViT-B/16 backbone was adopted as the default for all subsequent experiments. The key findings were: (1) ViT outperforms ResNet even with fewer cameras, (2) patch spatial resolution matters — ViT-B/16 substantially outperforms ViT-B/32, (3) frozen pretrained features are competitive with full fine-tuning, and (4) multi-camera ViT requires per-camera positional embeddings and shorter chunk sizes.

## Subtask Conditioning and Coordinate Tokens

Rather than training a monolithic policy for the entire pick-and-place trajectory, I decomposed the task into four subtasks — **MOVE_TO_SOURCE**, **PICK_UP**, **MOVE_TO_DEST**, and **DROP** — and provided the model with explicit conditioning signals via `observation.environment_state`. This 6-dimensional vector comprises a **4-dim one-hot subtask token** (indicating which phase the robot is in) and a **2-dim normalised pickup coordinate** (the block's XY position on the table).

Subtask boundaries are annotated offline using a proximity-plus-gripper heuristic: FK-computed end-effector distance to the block and bowl, combined with gripper open/close state, determines transitions. At inference time, the same geometric thresholds drive a state machine that updates the subtask token and calls `policy.reset()` to clear the action chunk queue at each transition — essential because ACT predicts 100 steps open-loop and stale actions from the previous subtask would otherwise continue executing.

### Selective Coordinate Masking

A key finding was that coordinates should **not** be provided during all subtasks. During PICK_UP and DROP the robot is already close enough to rely on wrist-camera feedback; providing absolute coordinates during these phases introduces a distribution mismatch between training and novel positions. **Selective masking** — zeroing the coordinate input during PICK_UP and DROP while providing it during MOVE_TO_SOURCE and MOVE_TO_DEST — yielded the best result:

| Configuration | Success | Pick Rate |
|---------------|---------|-----------|
| Subtask token only (Exp 7a) | 84% | 96% |
| Subtask + full coords (Exp 7b) | 86% | 96% |
| **Subtask + selective coords (Exp 7b)** | **90%** | **100%** |

The subtask token provides behavioural mode switching (the model learns distinct motion patterns for each phase), while selective coordinates guide navigation without contaminating the manipulation subtasks. This 90% model (on the 157-episode single-position dataset) became the baseline for all subsequent spatial generalisation work.

### Observation State Bug and Blinkering

Two sources of **position leakage** were identified that allowed the model to shortcut spatial reasoning rather than learning genuinely invariant behaviour:

1. **observation.state bug** — `get_observation()` read `qpos[:6]`, which is the Duplo block's freejoint (XYZ + quaternion), not the robot's joint angles at `qpos[7:13]`. The model was directly receiving the block's position as "proprioceptive state." Fixing this (`--fix_state`) dropped success from 90% to 55% on the single-position dataset, confirming the model had been relying on the leaked position.

2. **Overhead camera leakage** — during PICK_UP and DROP, the overhead camera shows the block at its absolute workspace position, allowing the model to bypass the wrist camera. **Blinkering** masks the overhead camera's 196 ViT patch tokens via `key_padding_mask` (encoder) and `memory_key_padding_mask` (decoder) during these subtasks, forcing reliance on the egocentric wrist view.

A 2×2 factorial experiment (Exp 11) on the 157-episode dataset showed that **neither fix alone improves spatial generalisation** with single-position data:

| | Buggy State | Fixed State |
|---|---|---|
| **No Blinkering** | 90% task / 12% spatial | 55% task / 4% spatial |
| **Blinkering** | 15–50% task | 60–65% task / 0% spatial |

The state bug was accidentally useful for navigation, and blinkering without diverse data simply removed information without providing an alternative learning signal. The real bottleneck was **training data diversity**.

### Combining Everything: Diverse Data + State Fix + Blinkering

With a **220-episode dataset** (two block positions plus 20 gap-filling episodes at random locations) and all fixes enabled (`--fix_state --blinkering --subtask --pickup_coords`), the picture changed dramatically (Exp 12):

| Condition | Success | Pick Rate | Drop Rate |
|-----------|---------|-----------|-----------|
| With blinkering | **65%** | 85% | 24% |
| Without blinkering | 35% | 90% | 50% |

Blinkering now **doubled** full-task success (35% → 65%) — with diverse data, the model has enough variety to learn position-invariant wrist-camera features when forced to rely on them.

TODO: Add diagram of the transformer mask for the blinkering.

### Auxiliary Completion Head

                         SINGLE OBSERVATION (one timestep)
        ┌──────────────────────┬──────────────────────┬──────────────────────────┐
        │ wrist_cam (RGB)       │ overhead_cam (RGB)   │ state (6) + env_state (6)│
        │ 640×480               │ 640×480               │                          │
        └──────────┬───────────┴──────────┬───────────┴──────────────┬───────────┘
                   │                      │                              │
            Resize 224×224         Resize 224×224                   Linear proj
                   │                      │                        (32/6/6 → 512)
                   v                      v                              v
        ┌─────────────────────────────────────────────────────────────────────────┐
        │                     ViT-B/16 (shared weights)                             │
        │                                                                           │
        │  • Patch embedding (16×16)                                                │
        │  • CLS token + positional embedding                                       │
        │  • 12 transformer layers                                                  │
        │  • Drop CLS, keep patch tokens                                            │
        └───────────────┬───────────────────────────────┬─────────────────────────┘
                        │                               │
                 (B,196,768)                      (B,196,768)
                        │                               │
                 Linear 768→512                   Linear 768→512
                        │                               │
             + camera_embed[0]               + camera_embed[1]
             + patch_pos_embed               + patch_pos_embed
                        │                               │
                        └───────────────┬───────────────┘
                                        │
        latent token (1)  ─ Linear 32→512 + 1d_pos[0]
        state token  (1)  ─ Linear 6 →512 + 1d_pos[1]
        env token    (1)  ─ Linear 6 →512 + 1d_pos[2]
                                        │
                                        v
        ┌─────────────────────────────────────────────────────────────────────────┐
        │                        TOKEN SEQUENCE (395 total)                         │
        │                                                                           │
        │  [ latent | state | env_state | wrist_cam ×196 | overhead_cam ×196 ]      │
        │                                                                           │
        │  Blinkering mask (PICK_UP / DROP):                                       │
        │   • wrist + non-visual tokens → visible                                  │
        │   • overhead tokens           → MASKED                                   │
        └───────────────────────────────┬─────────────────────────────────────────┘
                                        │
                                        v
        ┌─────────────────────────────────────────────────────────────────────────┐
        │                  TRANSFORMER ENCODER (4 layers)                          │
        │                                                                           │
        │  Self-attention + key_padding_mask                                       │
        │  (blinkering masks overhead tokens)                                      │
        └───────────────────────────────┬─────────────────────────────────────────┘
                                        │
                               395 encoder outputs
                                        │
                                        v
        ┌─────────────────────────────────────────────────────────────────────────┐
        │                  TRANSFORMER DECODER (7 layers)                          │
        │                                                                           │
        │  • 100 learned query embeddings (chunk steps)                            │
        │  • Self-attention on queries                                             │
        │  • Cross-attention to encoder outputs                                    │
        │    + memory_key_padding_mask (blinkering)                                │
        └───────────────────────────────┬─────────────────────────────────────────┘
                                        │
                           Decoder outputs: (B, 100, 512)
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
                    v                                       v
        ┌──────────────────────────┐        ┌────────────────────────────┐
        │ ACTION HEAD               │        │ COMPLETION HEAD (auxiliary)│
        │                           │        │                            │
        │ Linear 512 → 6            │        │ Linear 512 → 64             │
        │                           │        │ ReLU                        │
        │                           │        │ Linear 64 → 1               │
        │                           │        │ Sigmoid                     │
        └──────────┬───────────────┘        └──────────┬─────────────────┘
                   │                                   │
                   v                                   v
        (B,100,6) future joint actions      (B,100,1) future progress ∈ [0,1]

        Loss = action_L1
             + 0.1 × completion_MSE
             + KL_weight × KL_divergence



To address cross-subtask contamination in the action chunk (where a 100-step prediction at the start of an 18-step PICK_UP inevitably spans into MOVE_TO_DEST and beyond), I added a **completion head** — an MLP on the decoder output that predicts subtask progress (0 → 1) for each step of the chunk.

Two variants were tested (Exp 14):

- **Exp 14a (action masking):** Mask action loss beyond subtask boundaries. **Catastrophic failure (0% success)** — with subtask lengths of 18–37 steps, only ~12.5% of each chunk is supervised, starving the model of gradient signal.
- **Exp 14b (auxiliary only):** Keep full action supervision, add completion loss as auxiliary task (`loss = action_L1 + 0.1 × completion_MSE`). This produced the **best result: 85% success**.

| Metric | Exp 14b (completion) | Exp 12 (baseline) | Model 7b (original) |
|--------|----------------------|-------------------|---------------------|
| Success | **85%** | 65% | 90% |
| Pick rate | **95%** | 80% | ~95% |
| Drop rate | **10.5%** | 35% | ~20% |

The auxiliary completion loss acts as a **regulariser**, substantially improving drop reliability (10.5% vs 35%) without degrading action quality. Notably, using the completion head to actively reset the policy at predicted subtask boundaries *hurt* performance (65% → 20–25%), because the model learns smooth end-to-end trajectories that naturally cross boundaries — interrupting them forces cold restarts from unfamiliar intermediate states.

# Subtask Planning

I created some scripts to have a Very Large Language Model with Cot to create subtasks from a scene (using ChatGPT). 

## High-Level Planner: LLM-Driven Subtask Decomposition

The experiments above demonstrated that the low-level **ACT** policy achieves **position-invariant grasping** (*PICK_UP*), but fails to **navigate to arbitrary block positions** (*MOVE_TO_SOURCE* fails at **16 of 25** grid positions). This motivates a **hierarchical architecture**: a high-level planner that understands the scene and decomposes a natural-language instruction into a sequence of primitives, paired with a low-level ACT policy that executes each primitive.

To prototype this approach, I implemented a **three-stage pipeline** under `scripts/high-level-planner/`.

---

### Stage 1: Frame Extraction

`extract_first_frames.py` extracts the **first overhead camera frame** from each episode in a LeRobot dataset and saves it as a PNG. It also generates a `manifest.json` containing the **ground-truth block and bowl positions** (from `episode_scenes.json`) for each frame, enabling later comparison between detected and true object locations.

---

### Stage 2: Object Detection with GroundingDINO

`scan_scene.py` runs **GroundingDINO-tiny** (`IDEA-Research/grounding-dino-tiny`), a **zero-shot open-vocabulary object detector**, on the extracted frames. Given text labels (e.g. *"white block"*, *"bowl"*), it returns **normalised bounding-box coordinates** and confidence scores.

An optional `--annotate` flag saves images with overlaid bounding boxes and centre markers for visual verification. The detector is wrapped in a reusable `ObjectDetector` class (`utils/vision.py`), which handles model loading, GroundingDINO’s period-separated label format, and coordinate normalisation.

---

### Stage 3: LLM Task Planning

`chat-gpt/create-subtasks.py` sends an **overhead camera image** and a **natural-language instruction** (e.g. *"Put the white lego block on the left into the bin on the right"*) to **GPT-5** via the **OpenAI Responses API**. The prompt constrains the model to three primitive actions:

- `MOVE-TO(x, y)`
- `PICKUP`
- `DROP`

All coordinates are expressed in **normalised image space**. The response is enforced via **structured output** using a strict JSON schema, guaranteeing a well-formed plan:

```json
{
  "white_block_center": {"x": 0.35, "y": 0.42},
  "bin_center": {"x": 0.72, "y": 0.58},
  "plan": [
    {"action": "MOVE-TO", "x": 0.35, "y": 0.42},
    {"action": "PICKUP", "x": null, "y": null},
    {"action": "MOVE-TO", "x": 0.72, "y": 0.58},
    {"action": "DROP", "x": null, "y": null}
  ]
}


## Outstanding Challenges and Two-Week Plan

This section outlines the key open problems identified so far, along with concrete next steps. It forms the basis of the experimental plan for the final two weeks before submission.


So I should rerecord the data so that I know that the camera can SEE the block when it does the pickup, then it will have more information, and then redo the blinkering tests.

Also I should try a dataset with fewer samples, so 20 from position A, 20 from position B and then 20 random?

---

### 1. Subtask Isolation: Training ACT on Individual Primitives

ACT is currently trained on **full episodes spanning all subtasks**, with `chunk_size = 100`. This inevitably causes **temporal bleed across subtask boundaries**, where a single chunk spans multiple primitives. Earlier attempts to mitigate this using action masking (Exp 14a) proved ineffective.

The proposed solution is to **isolate subtasks** and train ACT on **individual primitives**, with a primary focus on **PICK_UP**:

- **MOVE_TO** can be handled by a scripted IK controller.
- **DROP** is a simpler variant of the same manipulation problem as PICK_UP.
- **PICK_UP** is the most critical and already demonstrates strong position invariance.

PICK_UP averages **~18 timesteps**, making `chunk_size = 100` severely oversized. Training on **subtask-segmented episodes** with `chunk_size = 20–25` ensures that each chunk lies entirely within a single primitive. This eliminates cross-boundary contamination and removes the need for action masking.

Additional benefits:
- **Faster inference** due to fewer wasted predictions
- Cleaner supervision signal aligned with the intended behaviour
- Better compatibility with a hierarchical controller

**Planned actions**
- Segment existing datasets into per-subtask episodes
- Train a PICK_UP-only ACT policy with `chunk_size ∈ {20, 25}`
- Use identical architecture and hyperparameters to prior ACT runs for comparability

---

### 2. Trajectory Blending at Subtask Boundaries

When transitioning from a scripted **MOVE_TO** controller to a learned **PICK_UP** policy, there is an inherent **trajectory discontinuity**: the scripted controller and ACT generate qualitatively different motion styles.

A cold `policy.reset()` at the boundary already works (Exp 13b achieved **100% pickup success** after a natural approach), but the transition is abrupt.

To smooth this handoff, introduce a **blending window** at the subtask boundary:

- During the final **N steps** of MOVE_TO, begin running the PICK_UP policy in parallel
- Linearly or exponentially ramp from scripted actions to learned actions
- This is equivalent to **temporal ensembling**, but applied *once* at the boundary rather than continuously

This can reuse the existing **exponential-weighted averaging** mechanism (`temporal_ensemble_coeff`) with minimal implementation effort.

**Planned actions**
- Implement a boundary-only blending mechanism
- Sweep blend window sizes (e.g. 5, 10, 15 steps)
- Evaluate smoothness and pickup success rate

---

### 3. Improving Position-Invariant Pickup

Although PICK_UP already generalises well spatially, two targeted changes may further improve invariance and robustness:

#### a) Delta Actions

Instead of predicting absolute joint targets, predict **relative joint displacements** (delta actions).

This failed catastrophically with `chunk_size = 100` (Exp 9b, 0% success), as small errors compounded over long horizons. However, with **short-horizon chunks** (20–25 steps) confined to PICK_UP only, error accumulation is far more limited.

Delta actions are naturally position-invariant: the policy learns *“close the gripper by this amount”* rather than *“move to joint angle X”*.

**Planned actions**
- Reintroduce delta actions for PICK_UP-only training
- Compare against absolute-action baseline at identical chunk sizes

#### b) Wrist Camera Geometry

The current wrist camera is pitched **40° downward** with a **71° FOV**. A wider FOV or altered pitch could provide a more consistent view of the block across approach directions.

If the block appears at a similar scale and location in wrist-camera space, visual features become more invariant and easier to learn.

This is a **single-parameter change** in the MuJoCo scene XML and can be tested rapidly.

**Planned actions**
- Test 2–3 alternative wrist camera pitches / FOVs
- Re-record a small PICK_UP-only dataset for comparison

---

### 4. Additional Improvements 

- **Velocity state**  
  Currently `observation.state` includes joint positions only. Adding joint velocities would help distinguish a dynamic approach from a static teleported start, which was identified as a failure mode in Exp 13c.

- **Wrist-camera data augmentation**  
  Apply random crops, colour jitter, or mild synthetic viewpoint perturbations during training to improve visual invariance without collecting more demonstrations.

- **Gripper-centric coordinate frame**  
  Express block coordinates relative to the end-effector position rather than in world space.  
  This reframes the task from *“go to (0.22, 0.23)”* to *“block is 3 cm left, 2 cm forward”*, which is inherently position-invariant.

- **Chunk-size sweep for isolated subtasks**  
  With subtask isolation, systematically test  
  `chunk_size ∈ {10, 15, 20, 25}`  
  to identify the shortest horizon that reliably completes PICK_UP without learning irrelevant future actions.

---


# NEXT STEPS

There are 2 main problems left:

Positional Invariance

Subtask partioning

We need to acheive seperate subtasks, where the 


## Delta vs Absolute

*Section to be completed.*


TODO investigate training with with temporal ensembling.
TODO investigate using delta robot positions
TODO wider angle on Wrist camera
TODO remove reflection on ground (it's very reflective, could this be confusing things?) try plain black ground (no grids).



References (to be tidied up)

"Attention is All You Need" (Vaswani et al., 2017) https://arxiv.org/abs/1706.03762
"Chain-of-Thought..." (Wei et al., 2022) https://arxiv.org/abs/2201.11903
"Scaling Laws for Neural Language Models" (Kaplan et al., 2020)  https://arxiv.org/abs/2001.08361
"Compute Trends Across Three Eras of Machine Learning" (Sevilla et al., 2022)  https://arxiv.org/abs/2202.05924
The 2024 AI Index Report (Stanford HAI, 2024) https://hai.stanford.edu/ai-index/2024-ai-index-report
"PyTorch..." (Paszke et al., 2019) https://arxiv.org/pdf/1912.01703
"SigLIP" (Zhai et al., 2023) https://arxiv.org/pdf/2303.15343
"PaliGemma" (Beyer et al., 2024) https://arxiv.org/abs/2407.07726
Zhou, K., Liu, Z. and Gao, P. (eds.) (2024) Large Vision-Language Models: Pre-training, Prompting, and Applications. Cham: Springer
OpenVLA (Kim et al., 2024) https://arxiv.org/abs/2406.09246

Integrated Task and
Motion Planning (Garrett et al ) https://arxiv.org/pdf/2010.01083
