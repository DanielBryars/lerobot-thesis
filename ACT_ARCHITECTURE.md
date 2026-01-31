# ACT (Action Chunking Transformer) Architecture Deep Dive

This document explains how ACT works internally, how gradients flow during training, and why certain failure modes (like mode collapse with coordinate conditioning) occur.

## Architecture Overview

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

The "z" is a **style variable** that captures the variability in how humans perform the same task.

### Why z Exists

Given the same starting state, there are many valid ways to pick up a block. You might approach from the left or right, go faster or slower, etc. The VAE learns to encode this "style" into z.

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

## Part 7: The Coordinate Conditioning Problem

### Why Coordinates Get Ignored

The pickup coordinates become **ONE token** among 603 tokens. That's 0.17% of the input. The model can easily learn to ignore it if the images provide enough information.

The critical issue: **ResNet processes images BEFORE coordinates are involved**:

```python
cam_features = self.backbone(img)["feature_map"]  # ResNet sees whole image
```

ResNet extracts features from BOTH blocks equally. Then the coordinate token has to somehow tell the transformer "ignore those 150 feature tokens that correspond to the confuser block." That's hard!

### Why Mode Collapse Happens

The L1 loss for a batch might look like:

```
Sample 1: coords=(0.2, 0.2), target picks up from pos1
Sample 2: coords=(0.3, -0.1), target picks up from pos2
```

If the model predicts the AVERAGE trajectory:
- Sample 1 error: moderate
- Sample 2 error: moderate
- **Total loss: acceptable!**

If the model tries to use coordinates:
- Early in training, coord encoding is noisy
- Model makes HIGH errors when it gets coords wrong
- Gradient says "coords are hurting, ignore them"

The gradient descent finds the local minimum of "output average, ignore coords."

### Evidence of Mode Collapse

In our experiments (see EXPERIMENTS.md, Experiment 5d), we observed:
- Loss converged to 0.056 (good!)
- Success rate: 0%
- Robot tries to pick up from **halfway between the two blocks**

The model learned to output the average trajectory, which minimizes MSE but accomplishes nothing.

## Part 8: Potential Solutions

### 1. Coordinate Prediction Auxiliary Loss (Easiest)

Add a head that must predict the input coordinates from the model's latent representation:

```python
# In ACT forward pass, after encoder
coord_pred = self.coord_predictor(encoder_output)  # MLP: hidden -> 2
coord_loss = F.mse_loss(coord_pred, input_coords)

total_loss = action_loss + 0.1 * coord_loss
```

This forces the model to actually encode coordinate information - if it ignores coords, it can't predict them.

### 2. Contrastive Coordinate Loss

Penalize when different coordinates produce similar actions:

```python
# Sample pairs with different coordinates
# If coords are different but actions are similar, add penalty
coord_distance = torch.norm(coords_a - coords_b)
action_distance = torch.norm(actions_a - actions_b)

# If coords differ, actions should differ proportionally
contrastive_loss = F.relu(margin - action_distance) * (coord_distance > threshold)
```

### 3. Coordinate-Guided Image Cropping

Crop images around target coordinates BEFORE ResNet sees them:

```python
crop_center = coords_to_pixel(pickup_coords)
cropped = crop_around(image, crop_center, size=128)
features = resnet(cropped)
```

This forces the model to only see the target block.

### 4. FiLM Conditioning

Inject coordinates INTO ResNet layers so it can focus on the right region:

```python
# At each ResNet block
gamma, beta = film_generator(coordinates)
features = gamma * features + beta
```

### 5. Increase Coordinate Signal Strength

Make coordinates more prominent:
- Increase `observation.environment_state` embedding dimension
- Add coordinate info to multiple layers (not just input)
- Concatenate coords with image features after each transformer layer

## Key Insight

The 157-episode model (trained WITHOUT confuser) works perfectly ON the confuser scene when given coordinates. But training WITH confuser data causes complete failure.

This suggests the model trained without confuser learned to focus on the target block's visual features. When evaluated with a confuser present, it treats the confuser as "unknown visual noise" and still uses coordinates correctly.

When trained WITH confuser visible, the model sees two identical blocks and can't learn which visual features correspond to which coordinates - so it gives up on coordinates entirely and outputs the average.
