#!/usr/bin/env python3
"""
Visualize ACT attention maps overlaid on camera images.

Shows what regions of the image the model attends to when predicting actions.

Usage:
    python scripts/tools/visualize_attention.py outputs/train/act_20260118_155135 --checkpoint checkpoint_045000
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Add project paths
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot_robot_sim import SO100Sim, SO100SimConfig

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class AttentionCapture:
    """Captures attention weights from MultiheadAttention layers using pre-hooks."""

    def __init__(self):
        self.attention_weights = {}
        self.hooks = []
        self._in_capture = False

    def pre_hook_fn(self, name):
        """Pre-hook to force need_weights=True."""
        def hook(module, args):
            # Force need_weights=True by modifying the module temporarily
            if not self._in_capture:
                return args
            return args
        return hook

    def post_hook_fn(self, name):
        """Post-hook to capture attention weights from output."""
        def hook(module, args, output):
            if self._in_capture:
                # output is (attn_output, attn_weights) when need_weights=True
                # But by default need_weights=True in PyTorch MHA
                if isinstance(output, tuple) and len(output) == 2:
                    attn_output, attn_weights = output
                    if attn_weights is not None:
                        self.attention_weights[name] = attn_weights.detach().cpu()
        return hook

    def register_hooks(self, model):
        """Register hooks on all MultiheadAttention layers."""
        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                # Store original need_weights setting
                handle = module.register_forward_hook(self.post_hook_fn(name))
                self.hooks.append(handle)
                print(f"Hooked: {name}")

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear(self):
        self.attention_weights = {}


def load_act_policy(model_path: Path, device: torch.device):
    """Load ACT policy."""
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    print(f"Loading ACT from {model_path}...")
    policy = ACTPolicy.from_pretrained(str(model_path))
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(model_path)
    )

    return policy, preprocessor, postprocessor


def prepare_obs(obs: dict, device: str = "cuda") -> dict:
    """Convert sim observation to policy input format."""
    batch = {}

    # State
    state = np.array([obs[m + ".pos"] for m in MOTOR_NAMES], dtype=np.float32)
    batch["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(device)

    # Images
    for key, value in obs.items():
        if isinstance(value, np.ndarray) and value.ndim == 3:
            img = torch.from_numpy(value).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            batch[f"observation.images.{key}"] = img.to(device)

    return batch


def visualize_attention_on_image(
    image: np.ndarray,
    attention: np.ndarray,
    title: str = "Attention",
    ax=None,
):
    """Overlay attention heatmap on image.

    Args:
        image: Original image (H, W, 3) in RGB, values 0-255
        attention: Attention weights (h, w) - will be resized to match image
        title: Plot title
        ax: Matplotlib axis (creates new figure if None)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Resize attention to image size
    h, w = image.shape[:2]
    attention_resized = cv2.resize(attention, (w, h), interpolation=cv2.INTER_LINEAR)

    # Normalize attention for visualization
    attention_norm = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min() + 1e-8)

    # Show image
    ax.imshow(image)

    # Overlay attention as heatmap
    ax.imshow(attention_norm, cmap='jet', alpha=0.5)
    ax.set_title(title)
    ax.axis('off')

    return ax


def extract_image_attention(
    attention_weights: dict,
    num_image_tokens: int,
    image_shape: tuple,  # (H, W) of feature map
    layer_name: str = None,
):
    """Extract attention weights for image tokens and reshape to spatial.

    Args:
        attention_weights: Dict of attention weights from capture
        num_image_tokens: Number of image tokens (H*W of feature map)
        image_shape: (H, W) of the feature map
        layer_name: Specific layer to extract (None = last decoder cross-attention)

    Returns:
        Dict mapping camera names to attention maps (H, W)
    """
    # Find the decoder cross-attention layer
    if layer_name is None:
        # Look for decoder multihead attention (cross-attention)
        for name in attention_weights:
            if 'decoder' in name and 'multihead_attn' in name:
                layer_name = name
                break

    if layer_name is None or layer_name not in attention_weights:
        print(f"Available layers: {list(attention_weights.keys())}")
        return None

    attn = attention_weights[layer_name]  # (batch, query_len, key_len)
    print(f"Attention shape from {layer_name}: {attn.shape}")

    # attn shape: (1, chunk_size, encoder_seq_len)
    # encoder_seq_len = 1 (latent) + 1 (state) + num_image_tokens * num_cameras

    # Average attention across all action queries (chunk dimension)
    attn_avg = attn[0].mean(dim=0)  # (encoder_seq_len,)

    # The image tokens start after latent and state tokens
    # Assuming: [latent, state, img1_tokens..., img2_tokens...]
    offset = 2  # latent + state

    h, w = image_shape

    # Extract attention for each camera's image tokens
    results = {}
    num_cameras = (len(attn_avg) - offset) // num_image_tokens

    for cam_idx in range(num_cameras):
        start = offset + cam_idx * num_image_tokens
        end = start + num_image_tokens
        cam_attn = attn_avg[start:end].numpy()

        # Reshape to spatial
        cam_attn_spatial = cam_attn.reshape(h, w)
        results[f"camera_{cam_idx}"] = cam_attn_spatial

    return results


def main():
    parser = argparse.ArgumentParser(description="Visualize ACT attention maps")
    parser.add_argument("model_path", type=str, help="Path to model directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_045000")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save", type=str, default=None, help="Save visualization to file")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model_path = Path(args.model_path) / args.checkpoint
    policy, preprocessor, postprocessor = load_act_policy(model_path, device)

    # Set up attention capture
    attention_capture = AttentionCapture()
    attention_capture.register_hooks(policy.model)

    # Create simulation
    scene_path = REPO_ROOT / "scenes" / "so101_with_wrist_cam.xml"
    sim_config = SO100SimConfig(
        scene_xml=str(scene_path),
        sim_cameras=["overhead_cam", "wrist_cam"],
        camera_width=640,
        camera_height=480,
    )
    sim = SO100Sim(sim_config)
    sim.connect()
    sim.reset_scene(randomize=False)

    # Get observation
    obs = sim.get_observation()

    # Store original images for visualization
    images = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray) and value.ndim == 3:
            images[key] = value.copy()  # RGB, 0-255

    # Prepare observation and run inference
    batch = prepare_obs(obs, device)
    batch = preprocessor(batch)

    attention_capture.clear()
    attention_capture._in_capture = True

    with torch.no_grad():
        # Use select_action which handles the batch format correctly
        actions = policy.select_action(batch)

    attention_capture._in_capture = False

    # Get image feature map dimensions from the model
    # ResNet18 with layer4 output: input_size / 32
    # For 640x480 input: 20x15 feature map
    img_h, img_w = 480 // 32, 640 // 32  # 15, 20
    num_image_tokens = img_h * img_w  # 300

    print(f"\nFeature map size: {img_h}x{img_w} = {num_image_tokens} tokens per camera")
    print(f"Captured attention from {len(attention_capture.attention_weights)} layers")

    # Extract attention maps
    attn_maps = extract_image_attention(
        attention_capture.attention_weights,
        num_image_tokens,
        (img_h, img_w),
    )

    if attn_maps is None:
        print("Could not extract attention maps")

        # Show what we have
        print("\nAvailable attention layers:")
        for name, weights in attention_capture.attention_weights.items():
            print(f"  {name}: {weights.shape}")

        attention_capture.remove_hooks()
        sim.disconnect()
        return

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    camera_names = list(images.keys())

    for idx, cam_name in enumerate(camera_names[:2]):
        # Original image
        ax = axes[0, idx]
        ax.imshow(images[cam_name])
        ax.set_title(f"{cam_name} - Original")
        ax.axis('off')

        # Attention overlay
        ax = axes[1, idx]
        if f"camera_{idx}" in attn_maps:
            visualize_attention_on_image(
                images[cam_name],
                attn_maps[f"camera_{idx}"],
                title=f"{cam_name} - Attention",
                ax=ax,
            )
        else:
            ax.imshow(images[cam_name])
            ax.set_title(f"{cam_name} - No attention data")
            ax.axis('off')

    plt.suptitle("ACT Decoder Cross-Attention Maps", fontsize=14)
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"Saved to {args.save}")
    else:
        plt.show()

    # Cleanup
    attention_capture.remove_hooks()
    sim.disconnect()


if __name__ == "__main__":
    main()
