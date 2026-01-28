#!/usr/bin/env python3
"""
Live ACT attention visualization during episode execution.

Shows camera views with attention heatmaps updating in real-time.

Usage:
    python scripts/tools/visualize_attention_live.py outputs/train/act_20260118_155135 --checkpoint checkpoint_045000
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive display
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
    """Captures attention weights from MultiheadAttention layers."""

    def __init__(self):
        self.attention_weights = {}
        self.hooks = []

    def post_hook_fn(self, name):
        def hook(module, args, output):
            if isinstance(output, tuple) and len(output) == 2:
                attn_output, attn_weights = output
                if attn_weights is not None:
                    self.attention_weights[name] = attn_weights.detach().cpu()
        return hook

    def register_hooks(self, model):
        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                handle = module.register_forward_hook(self.post_hook_fn(name))
                self.hooks.append(handle)

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

    state = np.array([obs[m + ".pos"] for m in MOTOR_NAMES], dtype=np.float32)
    batch["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(device)

    for key, value in obs.items():
        if isinstance(value, np.ndarray) and value.ndim == 3:
            img = torch.from_numpy(value).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            batch[f"observation.images.{key}"] = img.to(device)

    return batch


def extract_attention_grids(attention_weights: dict, layer_name: str = "decoder.layers.0.multihead_attn"):
    """Extract attention grids for each camera.

    Returns:
        tuple: (overhead_grid, wrist_grid) each as 15x20 numpy arrays
    """
    if layer_name not in attention_weights:
        return None, None

    attn = attention_weights[layer_name]  # [1, 100, 602]

    # Average across all action queries
    attn_avg = attn[0].mean(dim=0).numpy()  # [602]

    # Split: [latent, state, 300 overhead tiles, 300 wrist tiles]
    offset = 2
    overhead_attn = attn_avg[offset:offset+300]
    wrist_attn = attn_avg[offset+300:offset+600]

    # Reshape to 15x20 grids
    overhead_grid = overhead_attn.reshape(15, 20)
    wrist_grid = wrist_attn.reshape(15, 20)

    return overhead_grid, wrist_grid


def create_attention_overlay(image: np.ndarray, attention_grid: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Create attention heatmap overlay on image.

    Args:
        image: RGB image (H, W, 3), values 0-255
        attention_grid: Attention weights (15, 20)
        alpha: Overlay transparency

    Returns:
        BGR image with attention overlay (for OpenCV display)
    """
    h, w = image.shape[:2]

    # Normalize attention to 0-255
    attn_norm = attention_grid - attention_grid.min()
    attn_max = attn_norm.max()
    if attn_max > 0:
        attn_norm = attn_norm / attn_max
    attn_uint8 = (attn_norm * 255).astype(np.uint8)

    # Resize to image size (nearest neighbor to show tiles)
    attn_resized = cv2.resize(attn_uint8, (w, h), interpolation=cv2.INTER_NEAREST)

    # Apply colormap
    heatmap = cv2.applyColorMap(attn_resized, cv2.COLORMAP_JET)

    # Convert image to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Blend
    overlay = cv2.addWeighted(image_bgr, 1 - alpha, heatmap, alpha, 0)

    return overlay


def run_live_visualization(
    policy,
    preprocessor,
    postprocessor,
    attention_capture: AttentionCapture,
    device: torch.device,
    max_steps: int = 300,
    show_tiles: bool = True,
    scene: str = "so101_with_wrist_cam.xml",
    randomize: bool = True,
):
    """Run episode with live attention visualization using matplotlib."""

    # Create simulation
    scene_path = REPO_ROOT / "scenes" / scene
    sim_config = SO100SimConfig(
        scene_xml=str(scene_path),
        sim_cameras=["overhead_cam", "wrist_cam"],
        camera_width=640,
        camera_height=480,
    )
    sim = SO100Sim(sim_config)
    sim.connect()
    sim.reset_scene(randomize=randomize)

    # Reset policy state
    policy.reset()

    n_action_steps = policy.config.n_action_steps

    print("\n" + "="*60)
    print("LIVE ATTENTION VISUALIZATION")
    print("="*60)
    print("Close the window to stop")
    print("="*60 + "\n")

    # Set up matplotlib figure
    plt.ion()  # Interactive mode
    if show_tiles:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes = axes.reshape(1, -1)

    fig.suptitle("ACT Live Attention Visualization", fontsize=14)

    # Initialize image displays
    overhead_grid = np.zeros((15, 20))
    wrist_grid = np.zeros((15, 20))

    step = 0
    running = True
    chunk_step = 0  # Track position within chunk for display

    def on_close(event):
        nonlocal running
        running = False

    fig.canvas.mpl_connect('close_event', on_close)

    while running and step < max_steps:
        # Get observation
        obs = sim.get_observation()

        overhead_img = obs["overhead_cam"].copy()
        wrist_img = obs["wrist_cam"].copy()

        # Prepare batch and get action
        batch = prepare_obs(obs, device)
        batch = preprocessor(batch)

        attention_capture.clear()

        with torch.no_grad():
            action = policy.select_action(batch)

        # Extract attention grids (only when new chunk is predicted)
        # The policy predicts a new chunk every n_action_steps
        if step % n_action_steps == 0:
            overhead_grid, wrist_grid = extract_attention_grids(attention_capture.attention_weights)
            if overhead_grid is None:
                overhead_grid = np.zeros((15, 20))
                wrist_grid = np.zeros((15, 20))
            chunk_step = 0
        else:
            chunk_step += 1

        action = postprocessor(action)

        # Execute action
        action_np = action.cpu().numpy().flatten()
        action_dict = {m + ".pos": float(action_np[i]) for i, m in enumerate(MOTOR_NAMES)}
        sim.send_action(action_dict)

        # Check success
        success = sim.is_task_complete()
        if success:
            print(f"\n*** SUCCESS at step {step}! ***\n")

        # Update visualization
        for ax_row in axes:
            for ax in ax_row:
                ax.clear()

        # Create attention overlays
        overhead_attn_resized = cv2.resize(overhead_grid, (640, 480), interpolation=cv2.INTER_NEAREST)
        wrist_attn_resized = cv2.resize(wrist_grid, (640, 480), interpolation=cv2.INTER_NEAREST)

        # Normalize for display
        def normalize(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-8)

        if show_tiles:
            # Row 0: Camera images with attention overlay
            axes[0, 0].imshow(overhead_img)
            axes[0, 0].imshow(normalize(overhead_attn_resized), cmap='jet', alpha=0.4)
            axes[0, 0].set_title(f"Overhead + Attention (Step {step})")
            axes[0, 0].axis('off')

            axes[0, 1].imshow(wrist_img)
            axes[0, 1].imshow(normalize(wrist_attn_resized), cmap='jet', alpha=0.4)
            axes[0, 1].set_title(f"Wrist + Attention")
            axes[0, 1].axis('off')

            # Status display
            axes[0, 2].text(0.5, 0.7, f"Step: {step}", fontsize=20, ha='center', va='center')
            axes[0, 2].text(0.5, 0.5, f"Chunk: {chunk_step + 1}/{n_action_steps}",
                           fontsize=14, ha='center', va='center')
            if success:
                axes[0, 2].text(0.5, 0.3, "SUCCESS!", fontsize=24, ha='center', va='center',
                               color='green', fontweight='bold')
            axes[0, 2].set_xlim(0, 1)
            axes[0, 2].set_ylim(0, 1)
            axes[0, 2].axis('off')

            # Row 1: Raw attention tiles
            im1 = axes[1, 0].imshow(overhead_grid, cmap='hot', interpolation='nearest', aspect='auto')
            axes[1, 0].set_title("Overhead Tiles (15×20)")
            axes[1, 0].set_xlabel("20 tiles")
            axes[1, 0].set_ylabel("15 tiles")

            im2 = axes[1, 1].imshow(wrist_grid, cmap='hot', interpolation='nearest', aspect='auto')
            axes[1, 1].set_title("Wrist Tiles (15×20)")
            axes[1, 1].set_xlabel("20 tiles")

            # Attention distribution
            axes[1, 2].bar(['Overhead', 'Wrist'], [overhead_grid.sum(), wrist_grid.sum()], color=['#ff7f0e', '#2ca02c'])
            axes[1, 2].set_title("Total Attention per Camera")
            axes[1, 2].set_ylabel("Sum of attention weights")

        else:
            # Simple view without tiles
            axes[0, 0].imshow(overhead_img)
            axes[0, 0].imshow(normalize(overhead_attn_resized), cmap='jet', alpha=0.4)
            axes[0, 0].set_title(f"Overhead + Attention (Step {step})")
            axes[0, 0].axis('off')

            axes[0, 1].imshow(wrist_img)
            axes[0, 1].imshow(normalize(wrist_attn_resized), cmap='jet', alpha=0.4)
            axes[0, 1].set_title(f"Wrist + Attention")
            axes[0, 1].axis('off')

        plt.tight_layout()
        plt.pause(0.01)

        if success:
            plt.pause(2)  # Pause to show success
            break

        step += 1

    print(f"\nEpisode ended at step {step}")

    plt.ioff()
    plt.close(fig)
    sim.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Live ACT attention visualization")
    parser.add_argument("model_path", type=str, help="Path to model directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_045000")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--no-tiles", action="store_true", help="Hide raw tile view")
    parser.add_argument("--scene", type=str, default="so101_with_wrist_cam.xml",
                        help="Scene XML file (in scenes/ directory)")
    parser.add_argument("--no-randomize", action="store_true",
                        help="Disable block position randomization")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model_path = Path(args.model_path) / args.checkpoint
    policy, preprocessor, postprocessor = load_act_policy(model_path, device)

    # Set up attention capture
    attention_capture = AttentionCapture()
    attention_capture.register_hooks(policy.model)

    try:
        run_live_visualization(
            policy,
            preprocessor,
            postprocessor,
            attention_capture,
            device,
            max_steps=args.max_steps,
            show_tiles=not args.no_tiles,
            scene=args.scene,
            randomize=not args.no_randomize,
        )
    finally:
        attention_capture.remove_hooks()


if __name__ == "__main__":
    main()
