#!/usr/bin/env python
"""
Evaluate a trained TinyVLA model in simulation.

Provides a TinyVLAPolicy wrapper with select_action() interface compatible
with run_evaluation() and run_pickup_episodes().

Usage:
    # Full pick-and-place eval at training positions
    python scripts/inference/eval_tinyvla.py outputs/train/tinyvla_XXXX/final \
        --episodes 50

    # Pickup-only eval
    python scripts/inference/eval_tinyvla.py outputs/train/tinyvla_XXXX/final \
        --episodes 50 --pickup-only

    # Spatial grid eval (pickup)
    python scripts/inference/eval_tinyvla.py outputs/train/tinyvla_XXXX/final \
        --grid-size 5 --episodes 5 --pickup-only

    # With visualization
    python scripts/inference/eval_tinyvla.py outputs/train/tinyvla_XXXX/final \
        --episodes 10 --visualize
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# Reuse constrained decoding processor from training script
sys.path.insert(0, str(REPO_ROOT / "scripts" / "training"))
from train_tinyvla import NumberSpaceOnlyProcessor

from utils.training import run_evaluation, prepare_obs_for_policy
from utils.constants import MOTOR_NAMES, NUM_JOINTS


# ---------------------------------------------------------------------------
# Policy wrapper
# ---------------------------------------------------------------------------
class TinyVLAPolicy:
    """Wraps a fine-tuned Qwen2.5-VL model as a robot policy.

    Implements select_action(batch) to be compatible with run_evaluation()
    and run_pickup_episodes().
    """

    IMG_SIZE = 224

    def __init__(self, model_path: str, device: str = "cuda", instruction: str = None):
        model_path = Path(model_path)
        self.device = torch.device(device)

        # Check if this is a LoRA adapter checkpoint
        is_lora = (model_path / "adapter_config.json").exists()

        if is_lora:
            # Load base model + LoRA adapter
            with open(model_path / "adapter_config.json") as f:
                adapter_cfg = json.load(f)
            base_model_name = adapter_cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-VL-3B-Instruct")
            print(f"  Loading base model: {base_model_name}")
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(base_model, str(model_path))
            self.model = self.model.merge_and_unload()
            print(f"  LoRA adapter merged")
        else:
            # Load full model directly
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(model_path),
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
        self.model.to(self.device)
        self.model.eval()

        # Load processor (try checkpoint first, fall back to base model)
        if (model_path / "tokenizer_config.json").exists():
            self.processor = AutoProcessor.from_pretrained(str(model_path))
        else:
            self.processor = AutoProcessor.from_pretrained(base_model_name if is_lora else str(model_path))
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        # Load dataset stats
        stats_path = model_path / "dataset_stats.json"
        with open(stats_path) as f:
            stats = json.load(f)
        self.action_min = np.array(stats["action_min"])
        self.action_max = np.array(stats["action_max"])
        self.num_bins = stats["num_bins"]
        self.horizon = stats["horizon"]
        self.action_dim = stats["action_dim"]

        # Load training metadata
        meta_path = model_path / "training_metadata.json"
        self.training_meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                self.training_meta = json.load(f)

        # Instruction
        self.instruction = instruction or self.training_meta.get("instruction", "Pick up the block")

        # System prompt (must match training)
        self.system_prompt = (
            f"You are a robot control model. Analyze the input image and predict robot actions "
            f"for the next {self.horizon} timesteps. Each action has {self.action_dim} dimensions. "
            f"Output a single sequence of {self.horizon * self.action_dim} integers "
            f"(0-{self.num_bins} each), separated by spaces."
        )

        # Constrained decoding
        self.logits_processor = NumberSpaceOnlyProcessor(self.processor.tokenizer)

        # Camera names from training metadata
        camera_names = self.training_meta.get("cameras", ["wrist_cam", "overhead_cam"])
        self.cam_keys = [f"observation.images.{c}" for c in camera_names]

        # Action queue for serving horizon actions one by one
        self._action_queue = []

        # Build a minimal config for run_evaluation() camera detection
        input_features = {}
        for cam_name in camera_names:
            input_features[f"observation.images.{cam_name}"] = SimpleNamespace(
                shape=(3, self.IMG_SIZE, self.IMG_SIZE)
            )
        output_features = {
            "action": SimpleNamespace(shape=(self.action_dim,))
        }
        self.config = SimpleNamespace(
            input_features=input_features,
            output_features=output_features,
            chunk_size=self.horizon,
            n_action_steps=self.horizon,
        )

    def eval(self):
        self.model.eval()
        return self

    def reset(self):
        self._action_queue = []

    def _extract_images_from_batch(self, batch: dict) -> list:
        """Extract camera images from policy batch and return as PIL images."""
        pil_images = []
        for cam_key in self.cam_keys:
            if cam_key in batch:
                img_tensor = batch[cam_key]
                if img_tensor.dim() == 4:
                    img_tensor = img_tensor[0]  # Remove batch dim
                # Tensor is [C, H, W] in [0, 1] range after preprocessing
                img_np = (img_tensor.cpu().permute(1, 2, 0).float().numpy() * 255).clip(0, 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img_np))
        return pil_images

    def _tile_images(self, images: list) -> Image.Image:
        """Tile PIL images horizontally."""
        resized = [img.resize((self.IMG_SIZE, self.IMG_SIZE)) for img in images]
        total_w = self.IMG_SIZE * len(resized)
        tiled = Image.new("RGB", (total_w, self.IMG_SIZE))
        for i, img in enumerate(resized):
            tiled.paste(img, (i * self.IMG_SIZE, 0))
        return tiled

    def _decode_actions(self, text: str) -> np.ndarray:
        """Parse integer tokens from model output and denormalize to continuous actions."""
        tokens = text.strip().split()
        expected = self.horizon * self.action_dim

        ints = []
        for tok in tokens:
            try:
                val = int(tok)
                val = max(0, min(self.num_bins, val))
                ints.append(val)
            except ValueError:
                continue

        # Pad or truncate
        if len(ints) < expected:
            ints.extend([self.num_bins // 2] * (expected - len(ints)))
        ints = ints[:expected]

        # Reshape and denormalize
        disc = np.array(ints).reshape(self.horizon, self.action_dim)
        normed = disc.astype(float) / self.num_bins
        actions = normed * (self.action_max - self.action_min) + self.action_min
        return actions

    @torch.no_grad()
    def _generate_actions(self, batch: dict) -> np.ndarray:
        """Run model inference and return (horizon, action_dim) array."""
        pil_images = self._extract_images_from_batch(batch)
        if not pil_images:
            return np.zeros((self.horizon, self.action_dim))

        tiled = self._tile_images(pil_images)

        # Build chat messages (no assistant — we want the model to generate)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.instruction},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[[tiled]],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Generate with constrained decoding
        max_new = self.horizon * self.action_dim * 5  # ~5 chars per number
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            logits_processor=[self.logits_processor],
        )

        # Decode only the new tokens
        input_len = inputs["input_ids"].shape[1]
        new_ids = output_ids[0, input_len:]
        text_out = self.processor.tokenizer.decode(new_ids, skip_special_tokens=True)

        return self._decode_actions(text_out)

    def select_action(self, batch: dict) -> torch.Tensor:
        """Return a single action. Generates a full chunk when queue is empty."""
        if not self._action_queue:
            actions = self._generate_actions(batch)
            self._action_queue = list(actions)

        action = self._action_queue.pop(0)
        return torch.from_numpy(action).float().unsqueeze(0)


# ---------------------------------------------------------------------------
# No-op pre/post processors
# ---------------------------------------------------------------------------
def noop_preprocessor(batch):
    """Pass-through preprocessor — TinyVLAPolicy handles its own processing."""
    return batch


def noop_postprocessor(action):
    """Pass-through postprocessor — TinyVLAPolicy handles denormalization."""
    return action


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate TinyVLA model")
    parser.add_argument("path", type=str, help="Path to model checkpoint (e.g., outputs/train/tinyvla_XXXX/final)")
    parser.add_argument("--episodes", type=int, default=50, help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per episode")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--instruction", type=str, default=None, help="Override task instruction")
    parser.add_argument("--visualize", action="store_true", help="Show camera feeds")
    parser.add_argument("--mujoco-viewer", action="store_true", help="Open MuJoCo 3D viewer")
    parser.add_argument("--block-x", type=float, default=None, help="Fixed block X position")
    parser.add_argument("--block-y", type=float, default=None, help="Fixed block Y position")

    # Pickup-only mode
    parser.add_argument("--pickup-only", action="store_true", help="Run pickup-only evaluation")
    parser.add_argument("--grid-size", type=int, default=None, help="Spatial grid size (e.g., 5 for 5x5)")
    parser.add_argument("--lift-height", type=float, default=0.05, help="Lift height for pickup success (m)")
    parser.add_argument("--x-min", type=float, default=0.10)
    parser.add_argument("--x-max", type=float, default=0.35)
    parser.add_argument("--y-min", type=float, default=0.08)
    parser.add_argument("--y-max", type=float, default=0.38)

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model_path = Path(args.path)

    if not model_path.exists():
        print(f"ERROR: Model path not found: {model_path}")
        sys.exit(1)

    print(f"Loading TinyVLA from: {model_path}")
    policy = TinyVLAPolicy(str(model_path), device=device, instruction=args.instruction)
    print(f"  Horizon: {policy.horizon}, Action dim: {policy.action_dim}")
    print(f"  Cameras: {policy.cam_keys}")
    print(f"  Instruction: {policy.instruction}")

    if args.pickup_only:
        _run_pickup_eval(policy, args, device)
    else:
        _run_full_eval(policy, args, device)


def _run_full_eval(policy, args, device):
    """Full pick-and-place evaluation using run_evaluation()."""
    results = run_evaluation(
        policy=policy,
        preprocessor=noop_preprocessor,
        postprocessor=noop_postprocessor,
        device=torch.device(device),
        num_episodes=args.episodes,
        randomize=True,
        action_dim=policy.action_dim,
        max_steps=args.max_steps,
        verbose=True,
        analyze_failures=True,
        visualize=args.visualize,
        mujoco_viewer=args.mujoco_viewer,
        block_x=args.block_x,
        block_y=args.block_y,
    )

    success_rate, avg_steps, avg_time, _, _, failure_summary = results

    print("\n" + "=" * 60)
    print("TINYVLA EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Model: {args.path}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Success Rate: {success_rate * 100:.1f}%")
    print(f"  Avg Steps: {avg_steps:.1f}")
    print(f"  Avg Time: {avg_time:.2f}s")
    print("=" * 60)

    if failure_summary:
        print("\nFailure Analysis:")
        for key, value in failure_summary.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")


def _run_pickup_eval(policy, args, device):
    """Pickup-only evaluation using run_pickup_episodes()."""
    from scripts.experiments.eval_pickup_model_spatial import run_pickup_episodes
    from lerobot_robot_sim import SO100SimConfig, SO100Sim

    scene_path = REPO_ROOT / "scenes" / "so101_with_wrist_cam.xml"
    camera_names = policy.training_meta.get("cameras", ["wrist_cam", "overhead_cam"])

    config = SO100SimConfig(
        scene_xml=str(scene_path),
        sim_cameras=camera_names,
        camera_width=640,
        camera_height=480,
    )
    sim = SO100Sim(config)
    sim.connect()

    # Build position list
    if args.block_x is not None and args.block_y is not None:
        positions = [(args.block_x, args.block_y)]
    elif args.grid_size:
        xs = np.linspace(args.x_min, args.x_max, args.grid_size)
        ys = np.linspace(args.y_min, args.y_max, args.grid_size)
        positions = [(x, y) for x in xs for y in ys]
    else:
        positions = [
            (0.213, 0.254),
            (0.213, -0.047),
        ]

    print(f"\nTINYVLA PICKUP EVALUATION")
    print(f"  Positions: {len(positions)}")
    print(f"  Episodes per position: {args.episodes}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Lift height: {args.lift_height}m")
    print()

    all_results = []

    for pos_idx, (x, y) in enumerate(positions):
        print(f"  Position {pos_idx + 1}/{len(positions)}: ({x:.3f}, {y:.3f})...", end=" ", flush=True)

        succ, total, details, _ = run_pickup_episodes(
            sim, policy, noop_preprocessor, noop_postprocessor,
            torch.device(device),
            block_x=x, block_y=y,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            viewer=args.mujoco_viewer,
            lift_height=args.lift_height,
        )

        rate = succ / total if total > 0 else 0
        approach_count = sum(1 for d in details if d["approached"])
        print(f"{rate * 100:.0f}% ({succ}/{total}), approached={approach_count}")

        all_results.append({"x": x, "y": y, "rate": rate, "succ": succ, "total": total})

    sim.disconnect()

    # Summary
    print("\n" + "=" * 60)
    print("TINYVLA PICKUP SUMMARY")
    print("=" * 60)
    rates = [r["rate"] for r in all_results]
    total_succ = sum(r["succ"] for r in all_results)
    total_eps = sum(r["total"] for r in all_results)
    print(f"  Overall: {total_succ}/{total_eps} ({np.mean(rates) * 100:.1f}%)")
    print(f"  Positions >0%: {sum(1 for r in rates if r > 0)}/{len(rates)}")
    print(f"  Positions 100%: {sum(1 for r in rates if r >= 1.0)}/{len(rates)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
