#!/usr/bin/env python
"""
Train a TinyVLA (Qwen2.5-VL-3B) policy on a LeRobot dataset via SFT.

The model takes tiled camera images + a task instruction and outputs
discretized actions as space-separated integers.

Usage:
    python scripts/training/train_tinyvla.py danbhf/sim_pick_place_220ep_pickup_tight \
        --horizon 8 --epochs 32 --batch_size 4 --lr 4e-5 \
        --gradient_checkpointing --output_dir outputs/train/tinyvla_pickup

    # Quick local test
    python scripts/training/train_tinyvla.py danbhf/sim_pick_place_220ep_pickup_tight \
        --epochs 1 --batch_size 1 --gradient_checkpointing --no_wandb \
        --output_dir outputs/train/tinyvla_test
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from transformers import AutoProcessor, LogitsProcessor, Trainer, TrainingArguments
from transformers import Qwen2_5_VLForConditionalGeneration


# ---------------------------------------------------------------------------
# Constrained decoding: only allow digits, spaces, and EOS
# ---------------------------------------------------------------------------
class NumberSpaceOnlyProcessor(LogitsProcessor):
    """Restrict generation to digits 0-9, space, and EOS tokens."""

    def __init__(self, tokenizer):
        self.allowed_ids = set()
        for tok_str in list("0123456789 "):
            ids = tokenizer.encode(tok_str, add_special_tokens=False)
            self.allowed_ids.update(ids)
        self.allowed_ids.add(tokenizer.eos_token_id)
        self.allowed_ids = sorted(self.allowed_ids)

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.allowed_ids] = 0.0
        return scores + mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PickPlaceVLADataset(Dataset):
    """Wraps a LeRobotDataset for VLA-style chat training.

    Each sample returns a chat-formatted dict with a tiled camera image
    and discretized action string as the assistant response.
    """

    IMG_SIZE = 224
    NUM_BINS = 1000

    def __init__(
        self,
        repo_id: str,
        horizon: int = 8,
        cam_list: tuple = ("observation.images.wrist_cam", "observation.images.overhead_cam"),
        instruction: str = "Pick up the block",
        color_jitter: bool = True,
    ):
        self.horizon = horizon
        self.cam_list = list(cam_list)
        self.instruction = instruction
        self.color_jitter = color_jitter

        # Load dataset metadata for stats
        metadata = LeRobotDatasetMetadata(repo_id)
        self.fps = metadata.fps
        action_stats = metadata.stats["action"]
        self.action_min = np.array(action_stats["min"])
        self.action_max = np.array(action_stats["max"])
        self.action_dim = len(self.action_min)

        # Load dataset with delta_timestamps for future actions
        delta_timestamps = {
            "action": [i / self.fps for i in range(horizon)],
        }
        for cam_key in self.cam_list:
            delta_timestamps[cam_key] = [0.0]

        self.dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)
        self.system_prompt = (
            f"You are a robot control model. Analyze the input image and predict robot actions "
            f"for the next {horizon} timesteps. Each action has {self.action_dim} dimensions. "
            f"Output a single sequence of {horizon * self.action_dim} integers "
            f"(0-{self.NUM_BINS} each), separated by spaces."
        )

        # Color jitter for augmentation
        if color_jitter:
            from torchvision import transforms
            self.jitter = transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
            )

    def __len__(self):
        return len(self.dataset)

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Normalize action to [0, 1] then discretize to ints 0..NUM_BINS."""
        denom = self.action_max - self.action_min
        denom = np.where(denom < 1e-8, 1.0, denom)
        normed = (action - self.action_min) / denom
        normed = np.clip(normed, 0.0, 1.0)
        return np.round(normed * self.NUM_BINS).astype(int)

    def _tile_images(self, images: list) -> Image.Image:
        """Tile PIL images horizontally into a single image resized to 224xN."""
        resized = [img.resize((self.IMG_SIZE, self.IMG_SIZE)) for img in images]
        total_w = self.IMG_SIZE * len(resized)
        tiled = Image.new("RGB", (total_w, self.IMG_SIZE))
        for i, img in enumerate(resized):
            tiled.paste(img, (i * self.IMG_SIZE, 0))
        return tiled

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # Extract camera images -> PIL
        pil_images = []
        for cam_key in self.cam_list:
            img_tensor = sample[cam_key]  # (C, H, W) or (1, C, H, W)
            if img_tensor.dim() == 4:
                img_tensor = img_tensor[0]
            # Apply color jitter before converting to PIL
            if self.color_jitter:
                img_tensor = self.jitter(img_tensor)
            # Convert [0,1] float or [0,255] uint8 to PIL
            if img_tensor.dtype == torch.float32:
                img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            else:
                img_np = img_tensor.permute(1, 2, 0).numpy()
            pil_images.append(Image.fromarray(img_np))

        tiled = self._tile_images(pil_images)

        # Extract and discretize actions
        actions = sample["action"]  # (horizon, action_dim)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        actions_np = actions.numpy()[:self.horizon]
        disc_actions = np.array([self._normalize_action(a) for a in actions_np])
        action_str = " ".join(str(v) for v in disc_actions.flatten())

        # Build chat messages
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.instruction},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": action_str}]},
        ]

        return {"messages": messages, "images": [tiled]}


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------
class VLACollator:
    """Tokenizes chat messages via Qwen processor, masks non-assistant tokens."""

    def __init__(self, processor, action_mask_aug_pct: float = 0.4):
        self.processor = processor
        self.action_mask_aug_pct = action_mask_aug_pct

    def __call__(self, examples):
        texts = []
        all_images = []

        for ex in examples:
            msgs = ex["messages"]

            # Action mask augmentation: randomly replace some digits in assistant text
            if self.action_mask_aug_pct > 0 and random.random() < 0.5:
                aug_msgs = []
                for msg in msgs:
                    if msg["role"] == "assistant":
                        text = msg["content"][0]["text"]
                        tokens = text.split()
                        n_mask = int(len(tokens) * self.action_mask_aug_pct)
                        mask_indices = random.sample(range(len(tokens)), min(n_mask, len(tokens)))
                        for mi in mask_indices:
                            tokens[mi] = str(random.randint(0, 1000))
                        aug_msgs.append({"role": "assistant", "content": [{"type": "text", "text": " ".join(tokens)}]})
                    else:
                        aug_msgs.append(msg)
                msgs = aug_msgs

            text = self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            all_images.append(ex["images"])

        batch = self.processor(
            text=texts,
            images=all_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )

        # Create labels: mask everything except assistant response
        labels = batch["input_ids"].clone()

        # Find assistant token boundaries and mask everything else
        for i in range(len(examples)):
            input_ids = batch["input_ids"][i]

            # Find the assistant header token sequence
            # In Qwen chat format, assistant content follows a specific pattern
            # We'll mask all tokens before the action digits
            assistant_msgs = examples[i]["messages"]
            assistant_text = None
            for msg in assistant_msgs:
                if msg["role"] == "assistant":
                    assistant_text = msg["content"][0]["text"]
                    break

            if assistant_text:
                # Encode just the assistant text to find its length
                assistant_ids = self.processor.tokenizer.encode(
                    assistant_text, add_special_tokens=False
                )
                # Mask everything except the last N tokens (the action tokens)
                # Plus a small buffer for the assistant header
                n_action_tokens = len(assistant_ids)
                # Find where non-padding ends
                non_pad = (input_ids != self.processor.tokenizer.pad_token_id).sum().item()
                # Mask from start to (end - action_tokens)
                mask_end = max(0, non_pad - n_action_tokens)
                labels[i, :mask_end] = -100

            # Also mask padding
            pad_mask = input_ids == self.processor.tokenizer.pad_token_id
            labels[i, pad_mask] = -100

        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train TinyVLA on LeRobot dataset")
    parser.add_argument("dataset", type=str, help="HuggingFace dataset repo ID")
    parser.add_argument("--horizon", type=int, default=8, help="Action prediction horizon (default: 8)")
    parser.add_argument("--epochs", type=int, default=32, help="Training epochs (default: 32)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--lr", type=float, default=4e-5, help="Learning rate (default: 4e-5)")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for parameter-efficient training")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--instruction", type=str, default="Pick up the block",
                        help="Task instruction (default: 'Pick up the block')")
    parser.add_argument("--cameras", type=str, default="observation.images.wrist_cam,observation.images.overhead_cam",
                        help="Comma-separated camera keys")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="lerobot-thesis", help="WandB project name")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="Base VLM model (default: Qwen/Qwen2.5-VL-3B-Instruct)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps (default: 1)")
    args = parser.parse_args()

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"outputs/train/tinyvla_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cam_list = [c.strip() for c in args.cameras.split(",")]

    print("=" * 60)
    print("TinyVLA Training")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Base model: {args.base_model}")
    print(f"Horizon: {args.horizon}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Cameras: {cam_list}")
    print(f"Instruction: {args.instruction}")
    print(f"Gradient checkpointing: {args.gradient_checkpointing}")
    print(f"LoRA: {args.use_lora}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Load model and processor
    print("\nLoading base model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    processor = AutoProcessor.from_pretrained(args.base_model)

    # Ensure pad token
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # LoRA
    if args.use_lora:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Dataset
    print("\nLoading dataset...")
    dataset = PickPlaceVLADataset(
        repo_id=args.dataset,
        horizon=args.horizon,
        cam_list=tuple(cam_list),
        instruction=args.instruction,
    )
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Action dim: {dataset.action_dim}, FPS: {dataset.fps}")

    # Collator
    collator = VLACollator(processor, action_mask_aug_pct=0.4)

    # WandB
    report_to = "none" if args.no_wandb else "wandb"
    run_name = f"tinyvla_{args.dataset.split('/')[-1]}_h{args.horizon}"

    # Training config
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        report_to=report_to,
        run_name=run_name,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving final model to {final_dir}...")
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))

    # Save dataset stats for denormalization at inference
    stats = {
        "action_min": dataset.action_min.tolist(),
        "action_max": dataset.action_max.tolist(),
        "num_bins": dataset.NUM_BINS,
        "horizon": args.horizon,
        "action_dim": dataset.action_dim,
    }
    with open(final_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Save training metadata (project convention)
    camera_names = [c.replace("observation.images.", "") for c in cam_list]
    training_metadata = {
        "dataset_repo_id": args.dataset,
        "model_type": "tinyvla",
        "base_model": args.base_model,
        "cameras": camera_names,
        "action_space": f"discretized (0-{dataset.NUM_BINS})",
        "action_dim": dataset.action_dim,
        "horizon": args.horizon,
        "chunk_size": args.horizon,
        "fps": dataset.fps,
        "total_frames": len(dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "use_lora": args.use_lora,
        "instruction": args.instruction,
    }
    with open(final_dir / "training_metadata.json", "w") as f:
        json.dump(training_metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Final model: {final_dir}")
    print(f"Dataset stats: {final_dir / 'dataset_stats.json'}")
    print(f"Training metadata: {final_dir / 'training_metadata.json'}")


if __name__ == "__main__":
    main()
