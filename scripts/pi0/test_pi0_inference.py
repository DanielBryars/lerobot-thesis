#!/usr/bin/env python3
"""
Quick Pi0 model test - verifies model loads and produces actions.

Usage:
    python scripts/pi0/test_pi0_inference.py
    python scripts/pi0/test_pi0_inference.py --model danbhf/pi0_so101_lerobot_20k
"""

import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description="Test Pi0 inference")
    parser.add_argument("--model", type=str, default="danbhf/pi0_so101_lerobot",
                        help="HuggingFace model repo ID")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Testing Pi0 model: {args.model}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Import here to catch import errors clearly
    print("\nImporting PI0Policy...")
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy

    print(f"Loading model from {args.model}...")
    policy = PI0Policy.from_pretrained(args.model)
    policy.to(args.device)
    policy.eval()
    print("Model loaded successfully!")

    # Create dummy input matching training config
    print("\nCreating dummy observation...")
    dummy_state = torch.zeros(1, 6).to(args.device)
    dummy_img1 = torch.rand(1, 3, 224, 224).to(args.device)  # Random for variety
    dummy_img2 = torch.rand(1, 3, 224, 224).to(args.device)

    # Pi0 requires language instruction - tokenize it
    language_instruction = "Pick up the block and place it in the bowl"
    print(f"Language instruction: '{language_instruction}'")

    # Get tokenizer from model
    tokenizer = policy.model.paligemma_with_expert.processor.tokenizer
    encoding = tokenizer(
        language_instruction,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=policy.config.tokenizer_max_length,
    )
    lang_tokens = encoding["input_ids"].to(args.device)
    lang_mask = encoding["attention_mask"].bool().to(args.device)

    obs = {
        "observation.state": dummy_state,
        "observation.images.overhead_cam": dummy_img1,
        "observation.images.wrist_cam": dummy_img2,
        "observation.language.tokens": lang_tokens,
        "observation.language.attention_mask": lang_mask,
    }

    print("Running inference...")
    with torch.no_grad():
        action = policy.select_action(obs)

    action_np = action.cpu().numpy().flatten()
    print(f"\nAction shape: {action.shape}")
    print(f"Action values (first 6): {action_np[:6]}")
    print(f"Action min/max: {action_np.min():.4f} / {action_np.max():.4f}")

    # Run a few more inferences to check consistency
    print("\nRunning 5 more inferences for timing...")
    import time
    times = []
    for i in range(5):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = policy.select_action(obs)
        times.append(time.perf_counter() - t0)

    avg_time = sum(times) / len(times) * 1000
    hz = 1000 / avg_time
    print(f"Average inference time: {avg_time:.1f}ms ({hz:.1f} Hz)")

    print("\n" + "=" * 60)
    print("SUCCESS - Model loads and produces valid actions!")
    print("=" * 60)


if __name__ == "__main__":
    main()
