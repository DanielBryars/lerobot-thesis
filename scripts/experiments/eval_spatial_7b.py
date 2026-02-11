#!/usr/bin/env python
"""
Spatial generalization evaluation for ACT-ViT models with subtask/coords support.

Evaluates model at a grid of block positions and outputs results as CSV.

Usage:
    python scripts/experiments/eval_spatial_7b.py \
        outputs/train/act_vit_subtask_coords_157ep \
        --grid-size 7 --episodes 5
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add project root
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.inference.eval import load_policy_and_processors
from utils.training import run_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Model path")
    parser.add_argument("--policy", type=str, default="act_vit")
    parser.add_argument("--grid-size", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per grid position")
    parser.add_argument("--x-min", type=float, default=0.10)
    parser.add_argument("--x-max", type=float, default=0.35)
    parser.add_argument("--y-min", type=float, default=0.08)
    parser.add_argument("--y-max", type=float, default=0.38)
    parser.add_argument("--subtask", action="store_true")
    parser.add_argument("--pickup-coords", action="store_true")
    parser.add_argument("--selective-coords", action="store_true")
    parser.add_argument("--blinkering", action="store_true",
                        help="Enable blinkering: mask overhead camera during PICK_UP/DROP subtasks")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_path = Path(args.path)

    # Auto-detect final checkpoint
    final = model_path / "final"
    if final.exists():
        model_path = final

    # Load model once
    policy, preprocessor, postprocessor = load_policy_and_processors(
        model_path, args.policy, device, None
    )

    # Generate grid
    xs = np.linspace(args.x_min, args.x_max, args.grid_size)
    ys = np.linspace(args.y_min, args.y_max, args.grid_size)

    # Output CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.output or f"outputs/experiments/spatial_7b_{timestamp}.csv"
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    total_positions = args.grid_size ** 2
    print(f"Evaluating {total_positions} positions, {args.episodes} episodes each")
    print(f"X: {args.x_min:.2f} to {args.x_max:.2f} ({args.grid_size} steps)")
    print(f"Y: {args.y_min:.2f} to {args.y_max:.2f} ({args.grid_size} steps)")
    print(f"Output: {csv_path}")
    print()

    results = []
    pos_idx = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "success_rate", "episodes", "successes"])

        for x in xs:
            for y in ys:
                pos_idx += 1
                print(f"  Position {pos_idx}/{total_positions}: ({x:.3f}, {y:.3f})...", end=" ", flush=True)

                try:
                    result = run_evaluation(
                        policy=policy,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        device=device,
                        num_episodes=args.episodes,
                        randomize=True,
                        max_steps=300,
                        verbose=False,
                        analyze_failures=False,
                        block_x=x,
                        block_y=y,
                        pickup_coords=args.pickup_coords,
                        subtask=args.subtask,
                        selective_coords=args.selective_coords,
                        blinkering=args.blinkering,
                    )
                    success_rate = result[0]
                    successes = int(round(success_rate * args.episodes))
                except Exception as e:
                    print(f"ERROR: {e}")
                    success_rate = 0.0
                    successes = 0

                print(f"{success_rate*100:.0f}%")
                writer.writerow([f"{x:.4f}", f"{y:.4f}", f"{success_rate:.2f}", args.episodes, successes])
                f.flush()
                results.append({"x": x, "y": y, "rate": success_rate})

    # Print summary
    print()
    print("=" * 60)
    print("SPATIAL GENERALIZATION SUMMARY")
    print("=" * 60)

    rates = [r["rate"] for r in results]
    print(f"Overall success rate: {np.mean(rates)*100:.1f}%")
    print(f"Positions with >0% success: {sum(1 for r in rates if r > 0)}/{len(rates)}")
    print(f"Positions with >50% success: {sum(1 for r in rates if r > 0.5)}/{len(rates)}")
    print(f"Positions with 100% success: {sum(1 for r in rates if r >= 1.0)}/{len(rates)}")

    # Distance analysis from training center (0.22, 0.22)
    cx, cy = 0.22, 0.22
    print(f"\nBy distance from training center ({cx}, {cy}):")
    for d_max in [0.03, 0.05, 0.08, 0.10, 0.15]:
        in_range = [r for r in results if np.sqrt((r["x"]-cx)**2 + (r["y"]-cy)**2) <= d_max]
        if in_range:
            avg = np.mean([r["rate"] for r in in_range])
            print(f"  Within {d_max*100:.0f}cm: {avg*100:.1f}% ({len(in_range)} positions)")

    # Print grid
    print(f"\nSuccess Rate Grid:")
    print(f"{'Y\\X':>8}", end="")
    for x in xs:
        print(f" {x:.2f}", end="")
    print()

    for y in reversed(ys):
        print(f"{y:.2f}  ", end="")
        for x in xs:
            r = next((r for r in results if abs(r["x"]-x)<0.001 and abs(r["y"]-y)<0.001), None)
            if r:
                rate = r["rate"] * 100
                if rate >= 80:
                    marker = f"  {rate:3.0f}"
                elif rate > 0:
                    marker = f"  {rate:3.0f}"
                else:
                    marker = "    ."
                print(marker, end="")
            else:
                print("    ?", end="")
        print()

    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
