#!/usr/bin/env python
"""
Generate spatial generalization plots for ACT-ViT model 7b.

Creates:
1. Heatmap of success rate across grid positions
2. Success rate vs distance from training center
3. Comparison table with previous models

Usage:
    python scripts/experiments/plot_spatial_7b.py outputs/experiments/spatial_7b_20260207_164903.csv
    python scripts/experiments/plot_spatial_7b.py outputs/experiments/spatial_7b_20260207_164903.csv --output-dir outputs/experiments
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def load_csv(csv_path: str):
    """Load spatial eval CSV with columns: x, y, success_rate, episodes, successes."""
    data = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "x": float(row["x"]),
                "y": float(row["y"]),
                "success_rate": float(row["success_rate"]),
                "episodes": int(row["episodes"]),
                "successes": int(row["successes"]),
            })
    return data


def plot_heatmap(data, center_x=0.22, center_y=0.22, output_path=None):
    """Create heatmap of success rates across grid positions."""
    xs = sorted(set(d["x"] for d in data))
    ys = sorted(set(d["y"] for d in data))

    # Build 2D grid
    grid = np.zeros((len(ys), len(xs)))
    for d in data:
        xi = xs.index(d["x"])
        yi = ys.index(d["y"])
        grid[yi, xi] = d["success_rate"] * 100

    fig, ax = plt.subplots(figsize=(9, 7))

    # Custom colormap: red (0%) -> yellow (50%) -> green (100%)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rg", [(0.85, 0.1, 0.1), (1.0, 0.9, 0.1), (0.1, 0.75, 0.1)]
    )

    im = ax.imshow(
        grid[::-1],  # Flip so Y increases upward
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=100,
        interpolation="nearest",
    )

    # Add text annotations
    for d in data:
        rate = d["success_rate"] * 100
        color = "white" if rate < 30 or rate > 80 else "black"
        ax.text(d["x"], d["y"], f"{rate:.0f}%", ha="center", va="center",
                fontsize=9, fontweight="bold", color=color)

    # Mark training center
    ax.plot(center_x, center_y, "b*", markersize=15, markeredgecolor="white",
            markeredgewidth=1.0, label=f"Training center ({center_x}, {center_y})")

    # Draw training distribution box (approx)
    train_box = plt.Rectangle(
        (0.177, 0.185), 0.257 - 0.177, 0.265 - 0.185,
        linewidth=2, edgecolor="blue", facecolor="none",
        linestyle="--", label="Training distribution"
    )
    ax.add_patch(train_box)

    # Distance circles
    for r in [0.05, 0.10, 0.15]:
        circle = plt.Circle(
            (center_x, center_y), r, fill=False,
            linestyle=":", linewidth=1, color="blue", alpha=0.5
        )
        ax.add_patch(circle)
        ax.text(center_x + r * 0.71, center_y + r * 0.71, f"{r*100:.0f}cm",
                fontsize=7, color="blue", alpha=0.7)

    cbar = fig.colorbar(im, ax=ax, label="Success Rate (%)")
    ax.set_xlabel("X position (m)", fontsize=12)
    ax.set_ylabel("Y position (m)", fontsize=12)
    ax.set_title("Spatial Generalization: ACT-ViT 7b (Subtask + Selective Coords)\n"
                 "7x7 grid, 5 episodes per position", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Heatmap saved to: {output_path}")
    plt.show()


def plot_success_vs_distance(data, center_x=0.22, center_y=0.22, output_path=None):
    """Plot success rate vs distance from training center."""
    # Compute distances
    for d in data:
        d["distance"] = np.sqrt((d["x"] - center_x)**2 + (d["y"] - center_y)**2)

    # Bin by distance
    bins = [0, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    bin_labels = []
    bin_rates = []
    bin_counts = []

    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        in_bin = [d for d in data if lo <= d["distance"] < hi]
        if in_bin:
            total_eps = sum(d["episodes"] for d in in_bin)
            total_succ = sum(d["successes"] for d in in_bin)
            rate = total_succ / total_eps if total_eps > 0 else 0
            bin_labels.append(f"{lo*100:.0f}-{hi*100:.0f}cm")
            bin_rates.append(rate * 100)
            bin_counts.append(total_eps)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar chart of success rate
    x_pos = range(len(bin_labels))
    colors = [plt.cm.RdYlGn(r / 100) for r in bin_rates]
    bars = ax1.bar(x_pos, bin_rates, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for i, (bar, rate, count) in enumerate(zip(bars, bin_rates, bin_counts)):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{rate:.0f}%\n({count} ep)", ha="center", va="bottom", fontsize=9)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(bin_labels, rotation=0)
    ax1.set_xlabel("Distance from Training Center", fontsize=12)
    ax1.set_ylabel("Success Rate (%)", fontsize=12)
    ax1.set_title("Success Rate vs Distance: ACT-ViT 7b (Subtask + Selective Coords)", fontsize=13)
    ax1.set_ylim(0, 110)
    ax1.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="50% threshold")
    ax1.legend()

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Distance plot saved to: {output_path}")
    plt.show()


def plot_comparison(data, center_x=0.22, center_y=0.22, output_path=None):
    """Plot comparison between model 7b and previous models."""
    # Model 7b binned data
    for d in data:
        d["distance"] = np.sqrt((d["x"] - center_x)**2 + (d["y"] - center_y)**2)

    bins = [0, 0.05, 0.10, 0.15, 0.20]

    def bin_data(positions, bins):
        results = []
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            in_bin = [d for d in positions if lo <= d["distance"] < hi]
            if in_bin:
                total_eps = sum(d["episodes"] for d in in_bin)
                total_succ = sum(d["successes"] for d in in_bin)
                results.append(total_succ / total_eps * 100 if total_eps > 0 else 0)
            else:
                results.append(0)
        return results

    model_7b = bin_data(data, bins)

    # Previous model data (from experiments.md)
    # Single-position ACT (from extended spatial eval): ~97% within 1cm, ~79% at 3cm, ~64% at 5cm, ~25% at 7cm, 0-10% at 11+cm
    # Approximated from the combined analysis table
    single_pos = [85, 30, 5, 0]  # ACT single-position (0.217, 0.225)

    # 2-position ACT model (from multi-position experiment)
    two_pos = [70, 55, 20, 12]  # ACT 2-position (200ep)

    # 2-position + gap-filling (220ep, checkpoint 030000)
    two_pos_gap = [75, 58, 25, 15]  # ACT 2-position + gap (220ep)

    fig, ax = plt.subplots(figsize=(10, 6))

    bin_labels = [f"{bins[i]*100:.0f}-{bins[i+1]*100:.0f}cm" for i in range(len(bins)-1)]
    x = np.arange(len(bin_labels))
    width = 0.2

    bars1 = ax.bar(x - 1.5 * width, single_pos, width, label="ACT single-pos (40ep)", color="#d62728", alpha=0.8)
    bars2 = ax.bar(x - 0.5 * width, two_pos, width, label="ACT 2-pos (200ep)", color="#ff7f0e", alpha=0.8)
    bars3 = ax.bar(x + 0.5 * width, two_pos_gap, width, label="ACT 2-pos+gap (220ep)", color="#2ca02c", alpha=0.8)
    bars4 = ax.bar(x + 1.5 * width, model_7b, width, label="ACT-ViT 7b subtask+coords (157ep)", color="#1f77b4", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel("Distance from Training Center", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Spatial Generalization Comparison Across Models", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Comparison plot saved to: {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str, help="Path to spatial eval CSV")
    parser.add_argument("--center-x", type=float, default=0.22, help="Training center X")
    parser.add_argument("--center-y", type=float, default=0.22, help="Training center Y")
    parser.add_argument("--output-dir", type=str, default=None, help="Output dir for PNGs")
    parser.add_argument("--no-show", action="store_true", help="Don't show interactive plots")
    args = parser.parse_args()

    data = load_csv(args.csv_path)
    print(f"Loaded {len(data)} positions from {args.csv_path}")

    # Summary
    rates = [d["success_rate"] for d in data]
    print(f"Overall success rate: {np.mean(rates)*100:.1f}%")
    print(f"Positions with >0% success: {sum(1 for r in rates if r > 0)}/{len(rates)}")
    print(f"Positions with >50% success: {sum(1 for r in rates if r > 0.5)}/{len(rates)}")

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.csv_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    plot_heatmap(
        data, args.center_x, args.center_y,
        output_path=str(out_dir / "spatial_7b_heatmap.png")
    )
    plot_success_vs_distance(
        data, args.center_x, args.center_y,
        output_path=str(out_dir / "spatial_7b_distance.png")
    )
    plot_comparison(
        data, args.center_x, args.center_y,
        output_path=str(out_dir / "spatial_7b_comparison.png")
    )

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
