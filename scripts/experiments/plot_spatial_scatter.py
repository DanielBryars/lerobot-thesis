#!/usr/bin/env python3
"""
Generate scatter plot visualization of spatial evaluation results.

Each episode is a translucent circle at the actual block position.
- Green = success
- Red = failure
Overlapping circles create darker regions showing data density.

Usage:
    python scripts/experiments/plot_spatial_scatter.py outputs/experiments/spatial_eval_combined.csv
    python scripts/experiments/plot_spatial_scatter.py outputs/experiments/spatial_eval_combined.csv --output scatter.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


TRAINING_POS = (0.217, 0.225)
BOWL_POS = (0.217, -0.225)


def load_csv_data(csv_path: Path) -> list:
    """Load all episode data from CSV."""
    episodes = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Use actual block position (with noise), not target grid position
            x = float(row["block_x"])
            y = float(row["block_y"])
            episodes.append({
                "x": x,
                "y": y,
                "target_x": float(row["target_x"]),
                "target_y": float(row["target_y"]),
                "success": row["success"].lower() == "true",
                "steps": int(row["steps"]),
            })
    return episodes


def create_scatter_plot(episodes: list, title: str = "ACT Spatial Generalization",
                        output_path: Path = None, show: bool = True,
                        circle_size: float = 80, alpha: float = 0.15):
    """Create scatter plot with translucent circles."""

    # Separate successes and failures
    success_x = [ep["x"] for ep in episodes if ep["success"]]
    success_y = [ep["y"] for ep in episodes if ep["success"]]
    fail_x = [ep["x"] for ep in episodes if not ep["success"]]
    fail_y = [ep["y"] for ep in episodes if not ep["success"]]

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot failures first (red), then successes (green) on top
    ax.scatter(fail_x, fail_y, c='red', s=circle_size, alpha=alpha,
               edgecolors='none', label=f'Failure (n={len(fail_x)})')
    ax.scatter(success_x, success_y, c='green', s=circle_size, alpha=alpha,
               edgecolors='none', label=f'Success (n={len(success_x)})')

    # Mark training position
    ax.plot(TRAINING_POS[0], TRAINING_POS[1], 'b*', markersize=20,
            markeredgecolor='black', markeredgewidth=1.5,
            label='Training position', zorder=10)

    # Mark bowl position if in view
    y_min, y_max = ax.get_ylim()
    if BOWL_POS[1] >= y_min:
        ax.plot(BOWL_POS[0], BOWL_POS[1], 'ko', markersize=15,
                label='Bowl position', zorder=10)

    # Add distance rings from training position
    for radius in [0.05, 0.10, 0.15]:
        circle = plt.Circle(TRAINING_POS, radius, fill=False,
                           color='blue', linestyle='--', linewidth=1, alpha=0.5)
        ax.add_patch(circle)
        ax.annotate(f'{radius*100:.0f}cm',
                   xy=(TRAINING_POS[0] + radius * 0.7, TRAINING_POS[1] + radius * 0.7),
                   fontsize=9, color='blue', alpha=0.7)

    # Styling
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)

    # Calculate and display overall stats
    total = len(episodes)
    successes = len(success_x)
    rate = successes / total * 100 if total > 0 else 0
    ax.text(0.02, 0.98, f'Overall: {rate:.1f}% ({successes}/{total})',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved scatter plot to: {output_path}")

    if show:
        plt.show()

    return fig


def create_density_plot(episodes: list, title: str = "ACT Success Density",
                        output_path: Path = None, show: bool = True):
    """Create a 2D histogram showing success rate density."""

    # Get all positions
    all_x = np.array([ep["x"] for ep in episodes])
    all_y = np.array([ep["y"] for ep in episodes])
    success = np.array([ep["success"] for ep in episodes])

    # Create 2D histogram bins
    x_bins = np.linspace(all_x.min() - 0.01, all_x.max() + 0.01, 30)
    y_bins = np.linspace(all_y.min() - 0.01, all_y.max() + 0.01, 30)

    # Count total and successes in each bin
    total_hist, x_edges, y_edges = np.histogram2d(all_x, all_y, bins=[x_bins, y_bins])
    success_hist, _, _ = np.histogram2d(all_x[success], all_y[success], bins=[x_bins, y_bins])

    # Calculate success rate (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        rate_hist = np.where(total_hist > 0, success_hist / total_hist, np.nan)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot heatmap
    im = ax.imshow(rate_hist.T, origin='lower', cmap='RdYlGn',
                   extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                   vmin=0, vmax=1, aspect='auto')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Success Rate')
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])

    # Mark training position
    ax.plot(TRAINING_POS[0], TRAINING_POS[1], 'b*', markersize=20,
            markeredgecolor='white', markeredgewidth=2,
            label='Training position', zorder=10)

    # Distance rings
    for radius in [0.05, 0.10]:
        circle = plt.Circle(TRAINING_POS, radius, fill=False,
                           color='white', linestyle='--', linewidth=2, alpha=0.8)
        ax.add_patch(circle)

    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved density plot to: {output_path}")

    if show:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot spatial evaluation scatter")
    parser.add_argument("csv_path", type=str, help="Path to CSV file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output image path")
    parser.add_argument("--title", "-t", type=str, default="ACT Spatial Generalization",
                        help="Plot title")
    parser.add_argument("--size", "-s", type=float, default=80, help="Circle size")
    parser.add_argument("--alpha", "-a", type=float, default=0.15, help="Circle transparency")
    parser.add_argument("--density", action="store_true", help="Create density plot instead")
    parser.add_argument("--no-show", action="store_true", help="Don't display the plot")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(f"Loading data from: {csv_path}")
    episodes = load_csv_data(csv_path)
    print(f"Loaded {len(episodes)} episodes")

    successes = sum(1 for ep in episodes if ep["success"])
    print(f"Success rate: {successes/len(episodes)*100:.1f}%")

    output_path = Path(args.output) if args.output else None

    if args.density:
        create_density_plot(episodes, title=args.title,
                           output_path=output_path, show=not args.no_show)
    else:
        create_scatter_plot(episodes, title=args.title,
                           output_path=output_path, show=not args.no_show,
                           circle_size=args.size, alpha=args.alpha)


if __name__ == "__main__":
    main()
