#!/usr/bin/env python3
"""
Generate matplotlib heatmap from spatial evaluation CSV data.

Usage:
    python scripts/experiments/plot_spatial_heatmap.py outputs/experiments/spatial_eval_fine_grid.csv
    python scripts/experiments/plot_spatial_heatmap.py outputs/experiments/spatial_eval_20260121_154657.csv --output heatmap.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_csv_data(csv_path: Path) -> dict:
    """Load and aggregate CSV data by grid position."""
    position_data = {}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (float(row["target_x"]), float(row["target_y"]))
            if key not in position_data:
                position_data[key] = {"successes": 0, "total": 0}
            position_data[key]["total"] += 1
            if row["success"].lower() == "true":
                position_data[key]["successes"] += 1

    return position_data


def create_heatmap(position_data: dict, title: str = "ACT Spatial Generalization",
                   output_path: Path = None, show: bool = True):
    """Create a heatmap from position data."""

    # Extract unique x and y values
    x_vals = sorted(set(pos[0] for pos in position_data.keys()))
    y_vals = sorted(set(pos[1] for pos in position_data.keys()))

    # Create grid
    grid = np.zeros((len(y_vals), len(x_vals)))

    for (x, y), data in position_data.items():
        xi = x_vals.index(x)
        yi = y_vals.index(y)
        success_rate = data["successes"] / data["total"] if data["total"] > 0 else 0
        grid[yi, xi] = success_rate

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    im = ax.imshow(grid, cmap='RdYlGn', vmin=0, vmax=1,
                   extent=[min(x_vals), max(x_vals), min(y_vals), max(y_vals)],
                   origin='lower', aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Success Rate')
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])

    # Mark training position
    training_x, training_y = 0.217, 0.225
    ax.plot(training_x, training_y, 'b*', markersize=15, label='Training position')

    # Mark bowl position
    bowl_x, bowl_y = 0.217, -0.225
    if bowl_y >= min(y_vals) and bowl_y <= max(y_vals):
        ax.plot(bowl_x, bowl_y, 'ko', markersize=10, label='Bowl position')

    # Add grid lines
    ax.set_xticks(x_vals)
    ax.set_yticks(y_vals)
    ax.grid(True, alpha=0.3)

    # Labels
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')

    # Add text annotations for each cell
    for (x, y), data in position_data.items():
        xi = x_vals.index(x)
        yi = y_vals.index(y)
        success_rate = data["successes"] / data["total"] if data["total"] > 0 else 0
        text_color = 'white' if success_rate < 0.5 else 'black'
        ax.text(x, y, f'{success_rate*100:.0f}%',
                ha='center', va='center', fontsize=8, color=text_color)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved heatmap to: {output_path}")

    if show:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot spatial evaluation heatmap")
    parser.add_argument("csv_path", type=str, help="Path to CSV file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output image path")
    parser.add_argument("--title", "-t", type=str, default="ACT Spatial Generalization",
                        help="Plot title")
    parser.add_argument("--no-show", action="store_true", help="Don't display the plot")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(f"Loading data from: {csv_path}")
    position_data = load_csv_data(csv_path)

    print(f"Found {len(position_data)} unique positions")

    # Calculate overall success rate
    total_success = sum(d["successes"] for d in position_data.values())
    total_episodes = sum(d["total"] for d in position_data.values())
    overall_rate = total_success / total_episodes if total_episodes > 0 else 0
    print(f"Overall success rate: {overall_rate*100:.1f}% ({total_success}/{total_episodes})")

    output_path = Path(args.output) if args.output else None
    create_heatmap(position_data, title=args.title,
                   output_path=output_path, show=not args.no_show)


if __name__ == "__main__":
    main()
