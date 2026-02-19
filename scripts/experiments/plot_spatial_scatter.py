#!/usr/bin/env python3
"""
Generate scatter plot visualization of spatial evaluation results.

Each episode is a translucent circle at the actual block position.
- Green = success
- Red = failure
Overlapping circles create darker regions showing data density.

Usage:
    # Single plot
    python scripts/experiments/plot_spatial_scatter.py outputs/experiments/spatial_eval_combined.csv

    # Side-by-side comparison of two models
    python scripts/experiments/plot_spatial_scatter.py \
        outputs/experiments/spatial_scatter_checker_14b.csv \
        outputs/experiments/spatial_scatter_dark_ground_v4.csv \
        --side-by-side \
        --training-pos 0.213,0.254 --training-pos 0.213,-0.047 \
        --output docs/Images/spatial_scatter_comparison_220ep.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


DEFAULT_TRAINING_POS = [(0.217, 0.225)]
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


def _nearest_training_pos(x, y, training_positions):
    """Return the nearest training position to (x, y)."""
    dists = [np.sqrt((x - tx)**2 + (y - ty)**2) for tx, ty in training_positions]
    return training_positions[np.argmin(dists)]


def _plot_single(ax, episodes, training_positions, title, circle_size=80, alpha=0.15):
    """Plot a single scatter plot on the given axes."""
    # Separate successes and failures
    success_x = [ep["x"] for ep in episodes if ep["success"]]
    success_y = [ep["y"] for ep in episodes if ep["success"]]
    fail_x = [ep["x"] for ep in episodes if not ep["success"]]
    fail_y = [ep["y"] for ep in episodes if not ep["success"]]

    # Plot failures first (red), then successes (green) on top
    ax.scatter(fail_x, fail_y, c='red', s=circle_size, alpha=alpha,
               edgecolors='none', label=f'Failure (n={len(fail_x)})')
    ax.scatter(success_x, success_y, c='green', s=circle_size, alpha=alpha,
               edgecolors='none', label=f'Success (n={len(success_x)})')

    # Mark training positions
    for i, (tx, ty) in enumerate(training_positions):
        label = 'Training position' if i == 0 else None
        ax.plot(tx, ty, 'b*', markersize=20,
                markeredgecolor='black', markeredgewidth=1.5,
                label=label, zorder=10)

    # Mark bowl position if in view
    y_min, y_max = ax.get_ylim()
    if BOWL_POS[1] >= y_min:
        ax.plot(BOWL_POS[0], BOWL_POS[1], 'ko', markersize=15,
                label='Bowl position', zorder=10)

    # Add distance rings centered on nearest training position
    ring_radii = [0.05, 0.10, 0.15]
    for tx, ty in training_positions:
        for radius in ring_radii:
            circle = plt.Circle((tx, ty), radius, fill=False,
                               color='blue', linestyle='--', linewidth=1, alpha=0.5)
            ax.add_patch(circle)
        # Label rings on the first training position only
        if (tx, ty) == training_positions[0]:
            for radius in ring_radii:
                ax.annotate(f'{radius*100:.0f}cm',
                           xy=(tx + radius * 0.7, ty + radius * 0.7),
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


def create_scatter_plot(episodes: list, title: str = "ACT Spatial Generalization",
                        output_path: Path = None, show: bool = True,
                        circle_size: float = 80, alpha: float = 0.15,
                        training_positions: list = None):
    """Create scatter plot with translucent circles."""
    training_positions = training_positions or DEFAULT_TRAINING_POS

    fig, ax = plt.subplots(figsize=(12, 10))
    _plot_single(ax, episodes, training_positions, title, circle_size, alpha)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved scatter plot to: {output_path}")

    if show:
        plt.show()

    return fig


def create_side_by_side_plot(episodes_left: list, episodes_right: list,
                              title_left: str, title_right: str,
                              output_path: Path = None, show: bool = True,
                              circle_size: float = 80, alpha: float = 0.15,
                              training_positions: list = None):
    """Create side-by-side scatter plots comparing two models."""
    training_positions = training_positions or DEFAULT_TRAINING_POS

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))

    _plot_single(ax1, episodes_left, training_positions, title_left, circle_size, alpha)
    _plot_single(ax2, episodes_right, training_positions, title_right, circle_size, alpha)

    # Share axis limits
    all_x = [ep["x"] for ep in episodes_left + episodes_right]
    all_y = [ep["y"] for ep in episodes_left + episodes_right]
    margin = 0.02
    x_lim = (min(all_x) - margin, max(all_x) + margin)
    y_lim = (min(all_y) - margin, max(all_y) + margin)
    ax1.set_xlim(x_lim)
    ax1.set_ylim(y_lim)
    ax2.set_xlim(x_lim)
    ax2.set_ylim(y_lim)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved side-by-side scatter plot to: {output_path}")

    if show:
        plt.show()

    return fig


def create_density_plot(episodes: list, title: str = "ACT Success Density",
                        output_path: Path = None, show: bool = True,
                        training_positions: list = None):
    """Create a 2D histogram showing success rate density."""
    training_positions = training_positions or DEFAULT_TRAINING_POS

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

    # Mark training positions
    for i, (tx, ty) in enumerate(training_positions):
        label = 'Training position' if i == 0 else None
        ax.plot(tx, ty, 'b*', markersize=20,
                markeredgecolor='white', markeredgewidth=2,
                label=label, zorder=10)

    # Distance rings around first training position
    tx, ty = training_positions[0]
    for radius in [0.05, 0.10]:
        circle = plt.Circle((tx, ty), radius, fill=False,
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


def parse_training_pos(value: str) -> tuple:
    """Parse 'x,y' string into (float, float) tuple."""
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected x,y format, got: {value}")
    return (float(parts[0]), float(parts[1]))


def main():
    parser = argparse.ArgumentParser(description="Plot spatial evaluation scatter")
    parser.add_argument("csv_path", type=str, nargs="+",
                        help="Path to CSV file(s). Two files for --side-by-side mode.")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output image path")
    parser.add_argument("--title", "-t", type=str, default=None,
                        help="Plot title (default: auto-generated)")
    parser.add_argument("--size", "-s", type=float, default=80, help="Circle size")
    parser.add_argument("--alpha", "-a", type=float, default=0.15, help="Circle transparency")
    parser.add_argument("--density", action="store_true", help="Create density plot instead")
    parser.add_argument("--no-show", action="store_true", help="Don't display the plot")
    parser.add_argument("--side-by-side", action="store_true",
                        help="Plot two CSVs side by side for comparison")
    parser.add_argument("--training-pos", type=parse_training_pos, action="append", default=None,
                        help="Training position as x,y (can be repeated). "
                             "E.g. --training-pos 0.213,0.254 --training-pos 0.213,-0.047")
    args = parser.parse_args()

    training_positions = args.training_pos or DEFAULT_TRAINING_POS

    if args.side_by_side:
        if len(args.csv_path) != 2:
            parser.error("--side-by-side requires exactly 2 CSV files")

        csv1, csv2 = Path(args.csv_path[0]), Path(args.csv_path[1])
        if not csv1.exists():
            raise FileNotFoundError(f"CSV file not found: {csv1}")
        if not csv2.exists():
            raise FileNotFoundError(f"CSV file not found: {csv2}")

        print(f"Loading data from: {csv1}")
        episodes1 = load_csv_data(csv1)
        print(f"  {len(episodes1)} episodes, {sum(1 for e in episodes1 if e['success'])/len(episodes1)*100:.1f}% success")

        print(f"Loading data from: {csv2}")
        episodes2 = load_csv_data(csv2)
        print(f"  {len(episodes2)} episodes, {sum(1 for e in episodes2 if e['success'])/len(episodes2)*100:.1f}% success")

        # Auto-generate titles from filenames
        title1 = args.title or csv1.stem.replace("_", " ").title()
        title2 = csv2.stem.replace("_", " ").title()

        output_path = Path(args.output) if args.output else None
        create_side_by_side_plot(
            episodes1, episodes2,
            title_left=title1, title_right=title2,
            output_path=output_path, show=not args.no_show,
            circle_size=args.size, alpha=args.alpha,
            training_positions=training_positions,
        )
    else:
        csv_path = Path(args.csv_path[0])
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        print(f"Loading data from: {csv_path}")
        episodes = load_csv_data(csv_path)
        print(f"Loaded {len(episodes)} episodes")

        successes = sum(1 for ep in episodes if ep["success"])
        print(f"Success rate: {successes/len(episodes)*100:.1f}%")

        title = args.title or "ACT Spatial Generalization"
        output_path = Path(args.output) if args.output else None

        if args.density:
            create_density_plot(episodes, title=title,
                               output_path=output_path, show=not args.no_show,
                               training_positions=training_positions)
        else:
            create_scatter_plot(episodes, title=title,
                               output_path=output_path, show=not args.no_show,
                               circle_size=args.size, alpha=args.alpha,
                               training_positions=training_positions)


if __name__ == "__main__":
    main()
