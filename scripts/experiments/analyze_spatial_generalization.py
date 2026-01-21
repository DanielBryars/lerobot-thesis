#!/usr/bin/env python3
"""
Analyze spatial generalization results and generate publication-ready figures.

Usage:
    python scripts/experiments/analyze_spatial_generalization.py outputs/experiments/spatial_eval_combined.csv
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


TRAINING_POS = (0.217, 0.225)


def load_csv_data(csv_path: Path) -> list:
    """Load all episode data from CSV."""
    episodes = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = float(row["target_x"])
            y = float(row["target_y"])
            dist = np.sqrt((x - TRAINING_POS[0])**2 + (y - TRAINING_POS[1])**2)
            episodes.append({
                "x": x,
                "y": y,
                "dist": dist,
                "success": row["success"].lower() == "true",
                "steps": int(row["steps"]),
            })
    return episodes


def analyze_by_distance(episodes: list, bin_size: float = 0.02) -> dict:
    """Bin episodes by distance and compute statistics."""
    bins = defaultdict(lambda: {"successes": 0, "total": 0, "steps": []})

    for ep in episodes:
        bin_idx = int(ep["dist"] / bin_size)
        bin_center = (bin_idx + 0.5) * bin_size
        bins[bin_center]["total"] += 1
        if ep["success"]:
            bins[bin_center]["successes"] += 1
            bins[bin_center]["steps"].append(ep["steps"])

    # Compute rates
    results = {}
    for dist, data in sorted(bins.items()):
        rate = data["successes"] / data["total"] if data["total"] > 0 else 0
        avg_steps = np.mean(data["steps"]) if data["steps"] else 0
        results[dist] = {
            "success_rate": rate,
            "successes": data["successes"],
            "total": data["total"],
            "avg_steps": avg_steps,
        }

    return results


def plot_success_vs_distance(results: dict, output_path: Path = None, show: bool = True):
    """Plot success rate vs distance from training position."""
    distances = list(results.keys())
    rates = [results[d]["success_rate"] for d in distances]
    totals = [results[d]["total"] for d in distances]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar plot
    bars = ax.bar(distances, rates, width=0.015, alpha=0.7, color='steelblue',
                  edgecolor='black', linewidth=0.5)

    # Add sample size annotations
    for d, r, n in zip(distances, rates, totals):
        if n > 0:
            ax.annotate(f'n={n}', xy=(d, r + 0.02), ha='center', va='bottom',
                       fontsize=7, color='gray')

    # Add trend line
    distances_arr = np.array(distances)
    rates_arr = np.array(rates)

    # Fit exponential decay (linear in log space for positive rates)
    mask = rates_arr > 0
    if mask.sum() > 2:
        # Simple exponential fit: rate = a * exp(-b * dist)
        log_rates = np.log(rates_arr[mask] + 0.01)
        coeffs = np.polyfit(distances_arr[mask], log_rates, 1)
        trend_x = np.linspace(0, max(distances) * 1.1, 100)
        trend_y = np.exp(coeffs[1]) * np.exp(coeffs[0] * trend_x)
        trend_y = np.clip(trend_y, 0, 1)
        ax.plot(trend_x, trend_y, 'r--', linewidth=2, alpha=0.7, label='Exponential fit')

    # Add 50% threshold line
    ax.axhline(y=0.5, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='50% threshold')

    # Styling
    ax.set_xlabel('Distance from Training Position (m)', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title('ACT Policy: Success Rate vs Spatial Displacement', fontsize=14)
    ax.set_xlim(-0.01, max(distances) * 1.1)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")

    if show:
        plt.show()

    return fig


def print_summary(episodes: list, results: dict):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("ACT SPATIAL GENERALIZATION ANALYSIS")
    print("="*70)
    print(f"\nTotal episodes: {len(episodes)}")
    print(f"Training position: {TRAINING_POS}")

    overall_success = sum(1 for ep in episodes if ep["success"])
    print(f"Overall success rate: {overall_success/len(episodes)*100:.1f}%")

    print("\n" + "-"*70)
    print("SUCCESS RATE BY DISTANCE FROM TRAINING POSITION")
    print("-"*70)
    print(f"{'Distance (m)':<15} {'Success Rate':<15} {'N Episodes':<12} {'Avg Steps':<12}")
    print("-"*70)

    for dist, data in sorted(results.items()):
        rate_str = f"{data['success_rate']*100:.0f}%"
        steps_str = f"{data['avg_steps']:.0f}" if data['avg_steps'] > 0 else "N/A"
        print(f"{dist:.3f}          {rate_str:<15} {data['total']:<12} {steps_str:<12}")

    print("-"*70)

    # Find 50% threshold
    for dist, data in sorted(results.items()):
        if data["success_rate"] < 0.5:
            print(f"\n50% success threshold crossed at approximately {dist:.3f}m")
            break

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Analyze spatial generalization results")
    parser.add_argument("csv_path", type=str, help="Path to CSV file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output plot path")
    parser.add_argument("--bin-size", type=float, default=0.02, help="Distance bin size (m)")
    parser.add_argument("--no-show", action="store_true", help="Don't display the plot")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(f"Loading data from: {csv_path}")
    episodes = load_csv_data(csv_path)
    print(f"Loaded {len(episodes)} episodes")

    results = analyze_by_distance(episodes, bin_size=args.bin_size)
    print_summary(episodes, results)

    output_path = Path(args.output) if args.output else None
    plot_success_vs_distance(results, output_path=output_path, show=not args.no_show)


if __name__ == "__main__":
    main()
