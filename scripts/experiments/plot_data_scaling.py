#!/usr/bin/env python
"""Plot data scaling experiment results."""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def main():
    # Load data
    data_dir = Path("outputs/experiments/data_scaling")
    csv_path = data_dir / "summary.csv"

    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)

    # Extract episode counts and checkpoint columns
    episodes = df['episodes'].values
    checkpoint_cols = [c for c in df.columns if c.startswith('checkpoint_') or c == 'final']

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # === Plot 1: Line plot - Best success rate vs episodes ===
    ax1 = axes[0]

    # Calculate best success rate for each episode count
    best_success = df[checkpoint_cols].max(axis=1).values

    ax1.plot(episodes, best_success * 100, 'o-', linewidth=2, markersize=8, color='#2ecc71')
    ax1.axhline(y=100, color='#27ae60', linestyle='--', alpha=0.5, label='100% target')
    ax1.axhline(y=90, color='#f39c12', linestyle='--', alpha=0.5, label='90% threshold')

    # Highlight key points
    for ep, rate in zip(episodes, best_success):
        if rate == 1.0:
            ax1.annotate(f'{int(ep)} ep', (ep, rate * 100), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=9, color='#27ae60', fontweight='bold')

    ax1.set_xlabel('Number of Training Episodes', fontsize=12)
    ax1.set_ylabel('Best Success Rate (%)', fontsize=12)
    ax1.set_title('ACT Policy Performance vs Training Data Size', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 110)
    ax1.set_xlim(0, 165)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right')

    # Add annotations for key thresholds
    ax1.fill_between([0, 165], 90, 100, alpha=0.1, color='#f39c12')
    ax1.fill_between([0, 165], 100, 110, alpha=0.1, color='#27ae60')

    # === Plot 2: Heatmap - Full results matrix ===
    ax2 = axes[1]

    # Prepare data for heatmap
    heatmap_data = df[checkpoint_cols].values * 100

    # Create heatmap
    im = ax2.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    # Set ticks
    ax2.set_yticks(range(len(episodes)))
    ax2.set_yticklabels(episodes)
    checkpoint_labels = [c.replace('checkpoint_0', '').replace('000', 'k') if 'checkpoint' in c else c
                         for c in checkpoint_cols]
    ax2.set_xticks(range(len(checkpoint_cols)))
    ax2.set_xticklabels(checkpoint_labels, rotation=45, ha='right')

    # Add text annotations
    for i in range(len(episodes)):
        for j in range(len(checkpoint_cols)):
            value = heatmap_data[i, j]
            color = 'white' if value < 50 or value >= 90 else 'black'
            ax2.text(j, i, f'{int(value)}', ha='center', va='center', fontsize=8, color=color)

    ax2.set_xlabel('Training Checkpoint', fontsize=12)
    ax2.set_ylabel('Training Episodes', fontsize=12)
    ax2.set_title('Success Rate (%) by Episodes × Checkpoint', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Success Rate (%)', fontsize=10)

    plt.tight_layout()

    # Save
    output_path = data_dir / "data_scaling_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Also save a simplified version focusing on the key finding
    fig2, ax = plt.subplots(figsize=(10, 6))

    ax.plot(episodes, best_success * 100, 'o-', linewidth=3, markersize=10, color='#3498db', label='Peak Success Rate')

    # Fill regions
    ax.fill_between(episodes, 0, best_success * 100, alpha=0.2, color='#3498db')

    # Mark 100% achievements
    perfect_mask = best_success == 1.0
    ax.scatter(episodes[perfect_mask], best_success[perfect_mask] * 100,
               s=200, color='#27ae60', marker='*', zorder=5, label='100% Success')

    # Mark the first 100%
    first_100_idx = np.where(perfect_mask)[0][0] if any(perfect_mask) else None
    if first_100_idx is not None:
        ax.axvline(x=episodes[first_100_idx], color='#27ae60', linestyle=':', alpha=0.7)
        ax.annotate(f'First 100%\n({episodes[first_100_idx]} episodes)',
                   (episodes[first_100_idx], 50), fontsize=11, ha='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Reference lines
    ax.axhline(y=90, color='#e74c3c', linestyle='--', alpha=0.5, linewidth=2)
    ax.text(5, 92, '90% threshold', fontsize=10, color='#e74c3c')

    ax.set_xlabel('Number of Training Episodes', fontsize=14)
    ax.set_ylabel('Best Success Rate (%)', fontsize=14)
    ax.set_title('Data Scaling: How Much Training Data Does ACT Need?', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.set_xlim(0, 165)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=11)

    # Add key insight annotation
    ax.annotate('~100 episodes for\n95% success', xy=(100, 95), xytext=(60, 70),
                fontsize=11, ha='center',
                arrowprops=dict(arrowstyle='->', color='#7f8c8d'),
                bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))

    plt.tight_layout()
    output_path2 = data_dir / "data_scaling_key_finding.png"
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path2}")
    plt.close('all')


if __name__ == "__main__":
    main()
