#!/usr/bin/env python
"""
Plot spatial generalization learning curve across training checkpoints.

Shows how spatial generalization evolves during training compared to
standard evaluation performance.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data from act_2pos_220ep experiments
checkpoints = [10000, 30000, 50000, 60000, 100000]
checkpoint_labels = ['10K', '30K', '50K', '60K', '100K\n(final)']

# Standard evaluation (20 episodes, randomized positions near training data)
standard_eval = [95, 85, 90, 95, 80]

# Spatial evaluation (7x7 grid = 490 episodes total)
spatial_eval = [29.2, 36.1, 36.1, 33.3, 35.3]

# Baseline from 200ep model (no gap-filling)
baseline_spatial = 31.8

# Create figure
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot spatial evaluation (primary y-axis)
color1 = '#2ecc71'  # Green
ax1.plot(checkpoints, spatial_eval, 'o-', color=color1, linewidth=2, markersize=10, label='Spatial Eval (7x7 grid)')
ax1.axhline(y=baseline_spatial, color=color1, linestyle='--', alpha=0.5, label=f'200ep baseline ({baseline_spatial}%)')
ax1.fill_between(checkpoints, baseline_spatial, spatial_eval, alpha=0.2, color=color1)
ax1.set_xlabel('Training Steps', fontsize=12)
ax1.set_ylabel('Spatial Evaluation Success (%)', color=color1, fontsize=12)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, 100)

# Create secondary y-axis for standard evaluation
ax2 = ax1.twinx()
color2 = '#3498db'  # Blue
ax2.plot(checkpoints, standard_eval, 's--', color=color2, linewidth=2, markersize=8, label='Standard Eval (training positions)')
ax2.set_ylabel('Standard Evaluation Success (%)', color=color2, fontsize=12)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0, 100)

# Set x-axis ticks
ax1.set_xticks(checkpoints)
ax1.set_xticklabels(checkpoint_labels)

# Add annotations for key points
# Best spatial generalization
best_spatial_idx = np.argmax(spatial_eval)
ax1.annotate(f'Peak: {spatial_eval[best_spatial_idx]}%',
             xy=(checkpoints[best_spatial_idx], spatial_eval[best_spatial_idx]),
             xytext=(checkpoints[best_spatial_idx] + 10000, spatial_eval[best_spatial_idx] + 10),
             arrowprops=dict(arrowstyle='->', color=color1),
             color=color1, fontsize=10, fontweight='bold')

# Best standard eval
best_std_idx = np.argmax(standard_eval)
ax2.annotate(f'Best: {standard_eval[best_std_idx]}%',
             xy=(checkpoints[best_std_idx], standard_eval[best_std_idx]),
             xytext=(checkpoints[best_std_idx] + 15000, standard_eval[best_std_idx] - 8),
             arrowprops=dict(arrowstyle='->', color=color2),
             color=color2, fontsize=10, fontweight='bold')

# Title and legend
plt.title('ACT 220ep: Spatial Generalization vs Training Steps\n(Spatial peaks early while standard eval continues improving)', fontsize=14)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=10)

plt.tight_layout()

# Save figure
output_dir = Path(__file__).parent.parent.parent / 'plots'
output_dir.mkdir(exist_ok=True)
output_path = output_dir / 'spatial_learning_curve_220ep.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved plot to: {output_path}")
