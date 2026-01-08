#!/usr/bin/env python
"""Compare GPU training costs across different cards."""

import matplotlib.pyplot as plt
import numpy as np

# GPU data: (name, VRAM GB, relative speed vs 4090, $/hr on vast.ai)
gpus = [
    ("RTX 3090", 24, 0.7, 0.20),
    ("RTX 4090", 24, 1.0, 0.40),
    ("RTX 4090 D", 24, 0.9, 0.35),  # China variant
    ("A6000", 48, 1.2, 0.50),
    ("L40S", 48, 3.5, 1.00),
    ("A100 40GB", 40, 3.0, 1.20),
    ("A100 80GB", 80, 3.2, 1.80),
    ("H100 80GB", 80, 7.0, 3.00),
]

# Assume a 50k step training job takes 6 hours on RTX 4090
base_hours = 6.0  # hours on RTX 4090

# Calculate costs
names = []
costs = []
times = []
cost_per_speed = []

for name, vram, speed, price in gpus:
    hours = base_hours / speed
    total_cost = hours * price
    names.append(f"{name}\n({vram}GB)")
    costs.append(total_cost)
    times.append(hours)
    cost_per_speed.append(price / speed)

# Create figure with two subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Colors based on cost efficiency
colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(gpus)))  # Green=cheap, Red=expensive
sorted_indices = np.argsort(costs)
colors_sorted = [colors[i] for i in np.argsort(np.argsort(costs))]

# Plot 1: Total cost for job
ax1 = axes[0]
bars1 = ax1.bar(names, costs, color=colors_sorted)
ax1.set_ylabel('Total Cost ($)', fontsize=12)
ax1.set_title('Total Cost for 50k Step Training Job', fontsize=14)
ax1.tick_params(axis='x', rotation=45)
for bar, cost in zip(bars1, costs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'${cost:.2f}', ha='center', va='bottom', fontsize=9)

# Plot 2: Time to complete
ax2 = axes[1]
bars2 = ax2.bar(names, times, color=colors_sorted)
ax2.set_ylabel('Time (hours)', fontsize=12)
ax2.set_title('Time to Complete 50k Steps', fontsize=14)
ax2.tick_params(axis='x', rotation=45)
for bar, t in zip(bars2, times):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{t:.1f}h', ha='center', va='bottom', fontsize=9)

# Plot 3: Cost vs Time scatter
ax3 = axes[2]
for i, (name, vram, speed, price) in enumerate(gpus):
    hours = base_hours / speed
    total_cost = hours * price
    ax3.scatter(hours, total_cost, s=vram*10, alpha=0.7, label=name)
    ax3.annotate(name, (hours, total_cost), textcoords="offset points",
                 xytext=(5, 5), fontsize=8)

ax3.set_xlabel('Time (hours)', fontsize=12)
ax3.set_ylabel('Total Cost ($)', fontsize=12)
ax3.set_title('Cost vs Time Trade-off\n(bubble size = VRAM)', fontsize=14)
ax3.grid(True, alpha=0.3)

# Add "ideal" region annotation
ax3.annotate('Ideal: Low cost,\nFast time', xy=(1, 2), fontsize=10,
             color='green', alpha=0.7,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

plt.tight_layout()
plt.savefig('docs/Images/gpu_cost_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('docs/Images/gpu_cost_comparison.svg', bbox_inches='tight')
print("Saved to docs/Images/gpu_cost_comparison.png")

# Print summary table
print("\n" + "="*70)
print("GPU COST COMPARISON FOR 50K STEP TRAINING JOB")
print("="*70)
print(f"{'GPU':<15} {'VRAM':<8} {'Speed':<8} {'$/hr':<8} {'Hours':<8} {'Total $':<8}")
print("-"*70)
for name, vram, speed, price in sorted(gpus, key=lambda x: (base_hours/x[2])*x[3]):
    hours = base_hours / speed
    total = hours * price
    print(f"{name:<15} {vram:<8} {speed:<8.1f}x {price:<8.2f} {hours:<8.1f} ${total:<7.2f}")
print("-"*70)
print("\nKey insights:")
print("- L40S and A100 40GB offer best cost/performance balance")
print("- H100 is fastest but premium priced")
print("- RTX 3090 is cheapest but slowest")
print("- For same-day results, H100 wins despite higher hourly rate")
