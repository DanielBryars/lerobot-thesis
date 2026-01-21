"""
Real-time matplotlib plotting for joint trajectories.

Shows predicted vs actual joint positions over time.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class JointPlotter:
    """Real-time matplotlib plot showing predicted vs actual joint trajectories."""

    def __init__(self, chunk_size: int = 100, history_size: int = 200):
        self.chunk_size = chunk_size
        self.history_size = history_size

        # History of actual joint positions and desired positions
        self.actual_history = []  # List of (step, joint_values) tuples
        self.desired_history = []  # List of (step, joint_values) tuples
        self.global_step = 0

        # Set up interactive mode
        plt.ion()

        # Create figure with subplots for each joint
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.canvas.manager.set_window_title('Joint Predictions vs Actual')
        gs = GridSpec(3, 2, figure=self.fig, hspace=0.35, wspace=0.25)

        self.axes = []
        self.predicted_lines = []  # Current chunk prediction
        self.actual_lines = []  # Actual joint position history
        self.desired_lines = []  # Desired (commanded) position history
        self.current_step_lines = []  # Vertical lines showing current step

        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']

        for i, (motor, color) in enumerate(zip(MOTOR_NAMES, colors)):
            row, col = i // 2, i % 2
            ax = self.fig.add_subplot(gs[row, col])
            ax.set_title(motor, fontsize=10, fontweight='bold')
            ax.set_xlabel('Step', fontsize=8)
            ax.set_ylabel('Value (model-norm)', fontsize=8)
            ax.set_xlim(0, chunk_size)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

            # Create line for current chunk prediction (dashed, shows future)
            pred_line, = ax.plot([], [], color=color, linewidth=2, linestyle='--',
                                  alpha=0.7, label='Predicted chunk')
            # Create line for actual joint positions (solid, historical)
            actual_line, = ax.plot([], [], color='black', linewidth=1.5,
                                    alpha=0.8, label='Actual')
            # Create line for desired/commanded positions (dotted, historical)
            desired_line, = ax.plot([], [], color=color, linewidth=1.5, linestyle=':',
                                     alpha=0.6, label='Commanded')
            # Create vertical line for current execution point
            vline = ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)

            # Add legend only on first subplot
            if i == 0:
                ax.legend(loc='upper right', fontsize=7)

            self.axes.append(ax)
            self.predicted_lines.append(pred_line)
            self.actual_lines.append(actual_line)
            self.desired_lines.append(desired_line)
            self.current_step_lines.append(vline)

            # Set appropriate y-limits for model-normalized space (roughly -3 to 3)
            ax.set_ylim(-4, 4)

        self.fig.tight_layout()
        plt.show(block=False)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def record_actual(self, joint_positions: np.ndarray):
        """Record actual joint position at current step.

        Args:
            joint_positions: Array of 6 joint positions (normalized)
        """
        self.actual_history.append((self.global_step, joint_positions.copy()))
        # Trim history
        if len(self.actual_history) > self.history_size:
            self.actual_history.pop(0)

    def record_desired(self, joint_positions: np.ndarray):
        """Record desired/commanded joint position at current step.

        Args:
            joint_positions: Array of 6 joint positions (normalized, from action)
        """
        self.desired_history.append((self.global_step, joint_positions.copy()))
        # Trim history
        if len(self.desired_history) > self.history_size:
            self.desired_history.pop(0)
        self.global_step += 1

    def update(self, action_chunk: np.ndarray, executed_steps: int = 0):
        """Update the plot with new action chunk predictions and history.

        Args:
            action_chunk: Shape (chunk_size, 6) - predicted actions for current chunk
            executed_steps: How many steps have been executed in current chunk
        """
        if action_chunk is None or len(action_chunk) == 0:
            return

        # Calculate x-axis range to show: history leading up to now + future prediction
        # The current step is at global_step, chunk extends into future
        history_steps = min(len(self.actual_history), self.history_size // 2)
        future_steps = len(action_chunk) - executed_steps

        # X-axis: [global_step - history_steps, global_step + future_steps]
        x_min = max(0, self.global_step - history_steps)
        x_max = self.global_step + future_steps + 10

        # Prepare chunk prediction x-coords (starting from current global step - executed)
        chunk_start_step = self.global_step - executed_steps
        chunk_x = np.arange(len(action_chunk)) + chunk_start_step

        for i, (pred_line, actual_line, desired_line, vline, ax) in enumerate(
            zip(self.predicted_lines, self.actual_lines, self.desired_lines,
                self.current_step_lines, self.axes)):

            # Update predicted chunk line
            if i < action_chunk.shape[1]:
                pred_line.set_data(chunk_x, action_chunk[:, i])

            # Update actual history line
            if self.actual_history:
                actual_x = [h[0] for h in self.actual_history]
                actual_y = [h[1][i] for h in self.actual_history if i < len(h[1])]
                actual_x = actual_x[:len(actual_y)]
                actual_line.set_data(actual_x, actual_y)

            # Update desired history line
            if self.desired_history:
                desired_x = [h[0] for h in self.desired_history]
                desired_y = [h[1][i] for h in self.desired_history if i < len(h[1])]
                desired_x = desired_x[:len(desired_y)]
                desired_line.set_data(desired_x, desired_y)

            # Update x-limits to follow the action
            ax.set_xlim(x_min, x_max)

            # Auto-adjust y-limits based on visible data
            if MOTOR_NAMES[i] != 'gripper':
                all_y = list(action_chunk[:, i])
                if self.actual_history:
                    all_y.extend([h[1][i] for h in self.actual_history if i < len(h[1])])
                if all_y:
                    ymin, ymax = min(all_y), max(all_y)
                    margin = (ymax - ymin) * 0.15 + 1
                    ax.set_ylim(ymin - margin, ymax + margin)

            # Update vertical line to show current position
            vline.set_xdata([self.global_step, self.global_step])

        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def reset(self):
        """Reset history for new episode."""
        self.actual_history.clear()
        self.desired_history.clear()
        self.global_step = 0

    def close(self):
        """Close the plot window."""
        plt.close(self.fig)
