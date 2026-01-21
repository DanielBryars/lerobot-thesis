#!/usr/bin/env python3
"""
Visualize ACT temporal ensembling.

Shows how overlapping action chunks get combined into ensembled actions.

Usage:
    python scripts/tools/visualize_temporal_ensemble.py outputs/train/act_20260118_155135 --checkpoint checkpoint_045000
"""

import argparse
import sys
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project paths
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot_robot_sim import SO100Sim, SO100SimConfig

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def load_act_policy(model_path: Path, device: torch.device):
    """Load ACT policy."""
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    print(f"Loading ACT from {model_path}...")
    policy = ACTPolicy.from_pretrained(str(model_path))
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(model_path)
    )

    return policy, preprocessor, postprocessor


def prepare_obs(obs: dict, device: str = "cuda") -> dict:
    """Convert sim observation to policy input format."""
    batch = {}

    state = np.array([obs[m + ".pos"] for m in MOTOR_NAMES], dtype=np.float32)
    batch["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(device)

    for key, value in obs.items():
        if isinstance(value, np.ndarray) and value.ndim == 3:
            img = torch.from_numpy(value).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            batch[f"observation.images.{key}"] = img.to(device)

    return batch


class TemporalEnsemblerViz:
    """Temporal ensembler with visualization support."""

    def __init__(self, coeff: float, chunk_size: int):
        self.coeff = coeff
        self.chunk_size = chunk_size
        self.weights = np.exp(-coeff * np.arange(chunk_size))

        # Store history of chunks for visualization
        self.chunk_history = deque(maxlen=chunk_size)  # List of (start_step, chunk) tuples
        self.ensembled_history = []  # History of ensembled actions
        self.step = 0

    def reset(self):
        self.chunk_history.clear()
        self.ensembled_history = []
        self.step = 0

    def update(self, chunk: np.ndarray) -> np.ndarray:
        """Add new chunk and compute ensembled action.

        Args:
            chunk: (chunk_size, action_dim) array

        Returns:
            Ensembled action for current step
        """
        # Store chunk with its start step
        self.chunk_history.append((self.step, chunk.copy()))

        # Compute ensembled action for current step
        # Collect all predictions for current step from all chunks
        predictions = []
        weights = []

        for chunk_start, chunk_actions in self.chunk_history:
            idx_in_chunk = self.step - chunk_start
            if 0 <= idx_in_chunk < len(chunk_actions):
                predictions.append(chunk_actions[idx_in_chunk])
                # Older chunks have higher index -> different weights
                age = len(self.chunk_history) - 1 - list(self.chunk_history).index((chunk_start, chunk_actions))
                weights.append(self.weights[age])

        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize

        ensembled = (predictions * weights[:, None]).sum(axis=0)

        self.ensembled_history.append({
            'step': self.step,
            'ensembled': ensembled,
            'predictions': predictions,
            'weights': weights,
            'n_contributors': len(predictions),
        })

        self.step += 1
        return ensembled


def run_comparison(
    policy,
    preprocessor,
    postprocessor,
    device: torch.device,
    n_steps: int = 50,
    ensemble_coeff: float = 0.01,
):
    """Run episode collecting both raw and ensembled actions."""

    # Create simulation
    scene_path = REPO_ROOT / "scenes" / "so101_with_wrist_cam.xml"
    sim_config = SO100SimConfig(
        scene_xml=str(scene_path),
        sim_cameras=["overhead_cam", "wrist_cam"],
        camera_width=640,
        camera_height=480,
    )
    sim = SO100Sim(sim_config)
    sim.connect()
    sim.reset_scene(randomize=False)

    chunk_size = policy.config.chunk_size

    # Set up ensembler
    ensembler = TemporalEnsemblerViz(ensemble_coeff, chunk_size)

    # Storage for visualization
    raw_chunks = []  # Store (step, chunk) for each prediction
    raw_actions = []  # Actions without ensembling
    ensembled_actions = []  # Actions with ensembling

    print(f"\nRunning {n_steps} steps, predicting every step...")
    print(f"Chunk size: {chunk_size}, Ensemble coeff: {ensemble_coeff}")

    for step in range(n_steps):
        obs = sim.get_observation()
        batch = prepare_obs(obs, device)
        batch = preprocessor(batch)

        with torch.no_grad():
            # Get full chunk prediction
            chunk = policy.predict_action_chunk(batch)
            chunk = postprocessor(chunk)
            chunk_np = chunk.cpu().numpy()[0]  # (chunk_size, action_dim)

        # Store raw chunk
        raw_chunks.append((step, chunk_np.copy()))

        # Raw action (no ensembling) - just first action of chunk
        raw_action = chunk_np[0]
        raw_actions.append(raw_action)

        # Ensembled action
        ensembled_action = ensembler.update(chunk_np)
        ensembled_actions.append(ensembled_action)

        # Execute ensembled action (for simulation continuity)
        action_dict = {m + ".pos": float(ensembled_action[i]) for i, m in enumerate(MOTOR_NAMES)}
        sim.send_action(action_dict)

        if step % 10 == 0:
            print(f"  Step {step}: {ensembler.ensembled_history[-1]['n_contributors']} chunks contributing")

    sim.disconnect()

    return {
        'raw_chunks': raw_chunks,
        'raw_actions': np.array(raw_actions),
        'ensembled_actions': np.array(ensembled_actions),
        'ensembler': ensembler,
    }


def visualize_results(results: dict, save_path: str = None):
    """Create visualization of temporal ensembling."""

    raw_actions = results['raw_actions']
    ensembled_actions = results['ensembled_actions']
    ensembler = results['ensembler']
    raw_chunks = results['raw_chunks']

    n_steps = len(raw_actions)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # Plot 1: Raw vs Ensembled for first 3 joints
    joint_names = MOTOR_NAMES[:3]
    for i, name in enumerate(joint_names):
        ax = axes[0, 0] if i == 0 else (axes[0, 1] if i == 1 else axes[1, 0])
        ax.plot(raw_actions[:, i], 'r-', alpha=0.7, label='Raw (first of chunk)')
        ax.plot(ensembled_actions[:, i], 'b-', linewidth=2, label='Ensembled')
        ax.set_xlabel('Step')
        ax.set_ylabel('Position')
        ax.set_title(f'{name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 2: Number of contributors over time
    ax = axes[1, 1]
    n_contributors = [h['n_contributors'] for h in ensembler.ensembled_history]
    ax.plot(n_contributors, 'g-', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('# Chunks Contributing')
    ax.set_title('Ensemble Size Over Time')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(n_contributors) + 1)

    # Plot 3: Overlay of multiple chunk predictions for one joint
    ax = axes[2, 0]
    joint_idx = 0  # shoulder_pan

    # Plot each chunk's prediction for this joint
    colors = plt.cm.viridis(np.linspace(0, 1, min(10, len(raw_chunks))))
    for i, (start_step, chunk) in enumerate(raw_chunks[-10:]):  # Last 10 chunks
        steps = np.arange(start_step, start_step + len(chunk))
        ax.plot(steps, chunk[:, joint_idx], '-', color=colors[i], alpha=0.4, linewidth=1)

    # Overlay ensembled
    ax.plot(ensembled_actions[:, joint_idx], 'k-', linewidth=2, label='Ensembled')
    ax.set_xlabel('Step')
    ax.set_ylabel('Position')
    ax.set_title(f'Overlapping Chunks → Ensembled ({MOTOR_NAMES[joint_idx]})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(max(0, n_steps - 30), n_steps)  # Focus on last 30 steps

    # Plot 4: Difference between raw and ensembled
    ax = axes[2, 1]
    diff = np.abs(raw_actions - ensembled_actions).mean(axis=1)
    ax.plot(diff, 'purple', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean Absolute Difference')
    ax.set_title('Raw vs Ensembled Difference')
    ax.grid(True, alpha=0.3)

    plt.suptitle('ACT Temporal Ensembling Visualization', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize ACT temporal ensembling")
    parser.add_argument("model_path", type=str, help="Path to model directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_045000")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to run")
    parser.add_argument("--coeff", type=float, default=0.01, help="Ensemble coefficient")
    parser.add_argument("--save", type=str, default=None, help="Save path for visualization")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model_path = Path(args.model_path) / args.checkpoint
    policy, preprocessor, postprocessor = load_act_policy(model_path, device)

    print(f"\nModel config:")
    print(f"  chunk_size: {policy.config.chunk_size}")
    print(f"  n_action_steps: {policy.config.n_action_steps}")
    print(f"  temporal_ensemble_coeff: {policy.config.temporal_ensemble_coeff}")

    # Run comparison
    results = run_comparison(
        policy, preprocessor, postprocessor, device,
        n_steps=args.steps,
        ensemble_coeff=args.coeff,
    )

    # Visualize
    visualize_results(results, save_path=args.save)


if __name__ == "__main__":
    main()
