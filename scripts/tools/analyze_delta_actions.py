#!/usr/bin/env python
"""
Analyze and compare absolute vs delta (relative) action representations.

This script examines the pick-and-place dataset to understand:
1. Distribution of absolute actions (joint targets)
2. Distribution of delta actions (change from current state)
3. How these distributions vary by subtask phase
4. Implications for learning and generalization

Usage:
    python scripts/tools/analyze_delta_actions.py --dataset datasets/sim_pick_place_157ep
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_dataset(dataset_path: Path):
    """Load dataset from parquet files."""
    parquet_files = sorted(dataset_path.glob("data/**/*.parquet"))
    if not parquet_files:
        parquet_files = sorted(dataset_path.glob("**/*.parquet"))

    dfs = []
    for f in parquet_files:
        df = pd.read_parquet(f)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def load_subtask_annotations(dataset_path: Path):
    """Load subtask annotations if available."""
    annotations_path = dataset_path / "meta" / "subtask_annotations.json"
    if annotations_path.exists():
        with open(annotations_path) as f:
            return json.load(f)
    return None


def analyze_actions(df, subtask_annotations=None):
    """Analyze absolute and delta actions."""

    # Extract actions and states
    # Actions and states may be stored as array columns or separate columns
    if 'action' in df.columns:
        # Check if it's an array column
        first_action = df['action'].iloc[0]
        if hasattr(first_action, '__len__') and len(first_action) > 1:
            actions = np.stack(df['action'].values)
        else:
            action_cols = [c for c in df.columns if c.startswith('action')]
            actions = df[action_cols].values
    else:
        action_cols = [c for c in df.columns if c.startswith('action')]
        actions = df[action_cols].values

    if 'observation.state' in df.columns:
        first_state = df['observation.state'].iloc[0]
        if hasattr(first_state, '__len__') and len(first_state) > 1:
            states = np.stack(df['observation.state'].values)
        else:
            state_cols = [c for c in df.columns if c.startswith('observation.state')]
            states = df[state_cols].values
    else:
        state_cols = [c for c in df.columns if c.startswith('observation.state')]
        states = df[state_cols].values

    # Ensure 2D arrays
    if len(actions.shape) == 1:
        actions = actions.reshape(-1, 1)
    if len(states.shape) == 1:
        states = states.reshape(-1, 1)

    action_dim = actions.shape[1]
    state_dim = states.shape[1]

    print(f"Dataset size: {len(df)} frames")
    print(f"Action dimension: {action_dim}")
    print(f"State dimension: {state_dim}")
    print()

    # Compute delta actions: delta = action - current_state
    # For joint control: delta represents "how much to move from current position"
    deltas = actions[:, :6] - states[:, :6]  # Only first 6 dims (joints, not gripper)

    # === ABSOLUTE ACTION ANALYSIS ===
    print("=" * 60)
    print("ABSOLUTE ACTIONS (Joint Targets)")
    print("=" * 60)

    print("\nPer-joint statistics (normalized units):")
    print(f"{'Joint':<10} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Range':>10}")
    print("-" * 60)
    for i in range(min(6, action_dim)):
        joint_actions = actions[:, i]
        print(f"Joint {i+1:<4} {np.mean(joint_actions):>10.2f} {np.std(joint_actions):>10.2f} "
              f"{np.min(joint_actions):>10.2f} {np.max(joint_actions):>10.2f} "
              f"{np.max(joint_actions) - np.min(joint_actions):>10.2f}")

    print(f"\nOverall action magnitude: {np.mean(np.abs(actions[:, :6])):.2f} +/- {np.std(np.abs(actions[:, :6])):.2f}")

    # === DELTA ACTION ANALYSIS ===
    print("\n" + "=" * 60)
    print("DELTA ACTIONS (Change from Current State)")
    print("=" * 60)

    print("\nPer-joint statistics (normalized units):")
    print(f"{'Joint':<10} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Range':>10}")
    print("-" * 60)
    for i in range(6):
        joint_deltas = deltas[:, i]
        print(f"Joint {i+1:<4} {np.mean(joint_deltas):>10.2f} {np.std(joint_deltas):>10.2f} "
              f"{np.min(joint_deltas):>10.2f} {np.max(joint_deltas):>10.2f} "
              f"{np.max(joint_deltas) - np.min(joint_deltas):>10.2f}")

    print(f"\nOverall delta magnitude: {np.mean(np.abs(deltas)):.2f} +/- {np.std(np.abs(deltas)):.2f}")

    # === COMPARISON ===
    print("\n" + "=" * 60)
    print("COMPARISON: ABSOLUTE vs DELTA")
    print("=" * 60)

    abs_magnitude = np.mean(np.linalg.norm(actions[:, :6], axis=1))
    delta_magnitude = np.mean(np.linalg.norm(deltas, axis=1))

    print(f"\nAverage L2 norm:")
    print(f"  Absolute actions: {abs_magnitude:.2f}")
    print(f"  Delta actions:    {delta_magnitude:.2f}")
    print(f"  Ratio (delta/abs): {delta_magnitude/abs_magnitude:.3f}")

    # Variance analysis
    abs_var = np.mean(np.var(actions[:, :6], axis=0))
    delta_var = np.mean(np.var(deltas, axis=0))

    print(f"\nAverage variance per joint:")
    print(f"  Absolute actions: {abs_var:.2f}")
    print(f"  Delta actions:    {delta_var:.2f}")
    print(f"  Ratio (delta/abs): {delta_var/abs_var:.3f}")

    # === SUBTASK ANALYSIS ===
    if subtask_annotations:
        print("\n" + "=" * 60)
        print("ANALYSIS BY SUBTASK PHASE")
        print("=" * 60)

        subtask_names = ["MOVE_TO_SOURCE", "PICK_UP", "MOVE_TO_DEST", "DROP"]

        # Build subtask labels for each frame
        subtask_labels = []
        for ep_idx in df['episode_index'].unique():
            ep_str = str(int(ep_idx))
            if ep_str in subtask_annotations:
                subtask_labels.extend(subtask_annotations[ep_str])

        if len(subtask_labels) == len(df):
            subtask_labels = np.array(subtask_labels)

            print("\nDelta magnitude by subtask:")
            print(f"{'Subtask':<20} {'Mean |delta|':>15} {'Std |delta|':>15} {'Frames':>10}")
            print("-" * 60)

            for st in range(4):
                mask = subtask_labels == st
                if np.sum(mask) > 0:
                    st_deltas = deltas[mask]
                    mean_mag = np.mean(np.abs(st_deltas))
                    std_mag = np.std(np.abs(st_deltas))
                    print(f"{subtask_names[st]:<20} {mean_mag:>15.2f} {std_mag:>15.2f} {np.sum(mask):>10}")

            print("\nAbsolute action variance by subtask:")
            print(f"{'Subtask':<20} {'Variance':>15} {'Notes':<30}")
            print("-" * 60)

            for st in range(4):
                mask = subtask_labels == st
                if np.sum(mask) > 0:
                    st_actions = actions[mask, :6]
                    var = np.mean(np.var(st_actions, axis=0))
                    notes = ""
                    if st == 0:
                        notes = "(navigating to block)"
                    elif st == 1:
                        notes = "(fine manipulation)"
                    elif st == 2:
                        notes = "(transporting)"
                    elif st == 3:
                        notes = "(releasing)"
                    print(f"{subtask_names[st]:<20} {var:>15.2f} {notes:<30}")

    # === IMPLICATIONS ===
    print("\n" + "=" * 60)
    print("IMPLICATIONS FOR LEARNING")
    print("=" * 60)

    print("""
Key observations:

1. MAGNITUDE:
   - Delta actions are typically much smaller than absolute actions
   - Smaller targets may be easier to predict accurately
   - But small errors in deltas accumulate over time

2. VARIANCE:
   - If delta variance << absolute variance, deltas are more consistent
   - Consistent deltas = similar motion patterns regardless of start position
   - This could help with position generalization

3. SUBTASK DIFFERENCES:
   - Navigation phases (MOVE_TO_SOURCE, MOVE_TO_DEST) likely have larger deltas
   - Manipulation phases (PICK_UP, DROP) likely have smaller, more precise deltas
   - Could use different action scales per subtask

4. RECOMMENDATIONS:
   - If delta variance is much lower: delta actions may generalize better
   - If deltas are very small: may need careful normalization
   - Consider hybrid: absolute for coarse motion, delta for fine manipulation
""")

    return {
        'actions': actions,
        'states': states,
        'deltas': deltas,
        'abs_magnitude': abs_magnitude,
        'delta_magnitude': delta_magnitude,
        'abs_variance': abs_var,
        'delta_variance': delta_var,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze delta vs absolute actions")
    parser.add_argument("--dataset", type=str, default="datasets/sim_pick_place_157ep",
                        help="Path to dataset directory")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return

    print(f"Loading dataset from: {dataset_path}")
    df = load_dataset(dataset_path)

    subtask_annotations = load_subtask_annotations(dataset_path)
    if subtask_annotations:
        print(f"Loaded subtask annotations for {len(subtask_annotations)} episodes")

    print()
    results = analyze_actions(df, subtask_annotations)

    # Summary for documentation
    print("\n" + "=" * 60)
    print("SUMMARY FOR EXPERIMENTS2.MD")
    print("=" * 60)
    print(f"""
| Metric | Absolute Actions | Delta Actions | Ratio |
|--------|-----------------|---------------|-------|
| Mean L2 norm | {results['abs_magnitude']:.2f} | {results['delta_magnitude']:.2f} | {results['delta_magnitude']/results['abs_magnitude']:.3f} |
| Mean variance | {results['abs_variance']:.2f} | {results['delta_variance']:.2f} | {results['delta_variance']/results['abs_variance']:.3f} |
""")


if __name__ == "__main__":
    main()
