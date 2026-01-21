#!/usr/bin/env python3
"""
Evaluate policy spatial generalization by testing block pickup at different positions.

Creates a heatmap showing success rate across the workspace:
- Green = 100% success rate
- Red = 0% success rate

Usage:
    # Run evaluation (headless)
    python scripts/experiments/eval_spatial_generalization.py outputs/train/act_20260118_155135 --checkpoint checkpoint_045000 --episodes 20

    # Visualize results as heatmap in MuJoCo
    python scripts/experiments/eval_spatial_generalization.py --visualize outputs/experiments/spatial_eval_XXXXXX.json
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import mujoco
import mujoco.viewer

# Add project paths
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot_robot_sim import SO100Sim, SO100SimConfig
from utils.mujoco_viz import MujocoPathRenderer

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Default workspace bounds (robot reach)
# Robot base at origin, block default at (0.217, 0.225)
DEFAULT_X_RANGE = (0.15, 0.35)  # Forward from robot
DEFAULT_Y_RANGE = (-0.15, 0.30)  # Side to side (bowl at y=-0.225, block default at y=0.225)
DEFAULT_GRID_SIZE = 5  # 5x5 grid = 25 positions


def load_act_policy(model_path: Path, device: torch.device):
    """Load ACT policy and processors."""
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

    # State
    state = np.array([obs[m + ".pos"] for m in MOTOR_NAMES], dtype=np.float32)
    batch["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(device)

    # Images
    for key, value in obs.items():
        if isinstance(value, np.ndarray) and value.ndim == 3:
            img = torch.from_numpy(value).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            batch[f"observation.images.{key}"] = img.to(device)

    return batch


def run_single_episode(
    sim: SO100Sim,
    policy,
    preprocessor,
    postprocessor,
    device: torch.device,
    block_x: float,
    block_y: float,
    pos_noise: float = 0.02,
    rot_noise: float = np.pi,
    max_steps: int = 300,
) -> dict:
    """Run a single evaluation episode with block at specified position.

    Returns:
        dict with keys: success, steps, block_x, block_y, block_angle
    """
    # Reset scene without randomization first
    sim.reset_scene(randomize=False)

    # Set block position with small random noise
    actual_x = block_x + np.random.uniform(-pos_noise, pos_noise)
    actual_y = block_y + np.random.uniform(-pos_noise, pos_noise)

    # Random rotation
    angle = np.random.uniform(-rot_noise, rot_noise)
    quat = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]  # Rotation around Z

    sim.set_duplo_position(actual_x, actual_y, z=0.0096, quat=quat)

    # Let it settle
    for _ in range(50):
        mujoco.mj_step(sim.mj_model, sim.mj_data)

    # Get actual position after settling
    duplo_body_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "duplo")
    final_pos = sim.mj_data.xpos[duplo_body_id].copy()

    # Reset policy
    policy.reset()

    # Run episode
    for step in range(max_steps):
        obs = sim.get_observation()
        batch = prepare_obs(obs, device)

        if preprocessor is not None:
            batch = preprocessor(batch)

        with torch.no_grad():
            action = policy.select_action(batch)

        if postprocessor is not None:
            action = postprocessor(action)

        action_np = action.cpu().numpy().flatten()
        action_dict = {m + ".pos": float(action_np[i]) for i, m in enumerate(MOTOR_NAMES)}
        sim.send_action(action_dict)

        if sim.is_task_complete():
            return {
                "success": True,
                "steps": step + 1,
                "block_x": final_pos[0],
                "block_y": final_pos[1],
                "block_angle": np.degrees(angle),
            }

    return {
        "success": False,
        "steps": max_steps,
        "block_x": final_pos[0],
        "block_y": final_pos[1],
        "block_angle": np.degrees(angle),
    }


def run_spatial_evaluation(
    model_path: Path,
    device: torch.device,
    x_range: tuple,
    y_range: tuple,
    grid_size: int,
    episodes_per_position: int,
    pos_noise: float = 0.02,
    rot_noise: float = np.pi,
    max_steps: int = 300,
    csv_path: Path = None,
) -> dict:
    """Run full spatial generalization evaluation.

    Args:
        csv_path: If provided, write results incrementally to this CSV file.
                  If the file exists, will append to it.
    """

    # Load policy
    policy, preprocessor, postprocessor = load_act_policy(model_path, device)

    # Create simulation - use 640x480 to match normal eval
    scene_path = REPO_ROOT / "scenes" / "so101_with_wrist_cam.xml"
    sim_config = SO100SimConfig(
        scene_xml=str(scene_path),
        sim_cameras=["overhead_cam", "wrist_cam"],
        camera_width=640,
        camera_height=480,
    )
    sim = SO100Sim(sim_config)
    sim.connect()

    # Generate grid positions
    x_positions = np.linspace(x_range[0], x_range[1], grid_size)
    y_positions = np.linspace(y_range[0], y_range[1], grid_size)

    # Create all (xi, yi, x, y) tuples and sort by distance from training position
    # Training position was (0.217, 0.225) - start there and work outward
    TRAINING_POS = (0.217, 0.225)
    grid_order = []
    for xi, x in enumerate(x_positions):
        for yi, y in enumerate(y_positions):
            dist = np.sqrt((x - TRAINING_POS[0])**2 + (y - TRAINING_POS[1])**2)
            grid_order.append((xi, yi, x, y, dist))
    grid_order.sort(key=lambda p: p[4])  # Sort by distance from training position

    results = {
        "model_path": str(model_path),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "x_range": x_range,
            "y_range": y_range,
            "grid_size": grid_size,
            "episodes_per_position": episodes_per_position,
            "pos_noise": pos_noise,
            "rot_noise_deg": np.degrees(rot_noise),
            "max_steps": max_steps,
        },
        "grid_positions": [],
        "episodes": [],
    }

    total_positions = grid_size * grid_size
    total_episodes = total_positions * episodes_per_position

    print(f"\n{'='*70}")
    print(f"Spatial Generalization Evaluation")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Grid: {grid_size}x{grid_size} = {total_positions} positions")
    print(f"Episodes per position: {episodes_per_position}")
    print(f"Total episodes: {total_episodes}")
    print(f"X range: {x_range[0]:.3f} to {x_range[1]:.3f}")
    print(f"Y range: {y_range[0]:.3f} to {y_range[1]:.3f}")
    if csv_path:
        print(f"CSV output: {csv_path}")
    print(f"{'='*70}\n")

    # Set up CSV file for incremental saving
    csv_file = None
    csv_writer = None
    csv_fieldnames = [
        "timestamp", "model_path", "grid_x", "grid_y", "target_x", "target_y",
        "episode", "success", "steps", "block_x", "block_y", "block_angle",
        "pos_noise", "max_steps"
    ]

    if csv_path:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = csv_path.exists()
        csv_file = open(csv_path, "a", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
        if not file_exists:
            csv_writer.writeheader()
            print(f"  Created new CSV: {csv_path}")
        else:
            print(f"  Appending to existing CSV: {csv_path}")

    try:
        episode_count = 0
        run_timestamp = datetime.now().isoformat()

        for xi, yi, x, y, dist in grid_order:
            position_results = []
            successes = 0

            for ep in range(episodes_per_position):
                episode_count += 1
                result = run_single_episode(
                    sim, policy, preprocessor, postprocessor, device,
                    block_x=x, block_y=y,
                    pos_noise=pos_noise, rot_noise=rot_noise,
                    max_steps=max_steps,
                )
                result["grid_x"] = xi
                result["grid_y"] = yi
                result["target_x"] = x
                result["target_y"] = y
                result["episode"] = ep

                position_results.append(result)
                results["episodes"].append(result)

                if result["success"]:
                    successes += 1

                # Write to CSV immediately
                if csv_writer:
                    csv_row = {
                        "timestamp": run_timestamp,
                        "model_path": str(model_path),
                        "grid_x": xi,
                        "grid_y": yi,
                        "target_x": x,
                        "target_y": y,
                        "episode": ep,
                        "success": result["success"],
                        "steps": result["steps"],
                        "block_x": result["block_x"],
                        "block_y": result["block_y"],
                        "block_angle": result["block_angle"],
                        "pos_noise": pos_noise,
                        "max_steps": max_steps,
                    }
                    csv_writer.writerow(csv_row)
                    csv_file.flush()  # Ensure it's written immediately

                # Progress
                status = "OK" if result["success"] else "FAIL"
                print(f"\r  [{episode_count}/{total_episodes}] Pos ({x:.2f}, {y:.2f}) d={dist:.3f} ep {ep+1}: {status} ({result['steps']} steps)    ", end="", flush=True)

            success_rate = successes / episodes_per_position
            results["grid_positions"].append({
                "grid_x": xi,
                "grid_y": yi,
                "target_x": x,
                "target_y": y,
                "success_rate": success_rate,
                "successes": successes,
                "total": episodes_per_position,
            })

            print(f"\n  Position ({x:.2f}, {y:.2f}) dist={dist:.3f}: {success_rate*100:.0f}% ({successes}/{episodes_per_position})")

    finally:
        if csv_file:
            csv_file.close()
            print(f"\n  CSV saved: {csv_path}")

    sim.disconnect()

    # Summary
    overall_success = sum(1 for e in results["episodes"] if e["success"]) / len(results["episodes"])
    results["overall_success_rate"] = overall_success

    print(f"\n{'='*70}")
    print(f"OVERALL SUCCESS RATE: {overall_success*100:.1f}%")
    print(f"{'='*70}\n")

    return results


def load_results_from_csv(csv_path: Path) -> dict:
    """Load evaluation results from CSV and aggregate into the same format as JSON results."""
    results = {
        "model_path": None,
        "config": {},
        "grid_positions": [],
        "episodes": [],
    }

    # Read all rows from CSV
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No data in CSV: {csv_path}")

    # Get model path from first row
    results["model_path"] = rows[0]["model_path"]

    # Aggregate by grid position
    position_data = {}
    for row in rows:
        key = (int(row["grid_x"]), int(row["grid_y"]))
        if key not in position_data:
            position_data[key] = {
                "grid_x": int(row["grid_x"]),
                "grid_y": int(row["grid_y"]),
                "target_x": float(row["target_x"]),
                "target_y": float(row["target_y"]),
                "successes": 0,
                "total": 0,
            }
        position_data[key]["total"] += 1
        if row["success"].lower() == "true":
            position_data[key]["successes"] += 1

        # Track episode data
        results["episodes"].append({
            "success": row["success"].lower() == "true",
            "steps": int(row["steps"]),
            "grid_x": int(row["grid_x"]),
            "grid_y": int(row["grid_y"]),
            "target_x": float(row["target_x"]),
            "target_y": float(row["target_y"]),
        })

    # Convert to grid_positions format with success rates
    for key, data in position_data.items():
        data["success_rate"] = data["successes"] / data["total"] if data["total"] > 0 else 0
        results["grid_positions"].append(data)

    # Sort by grid position
    results["grid_positions"].sort(key=lambda p: (p["grid_x"], p["grid_y"]))

    # Compute config from data
    x_vals = [p["target_x"] for p in results["grid_positions"]]
    y_vals = [p["target_y"] for p in results["grid_positions"]]
    grid_x_vals = [p["grid_x"] for p in results["grid_positions"]]
    grid_y_vals = [p["grid_y"] for p in results["grid_positions"]]

    results["config"] = {
        "x_range": (min(x_vals), max(x_vals)) if x_vals else (0, 1),
        "y_range": (min(y_vals), max(y_vals)) if y_vals else (0, 1),
        "grid_size": max(max(grid_x_vals), max(grid_y_vals)) + 1 if grid_x_vals else 1,
    }

    # Overall success rate
    total_success = sum(1 for e in results["episodes"] if e["success"])
    results["overall_success_rate"] = total_success / len(results["episodes"]) if results["episodes"] else 0

    return results


def visualize_heatmap(results_path: Path):
    """Visualize spatial evaluation results as a heatmap in MuJoCo."""
    with open(results_path) as f:
        results = json.load(f)
    visualize_heatmap_from_results(results)


def visualize_heatmap_from_results(results: dict):
    """Visualize spatial evaluation results as a heatmap in MuJoCo."""
    config = results["config"]
    grid_positions = results["grid_positions"]

    # Create simulation for visualization
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

    # Hide the block by moving it far away
    sim.set_duplo_position(10.0, 10.0, 0.0)

    print(f"\nVisualizing spatial evaluation results")
    print(f"Model: {results['model_path']}")
    print(f"Overall success rate: {results['overall_success_rate']*100:.1f}%")
    print(f"\nColor legend: GREEN = 100% success, RED = 0% success")
    print("Press Q or ESC to close\n")

    # Pre-compute heatmap geometry
    heatmap_geoms = []
    for pos in grid_positions:
        x, y = pos["target_x"], pos["target_y"]
        success_rate = pos["success_rate"]

        # Color: red (0%) to green (100%)
        r = 1.0 - success_rate
        g = success_rate
        b = 0.0
        alpha = 0.6

        heatmap_geoms.append({
            "pos": np.array([x, y, 0.001]),  # Slightly above floor
            "color": (r, g, b, alpha),
            "success_rate": success_rate,
        })

    # Calculate cell size for visualization
    x_range = config["x_range"]
    y_range = config["y_range"]
    grid_size = config["grid_size"]
    cell_width = (x_range[1] - x_range[0]) / grid_size / 2
    cell_height = (y_range[1] - y_range[0]) / grid_size / 2

    with mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data) as viewer:
        while viewer.is_running():
            # Draw heatmap
            with viewer.lock():
                viewer.user_scn.ngeom = 0

                for geom in heatmap_geoms:
                    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                        break

                    g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                    mujoco.mjv_initGeom(
                        g,
                        mujoco.mjtGeom.mjGEOM_BOX,
                        np.array([cell_width, cell_height, 0.001], dtype=np.float64),
                        geom["pos"].astype(np.float64),
                        np.eye(3, dtype=np.float64).flatten(),
                        np.array(geom["color"], dtype=np.float64),
                    )
                    viewer.user_scn.ngeom += 1

                # Draw bowl position marker (target)
                if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                    g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                    mujoco.mjv_initGeom(
                        g,
                        mujoco.mjtGeom.mjGEOM_CYLINDER,
                        np.array([0.01, 0.01, 0.01], dtype=np.float64),
                        np.array([0.217, -0.225, 0.02], dtype=np.float64),
                        np.eye(3, dtype=np.float64).flatten(),
                        np.array([0.0, 0.0, 1.0, 0.8], dtype=np.float64),  # Blue
                    )
                    viewer.user_scn.ngeom += 1

            viewer.sync()
            time.sleep(0.05)

    sim.disconnect()
    print("Visualization closed")


def visualize_scatter_from_csv(csv_path: Path, sphere_radius: float = 0.008, alpha: float = 0.4):
    """Visualize individual episodes as colored spheres in MuJoCo.

    Green spheres = success, Red spheres = failure.
    Overlapping spheres create darker regions showing density.
    """
    # Load episode data from CSV
    episodes = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append({
                "x": float(row["block_x"]),
                "y": float(row["block_y"]),
                "success": row["success"].lower() == "true",
            })

    successes = sum(1 for ep in episodes if ep["success"])
    total = len(episodes)

    # Create simulation for visualization
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

    # Hide the block by moving it far away
    sim.set_duplo_position(10.0, 10.0, 0.0)

    print(f"\nVisualizing {total} episodes as scatter plot")
    print(f"Success rate: {successes/total*100:.1f}% ({successes}/{total})")
    print(f"\nColor legend: GREEN = success, RED = failure")
    print(f"Overlapping spheres create darker regions")
    print("Press Q or ESC to close\n")

    # Training position marker
    TRAINING_POS = (0.217, 0.225)

    # Pre-compute distance ring points (blue circles at 5cm, 10cm, 15cm, etc.)
    distance_rings = []
    ring_radii = [0.05, 0.10, 0.15, 0.20]  # 5cm, 10cm, 15cm, 20cm
    points_per_ring = 60  # Number of points to draw each ring
    ring_marker_radius = 0.003  # Size of each marker on the ring

    for radius in ring_radii:
        for i in range(points_per_ring):
            angle = 2 * np.pi * i / points_per_ring
            x = TRAINING_POS[0] + radius * np.cos(angle)
            y = TRAINING_POS[1] + radius * np.sin(angle)
            distance_rings.append((x, y))

    with mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data) as viewer:
        while viewer.is_running():
            with viewer.lock():
                viewer.user_scn.ngeom = 0

                # Draw episode circles first (flat ellipsoids on the floor)
                for ep in episodes:
                    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom - 5:
                        break

                    # Color based on success
                    if ep["success"]:
                        color = (0.0, 0.85, 0.0, alpha)  # Green
                    else:
                        color = (0.85, 0.0, 0.0, alpha)  # Red

                    g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                    # Flat ellipsoid: radius in x/y, nearly zero in z
                    mujoco.mjv_initGeom(
                        g,
                        mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                        np.array([sphere_radius, sphere_radius, 0.0001], dtype=np.float64),
                        np.array([ep["x"], ep["y"], 0.0005], dtype=np.float64),
                        np.eye(3, dtype=np.float64).flatten(),
                        np.array(color, dtype=np.float64),
                    )
                    viewer.user_scn.ngeom += 1

                # Draw distance rings on top (blue circles)
                for (rx, ry) in distance_rings:
                    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom - 10:
                        break
                    g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                    mujoco.mjv_initGeom(
                        g,
                        mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                        np.array([ring_marker_radius, ring_marker_radius, 0.0001], dtype=np.float64),
                        np.array([rx, ry, 0.001], dtype=np.float64),  # Higher z to be on top
                        np.eye(3, dtype=np.float64).flatten(),
                        np.array([0.2, 0.4, 1.0, 0.9], dtype=np.float64),  # Blue, more opaque
                    )
                    viewer.user_scn.ngeom += 1

                # Draw training position marker (blue dot)
                if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                    g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                    mujoco.mjv_initGeom(
                        g,
                        mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                        np.array([0.012, 0.012, 0.001], dtype=np.float64),
                        np.array([TRAINING_POS[0], TRAINING_POS[1], 0.002], dtype=np.float64),
                        np.eye(3, dtype=np.float64).flatten(),
                        np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64),  # Blue
                    )
                    viewer.user_scn.ngeom += 1

                # Draw bowl position marker
                if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                    g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                    mujoco.mjv_initGeom(
                        g,
                        mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                        np.array([0.01, 0.01, 0.001], dtype=np.float64),
                        np.array([0.217, -0.225, 0.002], dtype=np.float64),
                        np.eye(3, dtype=np.float64).flatten(),
                        np.array([0.5, 0.5, 0.5, 0.9], dtype=np.float64),  # Gray
                    )
                    viewer.user_scn.ngeom += 1

            viewer.sync()
            time.sleep(0.05)

    sim.disconnect()
    print("Visualization closed")


def main():
    parser = argparse.ArgumentParser(description="Evaluate spatial generalization")
    parser.add_argument("path", type=str, nargs="?",
                        help="Model path (for eval) or results JSON (for visualize)")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_045000",
                        help="Checkpoint name")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Episodes per grid position")
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE,
                        help=f"Grid size (default: {DEFAULT_GRID_SIZE})")
    parser.add_argument("--x-min", type=float, default=DEFAULT_X_RANGE[0],
                        help=f"X range min (default: {DEFAULT_X_RANGE[0]})")
    parser.add_argument("--x-max", type=float, default=DEFAULT_X_RANGE[1],
                        help=f"X range max (default: {DEFAULT_X_RANGE[1]})")
    parser.add_argument("--y-min", type=float, default=DEFAULT_Y_RANGE[0],
                        help=f"Y range min (default: {DEFAULT_Y_RANGE[0]})")
    parser.add_argument("--y-max", type=float, default=DEFAULT_Y_RANGE[1],
                        help=f"Y range max (default: {DEFAULT_Y_RANGE[1]})")
    parser.add_argument("--pos-noise", type=float, default=0.02,
                        help="Position noise around grid center (default: 0.02m)")
    parser.add_argument("--max-steps", type=int, default=300,
                        help="Max steps per episode")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    parser.add_argument("--csv", type=str, default=None,
                        help="CSV file for incremental results (appends if exists)")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize results from JSON file instead of running eval")
    parser.add_argument("--visualize-csv", action="store_true",
                        help="Visualize results from CSV file instead of running eval")
    parser.add_argument("--scatter", action="store_true",
                        help="Visualize as scatter plot (spheres) from CSV file")
    parser.add_argument("--sphere-radius", type=float, default=0.008,
                        help="Sphere radius for scatter visualization (default: 0.008)")
    parser.add_argument("--sphere-alpha", type=float, default=0.4,
                        help="Sphere transparency for scatter visualization (default: 0.4)")
    args = parser.parse_args()

    if args.visualize:
        if not args.path:
            parser.error("--visualize requires a results JSON path")
        visualize_heatmap(Path(args.path))
        return

    if args.visualize_csv:
        if not args.path:
            parser.error("--visualize-csv requires a CSV path")
        results = load_results_from_csv(Path(args.path))
        # Visualize using the same heatmap function but pass results directly
        visualize_heatmap_from_results(results)
        return

    if args.scatter:
        if not args.path:
            parser.error("--scatter requires a CSV path")
        visualize_scatter_from_csv(
            Path(args.path),
            sphere_radius=args.sphere_radius,
            alpha=args.sphere_alpha
        )
        return

    if not args.path:
        parser.error("Model path required for evaluation")

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Build model path
    model_path = Path(args.path) / args.checkpoint
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    # Set default CSV path if not provided
    csv_path = args.csv
    if csv_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = REPO_ROOT / "outputs" / "experiments" / f"spatial_eval_{timestamp}.csv"

    # Run evaluation
    results = run_spatial_evaluation(
        model_path=model_path,
        device=device,
        x_range=(args.x_min, args.x_max),
        y_range=(args.y_min, args.y_max),
        grid_size=args.grid_size,
        episodes_per_position=args.episodes,
        pos_noise=args.pos_noise,
        max_steps=args.max_steps,
        csv_path=csv_path,
    )

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = REPO_ROOT / "outputs" / "experiments" / f"spatial_eval_{timestamp}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")

    print(f"\nTo visualize the heatmap:")
    print(f"  python {__file__} --visualize {output_path}")


if __name__ == "__main__":
    main()
