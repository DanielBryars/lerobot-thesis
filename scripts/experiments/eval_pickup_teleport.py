#!/usr/bin/env python
"""
Test PICK_UP position invariance by teleporting the robot to a "looking at block" pose.

Instead of running MOVE_TO_SOURCE (which fails at many positions), this script:
1. Captures canonical joint configs from natural approaches at known-good positions
2. For each grid position, teleports the robot to the nearest canonical config
3. Runs PICK_UP subtask only and measures success

This isolates the PICK_UP policy's spatial generalization from MOVE_TO_SOURCE limitations.

Usage:
    python scripts/experiments/eval_pickup_teleport.py \
        outputs/train/act_vit_220ep_fixstate_blinkering \
        --grid-size 5 --episodes 5 --blinkering
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.inference.eval import load_policy_and_processors
from utils.training import prepare_obs_for_policy, MOTOR_NAMES, NUM_JOINTS, PickupCoordinateDataset
from utils.conversions import radians_to_normalized, normalized_to_radians

LIFT_HEIGHT = 0.05  # 5cm above table = successful lift


def capture_approach_config(sim, policy, preprocessor, postprocessor, device,
                            block_x, block_y, max_steps=200, selective_coords=True,
                            blinkering=False):
    """Run MOVE_TO_SOURCE at a position and capture joint config at PICK_UP transition.

    Returns (joint_angles, ee_pos, success) where joint_angles is the robot qpos[7:13]
    at the moment of MOVE_TO_SOURCE -> PICK_UP transition.
    """
    import mujoco

    NEAR_THRESHOLD = 0.06

    duplo_body_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "duplo")
    ee_site_id = None
    for site_name in ["gripperframe", "gripper_site", "ee_site"]:
        sid = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if sid != -1:
            ee_site_id = sid
            break

    # Compute pickup coords
    x_bounds = PickupCoordinateDataset.DEFAULT_X_BOUNDS
    y_bounds = PickupCoordinateDataset.DEFAULT_Y_BOUNDS
    x_norm = max(-1, min(1, 2 * (block_x - x_bounds[0]) / (x_bounds[1] - x_bounds[0]) - 1))
    y_norm = max(-1, min(1, 2 * (block_y - y_bounds[0]) / (y_bounds[1] - y_bounds[0]) - 1))
    pickup_coord_tensor = torch.tensor([[x_norm, y_norm]], dtype=torch.float32, device=device)

    policy.reset()
    sim.reset_scene(randomize=False, pos_range=0.0,
                    pos_center_x=block_x, pos_center_y=block_y)

    block_pos = sim.mj_data.xpos[duplo_body_id].copy()
    subtask_state = 0  # MOVE_TO_SOURCE

    for step in range(max_steps):
        obs = sim.get_observation()
        batch = prepare_obs_for_policy(obs, device)

        subtask_onehot = torch.zeros(4, dtype=torch.float32, device=device)
        subtask_onehot[subtask_state] = 1.0
        subtask_tensor = subtask_onehot.unsqueeze(0)

        if selective_coords and subtask_state in (1, 3):
            coords = torch.zeros_like(pickup_coord_tensor)
        else:
            coords = pickup_coord_tensor
        batch["observation.environment_state"] = torch.cat([coords, subtask_tensor], dim=1)

        batch = preprocessor(batch)
        with torch.no_grad():
            action = policy.select_action(batch)
            action = postprocessor(action)

        action_np = action.cpu().numpy().flatten()
        action_dict = {f"{MOTOR_NAMES[i]}.pos": float(action_np[i]) for i in range(NUM_JOINTS)}
        sim.send_action(action_dict)

        # Check transition
        ee_pos = sim.mj_data.site_xpos[ee_site_id].copy()
        block_pos = sim.mj_data.xpos[duplo_body_id].copy()
        dist_xy = np.linalg.norm(ee_pos[:2] - block_pos[:2])

        if subtask_state == 0 and dist_xy < NEAR_THRESHOLD:
            # Capture the joint configuration at transition
            joint_angles = sim.mj_data.qpos[7:13].copy()
            return joint_angles, ee_pos.copy(), True

    return None, None, False


def run_pickup_from_config(sim, policy, preprocessor, postprocessor, device,
                           joint_config, block_x, block_y,
                           num_episodes=5, max_steps=150, selective_coords=True):
    """Run PICK_UP from a given starting joint configuration.

    Sets robot joints to joint_config, places block at (block_x, block_y),
    then runs the PICK_UP subtask.

    Returns (successes, total, details).
    """
    import mujoco

    duplo_body_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "duplo")

    # Compute pickup coords (zeroed during PICK_UP if selective_coords)
    x_bounds = PickupCoordinateDataset.DEFAULT_X_BOUNDS
    y_bounds = PickupCoordinateDataset.DEFAULT_Y_BOUNDS
    x_norm = max(-1, min(1, 2 * (block_x - x_bounds[0]) / (x_bounds[1] - x_bounds[0]) - 1))
    y_norm = max(-1, min(1, 2 * (block_y - y_bounds[0]) / (y_bounds[1] - y_bounds[0]) - 1))
    pickup_coord_tensor = torch.tensor([[x_norm, y_norm]], dtype=torch.float32, device=device)

    successes = 0
    details = []

    for ep in range(num_episodes):
        policy.reset()

        # Reset scene — this resets robot to home and block to target
        sim.reset_scene(randomize=False, pos_range=0.0,
                        pos_center_x=block_x, pos_center_y=block_y)

        # Teleport robot to the canonical joint configuration
        sim.mj_data.qpos[7:13] = joint_config.copy()
        sim.mj_data.ctrl[:6] = joint_config.copy()

        # Step physics to settle (robot + block)
        for _ in range(20):
            mujoco.mj_step(sim.mj_model, sim.mj_data)

        # Record initial block position
        block_pos = sim.mj_data.xpos[duplo_body_id].copy()
        max_height = block_pos[2]

        # Run PICK_UP subtask
        lifted = False
        for step in range(max_steps):
            obs = sim.get_observation()
            batch = prepare_obs_for_policy(obs, device)

            # Always PICK_UP subtask (index 1)
            subtask_onehot = torch.zeros(4, dtype=torch.float32, device=device)
            subtask_onehot[1] = 1.0
            subtask_tensor = subtask_onehot.unsqueeze(0)

            if selective_coords:
                coords = torch.zeros_like(pickup_coord_tensor)
            else:
                coords = pickup_coord_tensor
            batch["observation.environment_state"] = torch.cat([coords, subtask_tensor], dim=1)

            batch = preprocessor(batch)
            with torch.no_grad():
                action = policy.select_action(batch)
                action = postprocessor(action)

            action_np = action.cpu().numpy().flatten()
            action_dict = {f"{MOTOR_NAMES[i]}.pos": float(action_np[i]) for i in range(NUM_JOINTS)}
            sim.send_action(action_dict)

            block_now = sim.mj_data.xpos[duplo_body_id].copy()
            max_height = max(max_height, block_now[2])

            if max_height > LIFT_HEIGHT:
                lifted = True
                break

        if lifted:
            successes += 1
        details.append({"success": lifted, "max_height": max_height, "steps": step + 1})

    return successes, num_episodes, details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Model path")
    parser.add_argument("--policy", type=str, default="act_vit")
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--x-min", type=float, default=0.10)
    parser.add_argument("--x-max", type=float, default=0.35)
    parser.add_argument("--y-min", type=float, default=0.08)
    parser.add_argument("--y-max", type=float, default=0.38)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--blinkering", action="store_true")
    parser.add_argument("--no-selective-coords", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_path = Path(args.path)
    final = model_path / "final"
    if final.exists():
        model_path = final

    policy, preprocessor, postprocessor = load_policy_and_processors(
        model_path, args.policy, device, None
    )

    if args.blinkering and hasattr(policy, 'model') and hasattr(policy.model, 'blinkering'):
        policy.model.blinkering = True
        print("Blinkering ENABLED on model")

    # Create simulation
    from lerobot_robot_sim import SO100SimConfig, SO100Sim

    scene_path = REPO_ROOT / "scenes" / "so101_with_wrist_cam.xml"
    config = SO100SimConfig(
        scene_xml=str(scene_path),
        sim_cameras=["wrist_cam", "overhead_cam"],
        camera_width=640,
        camera_height=480,
    )
    sim = SO100Sim(config)
    sim.connect()

    selective_coords = not args.no_selective_coords

    # ============================================================
    # Phase 1: Capture canonical approach configs at known-good positions
    # ============================================================
    # Use positions spread across the workspace that we know work
    calibration_positions = [
        (0.225, 0.230),  # Near training pos 1
        (0.225, 0.080),  # Bottom-center
        (0.287, 0.155),  # Center-right
        (0.350, 0.155),  # Right
        (0.350, 0.080),  # Bottom-right
        (0.163, 0.155),  # Left
        (0.287, 0.080),  # Bottom center-right
        (0.225, 0.155),  # Center
        (0.287, 0.230),  # Center-right upper
    ]

    print("=" * 60)
    print("PHASE 1: Capturing canonical approach configurations")
    print("=" * 60)

    canonical_configs = []  # (x, y, joint_angles, ee_pos)

    for cx, cy in calibration_positions:
        print(f"  Calibrating at ({cx:.3f}, {cy:.3f})...", end=" ", flush=True)
        joints, ee_pos, success = capture_approach_config(
            sim, policy, preprocessor, postprocessor, device,
            cx, cy, max_steps=200, selective_coords=selective_coords,
        )
        if success:
            canonical_configs.append((cx, cy, joints, ee_pos))
            joint_deg = np.degrees(joints[:5])
            print(f"OK  EE=({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})  "
                  f"Joints=[{', '.join(f'{d:.0f}' for d in joint_deg)}]°")
        else:
            print("FAIL (couldn't reach)")

    if not canonical_configs:
        print("\nERROR: No canonical configs captured!")
        sim.disconnect()
        return

    print(f"\nCaptured {len(canonical_configs)} canonical configs")

    # ============================================================
    # Phase 2: Test PICK_UP at grid positions using nearest canonical config
    # ============================================================
    xs = np.linspace(args.x_min, args.x_max, args.grid_size)
    ys = np.linspace(args.y_min, args.y_max, args.grid_size)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.output or f"outputs/experiments/pickup_teleport_{timestamp}.csv"
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    total_positions = args.grid_size ** 2
    print()
    print("=" * 60)
    print("PHASE 2: Testing PICK_UP with teleported start")
    print("=" * 60)
    print(f"Grid: {args.grid_size}x{args.grid_size} ({total_positions} positions)")
    print(f"X: {args.x_min:.2f} to {args.x_max:.2f}")
    print(f"Y: {args.y_min:.2f} to {args.y_max:.2f}")
    print(f"Episodes per position: {args.episodes}")
    print(f"Max PICK_UP steps: {args.max_steps}")
    print(f"Output: {csv_path}")
    print()

    results = []
    pos_idx = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "success_rate", "episodes", "successes",
                         "avg_height", "config_source_x", "config_source_y", "ee_block_dist"])

        for x in xs:
            for y in ys:
                pos_idx += 1

                # Find nearest canonical config (by EE-to-block distance)
                best_config = None
                best_dist = float('inf')
                best_source = None
                for cx, cy, joints, ee_pos in canonical_configs:
                    # Distance from this config's EE position to the target block
                    dist = np.sqrt((ee_pos[0] - x)**2 + (ee_pos[1] - y)**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_config = joints
                        best_source = (cx, cy)

                print(f"  Position {pos_idx}/{total_positions}: ({x:.3f}, {y:.3f}) "
                      f"[config from ({best_source[0]:.3f}, {best_source[1]:.3f}), "
                      f"EE-block: {best_dist*100:.1f}cm]...", end=" ", flush=True)

                try:
                    succ, total, details = run_pickup_from_config(
                        sim, policy, preprocessor, postprocessor, device,
                        best_config, x, y,
                        num_episodes=args.episodes,
                        max_steps=args.max_steps,
                        selective_coords=selective_coords,
                    )
                    rate = succ / total if total > 0 else 0
                    heights = [d["max_height"] for d in details]
                    avg_h = np.mean(heights)
                except Exception as e:
                    print(f"ERROR: {e}")
                    rate = 0
                    succ = 0
                    avg_h = 0

                print(f"{rate*100:.0f}% ({succ}/{args.episodes}) avg_h={avg_h:.3f}")

                writer.writerow([
                    f"{x:.4f}", f"{y:.4f}", f"{rate:.2f}", args.episodes, succ,
                    f"{avg_h:.4f}", f"{best_source[0]:.4f}", f"{best_source[1]:.4f}",
                    f"{best_dist:.4f}"
                ])
                f.flush()
                results.append({"x": x, "y": y, "rate": rate, "avg_h": avg_h,
                                "ee_dist": best_dist, "source": best_source})

    sim.disconnect()

    # ============================================================
    # Summary
    # ============================================================
    print()
    print("=" * 60)
    print("TELEPORTED PICKUP SPATIAL SUMMARY")
    print("=" * 60)

    rates = [r["rate"] for r in results]
    print(f"Overall pickup success: {np.mean(rates)*100:.1f}%")
    print(f"Positions with >0% pickup:  {sum(1 for r in rates if r > 0)}/{len(rates)}")
    print(f"Positions with >50% pickup: {sum(1 for r in rates if r > 0.5)}/{len(rates)}")
    print(f"Positions with 100% pickup: {sum(1 for r in rates if r >= 1.0)}/{len(rates)}")

    # Distance analysis: success rate vs EE-to-block distance at start
    print(f"\nBy EE-to-block distance at start:")
    for d_max in [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20]:
        in_range = [r for r in results if r["ee_dist"] <= d_max]
        if in_range:
            avg = np.mean([r["rate"] for r in in_range])
            print(f"  Within {d_max*100:.0f}cm: {avg*100:.1f}% ({len(in_range)} positions)")

    # Grid display
    print(f"\nPickup Success Grid:")
    print(f"{'Y\\X':>8}", end="")
    for x in xs:
        print(f" {x:.2f}", end="")
    print()
    for y in reversed(ys):
        print(f"{y:.2f}  ", end="")
        for x in xs:
            r = next((r for r in results if abs(r["x"]-x)<0.001 and abs(r["y"]-y)<0.001), None)
            if r:
                rate = r["rate"] * 100
                if rate >= 80:
                    marker = f"  {rate:3.0f}"
                elif rate > 0:
                    marker = f"  {rate:3.0f}"
                else:
                    marker = "    ."
                print(marker, end="")
            else:
                print("    ?", end="")
        print()

    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
