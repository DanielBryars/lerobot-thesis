#!/usr/bin/env python
"""
Test PICK_UP subtask position invariance.

Places block at various positions, uses IK to move the robot EE near the block
(so it's visible in the wrist camera), then tests if the PICK_UP subtask succeeds.

This tests the hypothesis: small BC subtasks can generalize spatially even if
the full pick-and-place task doesn't.

Usage:
    # 5x5 grid, 5 episodes each
    python scripts/experiments/eval_pickup_spatial.py \
        outputs/train/act_vit_subtask_coords_157ep --grid-size 5 --episodes 5

    # With viewer
    python scripts/experiments/eval_pickup_spatial.py \
        outputs/train/act_vit_subtask_coords_157ep --grid-size 3 --episodes 3 --viewer
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
from utils.training import prepare_obs_for_policy, MOTOR_NAMES, NUM_JOINTS
from utils.conversions import radians_to_normalized, normalized_to_radians
from utils.constants import SIM_ACTION_LOW, SIM_ACTION_HIGH

LIFT_HEIGHT = 0.05  # 5cm above table = successful lift
GRIPPER_OPEN_NORM = 80.0  # Normalized value for open gripper
APPROACH_HEIGHT = 0.06  # 6cm above table for IK target


def ik_teleport(sim, ik_solver, target_x, target_y, target_z=APPROACH_HEIGHT, gripper_open=True):
    """Use IK to teleport the robot EE above a target position.

    Returns True if IK succeeded and robot is positioned.
    """
    import mujoco

    # Solve IK for position above the block (position-only, no rotation constraint)
    target_pos = np.array([target_x, target_y, target_z])
    ik_joints, success, error = ik_solver.solve_ik(
        target_pos=target_pos,
        target_rot=None,  # Position-only IK for 5-DOF arm
    )

    if not success and error > 0.02:  # >2cm error is too much
        return False, error

    # Set gripper to open
    gripper_rad = SIM_ACTION_LOW[5] + (GRIPPER_OPEN_NORM / 100.0) * (SIM_ACTION_HIGH[5] - SIM_ACTION_LOW[5])

    # Combine arm joints (5) + gripper (1)
    full_joints_rad = np.concatenate([ik_joints, [gripper_rad]])

    # Set robot qpos directly (robot joints are at qpos[7:13])
    sim.mj_data.qpos[7:13] = full_joints_rad

    # Also set ctrl to hold this position
    sim.mj_data.ctrl[:6] = full_joints_rad

    # Step physics to settle
    for _ in range(50):
        mujoco.mj_step(sim.mj_model, sim.mj_data)

    return True, error


def run_pickup_at_position(
    sim, policy, preprocessor, postprocessor, device, ik_solver,
    block_x, block_y, num_episodes=5, max_steps=150, viewer=False,
    approach_max_steps=150, selective_coords=True,
):
    """Test pickup at a specific block position using natural approach.

    Uses the policy's own MOVE_TO_SOURCE to approach the block naturally,
    then measures PICK_UP success from there. This avoids the IK-teleport
    problem where the robot ends up in an out-of-distribution pose.

    Returns (successes, total, details).
    """
    import mujoco
    from utils.training import PickupCoordinateDataset

    NEAR_THRESHOLD = 0.06  # 6cm - same as subtask state machine
    FAR_THRESHOLD = 0.12   # 12cm - block lifted

    duplo_body_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "duplo")

    # Find EE site
    ee_site_id = None
    for site_name in ["gripperframe", "gripper_site", "ee_site"]:
        sid = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if sid != -1:
            ee_site_id = sid
            break

    successes = 0
    details = []

    # Compute pickup coords for conditioning
    x_bounds = PickupCoordinateDataset.DEFAULT_X_BOUNDS
    y_bounds = PickupCoordinateDataset.DEFAULT_Y_BOUNDS
    x_norm = max(-1, min(1, 2 * (block_x - x_bounds[0]) / (x_bounds[1] - x_bounds[0]) - 1))
    y_norm = max(-1, min(1, 2 * (block_y - y_bounds[0]) / (y_bounds[1] - y_bounds[0]) - 1))
    pickup_coord_tensor = torch.tensor([[x_norm, y_norm]], dtype=torch.float32, device=device)

    for ep in range(num_episodes):
        policy.reset()

        # Reset scene and place block at target
        sim.reset_scene(randomize=False, pos_range=0.0,
                        pos_center_x=block_x, pos_center_y=block_y)

        # Get actual block position (after settling)
        block_pos = sim.mj_data.xpos[duplo_body_id].copy()

        # Phase 1: MOVE_TO_SOURCE — let the policy naturally approach the block
        subtask_state = 0  # MOVE_TO_SOURCE
        approached = False

        for step in range(approach_max_steps):
            obs = sim.get_observation()

            if viewer:
                if not sim.render():
                    return successes, ep, details

            batch = prepare_obs_for_policy(obs, device)

            # Build environment_state matching run_evaluation
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

            # Check subtask transitions (same logic as run_evaluation)
            block_pos = sim.mj_data.xpos[duplo_body_id].copy()
            ee_pos = sim.mj_data.site_xpos[ee_site_id].copy()
            dist_to_block_xy = np.linalg.norm(ee_pos[:2] - block_pos[:2])
            dist_to_block_3d = np.linalg.norm(ee_pos - block_pos)

            if subtask_state == 0 and dist_to_block_xy < NEAR_THRESHOLD:
                subtask_state = 1  # -> PICK_UP
                approached = True
                # Reset action queue so PICK_UP starts fresh
                policy.reset()

            if subtask_state == 1 and dist_to_block_3d > FAR_THRESHOLD:
                # Block was lifted! Success
                subtask_state = 2
                break

        # Check result
        max_height = block_pos[2]
        lifted = subtask_state >= 2  # Got past PICK_UP

        if not approached:
            details.append({"success": False, "reason": "approach_fail",
                            "max_height": max_height, "steps": step + 1})
        else:
            # Phase 2: If approach succeeded but haven't lifted yet, continue PICK_UP
            if not lifted:
                for step2 in range(max_steps):
                    obs = sim.get_observation()

                    if viewer:
                        if not sim.render():
                            return successes, ep, details

                    batch = prepare_obs_for_policy(obs, device)

                    subtask_onehot = torch.zeros(4, dtype=torch.float32, device=device)
                    subtask_onehot[1] = 1.0  # PICK_UP
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
            details.append({"success": lifted, "max_height": max_height})

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
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--blinkering", action="store_true",
                        help="Enable blinkering: mask overhead camera during PICK_UP subtask")
    parser.add_argument("--no-selective-coords", action="store_true",
                        help="Don't zero coordinates during PICK_UP (pass actual coords)")
    parser.add_argument("--approach-steps", type=int, default=150,
                        help="Max steps for MOVE_TO_SOURCE approach phase (default: 150)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_path = Path(args.path)
    final = model_path / "final"
    if final.exists():
        model_path = final

    policy, preprocessor, postprocessor = load_policy_and_processors(
        model_path, args.policy, device, None
    )

    # Enable blinkering if requested
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

    # Generate grid
    xs = np.linspace(args.x_min, args.x_max, args.grid_size)
    ys = np.linspace(args.y_min, args.y_max, args.grid_size)

    # Output CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.output or f"outputs/experiments/pickup_spatial_{timestamp}.csv"
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    total_positions = args.grid_size ** 2
    print(f"\nPICKUP SUBTASK SPATIAL TEST")
    print(f"Model: {args.path}")
    print(f"Blinkering: {'enabled' if args.blinkering else 'disabled'}")
    print(f"Grid: {args.grid_size}x{args.grid_size} ({total_positions} positions)")
    print(f"X: {args.x_min:.2f} to {args.x_max:.2f}")
    print(f"Y: {args.y_min:.2f} to {args.y_max:.2f}")
    print(f"Episodes per position: {args.episodes}")
    print(f"Max steps for pickup: {args.max_steps}")
    print(f"Max steps for approach: {args.approach_steps}")
    print(f"Selective coords: {'disabled' if args.no_selective_coords else 'enabled'}")
    print(f"Output: {csv_path}")
    print()

    results = []
    pos_idx = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "success_rate", "episodes", "successes", "approach_failures", "avg_height"])

        for x in xs:
            for y in ys:
                pos_idx += 1
                print(f"  Position {pos_idx}/{total_positions}: ({x:.3f}, {y:.3f})...", end=" ", flush=True)

                try:
                    succ, total, details = run_pickup_at_position(
                        sim, policy, preprocessor, postprocessor, device, None,
                        block_x=x, block_y=y,
                        num_episodes=args.episodes,
                        max_steps=args.max_steps,
                        viewer=args.viewer,
                        approach_max_steps=args.approach_steps,
                        selective_coords=not args.no_selective_coords,
                    )
                    rate = succ / total if total > 0 else 0
                    ik_fails = sum(1 for d in details if d.get("reason") == "approach_fail")
                    heights = [d.get("max_height", 0) for d in details if "max_height" in d]
                    avg_h = np.mean(heights) if heights else 0
                except Exception as e:
                    print(f"ERROR: {e}")
                    rate = 0
                    succ = 0
                    ik_fails = 0
                    avg_h = 0

                print(f"{rate*100:.0f}% ({succ}/{args.episodes})" +
                      (f" [approach fail: {ik_fails}]" if ik_fails > 0 else ""))

                writer.writerow([f"{x:.4f}", f"{y:.4f}", f"{rate:.2f}", args.episodes, succ, ik_fails, f"{avg_h:.4f}"])
                f.flush()
                results.append({"x": x, "y": y, "rate": rate, "approach_fails": ik_fails, "avg_h": avg_h})

    sim.disconnect()

    # Summary
    print()
    print("=" * 60)
    print("PICKUP SPATIAL GENERALIZATION SUMMARY")
    print("=" * 60)

    rates = [r["rate"] for r in results]
    print(f"Overall pickup success: {np.mean(rates)*100:.1f}%")
    print(f"Positions with >0% pickup:  {sum(1 for r in rates if r > 0)}/{len(rates)}")
    print(f"Positions with >50% pickup: {sum(1 for r in rates if r > 0.5)}/{len(rates)}")
    print(f"Positions with 100% pickup: {sum(1 for r in rates if r >= 1.0)}/{len(rates)}")

    approach_total = sum(r["approach_fails"] for r in results)
    if approach_total > 0:
        print(f"\nApproach failures: {approach_total} (couldn't reach block within step limit)")

    # Distance analysis
    cx, cy = 0.22, 0.22
    print(f"\nBy distance from training center ({cx}, {cy}):")
    for d_max in [0.03, 0.05, 0.08, 0.10, 0.15]:
        in_range = [r for r in results if np.sqrt((r["x"]-cx)**2 + (r["y"]-cy)**2) <= d_max]
        if in_range:
            avg = np.mean([r["rate"] for r in in_range])
            print(f"  Within {d_max*100:.0f}cm: {avg*100:.1f}% ({len(in_range)} positions)")

    # Grid
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
                if r["approach_fails"] == args.episodes:
                    marker = "  NAP"
                elif rate >= 80:
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
