#!/usr/bin/env python
"""
Test the PICK_UP subtask in isolation.

Runs full episodes but tracks per-subtask success:
- Approach: Did the EE reach within 6cm of the block?
- Pickup: Did the block get lifted above 5cm?
- Transport: Did the EE reach within 6cm of the bowl?
- Drop: Did the block land in the bowl?

Can also start the robot already near the block (--teleport) to skip
the approach phase and test pickup in isolation.

Usage:
    # Full subtask breakdown
    python scripts/experiments/eval_pickup_only.py outputs/train/act_vit_subtask_coords_157ep --episodes 20

    # Teleport near block, test pickup only
    python scripts/experiments/eval_pickup_only.py outputs/train/act_vit_subtask_coords_157ep --episodes 20 --teleport

    # With MuJoCo viewer
    python scripts/experiments/eval_pickup_only.py outputs/train/act_vit_subtask_coords_157ep --episodes 5 --viewer

    # Test at a specific block position
    python scripts/experiments/eval_pickup_only.py outputs/train/act_vit_subtask_coords_157ep --episodes 10 --block-x 0.25 --block-y 0.25
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.inference.eval import load_policy_and_processors
from utils.training import prepare_obs_for_policy, MOTOR_NAMES, NUM_JOINTS


# Subtask names
SUBTASK_NAMES = ["MOVE_TO_SOURCE", "PICK_UP", "MOVE_TO_DEST", "DROP"]
NEAR_THRESHOLD = 0.06   # 6cm - transition to PICK_UP
FAR_THRESHOLD = 0.12    # 12cm - block lifted (transition to MOVE_TO_DEST)
BOWL_POS = np.array([0.217, -0.225, 0.0])
LIFT_HEIGHT = 0.05      # 5cm above table = successful lift


def run_pickup_test(
    policy, preprocessor, postprocessor, device,
    num_episodes=10, max_steps=300, block_x=None, block_y=None,
    teleport=False, viewer=False, verbose=True,
):
    """Run episodes tracking per-subtask success."""
    import mujoco
    from lerobot_robot_sim import SO100SimConfig, SO100Sim
    from utils.training import PickupCoordinateDataset

    scene_path = REPO_ROOT / "scenes" / "so101_with_wrist_cam.xml"
    config = SO100SimConfig(
        scene_xml=str(scene_path),
        sim_cameras=["wrist_cam", "overhead_cam"],
        camera_width=640,
        camera_height=480,
    )
    sim = SO100Sim(config)
    sim.connect()

    results = []
    policy.eval()

    for ep in range(num_episodes):
        print(f"  Episode {ep+1}/{num_episodes}...", end=" ", flush=True)
        policy.reset()

        # Reset scene
        fixed_pos = block_x is not None and block_y is not None
        sim.reset_scene(
            randomize=True,
            pos_range=0.0 if fixed_pos else 0.04,
            rot_range=np.pi,
            pos_center_x=block_x,
            pos_center_y=block_y,
        )

        # Get block position
        duplo_body_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "duplo")
        block_pos = sim.mj_data.xpos[duplo_body_id].copy()

        # Find EE site
        ee_site_id = None
        for site_name in ["gripperframe", "gripper_site", "ee_site"]:
            sid = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if sid != -1:
                ee_site_id = sid
                break

        # Get pickup coords for conditioning
        pickup_coord_tensor = None
        actual_x = sim.mj_data.xpos[duplo_body_id][0]
        actual_y = sim.mj_data.xpos[duplo_body_id][1]
        x_bounds = PickupCoordinateDataset.DEFAULT_X_BOUNDS
        y_bounds = PickupCoordinateDataset.DEFAULT_Y_BOUNDS
        x_norm = max(-1, min(1, 2 * (actual_x - x_bounds[0]) / (x_bounds[1] - x_bounds[0]) - 1))
        y_norm = max(-1, min(1, 2 * (actual_y - y_bounds[0]) / (y_bounds[1] - y_bounds[0]) - 1))
        pickup_coord_tensor = torch.tensor([[x_norm, y_norm]], dtype=torch.float32, device=device)

        # Teleport mode: run MOVE_TO_SOURCE with the model until near block, then start measuring
        teleport_steps = 0
        if teleport:
            # Run approach phase silently
            subtask_state = 0
            for pre_step in range(150):
                obs = sim.get_observation()
                batch = prepare_obs_for_policy(obs, device)

                # Build environment_state for MOVE_TO_SOURCE
                subtask_onehot = torch.zeros(4, dtype=torch.float32, device=device)
                subtask_onehot[0] = 1.0
                batch["observation.environment_state"] = torch.cat(
                    [pickup_coord_tensor, subtask_onehot.unsqueeze(0)], dim=1
                )

                batch = preprocessor(batch)
                with torch.no_grad():
                    action = policy.select_action(batch)
                    action = postprocessor(action)

                action_np = action.cpu().numpy().flatten()
                action_dict = {f"{MOTOR_NAMES[i]}.pos": float(action_np[i]) for i in range(NUM_JOINTS)}
                sim.send_action(action_dict)

                # Check if near block
                ee_pos = sim.mj_data.site_xpos[ee_site_id].copy()
                dist = np.linalg.norm(ee_pos[:2] - block_pos[:2])
                if dist < NEAR_THRESHOLD:
                    teleport_steps = pre_step + 1
                    break

                if viewer:
                    sim.render()
            else:
                # Failed to approach
                teleport_steps = 150

            if teleport_steps == 150:
                print(f"APPROACH_FAIL (couldn't reach block)")
                results.append({
                    "approach": False, "pickup": False,
                    "transport": False, "drop": False,
                    "max_height": 0, "teleport_steps": 150,
                })
                continue

            # Reset the policy chunking state for a fresh start at PICK_UP
            policy.reset()

        # Main evaluation loop
        subtask_state = 1 if teleport else 0  # Start at PICK_UP if teleported
        approach_done = teleport
        pickup_done = False
        transport_done = False
        drop_done = False
        max_block_height = 0.0
        pickup_start_step = None
        ep_start = time.time()

        for step in range(max_steps):
            obs = sim.get_observation()

            if viewer:
                if not sim.render():
                    print("\nViewer closed")
                    sim.disconnect()
                    return results

            batch = prepare_obs_for_policy(obs, device)

            # Get EE position
            ee_pos = sim.mj_data.site_xpos[ee_site_id].copy()
            block_now = sim.mj_data.xpos[duplo_body_id].copy()
            max_block_height = max(max_block_height, block_now[2])

            # Compute distances
            dist_to_block_xy = np.linalg.norm(ee_pos[:2] - block_pos[:2])
            dist_to_block_3d = np.linalg.norm(ee_pos - block_pos)
            dist_to_bowl_xy = np.linalg.norm(ee_pos[:2] - BOWL_POS[:2])

            # Forward-only state machine
            if subtask_state == 0:
                if dist_to_block_xy < NEAR_THRESHOLD:
                    subtask_state = 1
                    approach_done = True
                    pickup_start_step = step
                    if verbose:
                        print(f"[approach@{step}]", end=" ", flush=True)
            elif subtask_state == 1:
                if dist_to_block_3d > FAR_THRESHOLD:
                    subtask_state = 2
                    pickup_done = True
                    if verbose:
                        print(f"[pickup@{step} h={max_block_height:.3f}]", end=" ", flush=True)
            elif subtask_state == 2:
                if dist_to_bowl_xy < NEAR_THRESHOLD:
                    subtask_state = 3
                    transport_done = True
                    if verbose:
                        print(f"[transport@{step}]", end=" ", flush=True)

            # Build environment_state
            subtask_onehot = torch.zeros(4, dtype=torch.float32, device=device)
            subtask_onehot[subtask_state] = 1.0
            subtask_tensor = subtask_onehot.unsqueeze(0)

            # Selective coords: zero during PICK_UP and DROP
            if subtask_state in (1, 3):
                zeroed = torch.zeros_like(pickup_coord_tensor)
                batch["observation.environment_state"] = torch.cat([zeroed, subtask_tensor], dim=1)
            else:
                batch["observation.environment_state"] = torch.cat([pickup_coord_tensor, subtask_tensor], dim=1)

            batch = preprocessor(batch)
            with torch.no_grad():
                action = policy.select_action(batch)
                action = postprocessor(action)

            action_np = action.cpu().numpy().flatten()
            action_dict = {f"{MOTOR_NAMES[i]}.pos": float(action_np[i]) for i in range(NUM_JOINTS)}
            sim.send_action(action_dict)

            # Check task completion
            if sim.is_task_complete():
                drop_done = True
                transport_done = True
                pickup_done = True
                approach_done = True
                elapsed = time.time() - ep_start
                print(f"OK ({elapsed:.1f}s, {step+1} steps, h={max_block_height:.3f}m)")
                break
        else:
            elapsed = time.time() - ep_start
            # Check if block was at least lifted
            lifted = max_block_height > LIFT_HEIGHT
            status_parts = []
            if not approach_done:
                status_parts.append("no_approach")
            elif not pickup_done:
                if lifted:
                    status_parts.append(f"lifted_h={max_block_height:.3f}")
                else:
                    status_parts.append(f"no_lift_h={max_block_height:.3f}")
            elif not transport_done:
                status_parts.append("dropped_during_transport")
            else:
                status_parts.append("missed_drop")
            print(f"FAIL ({elapsed:.1f}s, {', '.join(status_parts)})")

        results.append({
            "approach": approach_done,
            "pickup": pickup_done or max_block_height > LIFT_HEIGHT,
            "transport": transport_done,
            "drop": drop_done,
            "max_height": max_block_height,
            "teleport_steps": teleport_steps,
        })

    sim.disconnect()
    return results


def main():
    parser = argparse.ArgumentParser(description="Test PICK_UP subtask in isolation")
    parser.add_argument("path", type=str, help="Model path")
    parser.add_argument("--policy", type=str, default="act_vit")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--block-x", type=float, default=None)
    parser.add_argument("--block-y", type=float, default=None)
    parser.add_argument("--teleport", action="store_true",
                        help="Use model to approach block first, then test pickup from near position")
    parser.add_argument("--viewer", action="store_true", help="Show MuJoCo viewer")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_path = Path(args.path)
    final = model_path / "final"
    if final.exists():
        model_path = final

    policy, preprocessor, postprocessor = load_policy_and_processors(
        model_path, args.policy, device, None
    )

    print(f"\nPICKUP SUBTASK TEST")
    print(f"Model: {args.path}")
    print(f"Mode: {'teleport (approach then test pickup)' if args.teleport else 'full episode with per-subtask tracking'}")
    if args.block_x is not None:
        print(f"Block position: ({args.block_x}, {args.block_y})")
    print(f"Episodes: {args.episodes}")
    print()

    results = run_pickup_test(
        policy, preprocessor, postprocessor, device,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        block_x=args.block_x,
        block_y=args.block_y,
        teleport=args.teleport,
        viewer=args.viewer,
    )

    # Summary
    print()
    print("=" * 60)
    print("PER-SUBTASK SUCCESS RATES")
    print("=" * 60)

    n = len(results)
    approach_ok = sum(1 for r in results if r["approach"])
    pickup_ok = sum(1 for r in results if r["pickup"])
    transport_ok = sum(1 for r in results if r["transport"])
    drop_ok = sum(1 for r in results if r["drop"])

    print(f"  Approach (EE near block):      {approach_ok}/{n} = {approach_ok/n*100:.0f}%")
    print(f"  Pickup (block lifted >5cm):    {pickup_ok}/{n} = {pickup_ok/n*100:.0f}%")
    if approach_ok > 0:
        print(f"    Pickup given approach:       {pickup_ok}/{approach_ok} = {pickup_ok/approach_ok*100:.0f}%")
    print(f"  Transport (EE near bowl):      {transport_ok}/{n} = {transport_ok/n*100:.0f}%")
    if pickup_ok > 0:
        print(f"    Transport given pickup:      {transport_ok}/{pickup_ok} = {transport_ok/pickup_ok*100:.0f}%")
    print(f"  Drop (block in bowl):          {drop_ok}/{n} = {drop_ok/n*100:.0f}%")
    if transport_ok > 0:
        print(f"    Drop given transport:        {drop_ok}/{transport_ok} = {drop_ok/transport_ok*100:.0f}%")

    heights = [r["max_height"] for r in results]
    print(f"\n  Max block height: {max(heights):.3f}m")
    print(f"  Avg block height: {np.mean(heights):.3f}m")

    if args.teleport:
        tele_steps = [r["teleport_steps"] for r in results if r["teleport_steps"] < 150]
        if tele_steps:
            print(f"\n  Avg approach steps: {np.mean(tele_steps):.0f}")
            print(f"  Approach failures:  {sum(1 for r in results if r['teleport_steps'] >= 150)}/{n}")


if __name__ == "__main__":
    main()
