#!/usr/bin/env python
"""
Evaluate a pickup-only ACT model (no subtask/coord conditioning) across spatial positions.

For models trained purely on PICK_UP segments. Simply places the block, runs the policy
from home pose, and checks if the block gets lifted.

Usage:
    # Test at training positions
    python scripts/experiments/eval_pickup_model_spatial.py outputs/train/act_vit_XXXX

    # 5x5 spatial grid, 5 episodes each
    python scripts/experiments/eval_pickup_model_spatial.py outputs/train/act_vit_XXXX \
        --grid-size 5 --episodes 5

    # Wait for training to finish then run full eval
    python scripts/experiments/eval_pickup_model_spatial.py outputs/train/act_vit_XXXX \
        --wait --grid-size 5 --episodes 5

    # Pre-position arm above block via IK (for models trained near-block)
    python scripts/experiments/eval_pickup_model_spatial.py outputs/train/act_vit_XXXX \
        --pre-position --episodes 20
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features, build_dataset_frame

from scripts.inference.eval import load_policy_and_processors
from utils.training import prepare_obs_for_policy
from utils.constants import MOTOR_NAMES, NUM_JOINTS, SIM_ACTION_HIGH, GRIPPER_IDX
from utils.ik_solver import IKSolver

LIFT_HEIGHT = 0.05  # 5cm above table = successful lift


def run_pickup_episodes(
    sim, policy, preprocessor, postprocessor, device,
    block_x, block_y, num_episodes=5, max_steps=150, viewer=False,
    pre_position=False, ik_solver=None,
    pre_position_height=0.08, ik_threshold_mm=30.0,
    record_dataset=None,
    lift_height=None,
):
    """Run pickup attempts at a specific block position. No conditioning — just raw policy.

    When pre_position=True, uses IK to teleport gripper above block before running policy.

    Args:
        lift_height: Height threshold for successful lift (default: LIFT_HEIGHT=0.05m).

    Returns (successes, total, details_list, ik_error_mm).
    ik_error_mm is None when pre_position=False.
    """
    import mujoco

    if lift_height is None:
        lift_height = LIFT_HEIGHT

    # Pre-position: solve IK to place gripper above block
    ik_error_mm = None
    ik_config = None  # 6-dim: 5 arm + 1 gripper

    if pre_position:
        assert ik_solver is not None, "ik_solver required when pre_position=True"
        target_pos = np.array([block_x, block_y, pre_position_height])
        ik_joints, _, ik_error = ik_solver.solve_ik(
            target_pos=target_pos,
            target_rot=np.eye(3),
        )
        ik_error_mm = ik_error * 1000
        if ik_error_mm > ik_threshold_mm:
            return 0, num_episodes, [], ik_error_mm
        gripper_open = SIM_ACTION_HIGH[GRIPPER_IDX]
        ik_config = np.concatenate([ik_joints, [gripper_open]])

    duplo_body_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "duplo")

    # Find EE site for distance tracking
    ee_site_id = None
    for site_name in ["gripperframe", "gripper_site", "ee_site"]:
        sid = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if sid != -1:
            ee_site_id = sid
            break

    successes = 0
    details = []

    task_label = f"Pick up block at ({block_x:.3f}, {block_y:.3f})"

    for ep in range(num_episodes):
        policy.reset()

        if record_dataset is not None:
            record_dataset.create_episode_buffer()

        # Reset scene, place block at target position
        sim.reset_scene(randomize=False, pos_range=0.0,
                        pos_center_x=block_x, pos_center_y=block_y)

        # Pre-position: teleport robot to IK solution above block
        if pre_position and ik_config is not None:
            sim.mj_data.qpos[7:13] = ik_config.copy()
            sim.mj_data.ctrl[:6] = ik_config.copy()
            for _ in range(50):
                mujoco.mj_step(sim.mj_model, sim.mj_data)

        max_height = 0.0
        lifted = False
        approached = False

        for step in range(max_steps):
            obs = sim.get_observation()

            if viewer:
                if not sim.render():
                    return successes, ep, details, ik_error_mm

            batch = prepare_obs_for_policy(obs, device)
            batch = preprocessor(batch)

            with torch.no_grad():
                action = policy.select_action(batch)
                action = postprocessor(action)

            action_np = action.cpu().numpy().flatten()
            action_dict = {f"{MOTOR_NAMES[i]}.pos": float(action_np[i])
                           for i in range(NUM_JOINTS)}
            sim.send_action(action_dict)

            # Record frame
            if record_dataset is not None:
                obs_frame = build_dataset_frame(record_dataset.features, obs, prefix="observation")
                action_frame = build_dataset_frame(record_dataset.features, action_dict, prefix="action")
                record_dataset.add_frame({**obs_frame, **action_frame, "task": task_label})

            # Track block height
            block_z = sim.mj_data.xpos[duplo_body_id][2]
            max_height = max(max_height, block_z)

            # Track approach (for diagnostics)
            if ee_site_id is not None and not approached:
                ee_pos = sim.mj_data.site_xpos[ee_site_id]
                block_pos = sim.mj_data.xpos[duplo_body_id]
                dist_xy = np.linalg.norm(ee_pos[:2] - block_pos[:2])
                if dist_xy < 0.06:  # 6cm
                    approached = True

            if max_height > lift_height:
                lifted = True
                break

        if lifted:
            successes += 1
        details.append({
            "success": lifted,
            "approached": approached,
            "max_height": float(max_height),
            "steps": step + 1,
        })

        if record_dataset is not None:
            record_dataset.save_episode()

    return successes, num_episodes, details, ik_error_mm


def wait_for_training(model_dir: Path, check_interval=60):
    """Wait until a 'final' checkpoint appears in model_dir."""
    final_path = model_dir / "final"
    print(f"Waiting for training to finish (checking for {final_path})...")

    while not final_path.exists():
        # Show latest checkpoint
        checkpoints = sorted(
            [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint_")],
            key=lambda x: int(x.name.split("_")[1])
        ) if model_dir.exists() else []

        if checkpoints:
            latest = checkpoints[-1].name
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] Latest: {latest}, waiting...")
        else:
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] No checkpoints yet, waiting...")

        time.sleep(check_interval)

    print(f"  Training complete! Found {final_path}")
    # Give a few seconds for files to finish writing
    time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description="Evaluate pickup-only model spatially")
    parser.add_argument("path", type=str, help="Training output directory")
    parser.add_argument("--checkpoint", type=str, default="final",
                        help="Checkpoint name (default: final)")
    parser.add_argument("--policy", type=str, default="act_vit")
    parser.add_argument("--grid-size", type=int, default=None,
                        help="Grid size for spatial test (e.g. 5 for 5x5)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=150,
                        help="Max steps per episode (default: 150)")
    parser.add_argument("--block-x", type=float, default=None)
    parser.add_argument("--block-y", type=float, default=None)
    parser.add_argument("--x-min", type=float, default=0.10)
    parser.add_argument("--x-max", type=float, default=0.35)
    parser.add_argument("--y-min", type=float, default=0.08)
    parser.add_argument("--y-max", type=float, default=0.38)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--pre-position", action="store_true",
                        help="Pre-position arm above block via IK before running policy")
    parser.add_argument("--pre-position-height", type=float, default=0.08,
                        help="Height above table for pre-positioning (default: 0.08m)")
    parser.add_argument("--ik-threshold", type=float, default=30.0,
                        help="Max IK error in mm before skipping position (default: 30)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset repo ID for loading stats (if needed)")
    parser.add_argument("--wait", action="store_true",
                        help="Wait for training to finish before evaluating")
    parser.add_argument("--record", type=str, default=None,
                        help="Record episodes to LeRobot dataset at this path")
    parser.add_argument("--scene", type=str, default=None,
                        help="Scene XML filename (e.g. so101_dark_ground.xml)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_dir = Path(args.path)

    # Wait for training if requested
    if args.wait:
        wait_for_training(model_dir)

    # Find checkpoint
    checkpoint_path = model_dir / args.checkpoint
    if not checkpoint_path.exists():
        checkpoints = sorted(
            [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint_")],
            key=lambda x: int(x.name.split("_")[1])
        )
        if checkpoints:
            checkpoint_path = checkpoints[-1]
            print(f"Using latest checkpoint: {checkpoint_path.name}")
        else:
            print(f"ERROR: No checkpoint found at {model_dir}")
            sys.exit(1)

    # Load training metadata
    meta_path = checkpoint_path / "training_metadata.json"
    training_meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            training_meta = json.load(f)

    dataset_id = args.dataset or training_meta.get("dataset_repo_id")

    print(f"Loading model from: {checkpoint_path}")
    policy, preprocessor, postprocessor = load_policy_and_processors(
        checkpoint_path, args.policy, device, dataset_id
    )

    # Create simulation
    from lerobot_robot_sim import SO100SimConfig, SO100Sim

    scene_name = args.scene or "so101_with_wrist_cam.xml"
    scene_path = REPO_ROOT / "scenes" / scene_name
    config = SO100SimConfig(
        scene_xml=str(scene_path),
        sim_cameras=["wrist_cam", "overhead_cam"],
        camera_width=640,
        camera_height=480,
    )
    sim = SO100Sim(config)
    sim.connect()

    # Create recording dataset if requested
    record_dataset = None
    if args.record:
        action_features = hw_to_dataset_features(sim.action_features, "action")
        obs_features = hw_to_dataset_features(sim.observation_features, "observation")
        features = {**action_features, **obs_features}
        record_dataset = LeRobotDataset.create(
            repo_id=f"danbhf/{Path(args.record).name}",
            fps=30,
            root=args.record,
            robot_type="so100_sim",
            features=features,
            image_writer_threads=4,
        )

    # Create IK solver if pre-positioning
    ik_solver = IKSolver(scene_xml=str(scene_path)) if args.pre_position else None

    # Build position list
    if args.block_x is not None and args.block_y is not None:
        positions = [(args.block_x, args.block_y)]
        mode = "single"
    elif args.grid_size:
        xs = np.linspace(args.x_min, args.x_max, args.grid_size)
        ys = np.linspace(args.y_min, args.y_max, args.grid_size)
        positions = [(x, y) for x in xs for y in ys]
        mode = "grid"
    else:
        # Default: test at training positions
        positions = [
            (0.213, 0.254),   # Position 1 (y > 0.1)
            (0.213, -0.047),  # Position 2 (y < 0.1)
        ]
        mode = "training"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prepos_tag = "_prepos" if args.pre_position else ""
    csv_path = args.output or f"outputs/experiments/pickup_model_{mode}{prepos_tag}_{timestamp}.csv"
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\nPICKUP-ONLY MODEL EVALUATION")
    print(f"Model: {args.path}")
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"Dataset: {dataset_id}")
    print(f"Mode: {mode}")
    print(f"Pre-position: {args.pre_position}")
    print(f"Positions: {len(positions)}")
    print(f"Episodes per position: {args.episodes}")
    print(f"Max steps: {args.max_steps}")
    print(f"Output: {csv_path}")
    print()

    results = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "success_rate", "episodes", "successes",
                          "approached", "avg_height", "avg_steps_success", "ik_error"])

        for pos_idx, (x, y) in enumerate(positions):
            print(f"  Position {pos_idx + 1}/{len(positions)}: ({x:.3f}, {y:.3f})...",
                  end=" ", flush=True)

            try:
                succ, total, details, ik_err = run_pickup_episodes(
                    sim, policy, preprocessor, postprocessor, device,
                    block_x=x, block_y=y,
                    num_episodes=args.episodes,
                    max_steps=args.max_steps,
                    viewer=args.viewer,
                    pre_position=args.pre_position,
                    ik_solver=ik_solver,
                    pre_position_height=args.pre_position_height,
                    ik_threshold_mm=args.ik_threshold,
                    record_dataset=record_dataset,
                )
                rate = succ / total if total > 0 else 0
                approach_count = sum(1 for d in details if d["approached"])
                heights = [d["max_height"] for d in details]
                avg_h = np.mean(heights) if heights else 0
                steps_succ = [d["steps"] for d in details if d["success"]]
                avg_s = np.mean(steps_succ) if steps_succ else 0
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
                rate, succ, approach_count, avg_h, avg_s, ik_err = 0, 0, 0, 0, 0, None

            ik_info = f", ik_err={ik_err:.1f}mm" if ik_err is not None else ""
            print(f"{rate * 100:.0f}% ({succ}/{args.episodes}), "
                  f"approached={approach_count}, avg_h={avg_h:.3f}m{ik_info}")

            ik_err_str = f"{ik_err:.2f}" if ik_err is not None else ""
            writer.writerow([f"{x:.4f}", f"{y:.4f}", f"{rate:.2f}",
                              args.episodes, succ, approach_count,
                              f"{avg_h:.4f}", f"{avg_s:.1f}", ik_err_str])
            f.flush()
            results.append({"x": x, "y": y, "rate": rate,
                            "approached": approach_count, "avg_h": avg_h})

    if record_dataset is not None:
        record_dataset.finalize()
        print(f"Recorded dataset saved to: {args.record}")

    sim.disconnect()

    # ==================== SUMMARY ====================
    print()
    print("=" * 60)
    print("PICKUP-ONLY MODEL EVALUATION SUMMARY")
    print("=" * 60)

    rates = [r["rate"] for r in results]
    print(f"Overall pickup success: {np.mean(rates) * 100:.1f}%")
    print(f"Positions with >0% pickup:  {sum(1 for r in rates if r > 0)}/{len(rates)}")
    print(f"Positions with >50% pickup: {sum(1 for r in rates if r > 0.5)}/{len(rates)}")
    print(f"Positions with 100% pickup: {sum(1 for r in rates if r >= 1.0)}/{len(rates)}")

    approach_rates = [r["approached"] / args.episodes for r in results]
    print(f"\nOverall approach rate: {np.mean(approach_rates) * 100:.1f}%")

    if mode == "grid" and args.grid_size:
        # Distance analysis from training positions
        train_positions = [(0.213, 0.254), (0.213, -0.047)]
        print(f"\nBy distance from nearest training position:")
        for d_max in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
            in_range = [r for r in results
                        if min(np.sqrt((r["x"] - tx) ** 2 + (r["y"] - ty) ** 2)
                               for tx, ty in train_positions) <= d_max]
            if in_range:
                avg = np.mean([r["rate"] for r in in_range])
                print(f"  Within {d_max * 100:.0f}cm: {avg * 100:.1f}% "
                      f"({len(in_range)} positions)")

        # Print grid
        xs = np.linspace(args.x_min, args.x_max, args.grid_size)
        ys = np.linspace(args.y_min, args.y_max, args.grid_size)

        print(f"\nPickup Success Grid (% success):")
        print(f"{'Y\\X':>8}", end="")
        for x in xs:
            print(f" {x:.2f}", end="")
        print()
        for y in reversed(ys):
            print(f"{y:.2f}  ", end="")
            for x in xs:
                r = next((r for r in results
                          if abs(r["x"] - x) < 0.001 and abs(r["y"] - y) < 0.001), None)
                if r:
                    pct = r["rate"] * 100
                    if pct >= 80:
                        marker = f"  {pct:3.0f}"
                    elif pct > 0:
                        marker = f"  {pct:3.0f}"
                    else:
                        marker = "    ."
                    print(marker, end="")
                else:
                    print("    ?", end="")
            print()

        print(f"\nApproach Rate Grid (% reached block):")
        print(f"{'Y\\X':>8}", end="")
        for x in xs:
            print(f" {x:.2f}", end="")
        print()
        for y in reversed(ys):
            print(f"{y:.2f}  ", end="")
            for x in xs:
                r = next((r for r in results
                          if abs(r["x"] - x) < 0.001 and abs(r["y"] - y) < 0.001), None)
                if r:
                    pct = r["approached"] / args.episodes * 100
                    if pct >= 80:
                        marker = f"  {pct:3.0f}"
                    elif pct > 0:
                        marker = f"  {pct:3.0f}"
                    else:
                        marker = "    ."
                    print(marker, end="")
                else:
                    print("    ?", end="")
            print()

    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
