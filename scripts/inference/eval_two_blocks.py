#!/usr/bin/env python
"""
Evaluate model behavior with TWO blocks in the scene.

This test places one white block at position 1 (training area 1) and one red block
at position 2 (training area 2) to see how the model behaves when both are present.

Key questions:
- Does the model consistently pick one block over the other?
- Does it get confused and fail?
- Does it sometimes pick white, sometimes red?

Usage:
    python eval_two_blocks.py --checkpoint outputs/train/act_2pos_220ep/checkpoint_030000
    python eval_two_blocks.py --checkpoint outputs/train/act_2pos_220ep/checkpoint_030000 --episodes 20
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch
import json
from datetime import datetime
import mujoco

# Add project root to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from utils.training import prepare_obs_for_policy
from utils.constants import MOTOR_NAMES, NUM_JOINTS


# Training positions (approximate centers)
POS1_CENTER = (0.22, 0.225)  # White block position
POS2_CENTER = (0.32, -0.03)  # Red block position


def load_policy(checkpoint_path: Path, device: str = "cuda"):
    """Load ACT policy from checkpoint."""
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    # Load policy
    policy = ACTPolicy.from_pretrained(str(checkpoint_path))
    policy.eval()
    policy.to(device)

    # Load pre/post processors
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint_path)
    )

    return policy, preprocessor, postprocessor


def get_block_positions(data, model):
    """Get positions of both blocks from simulation state.

    Returns:
        dict with 'white' and 'red' block positions (x, y, z)
    """
    # Get body IDs
    duplo_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "duplo")
    duplo2_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "duplo2")

    white_pos = data.xpos[duplo_body_id].copy()
    red_pos = data.xpos[duplo2_body_id].copy()

    return {
        'white': white_pos,
        'red': red_pos
    }


def check_block_in_bowl(block_pos, bowl_center=(0.217, -0.225), tolerance=0.06):
    """Check if a block is in the bowl."""
    dx = block_pos[0] - bowl_center[0]
    dy = block_pos[1] - bowl_center[1]
    return abs(dx) < tolerance and abs(dy) < tolerance and block_pos[2] < 0.05


def check_block_lifted(initial_z, current_z, threshold=0.03):
    """Check if block has been lifted significantly."""
    return current_z > initial_z + threshold


def set_arm_start_position(sim_robot, start_near: int):
    """Set the robot arm to start near a specific block position.

    Args:
        sim_robot: The simulation robot
        start_near: 1 for block at pos1, 2 for block at pos2
    """
    # In two-block scene, qpos layout:
    # [0:7] = duplo freejoint (pos xyz + quat wxyz)
    # [7:14] = duplo2 freejoint
    # [14:20] = robot joints (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper)

    # Robot joint indices
    SHOULDER_PAN_IDX = 14
    SHOULDER_LIFT_IDX = 15
    ELBOW_FLEX_IDX = 16
    WRIST_FLEX_IDX = 17
    WRIST_ROLL_IDX = 18
    GRIPPER_IDX = 19

    if start_near == 1:
        # Position arm toward block 1 (positive Y, x=0.22, y=0.225)
        # Rotate shoulder pan positive to face that direction
        sim_robot.mj_data.qpos[SHOULDER_PAN_IDX] = 0.8  # ~45 degrees toward pos1
        sim_robot.mj_data.qpos[SHOULDER_LIFT_IDX] = -0.3
        sim_robot.mj_data.qpos[ELBOW_FLEX_IDX] = 0.5
    elif start_near == 2:
        # Position arm toward block 2 (negative Y, x=0.32, y=-0.03)
        # Rotate shoulder pan negative to face that direction
        sim_robot.mj_data.qpos[SHOULDER_PAN_IDX] = -0.5  # ~-30 degrees toward pos2
        sim_robot.mj_data.qpos[SHOULDER_LIFT_IDX] = -0.3
        sim_robot.mj_data.qpos[ELBOW_FLEX_IDX] = 0.5

    # Step to apply the changes
    for _ in range(10):
        mujoco.mj_step(sim_robot.mj_model, sim_robot.mj_data)


def run_episode(sim_robot, policy, preprocessor, postprocessor, device, max_steps=300, start_near=None, viewer=False, skip_reset=False):
    """Run a single two-block episode.

    Returns:
        dict with results:
        - success: bool (any block in bowl)
        - white_picked: bool (white block was lifted)
        - red_picked: bool (red block was lifted)
        - white_in_bowl: bool
        - red_in_bowl: bool
        - steps: int
        - behavior: str (description of what happened)
    """
    # Reset environment (blocks are at fixed positions in the two-block scene)
    if not skip_reset:
        sim_robot.reset_scene(randomize=False)

        # Set arm starting position if specified
        if start_near is not None:
            set_arm_start_position(sim_robot, start_near)

    # Reset policy state
    policy.reset()

    # Get initial block positions
    initial_positions = get_block_positions(sim_robot.mj_data, sim_robot.mj_model)
    white_initial_z = initial_positions['white'][2]
    red_initial_z = initial_positions['red'][2]

    # Track if blocks were ever lifted
    white_ever_lifted = False
    red_ever_lifted = False

    # Run episode
    for step in range(max_steps):
        # Get observation
        obs = sim_robot.get_observation()

        # Prepare for policy
        batch = prepare_obs_for_policy(obs, device)

        # Apply preprocessor
        batch = preprocessor(batch)

        # Get action
        with torch.no_grad():
            action = policy.select_action(batch)

        # Apply postprocessor
        action = postprocessor(action)
        action_np = action.cpu().numpy()
        if action_np.ndim > 1:
            action_np = action_np.flatten()

        # Send action to simulation
        action_dict = {f"{MOTOR_NAMES[i]}.pos": float(action_np[i]) for i in range(NUM_JOINTS)}
        sim_robot.send_action(action_dict)

        # Render viewer if enabled
        if viewer:
            if not sim_robot.render():
                return None  # Viewer was closed

        # Check block states
        current_positions = get_block_positions(sim_robot.mj_data, sim_robot.mj_model)

        if check_block_lifted(white_initial_z, current_positions['white'][2]):
            white_ever_lifted = True
        if check_block_lifted(red_initial_z, current_positions['red'][2]):
            red_ever_lifted = True

        # Check success (either block in bowl)
        white_in_bowl = check_block_in_bowl(current_positions['white'])
        red_in_bowl = check_block_in_bowl(current_positions['red'])

        if white_in_bowl or red_in_bowl:
            # Determine behavior
            if white_in_bowl and not red_ever_lifted:
                behavior = "picked_white_only"
            elif red_in_bowl and not white_ever_lifted:
                behavior = "picked_red_only"
            elif white_in_bowl:
                behavior = "picked_white_after_touching_red"
            else:
                behavior = "picked_red_after_touching_white"

            return {
                'success': True,
                'white_picked': white_ever_lifted,
                'red_picked': red_ever_lifted,
                'white_in_bowl': white_in_bowl,
                'red_in_bowl': red_in_bowl,
                'steps': step + 1,
                'behavior': behavior
            }

    # Timeout - determine what happened
    if white_ever_lifted and red_ever_lifted:
        behavior = "confused_touched_both"
    elif white_ever_lifted:
        behavior = "failed_with_white"
    elif red_ever_lifted:
        behavior = "failed_with_red"
    else:
        behavior = "no_pickup_attempted"

    return {
        'success': False,
        'white_picked': white_ever_lifted,
        'red_picked': red_ever_lifted,
        'white_in_bowl': False,
        'red_in_bowl': False,
        'steps': max_steps,
        'behavior': behavior
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate model with two blocks")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=300,
                        help="Maximum steps per episode")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--scene", type=str, default="so101_two_blocks.xml",
                        help="Scene file to use (in scenes/ directory)")
    parser.add_argument("--start-near", type=int, choices=[1, 2], default=None,
                        help="Start arm near block 1 or 2 (default: neutral)")
    parser.add_argument("--viewer", action="store_true",
                        help="Show MuJoCo 3D viewer")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Two-Block Evaluation")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Scene: {args.scene}")
    print(f"Episodes: {args.episodes}")
    print(f"Max steps: {args.max_steps}")

    # Load policy
    print("\nLoading policy...")
    policy, preprocessor, postprocessor = load_policy(checkpoint_path, args.device)

    # Create environment with two-block scene
    print("Creating two-block environment...")
    from lerobot_robot_sim import SO100SimConfig, SO100Sim

    scene_path = repo_root / "scenes" / args.scene
    config = SO100SimConfig(
        scene_xml=str(scene_path),
        sim_cameras=['wrist_cam', 'overhead_cam'],
        depth_cameras=[],
        enable_vr=False,
        camera_width=640,
        camera_height=480,
    )
    sim_robot = SO100Sim(config)
    sim_robot.connect()

    print(f"\nBlock positions:")
    print(f"  White (pos1 area): x={POS1_CENTER[0]:.2f}, y={POS1_CENTER[1]:.2f}")
    print(f"  Red (pos2 area):   x={POS2_CENTER[0]:.2f}, y={POS2_CENTER[1]:.2f}")
    if args.start_near:
        print(f"\nArm starting position: Near block {args.start_near}")

    # Run episodes
    results = []
    print(f"\nRunning {args.episodes} episodes...")
    print("-" * 60)

    for ep in range(args.episodes):
        # Show starting position and wait for user before episode starts
        if args.viewer:
            # Reset and set starting position so user can see it
            sim_robot.reset_scene(randomize=False)
            if args.start_near is not None:
                set_arm_start_position(sim_robot, args.start_near)

            print(f"\n>>> Episode {ep+1}/{args.episodes} ready. Press ENTER to start (or 'q' to quit) <<<")
            import msvcrt
            while True:
                sim_robot.render()
                time.sleep(0.033)
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
                    if key == '\r' or key == ' ':
                        break
                    elif key == 'q':
                        print("\nQuitting...")
                        sim_robot.disconnect()
                        sys.exit(0)

        result = run_episode(
            sim_robot, policy, preprocessor, postprocessor,
            device=args.device,
            max_steps=args.max_steps,
            start_near=args.start_near,
            viewer=args.viewer,
            skip_reset=args.viewer,  # Skip reset in run_episode if we already did it
        )

        # Handle viewer closed
        if result is None:
            print("\nViewer closed, stopping evaluation.")
            break

        results.append(result)

        status = "SUCCESS" if result['success'] else "FAIL"
        block = "WHITE" if result['white_in_bowl'] else ("RED" if result['red_in_bowl'] else "NONE")
        print(f"Episode {ep+1:3d}: {status:7s} | Block: {block:5s} | Steps: {result['steps']:3d} | {result['behavior']}")

        # Wait for user input between episodes if viewer is active
        if args.viewer and ep < args.episodes - 1:
            print("    >>> Press ENTER for next episode (or 'q' to quit) <<<")
            # Keep rendering while waiting for input
            import sys
            import select
            if sys.platform == 'win32':
                # Windows: use msvcrt for non-blocking input
                import msvcrt
                while True:
                    sim_robot.render()
                    time.sleep(0.033)
                    if msvcrt.kbhit():
                        key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
                        if key == '\r' or key == ' ':  # Enter or Space
                            break
                        elif key == 'q':
                            print("\nQuitting...")
                            sim_robot.disconnect()
                            sys.exit(0)
            else:
                # Unix: just use input (blocks but viewer stays visible)
                user_input = input()
                if user_input.lower() == 'q':
                    print("\nQuitting...")
                    sim_robot.disconnect()
                    sys.exit(0)

    # Summarize results
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    success_count = sum(1 for r in results if r['success'])
    white_success = sum(1 for r in results if r['white_in_bowl'])
    red_success = sum(1 for r in results if r['red_in_bowl'])

    print(f"\nOverall success: {success_count}/{args.episodes} ({100*success_count/args.episodes:.1f}%)")
    print(f"  White block placed: {white_success} ({100*white_success/args.episodes:.1f}%)")
    print(f"  Red block placed:   {red_success} ({100*red_success/args.episodes:.1f}%)")

    # Behavior breakdown
    behavior_counts = {}
    for r in results:
        b = r['behavior']
        behavior_counts[b] = behavior_counts.get(b, 0) + 1

    print(f"\nBehavior breakdown:")
    for behavior, count in sorted(behavior_counts.items(), key=lambda x: -x[1]):
        print(f"  {behavior}: {count} ({100*count/args.episodes:.1f}%)")

    # Analysis
    print(f"\nAnalysis:")
    if white_success > red_success * 2:
        print("  -> Model strongly prefers WHITE block (position 1 area)")
    elif red_success > white_success * 2:
        print("  -> Model strongly prefers RED block (position 2 area)")
    elif abs(white_success - red_success) <= 2:
        print("  -> Model shows no strong preference (roughly equal)")
    else:
        print("  -> Model shows slight preference")

    if success_count < args.episodes * 0.5:
        print("  -> Model struggles with two-block scenario (< 50% success)")

    # Save results
    results_dir = checkpoint_path / "two_block_eval"
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    with open(results_file, 'w') as f:
        json.dump(convert_to_serializable({
            'checkpoint': str(checkpoint_path),
            'episodes': args.episodes,
            'success_rate': success_count / args.episodes,
            'white_success_rate': white_success / args.episodes,
            'red_success_rate': red_success / args.episodes,
            'behavior_counts': behavior_counts,
            'results': results
        }), f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Keep viewer open at the end if active
    if args.viewer:
        print("\nPress Ctrl+C or close viewer window to exit...")
        try:
            while sim_robot.render():
                time.sleep(0.033)
        except KeyboardInterrupt:
            print("\nExiting...")

    sim_robot.disconnect()


if __name__ == "__main__":
    main()
