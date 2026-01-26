#!/usr/bin/env python3
"""
Interactive object positioning tool for MuJoCo simulation.

Use keyboard to move the block around and see coordinates.

Usage:
    python scripts/tools/position_objects.py
    python scripts/tools/position_objects.py --scene scenes/so101_with_confuser.xml
"""

import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path


DEFAULT_SCENE = Path(__file__).parent.parent.parent / "scenes" / "so101_with_wrist_cam.xml"

# Movement step size in meters
MOVE_STEP = 0.01  # 1cm per keypress
MOVE_STEP_FINE = 0.002  # 2mm for fine adjustment


def main():
    parser = argparse.ArgumentParser(description="Interactive object positioning tool")
    parser.add_argument("--scene", type=str, default=None,
                        help="Scene XML path (default: so101_with_wrist_cam.xml)")
    args = parser.parse_args()

    # Determine scene path
    if args.scene:
        scene_path = Path(args.scene)
        if not scene_path.is_absolute():
            scene_path = Path(__file__).parent.parent.parent / args.scene
    else:
        scene_path = DEFAULT_SCENE

    print("=" * 60)
    print("OBJECT POSITIONING TOOL")
    print("=" * 60)
    print("\nLoading scene...")

    # Load model
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    # Reset
    mujoco.mj_resetData(model, data)

    print(f"\nScene loaded: {scene_path.name}")
    print("\n" + "-" * 60)
    print("CONTROLS:")
    print("  W/S     - Move block forward/backward (X axis)")
    print("  A/D     - Move block left/right (Y axis)")
    print("  Hold SHIFT for fine movement (2mm vs 1cm)")
    print("")
    print("  P       - Print block position")
    print("  R       - Reset scene")
    print("  ESC     - Quit")
    print("-" * 60)

    # Get initial position
    block_x = data.qpos[0]
    block_y = data.qpos[1]
    print(f"\nInitial block position: x={block_x:.4f}, y={block_y:.4f}")
    print(f"(Training was at x=0.217, y=0.225)")
    print("\nStarting viewer...\n")

    last_action_time = 0
    shift_held = False

    def print_position():
        block_x = data.qpos[0]
        block_y = data.qpos[1]
        block_z = data.qpos[2]

        # Get rotation
        qw, qx, qy, qz = data.qpos[3:7]
        angle = 2 * np.arctan2(qz, qw)
        angle_deg = np.degrees(angle)

        print(f"\n>>> BLOCK POSITION: x={block_x:.4f}, y={block_y:.4f}")
        print(f"    Height: z={block_z:.4f}, Rotation: {angle_deg:.1f} deg")
        print(f"    Command: --block-x {block_x:.3f} --block-y {block_y:.3f}")

    def move_block(dx, dy):
        """Move block by delta x, y."""
        data.qpos[0] += dx
        data.qpos[1] += dy
        # Reset velocity so it doesn't drift
        data.qvel[0] = 0
        data.qvel[1] = 0
        data.qvel[2] = 0
        mujoco.mj_forward(model, data)

        block_x = data.qpos[0]
        block_y = data.qpos[1]
        print(f"Block: x={block_x:.4f}, y={block_y:.4f}")

    def key_callback(keycode):
        nonlocal last_action_time, shift_held

        now = time.time()
        if now - last_action_time < 0.05:  # Debounce
            return

        step = MOVE_STEP_FINE if shift_held else MOVE_STEP

        # WASD for movement
        if keycode == ord('W') or keycode == ord('w'):
            move_block(step, 0)  # Forward (+X)
            last_action_time = now
        elif keycode == ord('S') or keycode == ord('s'):
            move_block(-step, 0)  # Backward (-X)
            last_action_time = now
        elif keycode == ord('A') or keycode == ord('a'):
            move_block(0, step)  # Left (+Y)
            last_action_time = now
        elif keycode == ord('D') or keycode == ord('d'):
            move_block(0, -step)  # Right (-Y)
            last_action_time = now

        # P = print position
        elif keycode == ord('P') or keycode == ord('p'):
            print_position()
            last_action_time = now

        # R = reset
        elif keycode == ord('R') or keycode == ord('r'):
            mujoco.mj_resetData(model, data)
            print("\n>>> Scene reset")
            last_action_time = now

        # Track shift key (keycode 340 = left shift, 344 = right shift)
        elif keycode == 340 or keycode == 344:
            shift_held = True

    # Launch viewer
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        print("Viewer opened. Use WASD to move block, P to print position.")

        while viewer.is_running():
            # Step simulation (but block position is controlled by us)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(1/60)

            # Reset shift held each frame (since we don't get key-up events)
            shift_held = False

    # Final position
    print_position()


if __name__ == "__main__":
    main()
