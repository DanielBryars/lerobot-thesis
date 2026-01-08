#!/usr/bin/env python
"""
MuJoCo Scene Viewer with Hot-Reload

Interactive viewer for testing MuJoCo XML scene files. Automatically reloads
when the XML file is modified, showing both RGB and depth views.

Usage:
    python scripts/scene_viewer.py                           # Default scene
    python scripts/scene_viewer.py scenes/so101_rgbd.xml     # Custom scene
    python scripts/scene_viewer.py --no-depth                # RGB only

Controls:
    Q/ESC    - Quit
    R        - Manual reload
    D        - Toggle depth view
    1-9      - Switch cameras
    SPACE    - Reset robot pose
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project paths
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

import mujoco


class SceneViewer:
    """MuJoCo scene viewer with hot-reload and depth rendering."""

    def __init__(self, xml_path: str, width: int = 640, height: int = 480):
        self.xml_path = Path(xml_path).resolve()
        self.width = width
        self.height = height

        self.model = None
        self.data = None
        self.renderer = None
        self.last_mtime = 0
        self.show_depth = True
        self.current_camera_idx = 0
        self.camera_names = []

        self._load_model()

    def _load_model(self):
        """Load or reload the MuJoCo model from XML."""
        try:
            print(f"Loading: {self.xml_path}")
            self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
            self.data = mujoco.MjData(self.model)
            self.renderer = mujoco.Renderer(self.model, self.height, self.width)
            self.last_mtime = os.path.getmtime(self.xml_path)

            # Get camera names
            self.camera_names = []
            for i in range(self.model.ncam):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                if name:
                    self.camera_names.append(name)

            print(f"  Cameras: {self.camera_names}")
            print(f"  Bodies: {self.model.nbody}, Geoms: {self.model.ngeom}")

            # Step simulation to settle
            for _ in range(100):
                mujoco.mj_step(self.model, self.data)

            return True
        except Exception as e:
            print(f"ERROR loading model: {e}")
            return False

    def check_reload(self) -> bool:
        """Check if XML file has been modified and reload if needed."""
        try:
            current_mtime = os.path.getmtime(self.xml_path)
            if current_mtime > self.last_mtime:
                print("\nFile changed, reloading...")
                return self._load_model()
        except Exception as e:
            print(f"Error checking file: {e}")
        return False

    def render_rgb(self, camera_id: int = None) -> np.ndarray:
        """Render RGB image from specified camera."""
        self.renderer.disable_depth_rendering()
        if camera_id is not None:
            self.renderer.update_scene(self.data, camera=camera_id)
        else:
            self.renderer.update_scene(self.data)
        return self.renderer.render().copy()

    def render_depth(self, camera_id: int = None) -> np.ndarray:
        """Render depth image from specified camera."""
        self.renderer.enable_depth_rendering()
        if camera_id is not None:
            self.renderer.update_scene(self.data, camera=camera_id)
        else:
            self.renderer.update_scene(self.data)
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        return depth

    def depth_to_display(self, depth: np.ndarray, max_depth: float = 1.5) -> np.ndarray:
        """Convert depth buffer to displayable BGR image."""
        # Clip and normalize
        depth_clipped = np.clip(depth, 0, max_depth)
        depth_normalized = depth_clipped / max_depth

        # Apply colormap (TURBO gives nice visualization)
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)

        return depth_colored

    def get_camera_id(self, camera_name: str) -> int:
        """Get camera ID from name."""
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

    def run(self):
        """Main viewer loop."""
        window_name = "Scene Viewer (Q=quit, D=depth, R=reload, 1-9=cameras)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("\n" + "="*60)
        print("Scene Viewer Controls:")
        print("  Q/ESC  - Quit")
        print("  R      - Manual reload")
        print("  D      - Toggle depth view")
        print("  1-9    - Switch cameras")
        print("  SPACE  - Reset robot pose")
        print("="*60 + "\n")

        running = True
        frame_count = 0
        last_fps_time = time.time()
        fps = 0

        while running:
            # Check for file changes every 30 frames
            if frame_count % 30 == 0:
                self.check_reload()

            # Get current camera
            camera_id = None
            camera_name = "default"
            if self.camera_names and self.current_camera_idx < len(self.camera_names):
                camera_name = self.camera_names[self.current_camera_idx]
                camera_id = self.get_camera_id(camera_name)

            # Render RGB
            rgb = self.render_rgb(camera_id)
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Render depth if enabled
            if self.show_depth:
                depth = self.render_depth(camera_id)
                depth_display = self.depth_to_display(depth)

                # Stack horizontally
                display = np.hstack([rgb_bgr, depth_display])
            else:
                display = rgb_bgr

            # Add info overlay
            info_lines = [
                f"Camera: {camera_name} ({self.current_camera_idx + 1}/{len(self.camera_names)})",
                f"FPS: {fps:.1f}",
                f"Depth: {'ON' if self.show_depth else 'OFF'}",
            ]

            y_offset = 25
            for line in info_lines:
                cv2.putText(display, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25

            # Show depth range info
            if self.show_depth:
                depth_min = depth.min()
                depth_max = depth.max()
                cv2.putText(display, f"Depth: {depth_min:.3f}m - {depth_max:.3f}m",
                           (self.width + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow(window_name, display)

            # Handle keyboard input
            key = cv2.waitKey(16) & 0xFF  # ~60 FPS

            if key == ord('q') or key == 27:  # Q or ESC
                running = False
            elif key == ord('r'):  # Manual reload
                print("Manual reload...")
                self._load_model()
            elif key == ord('d'):  # Toggle depth
                self.show_depth = not self.show_depth
                print(f"Depth: {'ON' if self.show_depth else 'OFF'}")
            elif key == ord(' '):  # Reset
                self.data.qpos[:] = 0
                mujoco.mj_forward(self.model, self.data)
                print("Reset pose")
            elif ord('1') <= key <= ord('9'):  # Camera selection
                idx = key - ord('1')
                if idx < len(self.camera_names):
                    self.current_camera_idx = idx
                    print(f"Camera: {self.camera_names[idx]}")

            # Update FPS counter
            frame_count += 1
            if frame_count % 30 == 0:
                now = time.time()
                fps = 30 / (now - last_fps_time)
                last_fps_time = now

            # Step simulation (keeps physics alive)
            mujoco.mj_step(self.model, self.data)

        cv2.destroyAllWindows()
        print("Viewer closed.")


def main():
    parser = argparse.ArgumentParser(
        description="MuJoCo Scene Viewer with Hot-Reload",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("xml_path", nargs="?",
                        default=str(repo_root / "scenes" / "so101_with_wrist_cam.xml"),
                        help="Path to MuJoCo XML scene file")
    parser.add_argument("--width", "-w", type=int, default=640,
                        help="Render width (default: 640)")
    parser.add_argument("--height", "-H", type=int, default=480,
                        help="Render height (default: 480)")
    parser.add_argument("--no-depth", action="store_true",
                        help="Start with depth view disabled")

    args = parser.parse_args()

    # Validate XML path
    xml_path = Path(args.xml_path)
    if not xml_path.exists():
        print(f"ERROR: XML file not found: {xml_path}")
        sys.exit(1)

    viewer = SceneViewer(str(xml_path), args.width, args.height)
    viewer.show_depth = not args.no_depth
    viewer.run()


if __name__ == "__main__":
    main()
