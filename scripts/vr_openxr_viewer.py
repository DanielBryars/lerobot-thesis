"""
MuJoCo VR Viewer using OpenXR with Leader Arm Teleoperation.

Proper head-tracked VR rendering with Quest Link.
Uses pyopenxr for OpenXR bindings and MuJoCo for physics/rendering.
Optionally connects to a physical leader arm for teleoperation.

Requirements:
    pip install pyopenxr glfw PyOpenGL numpy mujoco

Setup:
    1. Connect Quest 3 via Quest Link (USB or Air Link)
    2. (Optional) Connect leader arm for teleoperation
    3. Run this script
    4. Put on headset - you should see the simulation in VR

Usage:
    python vr_openxr_viewer.py                    # VR only (no teleop)
    python vr_openxr_viewer.py --teleop           # VR + leader arm teleop
    python vr_openxr_viewer.py --teleop --port COM8

Based on pyopenxr examples by Christopher Bruns.
"""

import argparse
import ctypes
import json
import threading
import time
import numpy as np
import mujoco
from pathlib import Path

try:
    import xr
    import xr.exception
except ImportError:
    print("ERROR: pyopenxr not installed. Run: pip install pyopenxr")
    exit(1)

try:
    import glfw
    from OpenGL import GL
except ImportError:
    print("ERROR: OpenGL dependencies missing. Run: pip install glfw PyOpenGL")
    exit(1)


SCENE_XML = Path(__file__).parent.parent / "scenes" / "so101_with_wrist_cam.xml"

# Eye separation (IPD will come from OpenXR runtime)
DEFAULT_IPD = 0.063

# Sim action space bounds (radians)
SIM_ACTION_LOW = np.array([-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.17453])
SIM_ACTION_HIGH = np.array([1.91986, 1.74533, 1.69, 1.65806, 2.84121, 1.74533])


def normalized_to_radians(normalized_values: np.ndarray) -> np.ndarray:
    """Convert from lerobot normalized values to sim radians."""
    radians = np.zeros(6, dtype=np.float32)
    # Joints 0-4: map [-100, 100] -> [low, high]
    for i in range(5):
        t = (normalized_values[i] + 100) / 200.0
        radians[i] = SIM_ACTION_LOW[i] + t * (SIM_ACTION_HIGH[i] - SIM_ACTION_LOW[i])
    # Gripper: map [0, 100] -> [low, high]
    t = normalized_values[5] / 100.0
    radians[5] = SIM_ACTION_LOW[5] + t * (SIM_ACTION_HIGH[5] - SIM_ACTION_LOW[5])
    return radians


def load_config():
    """Load config.json for COM port settings."""
    repo_root = Path(__file__).parent.parent
    config_paths = [
        repo_root / "configs" / "config.json",
        Path("config.json"),
    ]
    for path in config_paths:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


def create_leader_bus(port: str):
    """Create motor bus for leader arm with STS3250 motors."""
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    bus = FeetechMotorsBus(
        port=port,
        motors={
            "shoulder_pan": Motor(1, "sts3250", MotorNormMode.RANGE_M100_100),
            "shoulder_lift": Motor(2, "sts3250", MotorNormMode.RANGE_M100_100),
            "elbow_flex": Motor(3, "sts3250", MotorNormMode.RANGE_M100_100),
            "wrist_flex": Motor(4, "sts3250", MotorNormMode.RANGE_M100_100),
            "wrist_roll": Motor(5, "sts3250", MotorNormMode.RANGE_M100_100),
            "gripper": Motor(6, "sts3250", MotorNormMode.RANGE_0_100),
        },
    )
    return bus


def load_calibration(arm_id: str = "leader_so100"):
    """Load calibration from JSON file."""
    import draccus
    from lerobot.motors import MotorCalibration
    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION

    calib_path = HF_LEROBOT_CALIBRATION / "teleoperators" / "so100_leader_sts3250" / f"{arm_id}.json"

    if not calib_path.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {calib_path}\n"
            f"Run 'python calibrate_from_zero.py' first."
        )

    with open(calib_path) as f, draccus.config_type("json"):
        return draccus.load(dict[str, MotorCalibration], f)


class MuJoCoVRViewer:
    def __init__(self, teleop_enabled=False, leader_port=None):
        print(f"Loading MuJoCo scene: {SCENE_XML}")
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)

        # Joint positions
        self.joint_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
        self.data.ctrl[:] = self.joint_pos

        # Step simulation to initialize
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

        # MuJoCo rendering objects (will be created after OpenGL context)
        self.mj_scene = None
        self.mj_context = None
        self.mj_camera = mujoco.MjvCamera()
        self.mj_option = mujoco.MjvOption()
        # Offset between XR space (right/up/-forward) and MuJoCo scene origin to place the robot under the user.
        self.stage_offset_xr = xr.Vector3f(-0.3, -0.8, -0.3)

        # Default camera setup
        self.mj_camera.lookat[:] = [0.1, 0.0, 0.1]
        self.mj_camera.distance = 0.6
        self.mj_camera.azimuth = -120
        self.mj_camera.elevation = -20

        # OpenXR state
        self.instance = None
        self.system_id = None
        self.session = None
        self.swapchains = []
        self.swapchain_images = []
        self.frame_buffers = []
        self.projection_views = []
        self.running = True

        # Teleop state
        self.teleop_enabled = teleop_enabled
        self.leader_port = leader_port
        self.leader_bus = None
        self.step_count = 0

        # Async teleop - read motors in background thread
        self.teleop_thread = None
        self.teleop_running = False
        self.latest_joint_radians = None
        self.latest_joint_timestamp = 0.0  # When the reading was taken
        self.joint_radians_lock = threading.Lock()

        # Timing stats
        self.frame_times = []
        self.motor_read_times = []
        self.last_frame_time = time.perf_counter()
        self.last_stats_time = time.perf_counter()

        # Scene position offset (can be adjusted at runtime)
        # For STAGE (roomscale): Y=0 is floor, standing head ~1.2-1.6m
        # Robot is at Z≈0.1m. We want standing eye level at Z≈0.5m
        self.base_pos = np.array([0.4, 0.3, -0.9], dtype=np.float64)
        self.scene_yaw = 0.0  # Rotation around Z axis in degrees

        # Store last head position for recentering
        self.last_head_pos_xr = None
        self.last_head_yaw_xr = None

    def init_teleop(self):
        """Initialize leader arm for teleoperation."""
        if not self.teleop_enabled:
            return

        print(f"\nConnecting to leader arm on {self.leader_port}...")
        self.leader_bus = create_leader_bus(self.leader_port)
        self.leader_bus.connect()

        # Load calibration
        print("Loading calibration...")
        self.leader_bus.calibration = load_calibration("leader_so100")
        self.leader_bus.disable_torque()
        print("Leader arm connected! Move it to control the simulation.")

        # Start background thread for reading motor positions
        self.teleop_running = True
        self.teleop_thread = threading.Thread(target=self._teleop_read_loop, daemon=True)
        self.teleop_thread.start()
        print("Teleop read thread started.\n")

    def _teleop_read_loop(self):
        """Background thread that continuously reads motor positions."""
        consecutive_errors = 0
        max_errors = 5
        target_hz = 200  # Limit read rate to reduce bus saturation
        min_interval = 1.0 / target_hz

        while self.teleop_running and self.leader_bus is not None:
            loop_start = time.perf_counter()
            try:
                read_start = time.perf_counter()
                positions = self.leader_bus.sync_read("Present_Position")
                read_end = time.perf_counter()

                consecutive_errors = 0  # Reset on success

                # Track read time
                read_duration_ms = (read_end - read_start) * 1000
                with self.joint_radians_lock:
                    self.motor_read_times.append(read_duration_ms)
                    if len(self.motor_read_times) > 100:
                        self.motor_read_times.pop(0)

                normalized = np.array([
                    positions["shoulder_pan"],
                    positions["shoulder_lift"],
                    positions["elbow_flex"],
                    positions["wrist_flex"],
                    positions["wrist_roll"],
                    positions["gripper"],
                ], dtype=np.float32)

                # Convert to radians and clip
                joint_radians = normalized_to_radians(normalized)
                joint_radians = np.clip(joint_radians, SIM_ACTION_LOW, SIM_ACTION_HIGH)

                # Store for main thread to use
                with self.joint_radians_lock:
                    self.latest_joint_radians = joint_radians
                    self.latest_joint_timestamp = read_end

            except Exception as e:
                consecutive_errors += 1
                print(f"Teleop read error ({consecutive_errors}/{max_errors}): {e}")
                if consecutive_errors >= max_errors:
                    print("Too many consecutive errors, stopping teleop thread")
                    break
                time.sleep(0.05)  # Brief pause before retry
                continue

            # Rate limit to avoid saturating the serial bus
            elapsed = time.perf_counter() - loop_start
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

    def get_latest_joint_radians(self):
        """Get the latest joint positions and timestamp (non-blocking)."""
        with self.joint_radians_lock:
            return self.latest_joint_radians, self.latest_joint_timestamp

    def cleanup_teleop(self):
        """Disconnect leader arm and stop thread."""
        self.teleop_running = False

        if self.teleop_thread is not None:
            self.teleop_thread.join(timeout=1.0)
            self.teleop_thread = None

        if self.leader_bus is not None:
            print("Disconnecting leader arm...")
            self.leader_bus.disconnect()
            self.leader_bus = None

    def recenter_scene(self):
        """Move the scene so the robot is directly in front of current head position."""
        if self.last_head_pos_xr is None:
            print("Cannot recenter - no head position yet")
            return

        # Convert current head position to MuJoCo coordinates
        head_mj = self.xr_to_mj(self.last_head_pos_xr)

        # Set base_pos so robot appears at a comfortable distance in front
        # Robot is at MuJoCo origin [0, 0, 0.1]
        # We want it ~0.5m in front of us at table height
        robot_pos = np.array([0.1, 0.0, 0.1])
        desired_offset = np.array([0.5, 0.0, 0.0])  # 0.5m in front

        # New base_pos places robot in front of head
        self.base_pos = robot_pos - head_mj - desired_offset

        print(f"Scene recentered! base_pos: [{self.base_pos[0]:.2f}, {self.base_pos[1]:.2f}, {self.base_pos[2]:.2f}]")

    def move_scene(self, dx=0.0, dy=0.0, dz=0.0):
        """Move the scene by the given offsets (in MuJoCo coordinates)."""
        self.base_pos[0] += dx  # Forward/back
        self.base_pos[1] += dy  # Left/right
        self.base_pos[2] += dz  # Up/down

    def handle_keyboard(self):
        """Handle keyboard input for scene controls."""
        # Movement amount per key press
        move_step = 0.05  # 5cm

        # Check for key presses (GLFW key states)
        # WASD for horizontal movement, QE for up/down, R for recenter
        if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
            self.move_scene(dx=move_step)  # Move scene forward (robot appears closer)
        if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS:
            self.move_scene(dx=-move_step)  # Move scene back
        if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS:
            self.move_scene(dy=move_step)  # Move scene left
        if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS:
            self.move_scene(dy=-move_step)  # Move scene right
        if glfw.get_key(self.window, glfw.KEY_Q) == glfw.PRESS:
            self.move_scene(dz=move_step)  # Move scene up
        if glfw.get_key(self.window, glfw.KEY_E) == glfw.PRESS:
            self.move_scene(dz=-move_step)  # Move scene down

        # R to recenter (with debounce)
        if glfw.get_key(self.window, glfw.KEY_R) == glfw.PRESS:
            if not hasattr(self, '_r_pressed'):
                self._r_pressed = False
            if not self._r_pressed:
                self.recenter_scene()
                self._r_pressed = True
        else:
            self._r_pressed = False

        # P to print current position
        if glfw.get_key(self.window, glfw.KEY_P) == glfw.PRESS:
            if not hasattr(self, '_p_pressed'):
                self._p_pressed = False
            if not self._p_pressed:
                print(f"base_pos: [{self.base_pos[0]:.2f}, {self.base_pos[1]:.2f}, {self.base_pos[2]:.2f}]")
                self._p_pressed = True
        else:
            self._p_pressed = False

    def init_openxr(self):
        """Initialize OpenXR instance, system, and session."""
        print("Initializing OpenXR...")

        # Request OpenGL extension
        extensions = [xr.KHR_OPENGL_ENABLE_EXTENSION_NAME]

        # Create instance (pyopenxr uses simplified API)
        create_info = xr.InstanceCreateInfo(
            enabled_extension_names=extensions,
        )

        self.instance = xr.create_instance(create_info)
        instance_props = xr.get_instance_properties(self.instance)
        print(f"OpenXR Runtime: {instance_props.runtime_name}")

        # Get system (HMD)
        system_get_info = xr.SystemGetInfo(
            form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY,
        )
        self.system_id = xr.get_system(self.instance, system_get_info)

        # Get view configuration
        view_configs = xr.enumerate_view_configurations(self.instance, self.system_id)
        print(f"View configurations: {view_configs}")

        self.view_config_type = xr.ViewConfigurationType.PRIMARY_STEREO
        view_config_views = xr.enumerate_view_configuration_views(
            self.instance, self.system_id, self.view_config_type
        )
        print(f"View config views: {len(view_config_views)}")
        for i, view in enumerate(view_config_views):
            print(f"  Eye {i}: {view.recommended_image_rect_width}x{view.recommended_image_rect_height}")

        self.view_config_views = view_config_views

    def init_glfw(self):
        """Initialize GLFW window for OpenGL context."""
        print("Initializing GLFW...")

        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        # OpenGL 3.3 Compatibility Profile (MuJoCo needs compatibility)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)
        glfw.window_hint(glfw.DOUBLEBUFFER, True)
        glfw.window_hint(glfw.VISIBLE, True)  # Visible window for MuJoCo

        self.window = glfw.create_window(800, 600, "MuJoCo VR", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        print(f"OpenGL Version: {GL.glGetString(GL.GL_VERSION).decode()}")

    def init_session(self):
        """Create OpenXR session with OpenGL graphics binding."""
        print("Creating OpenXR session...")

        # Get OpenGL requirements
        pfn_get_opengl_graphics_requirements = ctypes.cast(
            xr.get_instance_proc_addr(
                self.instance,
                "xrGetOpenGLGraphicsRequirementsKHR"
            ),
            xr.PFN_xrGetOpenGLGraphicsRequirementsKHR
        )

        graphics_requirements = xr.GraphicsRequirementsOpenGLKHR()
        result = pfn_get_opengl_graphics_requirements(
            self.instance,
            self.system_id,
            ctypes.byref(graphics_requirements)
        )

        # Create graphics binding (Windows-specific)
        import platform
        if platform.system() == "Windows":
            from OpenGL import WGL
            graphics_binding = xr.GraphicsBindingOpenGLWin32KHR(
                h_dc=WGL.wglGetCurrentDC(),
                h_glrc=WGL.wglGetCurrentContext(),
            )
        else:
            raise RuntimeError("Only Windows is currently supported")

        # Create session
        session_create_info = xr.SessionCreateInfo(
            system_id=self.system_id,
            next=ctypes.cast(ctypes.pointer(graphics_binding), ctypes.c_void_p),
        )

        self.session = xr.create_session(self.instance, session_create_info)
        print("Session created!")

    def init_swapchains(self):
        """Create swapchains for each eye."""
        print("Creating swapchains...")

        # Query swapchain formats
        swapchain_formats = xr.enumerate_swapchain_formats(self.session)

        # Prefer SRGB format
        chosen_format = GL.GL_SRGB8_ALPHA8
        if chosen_format not in swapchain_formats:
            chosen_format = swapchain_formats[0]

        self.swapchains = []
        self.swapchain_images = []

        for i, view_config in enumerate(self.view_config_views):
            width = view_config.recommended_image_rect_width
            height = view_config.recommended_image_rect_height

            swapchain_create_info = xr.SwapchainCreateInfo(
                usage_flags=xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT | xr.SwapchainUsageFlags.SAMPLED_BIT,
                format=chosen_format,
                sample_count=1,
                width=width,
                height=height,
                face_count=1,
                array_size=1,
                mip_count=1,
            )

            swapchain = xr.create_swapchain(self.session, swapchain_create_info)
            self.swapchains.append(swapchain)

            # Get swapchain images
            images = xr.enumerate_swapchain_images(swapchain, xr.SwapchainImageOpenGLKHR)
            self.swapchain_images.append(images)

            print(f"  Eye {i}: {width}x{height}, {len(images)} images")

        # Create framebuffers
        self.create_framebuffers()

    def create_framebuffers(self):
        """Create OpenGL framebuffers for rendering."""
        self.frame_buffers = []

        for eye_idx, (swapchain, images) in enumerate(zip(self.swapchains, self.swapchain_images)):
            view_config = self.view_config_views[eye_idx]
            width = view_config.recommended_image_rect_width
            height = view_config.recommended_image_rect_height

            eye_fbs = []
            for img in images:
                # Create framebuffer
                fbo = GL.glGenFramebuffers(1)
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)

                # Attach swapchain image as color attachment
                GL.glFramebufferTexture2D(
                    GL.GL_FRAMEBUFFER,
                    GL.GL_COLOR_ATTACHMENT0,
                    GL.GL_TEXTURE_2D,
                    img.image,
                    0
                )

                # Create depth buffer
                depth_buffer = GL.glGenRenderbuffers(1)
                GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, depth_buffer)
                GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH24_STENCIL8, width, height)
                GL.glFramebufferRenderbuffer(
                    GL.GL_FRAMEBUFFER,
                    GL.GL_DEPTH_STENCIL_ATTACHMENT,
                    GL.GL_RENDERBUFFER,
                    depth_buffer
                )

                status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
                if status != GL.GL_FRAMEBUFFER_COMPLETE:
                    print(f"Warning: Framebuffer incomplete: {status}")

                eye_fbs.append((fbo, depth_buffer))

            self.frame_buffers.append(eye_fbs)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def init_mujoco_rendering(self):
        """Initialize MuJoCo rendering context."""
        print("Initializing MuJoCo rendering...")

        self.mj_scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.mj_context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

    def quat_rotate(self, quat, v):
        """Rotate vector v by OpenXR quaternion (x,y,z,w)."""
        x, y, z, w = quat.x, quat.y, quat.z, quat.w
        qv = np.array([x, y, z], dtype=np.float64)
        v = np.array(v, dtype=np.float64)
        t = 2.0 * np.cross(qv, v)
        return v + w * t + np.cross(qv, t)

    def xr_to_mj(self, v):
        """Convert vector from OpenXR (+X right, +Y up, -Z forward) to MuJoCo (+X forward, +Y left, +Z up)."""
        v = np.array(v, dtype=np.float64)
        return np.array([-v[2], -v[0], v[1]], dtype=np.float64)

    def render_eye(self, eye_idx, view, fbo, width, height):
        """Render scene for one eye using the OpenXR eye pose + asymmetric FOV."""
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glViewport(0, 0, width, height)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClearColor(0.2, 0.3, 0.4, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        # OpenXR provides a per-eye pose (already includes IPD offset) and per-eye FOV.
        pos = view.pose.position
        quat = view.pose.orientation

        # Store head position for recentering (left eye only)
        if eye_idx == 0:
            self.last_head_pos_xr = np.array([pos.x, pos.y, pos.z])
            # Compute head yaw from forward vector
            fwd_temp = self.quat_rotate(quat, [0.0, 0.0, -1.0])
            self.last_head_yaw_xr = np.degrees(np.arctan2(-fwd_temp[0], -fwd_temp[2]))

        # Eye position in MuJoCo coordinates (using adjustable base_pos)
        xr_pos = np.array([pos.x, pos.y, pos.z])
        eye_pos_mj = self.base_pos + self.xr_to_mj(xr_pos)

        # Forward, up, and right vectors from OpenXR quaternion
        fwd_xr = self.quat_rotate(quat, [0.0, 0.0, -1.0])   # OpenXR forward is -Z
        up_xr  = self.quat_rotate(quat, [0.0, 1.0,  0.0])   # OpenXR up is +Y
        right_xr = self.quat_rotate(quat, [1.0, 0.0, 0.0])  # OpenXR right is +X

        # Convert to MuJoCo coordinates
        fwd_mj = self.xr_to_mj(fwd_xr)
        right_mj = self.xr_to_mj(right_xr)
        fwd_mj = fwd_mj / (np.linalg.norm(fwd_mj) + 1e-12)
        right_mj = right_mj / (np.linalg.norm(right_mj) + 1e-12)

        # Compute up from cross product to ensure correct handedness
        up_mj = np.cross(right_mj, fwd_mj)
        up_mj = up_mj / (np.linalg.norm(up_mj) + 1e-12)

        # Use a temporary MjvCamera for mjv_updateScene (it populates the scene geoms)
        # Must match fusion_fix.py approach for consistent behavior
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE

        # Choose a focus distance (metres). 1.0 tends to be comfortable.
        dist = 1.0
        cam.lookat[:] = eye_pos_mj + fwd_mj * dist
        cam.distance = dist

        # Derive azimuth/elevation from forward direction (MuJoCo uses azimuth about +Z; +Y is left)
        cam.azimuth = np.degrees(np.arctan2(fwd_mj[1], fwd_mj[0]))
        # elevation: positive = looking up, negative = looking down
        # fwd_mj[2] > 0 means looking up in MuJoCo (+Z is up)
        cam.elevation = np.degrees(np.arctan2(fwd_mj[2], np.sqrt(fwd_mj[0]**2 + fwd_mj[1]**2)))

        mujoco.mjv_updateScene(
            self.model, self.data, self.mj_option, None,
            cam, mujoco.mjtCatBit.mjCAT_ALL, self.mj_scene
        )

        # Override the actual render camera pose (mjv_updateScene may overwrite it)
        self.mj_scene.camera[0].pos[:] = eye_pos_mj
        self.mj_scene.camera[0].forward[:] = fwd_mj
        self.mj_scene.camera[0].up[:] = up_mj

        # Set an *asymmetric* frustum from OpenXR FOV.
        # OpenXR fov angles are radians about the view axis.
        fov = view.fov
        near = 0.05
        far = 50.0

        left   = near * np.tan(fov.angle_left)   # typically negative
        right  = near * np.tan(fov.angle_right)  # typically positive
        bottom = near * np.tan(fov.angle_down)   # typically negative
        top    = near * np.tan(fov.angle_up)     # typically positive

        # MuJoCo camera frustum params encode left/right via center+width.
        center = (left + right) / 2.0
        half_width = (right - left) / 2.0

        self.mj_scene.camera[0].frustum_near = near
        self.mj_scene.camera[0].frustum_far = far
        self.mj_scene.camera[0].frustum_center = center
        self.mj_scene.camera[0].frustum_width = half_width
        self.mj_scene.camera[0].frustum_bottom = bottom
        self.mj_scene.camera[0].frustum_top = top

        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjr_render(viewport, self.mj_scene, self.mj_context)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def run(self):
        """Main VR render loop."""
        print("\n" + "="*60)
        print("MuJoCo VR Viewer - OpenXR")
        if self.teleop_enabled:
            print("WITH LEADER ARM TELEOPERATION")
        print("="*60)
        print("Make sure Quest Link is active!")
        if self.teleop_enabled:
            print("Move leader arm to control the simulation")
        print("\nScene Controls (focus desktop window):")
        print("  WASD  - Move scene forward/back/left/right")
        print("  Q/E   - Move scene up/down")
        print("  R     - Recenter robot in front of you")
        print("  P     - Print current position")
        print("\nPress Ctrl+C to exit")
        print("="*60 + "\n")

        try:
            self.init_glfw()
            self.init_openxr()
            self.init_session()
            self.init_swapchains()
            self.init_mujoco_rendering()
            self.init_teleop()

            # Begin session
            session_begin_info = xr.SessionBeginInfo(
                primary_view_configuration_type=self.view_config_type
            )
            xr.begin_session(self.session, session_begin_info)
            print("Session started!")

                        # Reference space
            # Prefer room-/floor-scale reference spaces so standing up / walking works as expected.
            # Fallback to LOCAL if STAGE/LOCAL_FLOOR are not supported by the runtime.
            try:
                supported_spaces = xr.enumerate_reference_spaces(self.session)
            except Exception:
                supported_spaces = []

            preferred_space = None
            for t in (
                xr.ReferenceSpaceType.STAGE,
                getattr(xr.ReferenceSpaceType, "LOCAL_FLOOR", None),
                xr.ReferenceSpaceType.LOCAL,
            ):
                if t is None:
                    continue
                if supported_spaces and t in supported_spaces:
                    preferred_space = t
                    break
                # If we couldn't enumerate, just pick STAGE first and let create fail/fallback below.
                if not supported_spaces and preferred_space is None:
                    preferred_space = t

            def _try_create_space(space_type):
                info = xr.ReferenceSpaceCreateInfo(
                    reference_space_type=space_type,
                    pose_in_reference_space=xr.Posef(
                        orientation=xr.Quaternionf(0, 0, 0, 1),
                        position=xr.Vector3f(0, 0, 0),  # No offset - handled by base_pos in render
                    ),
                )
                return xr.create_reference_space(self.session, info)

            self.reference_space = None
            for candidate in [preferred_space, xr.ReferenceSpaceType.LOCAL]:
                if candidate is None:
                    continue
                try:
                    self.reference_space = _try_create_space(candidate)
                    print(f"Using reference space: {candidate}")
                    break
                except Exception as e:
                    print(f"Failed to create reference space {candidate}: {e}")

            if self.reference_space is None:
                raise RuntimeError("Could not create any OpenXR reference space (STAGE/LOCAL_FLOOR/LOCAL).")

            # Main loop
            while self.running:
                glfw.poll_events()
                self.handle_keyboard()

                if glfw.window_should_close(self.window):
                    break

                # Poll OpenXR events
                self.poll_events()

                if not self.running:
                    break

                # Wait for frame
                frame_state = xr.wait_frame(self.session)
                xr.begin_frame(self.session)

                # Get view poses
                view_locate_info = xr.ViewLocateInfo(
                    view_configuration_type=self.view_config_type,
                    display_time=frame_state.predicted_display_time,
                    space=self.reference_space,
                )

                view_state, views = xr.locate_views(self.session, view_locate_info)

                # Render each eye
                projection_views = []

                for eye_idx, (view, swapchain, images, fbs) in enumerate(
                    zip(views, self.swapchains, self.swapchain_images, self.frame_buffers)
                ):
                    view_config = self.view_config_views[eye_idx]
                    width = view_config.recommended_image_rect_width
                    height = view_config.recommended_image_rect_height

                    # Acquire swapchain image
                    swapchain_image_index = xr.acquire_swapchain_image(swapchain)

                    wait_info = xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION)
                    xr.wait_swapchain_image(swapchain, wait_info)

                    # Get framebuffer for this image
                    fbo, _ = fbs[swapchain_image_index]

                    # Render
                    self.render_eye(eye_idx, view, fbo, width, height)

                    # Release swapchain image
                    xr.release_swapchain_image(swapchain)

                    # Build projection view
                    projection_view = xr.CompositionLayerProjectionView(
                        pose=view.pose,
                        fov=view.fov,
                        sub_image=xr.SwapchainSubImage(
                            swapchain=swapchain,
                            image_rect=xr.Rect2Di(
                                offset=xr.Offset2Di(0, 0),
                                extent=xr.Extent2Di(width, height),
                            ),
                        ),
                    )
                    projection_views.append(projection_view)

                # Track frame timing
                now = time.perf_counter()
                frame_time_ms = (now - self.last_frame_time) * 1000
                self.last_frame_time = now
                self.frame_times.append(frame_time_ms)
                if len(self.frame_times) > 100:
                    self.frame_times.pop(0)

                # Read from leader arm if teleop enabled (non-blocking)
                data_age_ms = 0.0
                if self.teleop_enabled:
                    joint_radians, timestamp = self.get_latest_joint_radians()
                    if joint_radians is not None:
                        # Direct qpos setting for instant response
                        self.data.qpos[:6] = joint_radians
                        data_age_ms = (now - timestamp) * 1000
                    # Use mj_forward (kinematics only) - no physics stepping
                    mujoco.mj_forward(self.model, self.data)
                else:
                    # Full physics simulation when not in teleop mode
                    mujoco.mj_step(self.model, self.data)
                self.step_count += 1

                # Print timing stats every 2 seconds
                if now - self.last_stats_time >= 2.0:
                    self.last_stats_time = now

                    avg_frame = np.mean(self.frame_times) if self.frame_times else 0
                    fps = 1000.0 / avg_frame if avg_frame > 0 else 0

                    if self.teleop_enabled:
                        with self.joint_radians_lock:
                            avg_read = np.mean(self.motor_read_times) if self.motor_read_times else 0
                            motor_hz = 1000.0 / avg_read if avg_read > 0 else 0

                        print(f"VR: {fps:.1f} fps ({avg_frame:.1f}ms) | "
                              f"Motor read: {avg_read:.1f}ms ({motor_hz:.1f} Hz) | "
                              f"Data age: {data_age_ms:.1f}ms")
                    else:
                        print(f"VR: {fps:.1f} fps ({avg_frame:.1f}ms)")

                # Submit frame
                projection_layer = xr.CompositionLayerProjection(
                    space=self.reference_space,
                    views=projection_views,
                )

                layers = [ctypes.byref(projection_layer)]

                frame_end_info = xr.FrameEndInfo(
                    display_time=frame_state.predicted_display_time,
                    environment_blend_mode=xr.EnvironmentBlendMode.OPAQUE,
                    layers=layers,
                )

                xr.end_frame(self.session, frame_end_info)

        except KeyboardInterrupt:
            print("\nShutting down...")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def poll_events(self):
        """Poll and handle OpenXR events."""
        while True:
            try:
                # pyopenxr poll_event returns the event directly
                event = xr.poll_event(self.instance)

                if event is None:
                    break

                # Check event type
                if isinstance(event, xr.EventDataSessionStateChanged):
                    print(f"Session state changed: {event.state}")

                    if event.state == xr.SessionState.STOPPING:
                        xr.end_session(self.session)
                        self.running = False
                    elif event.state == xr.SessionState.EXITING:
                        self.running = False
                    elif event.state == xr.SessionState.LOSS_PENDING:
                        self.running = False

            except xr.exception.EventUnavailable:
                break
            except Exception:
                break

    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up...")

        # Disconnect leader arm first
        self.cleanup_teleop()

        if self.session:
            try:
                xr.destroy_session(self.session)
            except:
                pass

        if self.instance:
            try:
                xr.destroy_instance(self.instance)
            except:
                pass

        if hasattr(self, 'window') and self.window:
            glfw.destroy_window(self.window)
            glfw.terminate()


def main():
    parser = argparse.ArgumentParser(description="MuJoCo VR Viewer with optional teleoperation")
    parser.add_argument("--teleop", "-t", action="store_true",
                        help="Enable leader arm teleoperation")
    parser.add_argument("--port", "-p", type=str, default=None,
                        help="Serial port for leader arm (default: from config.json or COM8)")

    args = parser.parse_args()

    # Get leader port if teleop enabled
    leader_port = args.port
    if args.teleop and leader_port is None:
        config = load_config()
        if config and "leader" in config:
            leader_port = config["leader"]["port"]
            print(f"Using leader port from config: {leader_port}")
        else:
            leader_port = "COM8"
            print(f"Using default leader port: {leader_port}")

    viewer = MuJoCoVRViewer(
        teleop_enabled=args.teleop,
        leader_port=leader_port
    )
    viewer.run()


if __name__ == "__main__":
    main()
