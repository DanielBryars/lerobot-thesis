#!/usr/bin/env python
"""
Teleoperate the SO101 simulation using a physical arm with VR display.

Uses the gym environment for physics (like teleop_sim.py) but renders to VR headset.
This can be integrated with lerobot record for dataset collection.

Usage:
    python teleop_sim_vr.py                    # Use leader arm (default)
    python teleop_sim_vr.py --port COM8
"""
import argparse
import ctypes
import json
import time
from pathlib import Path

import numpy as np
import mujoco

try:
    import xr
    import xr.exception
except ImportError:
    print("ERROR: pyopenxr not installed. Run: pip install pyopenxr")
    exit(1)

try:
    import msvcrt
    _msvcrt_available = True
except ImportError:
    _msvcrt_available = False

try:
    import glfw
    from OpenGL import GL
except ImportError:
    print("ERROR: OpenGL dependencies missing. Run: pip install glfw PyOpenGL")
    exit(1)


# Sim action space bounds (radians)
SIM_ACTION_LOW = np.array([-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.17453])
SIM_ACTION_HIGH = np.array([1.91986, 1.74533, 1.69, 1.65806, 2.84121, 1.74533])


def normalized_to_radians(normalized_values: np.ndarray) -> np.ndarray:
    """Convert from lerobot normalized values to sim radians."""
    radians = np.zeros(6, dtype=np.float32)
    for i in range(5):
        t = (normalized_values[i] + 100) / 200.0
        radians[i] = SIM_ACTION_LOW[i] + t * (SIM_ACTION_HIGH[i] - SIM_ACTION_LOW[i])
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
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    with open(calib_path) as f, draccus.config_type("json"):
        return draccus.load(dict[str, MotorCalibration], f)


class VRRenderer:
    """OpenXR VR renderer that displays a MuJoCo simulation."""

    def __init__(self, mj_model, mj_data):
        self.model = mj_model
        self.data = mj_data

        # MuJoCo rendering
        self.mj_scene = None
        self.mj_context = None
        self.mj_option = mujoco.MjvOption()

        # OpenXR state
        self.instance = None
        self.session = None
        self.swapchains = []
        self.swapchain_images = []
        self.frame_buffers = []
        self.reference_space = None
        self.running = True

        # Controller input
        self.action_set = None
        self.thumbstick_x_action = None
        self.thumbstick_y_action = None
        self.recenter_action = None
        self.hand_paths = []

        # Session state tracking
        self.session_state = None
        self.is_focused = False

        # Scene positioning (MuJoCo coords: X=forward, Y=left, Z=up)
        # Z=0.5 puts camera ~0.5m above the floor (table height)
        self.base_pos = np.array([0.4, 0.3, 0.5], dtype=np.float64)
        self.view_yaw = 0.0  # Rotation around Z axis (radians)
        self.last_head_pos = None
        self.last_head_fwd = None  # Head forward direction in XR coords

    def init_all(self):
        """Initialize GLFW, OpenXR, and MuJoCo rendering."""
        self._init_glfw()
        self._init_openxr()
        self._init_session()
        self._init_actions()
        self._init_swapchains()
        self._init_mujoco_rendering()
        self._init_reference_space()

    def _init_glfw(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)
        glfw.window_hint(glfw.DOUBLEBUFFER, True)
        glfw.window_hint(glfw.VISIBLE, True)

        self.window = glfw.create_window(800, 600, "VR Teleop", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)

    def _init_openxr(self):
        extensions = [xr.KHR_OPENGL_ENABLE_EXTENSION_NAME]
        create_info = xr.InstanceCreateInfo(enabled_extension_names=extensions)
        self.instance = xr.create_instance(create_info)

        system_get_info = xr.SystemGetInfo(form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY)
        self.system_id = xr.get_system(self.instance, system_get_info)

        self.view_config_type = xr.ViewConfigurationType.PRIMARY_STEREO
        self.view_config_views = xr.enumerate_view_configuration_views(
            self.instance, self.system_id, self.view_config_type
        )

    def _init_session(self):
        import sys
        if sys.platform != "win32":
            raise RuntimeError(f"VR renderer only supports Windows (current platform: {sys.platform})")

        # Get OpenGL requirements
        pfn = ctypes.cast(
            xr.get_instance_proc_addr(self.instance, "xrGetOpenGLGraphicsRequirementsKHR"),
            xr.PFN_xrGetOpenGLGraphicsRequirementsKHR
        )
        graphics_requirements = xr.GraphicsRequirementsOpenGLKHR()
        pfn(self.instance, self.system_id, ctypes.byref(graphics_requirements))

        # Create graphics binding (Windows only)
        from OpenGL import WGL
        graphics_binding = xr.GraphicsBindingOpenGLWin32KHR(
            h_dc=WGL.wglGetCurrentDC(),
            h_glrc=WGL.wglGetCurrentContext(),
        )

        session_create_info = xr.SessionCreateInfo(
            system_id=self.system_id,
            next=ctypes.cast(ctypes.pointer(graphics_binding), ctypes.c_void_p),
        )
        self.session = xr.create_session(self.instance, session_create_info)

    def _init_actions(self):
        """Initialize OpenXR controller actions for scene movement."""
        # Create action set
        action_set_info = xr.ActionSetCreateInfo(
            action_set_name="scene_control",
            localized_action_set_name="Scene Control",
            priority=0,
        )
        self.action_set = xr.create_action_set(self.instance, action_set_info)

        # Get hand paths
        self.hand_paths = [
            xr.string_to_path(self.instance, "/user/hand/left"),
            xr.string_to_path(self.instance, "/user/hand/right"),
        ]

        # Create thumbstick X action (left/right movement)
        thumbstick_x_info = xr.ActionCreateInfo(
            action_name="thumbstick_x",
            action_type=xr.ActionType.FLOAT_INPUT,
            count_subaction_paths=len(self.hand_paths),
            subaction_paths=self.hand_paths,
            localized_action_name="Thumbstick X",
        )
        self.thumbstick_x_action = xr.create_action(self.action_set, thumbstick_x_info)

        # Create thumbstick Y action (forward/back movement)
        thumbstick_y_info = xr.ActionCreateInfo(
            action_name="thumbstick_y",
            action_type=xr.ActionType.FLOAT_INPUT,
            count_subaction_paths=len(self.hand_paths),
            subaction_paths=self.hand_paths,
            localized_action_name="Thumbstick Y",
        )
        self.thumbstick_y_action = xr.create_action(self.action_set, thumbstick_y_info)

        # Create recenter action (button press)
        recenter_info = xr.ActionCreateInfo(
            action_name="recenter",
            action_type=xr.ActionType.BOOLEAN_INPUT,
            count_subaction_paths=len(self.hand_paths),
            subaction_paths=self.hand_paths,
            localized_action_name="Recenter Scene",
        )
        self.recenter_action = xr.create_action(self.action_set, recenter_info)

        # Right thumbstick X (left/right strafe) - only for right hand
        right_x_info = xr.ActionCreateInfo(
            action_name="right_thumbstick_x",
            action_type=xr.ActionType.FLOAT_INPUT,
            count_subaction_paths=1,
            subaction_paths=[self.hand_paths[1]],  # Right hand only
            localized_action_name="Right Thumbstick X",
        )
        self.right_thumbstick_x_action = xr.create_action(self.action_set, right_x_info)

        # Right thumbstick Y (up/down) - only for right hand
        right_y_info = xr.ActionCreateInfo(
            action_name="right_thumbstick_y",
            action_type=xr.ActionType.FLOAT_INPUT,
            count_subaction_paths=1,
            subaction_paths=[self.hand_paths[1]],  # Right hand only
            localized_action_name="Right Thumbstick Y",
        )
        self.right_thumbstick_y_action = xr.create_action(self.action_set, right_y_info)

        # Suggest bindings for Oculus Touch controllers
        oculus_path = xr.string_to_path(self.instance, "/interaction_profiles/oculus/touch_controller")
        bindings = [
            # Left thumbstick: forward/back, left/right
            xr.ActionSuggestedBinding(
                self.thumbstick_x_action,
                xr.string_to_path(self.instance, "/user/hand/left/input/thumbstick/x"),
            ),
            xr.ActionSuggestedBinding(
                self.thumbstick_y_action,
                xr.string_to_path(self.instance, "/user/hand/left/input/thumbstick/y"),
            ),
            # Right thumbstick: strafe, up/down
            xr.ActionSuggestedBinding(
                self.right_thumbstick_x_action,
                xr.string_to_path(self.instance, "/user/hand/right/input/thumbstick/x"),
            ),
            xr.ActionSuggestedBinding(
                self.right_thumbstick_y_action,
                xr.string_to_path(self.instance, "/user/hand/right/input/thumbstick/y"),
            ),
            # X button (left controller): recenter
            xr.ActionSuggestedBinding(
                self.recenter_action,
                xr.string_to_path(self.instance, "/user/hand/left/input/x/click"),
            ),
        ]

        suggested_bindings = xr.InteractionProfileSuggestedBinding(
            interaction_profile=oculus_path,
            suggested_bindings=bindings,
        )
        try:
            xr.suggest_interaction_profile_bindings(self.instance, suggested_bindings)
        except Exception as e:
            print(f"Warning: Could not bind Oculus Touch profile: {e}")

        # Attach action set to session
        attach_info = xr.SessionActionSetsAttachInfo(action_sets=[self.action_set])
        xr.attach_session_action_sets(self.session, attach_info)

        print("Controller actions initialized (left stick = move, X = recenter)")

    def _init_swapchains(self):
        # Ensure our context is current before creating FBOs
        glfw.make_context_current(self.window)

        swapchain_formats = xr.enumerate_swapchain_formats(self.session)
        chosen_format = GL.GL_SRGB8_ALPHA8 if GL.GL_SRGB8_ALPHA8 in swapchain_formats else swapchain_formats[0]

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

            images = xr.enumerate_swapchain_images(swapchain, xr.SwapchainImageOpenGLKHR)
            self.swapchain_images.append(images)

            # Create framebuffers
            eye_fbs = []
            for img in images:
                fbo = int(GL.glGenFramebuffers(1))
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
                GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, img.image, 0)

                depth_buffer = int(GL.glGenRenderbuffers(1))
                GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, depth_buffer)
                GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH24_STENCIL8, width, height)
                GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_STENCIL_ATTACHMENT, GL.GL_RENDERBUFFER, depth_buffer)

                eye_fbs.append((fbo, depth_buffer))
            self.frame_buffers.append(eye_fbs)

    def _init_mujoco_rendering(self):
        self.mj_scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.mj_context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

    def _poll_session_events(self):
        """Poll and process OpenXR session events."""
        while True:
            try:
                event = xr.poll_event(self.instance)
                if event is None:
                    break
                if isinstance(event, xr.EventDataSessionStateChanged):
                    self.session_state = event.state
                    was_focused = self.is_focused
                    self.is_focused = (event.state == xr.SessionState.FOCUSED)
                    if self.is_focused and not was_focused:
                        print("Session FOCUSED - controllers now active")
                    elif not self.is_focused and was_focused:
                        print(f"Session lost focus (state: {event.state})")
            except xr.exception.EventUnavailable:
                break
            except:
                break

    def _init_reference_space(self):
        # Begin session
        session_begin_info = xr.SessionBeginInfo(primary_view_configuration_type=self.view_config_type)
        xr.begin_session(self.session, session_begin_info)

        # Poll events to catch early session state changes
        self._poll_session_events()

        # Use STAGE for roomscale
        try:
            supported_spaces = xr.enumerate_reference_spaces(self.session)
        except:
            supported_spaces = []

        for space_type in [xr.ReferenceSpaceType.STAGE, xr.ReferenceSpaceType.LOCAL]:
            try:
                info = xr.ReferenceSpaceCreateInfo(
                    reference_space_type=space_type,
                    pose_in_reference_space=xr.Posef(
                        orientation=xr.Quaternionf(0, 0, 0, 1),
                        position=xr.Vector3f(0, 0, 0),
                    ),
                )
                self.reference_space = xr.create_reference_space(self.session, info)
                print(f"Using reference space: {space_type}")
                break
            except:
                continue

        # Poll again after setup
        self._poll_session_events()
        print(f"Initial session state: {self.session_state}, focused: {self.is_focused}")

    def quat_rotate(self, quat, v):
        """Rotate vector by OpenXR quaternion (x,y,z,w)."""
        x, y, z, w = quat.x, quat.y, quat.z, quat.w
        qv = np.array([x, y, z], dtype=np.float64)
        v = np.array(v, dtype=np.float64)
        t = 2.0 * np.cross(qv, v)
        return v + w * t + np.cross(qv, t)

    def xr_to_mj(self, v):
        """OpenXR (+X right, +Y up, -Z forward) -> MuJoCo (+X forward, +Y left, +Z up)."""
        v = np.array(v, dtype=np.float64)
        return np.array([-v[2], -v[0], v[1]], dtype=np.float64)

    def rotate_z(self, v, angle):
        """Rotate vector v around Z axis by angle (radians)."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            v[0] * c - v[1] * s,
            v[0] * s + v[1] * c,
            v[2]
        ], dtype=np.float64)

    def render_eye(self, eye_idx, view, fbo, width, height):
        """Render scene for one eye."""
        # Ensure our context is current (gym env may have changed it)
        glfw.make_context_current(self.window)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glViewport(0, 0, width, height)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClearColor(0.2, 0.3, 0.4, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        pos = view.pose.position
        quat = view.pose.orientation

        xr_pos = np.array([pos.x, pos.y, pos.z])
        eye_pos_mj_raw = self.base_pos + self.xr_to_mj(xr_pos)

        fwd_xr = self.quat_rotate(quat, [0.0, 0.0, -1.0])
        right_xr = self.quat_rotate(quat, [1.0, 0.0, 0.0])

        fwd_mj_raw = self.xr_to_mj(fwd_xr)
        right_mj_raw = self.xr_to_mj(right_xr)

        # Apply view_yaw rotation around Z axis (orbit around robot at origin)
        eye_pos_mj = self.rotate_z(eye_pos_mj_raw, self.view_yaw)
        fwd_mj = self.rotate_z(fwd_mj_raw, self.view_yaw)
        right_mj = self.rotate_z(right_mj_raw, self.view_yaw)

        fwd_mj = fwd_mj / (np.linalg.norm(fwd_mj) + 1e-12)
        right_mj = right_mj / (np.linalg.norm(right_mj) + 1e-12)
        up_mj = np.cross(right_mj, fwd_mj)
        up_mj = up_mj / (np.linalg.norm(up_mj) + 1e-12)

        # Update scene
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        dist = 1.0
        cam.lookat[:] = eye_pos_mj + fwd_mj * dist
        cam.distance = dist
        cam.azimuth = np.degrees(np.arctan2(fwd_mj[1], fwd_mj[0]))
        cam.elevation = np.degrees(np.arctan2(fwd_mj[2], np.sqrt(fwd_mj[0]**2 + fwd_mj[1]**2)))

        mujoco.mjv_updateScene(self.model, self.data, self.mj_option, None, cam, mujoco.mjtCatBit.mjCAT_ALL, self.mj_scene)

        # Override camera
        self.mj_scene.camera[0].pos[:] = eye_pos_mj
        self.mj_scene.camera[0].forward[:] = fwd_mj
        self.mj_scene.camera[0].up[:] = up_mj

        # Asymmetric frustum
        fov = view.fov
        near, far = 0.05, 50.0
        left = near * np.tan(fov.angle_left)
        right = near * np.tan(fov.angle_right)
        bottom = near * np.tan(fov.angle_down)
        top = near * np.tan(fov.angle_up)

        self.mj_scene.camera[0].frustum_near = near
        self.mj_scene.camera[0].frustum_far = far
        self.mj_scene.camera[0].frustum_center = (left + right) / 2.0
        self.mj_scene.camera[0].frustum_width = (right - left) / 2.0
        self.mj_scene.camera[0].frustum_bottom = bottom
        self.mj_scene.camera[0].frustum_top = top

        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjr_render(viewport, self.mj_scene, self.mj_context)

    def _handle_controller_input(self, display_time):
        """Poll controller actions and update scene position."""
        if self.action_set is None:
            return

        # Try to sync actions - this will fail if session is not focused
        active_action_set = xr.ActiveActionSet(action_set=self.action_set, subaction_path=xr.NULL_PATH)
        sync_info = xr.ActionsSyncInfo(active_action_sets=[active_action_set])
        try:
            xr.sync_actions(self.session, sync_info)
            # Sync succeeded - session must be focused
            if not self.is_focused:
                self.is_focused = True
                print("Session FOCUSED - controllers now active (detected via sync)")
        except Exception as e:
            # Sync failed - session probably not focused, silently skip
            if self.is_focused:
                self.is_focused = False
                print(f"Controller sync failed (session lost focus?): {e}")
            return

        # Movement speed
        move_speed = 0.02  # meters per frame

        # Get thumbstick X (left/right in MuJoCo = Y axis)
        try:
            state_info = xr.ActionStateGetInfo(action=self.thumbstick_x_action, subaction_path=self.hand_paths[0])
            state = xr.get_action_state_float(self.session, state_info)
            if state.is_active and abs(state.current_state) > 0.1:
                self.base_pos[1] -= state.current_state * move_speed  # Left/right
        except Exception as e:
            if not hasattr(self, '_thumbstick_x_error_shown'):
                print(f"Warning: Left thumbstick X read failed: {e}")
                self._thumbstick_x_error_shown = True

        # Get thumbstick Y (forward/back in MuJoCo = X axis)
        try:
            state_info = xr.ActionStateGetInfo(action=self.thumbstick_y_action, subaction_path=self.hand_paths[0])
            state = xr.get_action_state_float(self.session, state_info)
            if state.is_active and abs(state.current_state) > 0.1:
                self.base_pos[0] += state.current_state * move_speed  # Forward/back
        except Exception as e:
            if not hasattr(self, '_thumbstick_y_error_shown'):
                print(f"Warning: Left thumbstick Y read failed: {e}")
                self._thumbstick_y_error_shown = True

        # Right thumbstick X: rotate view around robot (orbit)
        try:
            state_info = xr.ActionStateGetInfo(action=self.right_thumbstick_x_action, subaction_path=self.hand_paths[1])
            state = xr.get_action_state_float(self.session, state_info)
            if state.is_active and abs(state.current_state) > 0.1:
                rot_speed = 0.03  # radians per frame
                self.view_yaw += state.current_state * rot_speed
        except Exception as e:
            if not hasattr(self, '_right_x_error_shown'):
                print(f"Warning: Right thumbstick X read failed: {e}")
                self._right_x_error_shown = True

        # Right thumbstick Y (up/down in MuJoCo = Z axis)
        try:
            state_info = xr.ActionStateGetInfo(action=self.right_thumbstick_y_action, subaction_path=self.hand_paths[1])
            state = xr.get_action_state_float(self.session, state_info)
            if state.is_active and abs(state.current_state) > 0.1:
                self.base_pos[2] += state.current_state * move_speed  # Up/down
        except Exception as e:
            if not hasattr(self, '_right_y_error_shown'):
                print(f"Warning: Right thumbstick Y read failed: {e}")
                self._right_y_error_shown = True

        # Get recenter button (X on left controller)
        try:
            state_info = xr.ActionStateGetInfo(action=self.recenter_action, subaction_path=self.hand_paths[0])
            state = xr.get_action_state_boolean(self.session, state_info)
            if state.is_active and state.current_state and state.changed_since_last_sync:
                self._recenter_scene()
        except Exception as e:
            if not hasattr(self, '_recenter_error_shown'):
                print(f"Warning: Recenter button read failed: {e}")
                self._recenter_error_shown = True

    def _debug_controller_status(self):
        """Print controller status for debugging."""
        if self.action_set is None:
            print("Controller debug: action_set is None")
            return

        if not self.is_focused:
            print(f"Controller status: Waiting for focus (state: {self.session_state}) - put on headset")
            return

        try:
            # Try to get state of left thumbstick
            state_info = xr.ActionStateGetInfo(action=self.thumbstick_x_action, subaction_path=self.hand_paths[0])
            state = xr.get_action_state_float(self.session, state_info)
            left_active = state.is_active
            left_x = state.current_state if state.is_active else 0

            state_info = xr.ActionStateGetInfo(action=self.thumbstick_y_action, subaction_path=self.hand_paths[0])
            state = xr.get_action_state_float(self.session, state_info)
            left_y = state.current_state if state.is_active else 0

            state_info = xr.ActionStateGetInfo(action=self.right_thumbstick_x_action, subaction_path=self.hand_paths[1])
            state = xr.get_action_state_float(self.session, state_info)
            right_active = state.is_active
            right_x = state.current_state if state.is_active else 0

            state_info = xr.ActionStateGetInfo(action=self.right_thumbstick_y_action, subaction_path=self.hand_paths[1])
            state = xr.get_action_state_float(self.session, state_info)
            right_y = state.current_state if state.is_active else 0

            print(f"Controller status: Left={'active' if left_active else 'INACTIVE'}({left_x:.2f},{left_y:.2f}) "
                  f"Right={'active' if right_active else 'INACTIVE'}({right_x:.2f},{right_y:.2f}) "
                  f"base_pos=[{self.base_pos[0]:.2f},{self.base_pos[1]:.2f},{self.base_pos[2]:.2f}]")
        except Exception as e:
            print(f"Controller debug failed: {e}")

    def check_keyboard(self):
        """Check for keyboard input (spacebar to recenter)."""
        if not _msvcrt_available:
            return

        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b' ':  # Spacebar
                print("Spacebar pressed - recentering...")
                self._recenter_scene()
            elif key == b'r':  # R key as alternative
                print("R pressed - recentering...")
                self._recenter_scene()

    def _recenter_scene(self):
        """Move scene so robot is 30cm in front at waist height, aligned with user's facing direction."""
        if self.last_head_pos is None:
            return

        # Convert head position to MuJoCo coordinates
        head_mj = self.xr_to_mj(self.last_head_pos)

        # Compute yaw from head forward direction so robot aligns with user's facing
        if self.last_head_fwd is not None:
            fwd_mj = self.xr_to_mj(self.last_head_fwd)
            # Yaw angle: how much user is rotated from looking at +X (robot front)
            # We want to counter-rotate the scene so robot appears in front
            self.view_yaw = -np.arctan2(fwd_mj[1], fwd_mj[0])
        else:
            self.view_yaw = 0.0

        # Final eye position is: rotate_z(base_pos + head_mj, view_yaw)
        # We want this to equal desired_eye_pos
        # So: base_pos + head_mj = rotate_z(desired_eye_pos, -view_yaw)
        # Thus: base_pos = rotate_z(desired_eye_pos, -view_yaw) - head_mj
        desired_eye_pos = np.array([-0.2, 0.0, 0.55])  
        self.base_pos = self.rotate_z(desired_eye_pos, -self.view_yaw) - head_mj
        print(f"Scene recentered! base_pos: [{self.base_pos[0]:.2f}, {self.base_pos[1]:.2f}, {self.base_pos[2]:.2f}], yaw: {np.degrees(self.view_yaw):.1f}°")

    def render_frame(self):
        """Render one VR frame. Returns False if should exit."""
        # Ensure our GL context is current (may have been changed by other code)
        glfw.make_context_current(self.window)

        glfw.poll_events()
        if glfw.window_should_close(self.window):
            return False

        # Poll OpenXR events
        self._poll_session_events()
        if self.session_state in [xr.SessionState.STOPPING, xr.SessionState.EXITING, xr.SessionState.LOSS_PENDING]:
            return False

        # Wait for frame
        frame_state = xr.wait_frame(self.session)
        xr.begin_frame(self.session)

        # Get views
        view_locate_info = xr.ViewLocateInfo(
            view_configuration_type=self.view_config_type,
            display_time=frame_state.predicted_display_time,
            space=self.reference_space,
        )
        view_state, views = xr.locate_views(self.session, view_locate_info)

        # Store head position and forward direction for recentering (from left eye)
        if len(views) > 0:
            pos = views[0].pose.position
            quat = views[0].pose.orientation
            self.last_head_pos = np.array([pos.x, pos.y, pos.z])
            # Forward direction in XR is -Z
            self.last_head_fwd = self.quat_rotate(quat, [0.0, 0.0, -1.0])

        # Poll controller input
        self._handle_controller_input(frame_state.predicted_display_time)

        # Check keyboard fallback (spacebar to recenter)
        self.check_keyboard()

        # Debug: show controller status on first frame and periodically
        if not hasattr(self, '_frame_count'):
            self._frame_count = 0
            print("First VR frame - checking controller status...")
            self._debug_controller_status()
        self._frame_count += 1
        if self._frame_count % 300 == 0:  # Every ~10 seconds at 30fps
            self._debug_controller_status()

        # Render each eye
        projection_views = []
        for eye_idx, (view, swapchain, images, fbs) in enumerate(
            zip(views, self.swapchains, self.swapchain_images, self.frame_buffers)
        ):
            view_config = self.view_config_views[eye_idx]
            width = view_config.recommended_image_rect_width
            height = view_config.recommended_image_rect_height

            swapchain_image_index = xr.acquire_swapchain_image(swapchain)
            xr.wait_swapchain_image(swapchain, xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION))

            fbo, _ = fbs[swapchain_image_index]
            self.render_eye(eye_idx, view, fbo, width, height)

            xr.release_swapchain_image(swapchain)

            projection_views.append(xr.CompositionLayerProjectionView(
                pose=view.pose,
                fov=view.fov,
                sub_image=xr.SwapchainSubImage(
                    swapchain=swapchain,
                    image_rect=xr.Rect2Di(offset=xr.Offset2Di(0, 0), extent=xr.Extent2Di(width, height)),
                ),
            ))

        # Submit frame
        projection_layer = xr.CompositionLayerProjection(space=self.reference_space, views=projection_views)
        frame_end_info = xr.FrameEndInfo(
            display_time=frame_state.predicted_display_time,
            environment_blend_mode=xr.EnvironmentBlendMode.OPAQUE,
            layers=[ctypes.byref(projection_layer)],
        )
        xr.end_frame(self.session, frame_end_info)

        return True

    def cleanup(self):
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


def run_teleop_vr(port: str, fps: int = 30):
    """Run VR teleoperation with leader arm controlling sim."""

    print(f"Connecting to leader arm on {port}...")
    bus = create_leader_bus(port)
    bus.connect()

    print("Loading calibration...")
    bus.calibration = load_calibration("leader_so100")
    bus.disable_torque()
    print("Leader arm connected!")

    # Load MuJoCo model directly (not through gym to avoid OpenGL context conflicts)
    print("Loading MuJoCo model...")
    scene_xml = Path(__file__).parent.parent / "scenes" / "so101_with_wrist_cam.xml"
    mj_model = mujoco.MjModel.from_xml_path(str(scene_xml))
    mj_data = mujoco.MjData(mj_model)

    # Initialize simulation
    for _ in range(100):
        mujoco.mj_step(mj_model, mj_data)

    # Create VR renderer FIRST (before any other OpenGL contexts)
    print("Initializing VR...")
    vr = VRRenderer(mj_model, mj_data)
    vr.init_all()

    print("\n" + "="*50)
    print("VR Teleop Started!")
    print("Move leader arm to control the simulation")
    print("")
    print("Controller Controls:")
    print("  Left Thumbstick:  Forward/back (Y), Left/right (X)")
    print("  Right Thumbstick: Up/down (Y), Rotate view (X)")
    print("  X Button (left):  Recenter robot in front of you")
    print("")
    print("Keyboard Fallback (in console window):")
    print("  SPACEBAR or R:    Recenter robot in front of you")
    print("")
    print("Press Ctrl+C to exit")
    print("="*50 + "\n")

    frame_time = 1.0 / fps
    step_count = 0
    n_sim_steps = 10  # Same as gym env default
    last_normalized = np.zeros(6, dtype=np.float32)  # For recovery from disconnects
    consecutive_errors = 0
    max_consecutive_errors = 30  # Give up after ~1 second at 30fps

    try:
        while True:
            loop_start = time.time()

            # Read leader arm (with error recovery)
            try:
                positions = bus.sync_read("Present_Position")
                normalized = np.array([
                    positions["shoulder_pan"],
                    positions["shoulder_lift"],
                    positions["elbow_flex"],
                    positions["wrist_flex"],
                    positions["wrist_roll"],
                    positions["gripper"],
                ], dtype=np.float32)
                last_normalized = normalized.copy()
                if consecutive_errors > 0:
                    print(f"✅ Leader arm reconnected after {consecutive_errors} missed frames")
                consecutive_errors = 0
            except ConnectionError as e:
                consecutive_errors += 1
                if consecutive_errors == 1:
                    print(f"\n⚠️  Leader arm read failed, using last position...")
                if consecutive_errors >= max_consecutive_errors:
                    print(f"\n❌ Too many consecutive read errors ({consecutive_errors}), exiting.")
                    break
                normalized = last_normalized  # Use last known position

            joint_radians = normalized_to_radians(normalized)
            joint_radians = np.clip(joint_radians, SIM_ACTION_LOW, SIM_ACTION_HIGH)

            # Apply control and step physics
            mj_data.ctrl[:] = joint_radians
            for _ in range(n_sim_steps):
                mujoco.mj_step(mj_model, mj_data)
            step_count += 1

            # Render to VR
            if not vr.render_frame():
                break

            # Print status periodically with position debug info
            if step_count % 100 == 0:
                elapsed = time.time() - loop_start
                actual_qpos = mj_data.qpos[:6]
                print(f"Step: {step_count}, FPS: {1/elapsed:.1f}")
                print(f"  Normalized:  [{normalized[0]:6.1f}, {normalized[1]:6.1f}, {normalized[2]:6.1f}, {normalized[3]:6.1f}, {normalized[4]:6.1f}, {normalized[5]:5.1f}]")
                print(f"  Target rad:  [{joint_radians[0]:6.3f}, {joint_radians[1]:6.3f}, {joint_radians[2]:6.3f}, {joint_radians[3]:6.3f}, {joint_radians[4]:6.3f}, {joint_radians[5]:6.3f}]")
                print(f"  Actual qpos: [{actual_qpos[0]:6.3f}, {actual_qpos[1]:6.3f}, {actual_qpos[2]:6.3f}, {actual_qpos[3]:6.3f}, {actual_qpos[4]:6.3f}, {actual_qpos[5]:6.3f}]")
                print(f"  VR pos: [{vr.base_pos[0]:.2f}, {vr.base_pos[1]:.2f}, {vr.base_pos[2]:.2f}]")

            # Maintain frame rate
            elapsed = time.time() - loop_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        vr.cleanup()
        bus.disconnect()
        print("Done.")


def run_vr_test(fps: int = 30):
    """Run VR test mode without physical arm - just view the scene and test controllers."""

    # Load MuJoCo model
    print("Loading MuJoCo model...")
    scene_xml = Path(__file__).parent.parent / "scenes" / "so101_with_wrist_cam.xml"
    mj_model = mujoco.MjModel.from_xml_path(str(scene_xml))
    mj_data = mujoco.MjData(mj_model)

    # Initialize simulation to rest position
    for _ in range(100):
        mujoco.mj_step(mj_model, mj_data)

    # Create VR renderer
    print("Initializing VR...")
    vr = VRRenderer(mj_model, mj_data)
    vr.init_all()

    print("\n" + "="*50)
    print("VR Test Mode (no arm required)")
    print("")
    print("Controller Controls:")
    print("  Left Thumbstick:  Forward/back (Y), Left/right (X)")
    print("  Right Thumbstick: Up/down (Y), Rotate view (X)")
    print("  X Button (left):  Recenter robot in front of you")
    print("")
    print("Keyboard Fallback (in console window):")
    print("  SPACEBAR or R:    Recenter robot in front of you")
    print("")
    print("Press Ctrl+C to exit")
    print("="*50 + "\n")

    frame_time = 1.0 / fps
    step_count = 0

    try:
        while True:
            loop_start = time.time()
            step_count += 1

            # Render to VR
            if not vr.render_frame():
                break

            # Print status periodically
            if step_count % 100 == 0:
                elapsed = time.time() - loop_start
                print(f"Step: {step_count}, FPS: {1/elapsed:.1f}, VR pos: [{vr.base_pos[0]:.2f}, {vr.base_pos[1]:.2f}, {vr.base_pos[2]:.2f}]")

            # Maintain frame rate
            elapsed = time.time() - loop_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        vr.cleanup()
        print("Done.")


def main():
    parser = argparse.ArgumentParser(description="VR Teleop for SO101 sim")
    parser.add_argument("--port", "-p", type=str, default=None,
                        help="Serial port for leader arm (default: from config.json)")
    parser.add_argument("--fps", "-f", type=int, default=30,
                        help="Target frame rate (default: 30)")
    parser.add_argument("--test", "-t", action="store_true",
                        help="Test mode: VR only, no arm required")

    args = parser.parse_args()

    if args.test:
        run_vr_test(args.fps)
        return

    port = args.port
    if port is None:
        config = load_config()
        if config and "leader" in config:
            port = config["leader"]["port"]
            print(f"Using leader port from config: {port}")
        else:
            port = "COM8"
            print(f"Using default leader port: {port}")

    run_teleop_vr(port, args.fps)


if __name__ == "__main__":
    main()
