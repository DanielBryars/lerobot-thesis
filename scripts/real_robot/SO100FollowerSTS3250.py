"""
SO100 Follower with STS3250 motors.
Registered as 'so100_follower_sts3250' for use with lerobot CLI tools.

Uses lerobot's standard file-based calibration (JSON files) instead of EEPROM.
Calibration files stored in: ~/.cache/huggingface/lerobot/calibration/robots/
"""

import platform
from dataclasses import dataclass

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.robots.so100_follower import SO100FollowerConfig, SO100Follower
from lerobot.robots.config import RobotConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError


def make_cameras_with_dshow(camera_configs: dict[str, CameraConfig]) -> dict:
    """
    Create cameras, using DirectShow backend on Windows for OpenCV cameras.
    Some cameras don't work with MSMF but work fine with DirectShow.
    """
    import cv2
    from lerobot.cameras.camera import Camera
    from lerobot.cameras.opencv import OpenCVCamera

    cameras: dict[str, Camera] = {}
    for key, cfg in camera_configs.items():
        if cfg.type == "opencv" and platform.system() == "Windows":
            # Create camera and override backend to DirectShow
            cam = OpenCVCamera(cfg)
            cam.backend = cv2.CAP_DSHOW
            cameras[key] = cam
        else:
            # Use default camera creation for non-Windows or non-opencv
            cameras.update(make_cameras_from_configs({key: cfg}))
    return cameras


@RobotConfig.register_subclass("so100_follower_sts3250")
@dataclass
class SO100FollowerSTS3250Config(SO100FollowerConfig):
    """Config for SO100 Follower with STS3250 motors."""
    pass  # Inherits all fields from SO100FollowerConfig


class SO100FollowerSTS3250(SO100Follower):
    """SO100 Follower robot with STS3250 motors.

    Uses lerobot's standard file-based calibration mechanism.
    Run calibration with:
        lerobot-calibrate --robot.type=so100_follower_sts3250 --robot.port=COM7 --robot.id=follower
    """

    config_class = SO100FollowerSTS3250Config
    name = "so100_follower_sts3250"

    def __init__(self, config: SO100FollowerSTS3250Config):
        # Call grandparent init to set up base robot properties
        # This loads calibration from JSON file if it exists
        super(SO100Follower, self).__init__(config)
        self.config = config

        norm_mode = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

        # Use sts3250 motor model instead of sts3215
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3250", norm_mode),
                "shoulder_lift": Motor(2, "sts3250", norm_mode),
                "elbow_flex": Motor(3, "sts3250", norm_mode),
                "wrist_flex": Motor(4, "sts3250", norm_mode),
                "wrist_roll": Motor(5, "sts3250", norm_mode),
                "gripper": Motor(6, "sts3250", MotorNormMode.RANGE_0_100),
            },
        )
        self.cameras = make_cameras_with_dshow(config.cameras)

    @property
    def is_calibrated(self) -> bool:
        """Check if calibration file exists and matches motors."""
        # Check we have calibration data loaded from file
        if not self.calibration:
            return False
        # Check all motors have calibration
        return all(motor in self.calibration for motor in self.bus.motors)

    def connect(self, calibrate: bool = True) -> None:
        """Connect and apply calibration from JSON file."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()

        # Apply calibration from JSON file (loaded in __init__ by parent class)
        if self.calibration:
            self.bus.calibration = self.calibration
        elif calibrate:
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()

    def calibrate(self) -> None:
        """Run interactive calibration and save to JSON file."""
        print(f"\nCalibrating {self.name}...")
        print("Move arm to the middle position (all joints at center of range)")
        print("Press Enter when ready...")
        input()

        # Disable torque for manual positioning
        self.bus.disable_torque()

        # Set homing offsets so current position = middle
        self.bus.set_half_turn_homings()

        # Record range by moving to limits
        print("\nNow move each joint to its limits to record the range.")
        print("Press Enter when done...")
        input()

        mins, maxes = self.bus.record_ranges_of_motion()

        # Build calibration dict
        self.calibration = {}
        for motor in self.bus.motors:
            from lerobot.motors import MotorCalibration
            self.calibration[motor] = MotorCalibration(
                id=self.bus.motors[motor].id,
                drive_mode=0,
                homing_offset=0,  # Not using EEPROM offset
                range_min=mins[motor],
                range_max=maxes[motor],
            )

        # Save to JSON file
        self._save_calibration()
        self.bus.calibration = self.calibration
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        """Configure motors for position control."""
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
            self.bus.write("P_Coefficient", motor, 16)
            self.bus.write("I_Coefficient", motor, 0)
            self.bus.write("D_Coefficient", motor, 32)
            if motor == "gripper":
                self.bus.write("Max_Torque_Limit", motor, 500)
                self.bus.write("Protection_Current", motor, 250)
                self.bus.write("Overload_Torque", motor, 25)
