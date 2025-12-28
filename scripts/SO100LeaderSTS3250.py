"""
SO100 Leader with STS3250 motors.
Registered as 'so100_leader_sts3250' for use with lerobot CLI tools.

Uses lerobot's standard file-based calibration (JSON files) instead of EEPROM.
Calibration files stored in: ~/.cache/huggingface/lerobot/calibration/teleoperators/
"""

from dataclasses import dataclass

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.teleoperators.so100_leader import SO100LeaderConfig, SO100Leader
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError


@TeleoperatorConfig.register_subclass("so100_leader_sts3250")
@dataclass
class SO100LeaderSTS3250Config(SO100LeaderConfig):
    """Config for SO100 Leader with STS3250 motors."""
    pass  # Inherits all fields from SO100LeaderConfig


class SO100LeaderSTS3250(SO100Leader):
    """SO100 Leader teleoperator with STS3250 motors.

    Uses lerobot's standard file-based calibration mechanism.
    Run calibration with:
        lerobot-calibrate --teleop.type=so100_leader_sts3250 --teleop.port=COM8 --teleop.id=leader
    """

    config_class = SO100LeaderSTS3250Config
    name = "so100_leader_sts3250"

    def __init__(self, config: SO100LeaderSTS3250Config):
        # Call grandparent init to set up base teleoperator properties
        # This loads calibration from JSON file if it exists
        super(SO100Leader, self).__init__(config)
        self.config = config

        # Use sts3250 motor model instead of sts3215
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3250", MotorNormMode.RANGE_M100_100),
                "shoulder_lift": Motor(2, "sts3250", MotorNormMode.RANGE_M100_100),
                "elbow_flex": Motor(3, "sts3250", MotorNormMode.RANGE_M100_100),
                "wrist_flex": Motor(4, "sts3250", MotorNormMode.RANGE_M100_100),
                "wrist_roll": Motor(5, "sts3250", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(6, "sts3250", MotorNormMode.RANGE_0_100),
            },
        )

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

        self.configure()

    def calibrate(self) -> None:
        """Run interactive calibration and save to JSON file."""
        print(f"\nCalibrating {self.name}...")
        print("Move arm to the middle position (all joints at center of range)")
        print("Press Enter when ready...")
        input()

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
        """Configure leader arm (torque disabled for manual movement)."""
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
