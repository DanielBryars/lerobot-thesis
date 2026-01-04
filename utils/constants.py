"""Shared constants for SO100/SO101 robot simulation."""

import numpy as np

# Motor names in order (matches MuJoCo model)
MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Simulation action space bounds (radians)
# These define the joint limits for the SO100/SO101 robot
SIM_ACTION_LOW = np.array([
    -1.91986,  # shoulder_pan
    -1.74533,  # shoulder_lift
    -1.69,     # elbow_flex
    -1.65806,  # wrist_flex
    -2.74385,  # wrist_roll
    -0.17453,  # gripper (closed)
])

SIM_ACTION_HIGH = np.array([
    1.91986,   # shoulder_pan
    1.74533,   # shoulder_lift
    1.69,      # elbow_flex
    1.65806,   # wrist_flex
    2.84121,   # wrist_roll
    1.74533,   # gripper (open)
])

# Gripper index
GRIPPER_IDX = 5

# Number of joints (excluding gripper)
NUM_ARM_JOINTS = 5

# Number of total joints (including gripper)
NUM_JOINTS = 6

# IK default tolerance (5mm)
DEFAULT_IK_TOLERANCE = 5e-3

# IK max iterations
DEFAULT_IK_MAX_ITER = 100
