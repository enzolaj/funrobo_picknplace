"""
Utility functions and lightweight data containers for FunRobo Kinematics.

This module groups together small helpers that are used across the project:
- Simple dataclasses for state, controls, and time-series logs
- End-effector pose container
- Rotation and homogeneous transform helpers
- Joint-limit checks and IK validation helpers
- Small geometry helpers (distance/heading)
- Angle wrapping and near-zero cleanup utilities

Conventions:
- Angles are in radians unless explicitly stated.
- Homogeneous transforms are 4x4 matrices.
- Rotation matrices are 3x3 and assumed to be proper rotation matrices.

Typical student use:
- Use `EndEffector` to represent desired / computed end-effector poses.
- Use `check_joint_limits` and `sample_valid_joints` when testing IK/FK.
- Use `wraptopi` when comparing angles.
"""

from typing import List
from dataclasses import dataclass, field
from functools import singledispatch
import math
import random

import numpy as np
from math import sqrt, sin, cos, atan2


PI = 3.1415926535897932384

@dataclass
class State:
    """
    System state container for planar motion.

    This is intended for simple robotics / controls demos where the robot state
    includes pose and velocity in 2D.

    Attributes:
        x: x-position (meters).
        y: y-position (meters).
        theta: heading/orientation (radians).
        x_dot: x-velocity (m/s).
        y_dot: y-velocity (m/s).
        theta_dot: angular velocity (rad/s).
    """
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    x_dot: float = 0.0
    y_dot: float = 0.0
    theta_dot: float = 0.0


@dataclass
class Controls:
    """
    Control input container for planar motion.

    This supports two common representations:
    - (v, w): linear and angular velocity (unicycle model)
    - (vx, vy): x and y velocity components (holonomic representation)

    Attributes:
        v: forward linear velocity (m/s).
        w: angular velocity (rad/s).
        vx: x-component of linear velocity (m/s).
        vy: y-component of linear velocity (m/s).
    """
    v: float = 0.0
    w: float = 0.0
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class GamepadCmds:
    """
    Gamepad command container.

    This dataclass groups all command signals coming from a gamepad or joystick
    interface. It is intended as an intermediate representation between raw
    gamepad inputs (axes, buttons) and higher-level robot control logic.

    Commands are separated into:
    - Base motion commands (mobile base velocities)
    - Arm Cartesian motion commands
    - Arm joint and end-effector commands
    - Utility and mode-selection buttons

    All fields are typically integers coming directly from the gamepad
    (e.g., axis values or button states) and are expected to be scaled,
    filtered, or mapped before being sent to actuators.

    Attributes:
        base_vx: Base x-direction velocity command (e.g., forward/backward).
        base_vy: Base y-direction velocity command (e.g., left/right).
        base_w: Base angular velocity command (yaw/turn rate).

        arm_vx: Arm Cartesian x-direction velocity command.
        arm_vy: Arm Cartesian y-direction velocity command.
        arm_vz: Arm Cartesian z-direction velocity command.

        arm_j1: Joint 1 command (e.g., shoulder rotation).
        arm_j2: Joint 2 command.
        arm_j3: Joint 3 command.
        arm_j4: Joint 4 command.
        arm_j5: Joint 5 command.

        arm_ee: End-effector command (e.g., gripper open/close).
        arm_home: Arm homing command (return to home pose).

        utility_btn: General-purpose or mode-selection button.
    """
    base_vx: int = 0
    base_vy: int = 0
    base_w: int = 0
    arm_vx: int = 0
    arm_vy: int = 0
    arm_vz: int = 0
    arm_j1: int = 0
    arm_j2: int = 0
    arm_j3: int = 0
    arm_j4: int = 0
    arm_j5: int = 0
    arm_ee: int = 0
    arm_home: int = 0
    utility_btn: int = 0



def print_dataclass(obj):
    """
    Pretty-print the fields and values of a dataclass instance.

    This utility function prints all fields of a dataclass in a readable,
    labeled format. It is intended for quick debugging, logging, and
    instructional use (e.g., inspecting robot state or control inputs).

    Numerical values are rounded to three decimal places for readability.

    Args:
        obj: A dataclass instance whose fields will be printed.

    Raises:
        AttributeError: If the provided object is not a dataclass.
    """
    print("------------------------------------")
    for field in obj.__dataclass_fields__:
        print(f"{field}: {round(getattr(obj, field), 3)}")
    print("------------------------------------ \n")


class EndEffector:
    """
    Minimal end-effector pose container.

    This class is used throughout the kinematics code to represent a robot end-effector
    pose (position + orientation).

    Attributes:
        x: x-position (meters).
        y: y-position (meters).
        z: z-position (meters).
        rotx: roll angle about x-axis (radians).
        roty: pitch angle about y-axis (radians).
        rotz: yaw angle about z-axis (radians).
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    rotx: float = 0.0
    roty: float = 0.0
    rotz: float = 0.0


def rotm_to_euler(R: np.ndarray) -> tuple:
    """
    Convert a rotation matrix to Euler angles (roll, pitch, yaw).

    This function assumes the rotation matrix uses the common Z-Y-X convention
    (yaw-pitch-roll composition). The implementation also includes handling for
    near-singular configurations (gimbal lock), where multiple Euler solutions exist.

    Args:
        R: 3x3 rotation matrix.

    Returns:
        A tuple (roll, pitch, yaw) in radians.

    Notes:
        - If `r31` is close to ±1, pitch is near ±90° and the solution is not unique.
        - This function chooses a reasonable representative solution in those cases.
    """
    r11 = R[0,0] if abs(R[0,0]) > 1e-7 else 0.0
    r12 = R[0,1] if abs(R[0,1]) > 1e-7 else 0.0
    r21 = R[1,0] if abs(R[1,0]) > 1e-7 else 0.0
    r22 = R[1,1] if abs(R[1,1]) > 1e-7 else 0.0
    r32 = R[2,1] if abs(R[2,1]) > 1e-7 else 0.0
    r33 = R[2,2] if abs(R[2,2]) > 1e-7 else 0.0
    r31 = R[2,0] if abs(R[2,0]) > 1e-7 else 0.0

    if abs(r31) != 1:
        roll = math.atan2(r32, r33)        
        yaw = math.atan2(r21, r11)
        denom = math.sqrt(r11 ** 2 + r21 ** 2)
        pitch = math.atan2(-r31, denom)
    
    elif r31 == 1:
        # pitch is close to -90 deg, i.e. cos(pitch) = 0.0
        # there are an infinitely many solutions, so we choose one possible solution where yaw = 0
        pitch, yaw = -PI/2, 0.0
        roll = -math.atan2(r12, r22)
    
    elif r31 == -1:
        # pitch is close to 90 deg, i.e. cos(pitch) = 0.0
        # there are an infinitely many solutions, so we choose one possible solution where yaw = 0
        pitch, yaw = PI/2, 0.0
        roll = math.atan2(r12, r22)

    return roll, pitch, yaw


def dh_to_matrix(dh_params: list) -> np.ndarray:
    """
    Convert Denavit–Hartenberg (DH) parameters to a homogeneous transform using the classic
    DH convention.

    Reference: https://en.wikipedia.org/wiki/Denavit%E2%80%93Hartenberg_parameters

    Args:
        dh_params: DH parameters [theta, d, a, alpha], where:
            - theta: joint angle (rad)
            - d: link offset along previous z (m)
            - a: link length along current x (m)
            - alpha: link twist about current x (rad)

    Returns:
        4x4 homogeneous transformation matrix.

    Notes:
        This is the "standard" DH transform convention.
    """
    theta, d, a, alpha = dh_params
    return np.array([
        [cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta)],
        [sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta)],
        [0, sin(alpha), cos(alpha), d],
        [0, 0, 0, 1]
    ])


def euler_to_rotm(rpy: tuple) -> np.ndarray:
    """
    Convert Euler angles (roll, pitch, yaw) to a rotation matrix.

    This uses Z-Y-X composition (yaw then pitch then roll):
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    Args:
        rpy: (roll, pitch, yaw) in radians.

    Returns:
        3x3 rotation matrix.
    """
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(rpy[0]), -math.sin(rpy[0])],
                    [0, math.sin(rpy[0]), math.cos(rpy[0])]])
    R_y = np.array([[math.cos(rpy[1]), 0, math.sin(rpy[1])],
                    [0, 1, 0],
                    [-math.sin(rpy[1]), 0, math.cos(rpy[1])]])
    R_z = np.array([[math.cos(rpy[2]), -math.sin(rpy[2]), 0],
                    [math.sin(rpy[2]), math.cos(rpy[2]), 0],
                    [0, 0, 1]])
    return R_z @ R_y @ R_x


@dataclass
class SimData:
    """Captures simulation data for storage.

    Attributes:
        x (List[float]): x-coordinates over time.
        y (List[float]): y-coordinates over time.
        theta (List[float]): Angles over time.
        x_dot (List[float]): x-velocity over time.
        y_dot (List[float]): y-velocity over time.
        theta_dot (List[float]): Angular velocity over time.
        v (List[float]): Linear velocity over time.
        w (List[float]): Angular velocity over time.
        vx (List[float]): x-component of linear velocity over time.
        vy (List[float]): y-component of linear velocity over time.
    """
    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    theta: List[float] = field(default_factory=list)
    x_dot: List[float] = field(default_factory=list)
    y_dot: List[float] = field(default_factory=list)
    theta_dot: List[float] = field(default_factory=list)
    v: List[float] = field(default_factory=list)
    w: List[float] = field(default_factory=list)
    vx: List[float] = field(default_factory=list)
    vy: List[float] = field(default_factory=list)


def check_joint_limits(theta: List[float], theta_limits: List[List[float]]) -> bool:
    """Checks if the joint angles are within the specified limits.

    Args:
        theta (List[float]): Current joint angles.
        theta_limits (List[List[float]]): Joint limits for each joint.

    Returns:
        bool: True if all joint angles are within limits, False otherwise.
    """
    for i, th in enumerate(theta):
        if not (theta_limits[i][0] <= th <= theta_limits[i][1]):
            return False
    return True


def check_valid_ik_soln(
    joint_values: List[float],
    actual_ee: EndEffector,
    robot_model=None,
    tol: float = 0.002,
) -> bool:
    """
    Validate an IK solution by checking joint limits and forward-kinematics error.

    This helper is useful when:
    - You compute candidate IK solutions
    - You want to filter out solutions that violate joint limits
    - You want to check that FK(q) matches the desired end-effector pose

    Args:
        joint_values: Candidate joint configuration (radians/meters depending on robot).
        actual_ee: Desired end-effector pose (ground truth target).
        robot_model: Robot model instance providing:
            - joint_limits
            - calc_forward_kinematics(...)
        tol: Maximum allowed Euclidean position error (meters). Defaults to 0.002.

    Returns:
        True if:
            - joint values are within joint limits, and
            - FK(q) is within `tol` of `actual_ee` in position.
        False otherwise.

    Notes:
        This checks only (x, y, z) position error, not orientation error.
    """
    if robot_model is None:
        raise ValueError("robot_model must be provided to validate an IK solution.")

    if not check_joint_limits(joint_values, robot_model.joint_limits):
        return False
    
    calc_ee, _ = robot_model.calc_forward_kinematics(joint_values, radians=True)

    # calculate the EE position error
    e = [0, 0, 0]
    e[0] = calc_ee.x - actual_ee.x
    e[1] = calc_ee.y - actual_ee.y
    e[2] = calc_ee.z - actual_ee.z

    return True if np.linalg.norm(e) < tol else False


def sample_valid_joints(robot, n_tries: int = 1000) -> List[float]:
    """
    Sample a random joint configuration that satisfies the robot's joint limits.

    This is useful for generating random test cases or sanity checks.

    Args:
        robot: Robot model instance that provides:
            - num_dof (int): number of joints
            - joint_limits (list[list[float]]): joint limits in radians/meters
        n_tries: Maximum number of random samples to attempt before failing.

    Returns:
        list[float]: A joint configuration `q` (length = robot.num_dof) that
        satisfies `ut.check_joint_limits(q, robot.joint_limits)`.

    Raises:
        RuntimeError: If no valid configuration is found after `n_tries` attempts.
    """
    for _ in range(n_tries):
        q = [random.uniform(-math.pi, math.pi) for _ in range(robot.num_dof)]
        if check_joint_limits(q, robot.joint_limits):
            return q
    raise RuntimeError("Could not sample valid joint values; check joint limits/ranges.")



def calc_distance(p1: State, p2: State) -> float:
    """Calculates the Euclidean distance between two states.

    Args:
        p1 (State): The first state.
        p2 (State): The second state.

    Returns:
        float: The Euclidean distance between p1 and p2.
    """
    return sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def calc_heading(p1: State, p2: State) -> float:
    """Calculates the heading (angle) between two states.

    Args:
        p1 (State): The first state.
        p2 (State): The second state.

    Returns:
        float: The heading angle in radians.
    """
    return atan2(p1.y - p2.y, p1.x - p2.x)


@singledispatch
def calc_angdiff(p1: State, p2: State) -> float:
    """Calculates the angular difference between two states.

    Args:
        p1 (State): The first state.
        p2 (State): The second state.

    Returns:
        float: The angular difference in radians.
    """
    d = p1.theta - p2.theta
    return math.fmod(d, 2 * math.pi)


@calc_angdiff.register
def _(th1: float, th2: float) -> float:
    """Calculates the angular difference between two angles.

    Args:
        th1 (float): The first angle.
        th2 (float): The second angle.

    Returns:
        float: The angular difference in radians.
    """
    return math.fmod(th1 - th2, 2 * math.pi)


def near_zero(arr: np.ndarray) -> np.ndarray:
    """
    Replace values close to zero with exact zeros.

    This can be useful when printing matrices (e.g., transforms) so that tiny
    floating point noise doesn't clutter results.

    Args:
        arr (np.ndarray): The input array.

    Returns:
        np.ndarray: An array with zeros where values are near zero, otherwise the original values.
    """
    tol = 1e-6
    return np.where(np.isclose(arr, 0, atol=tol), 0, arr)


def wraptopi(angle_rad):
  """
    Wrap an angle to the range [-pi, pi).

    Args:
        angle_rad: Angle in radians.

    Returns:
        Equivalent angle in radians in the interval [-pi, pi).
  """
  return (angle_rad + math.pi) % (2 * math.pi) - math.pi