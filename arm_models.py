"""
Arm model templates for the FunRobo kinematics library.

This module defines template classes for several common robot arm types:
- A planar 2-DOF robot arm
- A SCARA robot
- A 5-DOF articulated robot arm

Each class provides a common interface for:
- Forward kinematics
- Inverse kinematics (analytical or numerical)
- Velocity kinematics
- Computing joint and end-effector positions for visualization

"""

import math
import numpy as np
from typing import List, Tuple
import utils as ut



class BaseRobot():
    """
    Template model for a generic robotic arm. It serves as a parent class for all subsequent
    robot arm models.

    Attributes:
        joint_values (list[float]): Current joint angles [theta1, theta2, ...] in radians.
        joint_limits (list[list[float]]): Joint angle limits [[min, max], ...] in radians.
        joint_vel_limits (list[list[float]]): Joint velocity limits (rad/s).
        ee (EndEffector): End-effector pose container.
        num_dof (int): Number of degrees of freedom (default is 1).
        points (list[np.ndarray]): Homogeneous coordinates of joint positions.
    """


    def __init__(self) -> None:
        """
        Initialize the generic robot model with default geometry and joint limits.
        """
        self.joint_values = []
        self.joint_limits = []  # Joint limits
        self.joint_vel_limits = []

        self.ee = ut.EndEffector()  # The end-effector object
        self.num_dof = 1  # Number of degrees of freedom
        self.points = [None] * (self.num_dof + 1)  # List to store robot points


    def calc_forward_kinematics(
        self, joint_values: List[float], radians: bool = True
    ) -> Tuple[ut.EndEffector, List[np.ndarray]]:
        """
        Compute the forward kinematics of the robot.

        Given a set of joint angles, this method computes the pose of the
        end-effector and the corresponding homogeneous transformation matrices.

        Args:
            joint_values (list[float]): Joint angles [theta1, theta2].
            radians (bool, optional): If False, joint angles are assumed to be
                in degrees and will be converted to radians. Defaults to True.

        Returns:
            tuple:
                - EndEffector: End-effector pose container (position/orientation).
                - list[np.ndarray]: List of 4x4 individual link transforms (length = num_dof).

        """
        ee = ut.EndEffector()
        Hlist = [np.eye(4,4)] * self.num_dof
        return ee, Hlist


    def calc_inverse_kinematics(
        self, ee: ut.EndEffector, joint_values: List[float], soln: int = 0
    ) -> List[float]:
        """
        Compute an analytical inverse kinematics solution.

        Given a desired end-effector pose, this method computes a set of joint
        angles that achieve the pose, if a valid solution exists.

        Args:
            ee (EndEffector): Desired end-effector pose.
            joint_values (List[float]): Initial or previous joint angles, used
                for solution selection or continuity.
            soln (int, optional): Solution branch index (e.g., elbow-up vs
                elbow-down). Defaults to 0.

        Returns:
            list[float]: Joint angles [theta1, theta2] in radians that achieve
            the desired end-effector pose.
        """
        new_joint_values = joint_values.copy()
        return new_joint_values


    def calc_numerical_ik(
        self, ee: ut.EndEffector, joint_values: List[float], tol: float = 0.01, ilimit: int = 100
    ) -> List[float]:
        """
        Calculates numerical inverse kinematics (IK) based on input end effector coordinates.

        Args:
            ee (EndEffector): Desired end-effector pose.
            joint_values (list[float]): Initial guess for joint angles.
            tol (float, optional): Convergence tolerance on pose/position error. Defaults to 0.01.
            ilimit (int, optional): Maximum number of iterations. Defaults to 100.

        Returns:
            list[float]: Estimated joint angles in radians.
        """
        new_joint_values = joint_values.copy()
        return new_joint_values


    def calc_velocity_kinematics(
        self, joint_values: List[float], vel: List[float], dt: float = 0.02
    ) -> List[float]:
        """
        Update joint angles based on a desired end-effector velocity.

        This method maps a desired Cartesian end-effector velocity into joint
        space and integrates the result over a single time step.

        Args:
            joint_values (List[float]): Current joint angles in radians.
            vel (List[float]): Desired end-effector linear velocity [vx, vy].
            dt (float, optional): Integration time step in seconds.
                Defaults to 0.02.

        Returns:
            List[float]: Updated joint angles in radians after one time step.
        """
        new_joint_values = joint_values.copy()
        return new_joint_values


    def calc_robot_points(self, joint_values: List[float], Hlist: List[np.ndarray], radians: bool = True) -> None:
        """
        Compute joint and end-effector positions for visualization.

        This method chains a set of **individual link transformation matrices**
        to compute cumulative transforms and determine the positions of all
        joints and the end effector in the base frame.

        It updates internal state in-place:
            - `self.points`: base/joint/EE points (homogeneous coordinates)
            - `self.ee`: end-effector position and orientation
            - `self.H_ee`: cumulative base->EE transform
            - `self.EE_axes`: end-effector axis endpoints for visualization

        Args:
            joint_values (list[float]): Joint angles. Units depend on `radians`.
            H (np.ndarray | None): Array of individual 4x4 transforms with shape (num_dof, 4, 4),
                where H[i] is the transform contributed by joint i alone. If None, zeros are used.
            radians (bool, optional): If False, joint angles are assumed to be degrees and will be
                converted to radians. Defaults to True.

        Returns:
            None
        """
        return None


class TwoDOFRobotTemplate(BaseRobot):
    """
    Template model of a planar 2-degree-of-freedom (2-DOF) robotic arm.

    This class represents a simple planar arm with two revolute joints.

    The robot operates in the XY-plane with the base located at the origin.

    Attributes:
        l1 (float): Length of the first link (meters).
        l2 (float): Length of the second link (meters).
        joint_values (list[float]): Current joint angles [theta1, theta2] in radians.
        joint_limits (list[list[float]]): Joint angle limits [[min, max], ...] in radians.
        joint_vel_limits (list[list[float]]): Joint velocity limits (rad/s).
        ee (EndEffector): End-effector pose container.
        num_dof (int): Number of degrees of freedom (2).
        points (list[np.ndarray]): Homogeneous coordinates of joint positions.
    """


    def __init__(self) -> None:
        """
        Initialize the 2-DOF robot model with default geometry and joint limits.
        """
        super().__init__()

        self.l1 = 0.30  # Length of the first arm segment
        self.l2 = 0.25  # Length of the second arm segment

        self.joint_values = [0.0, 0.0]  # Joint angles (in radians)
        self.joint_limits = [[-math.pi, math.pi], [-math.pi + 0.261, math.pi - 0.261]]  # Joint limits
        self.joint_vel_limits = [[-math.pi, math.pi], [-math.pi, math.pi]]

        self.ee = ut.EndEffector()  # The end-effector object
        self.num_dof = 2  # Number of degrees of freedom
        self.points = [None] * (self.num_dof + 1)  # List to store robot points


    def calc_robot_points(self, joint_values: List[float], Hlist: List[np.ndarray], radians: bool = True) -> None:
        """
        Compute joint and end-effector positions for visualization.

        This method chains a set of **individual link transformation matrices**
        to compute cumulative transforms and determine the positions of all
        joints and the end effector in the base frame.

        It updates internal state in-place:
            - `self.points`: base/joint/EE points (homogeneous coordinates)
            - `self.ee`: end-effector position and orientation
            - `self.H_ee`: cumulative base->EE transform
            - `self.EE_axes`: end-effector axis endpoints for visualization

        Args:
            joint_values (list[float]): Joint angles. Units depend on `radians`.
            H (np.ndarray | None): Array of individual 4x4 transforms with shape (num_dof, 4, 4),
                where H[i] is the transform contributed by joint i alone. If None, zeros are used.
            radians (bool, optional): If False, joint angles are assumed to be degrees and will be
                converted to radians. Defaults to True.

        Returns:
            None
        """
        if not radians: # Convert degrees to radians if the input is in degrees
            joint_values = [np.deg2rad(theta) for theta in joint_values]

        self.joint_values = joint_values.copy()

        # Initialize points[0] to the base (origin)
        self.points[0] = np.array([0, 0, 0, 1])

        # Precompute cumulative transformations to avoid redundant calculations
        H_cumulative = [np.eye(4)]
        for i in range(self.num_dof):
            H_cumulative.append(H_cumulative[-1] @ Hlist[i])

        # Calculate the robot points by applying the cumulative transformations
        for i in range(1, len(self.points)):
            self.points[i] = H_cumulative[i] @ self.points[0]

        # Calculate EE position and rotation
        self.EE_axes = H_cumulative[-1] @ np.array([0.075, 0.075, 0.075, 1])  # End-effector axes
        self.H_ee = H_cumulative[-1]  # Final transformation matrix for EE

        # Set the end effector (EE) position
        self.ee.x, self.ee.y, self.ee.z = self.points[-1][:3]
        
        # Extract and assign the RPY (roll, pitch, yaw) from the rotation matrix
        rpy = ut.rotm_to_euler(self.H_ee[:3, :3])
        self.ee.rotx, self.ee.roty, self.ee.rotz = rpy[0], rpy[1], rpy[2]

        # Calculate the EE axes in space (in the base frame)
        self.EE_axes = np.array([self.H_ee[:3, i] * 0.075 + self.points[-1][:3] for i in range(3)])


class ScaraRobotTemplate(BaseRobot):
    """
    Template model of a SCARA (Selective Compliance Assembly Robot Arm) robot.

    Attributes:
        l1, l2, l3, l4, l5 (float): Geometric parameters (meters).
        joint_values (list[float]): Current joint variables.
        joint_limits (list[list[float]]): Joint limits for each joint.
        ee (EndEffector): End-effector pose container.
        num_dof (int): Number of degrees of freedom (3).
        points (list[np.ndarray | None]): Points describing robot configuration.
        EE_axes (np.ndarray | None): End-effector axis endpoints for visualization.
    """
    
    def __init__(self) -> None:
        """
        Initialize the SCARA robot model with default geometry, joint values, and limits.
        """
        super().__init__()

        # Geometry of the robot (link lengths in meters)
        self.l1 = 0.35  # Base to 1st joint
        self.l2 = 0.18  # 1st joint to 2nd joint
        self.l3 = 0.15  # 2nd joint to 3rd joint
        self.l4 = 0.30  # 3rd joint to 4th joint (tool or end-effector)
        self.l5 = 0.12  # Tool offset

        # Joint variables (angles in radians)
        self.joint_values = [0.0, 0.0, 0.0]

        # Joint angle limits (min, max) for each joint
        self.joint_limits = [
            [-np.pi, np.pi],
            [-np.pi + 0.261, np.pi - 0.261],
            [0, self.l1 + self.l3 - self.l5]
        ]

        # End-effector (EE) object to store EE position and orientation
        self.ee = ut.EndEffector()

        # Number of degrees of freedom and number of points to store robot configuration
        self.num_dof = 3
        self.num_points = 7
        self.points = [None] * self.num_points


    def calc_robot_points(self, joint_values: List[float], Hlist: List[np.ndarray], radians: bool = True) -> None:
        """
        Compute joint and end-effector positions for visualization.

        This method chains a set of **individual link transformation matrices**
        to compute cumulative transforms and determine the positions of all
        joints and the end effector in the base frame.

        It updates internal state in-place:
            - `self.points`: base/joint/EE points (homogeneous coordinates)
            - `self.ee`: end-effector position and orientation
            - `self.H_ee`: cumulative base->EE transform
            - `self.EE_axes`: end-effector axis endpoints for visualization

        Args:
            joint_values (list[float]): Joint angles. Units depend on `radians`.
            H (np.ndarray | None): Array of individual 4x4 transforms with shape (num_dof, 4, 4),
                where H[i] is the transform contributed by joint i alone. If None, zeros are used.
            radians (bool, optional): If False, joint angles are assumed to be degrees and will be
                converted to radians. Defaults to True.

        Returns:
            None
        """
        if not radians: # Convert degrees to radians if the input is in degrees
            joint_values = [np.deg2rad(theta) for theta in joint_values]

        self.joint_values = joint_values.copy()
            
        # Calculate transformation matrices for each joint and end-effector
        self.points[0] = np.array([0, 0, 0, 1])
        self.points[1] = np.array([0, 0, self.l1, 1])
        self.points[2] = Hlist[0]@self.points[0]
        self.points[3] = self.points[2] + np.array([0, 0, self.l3, 1])
        self.points[4] = Hlist[0]@Hlist[1]@self.points[0] + np.array([0, 0, self.l5, 1])
        self.points[5] = Hlist[0]@Hlist[1]@self.points[0]
        self.points[6] = Hlist[0]@Hlist[1]@Hlist[2]@self.points[0]

        self.EE_axes = Hlist[0]@Hlist[1]@Hlist[2]@np.array([0.075, 0.075, 0.075, 1])
        H_ee = Hlist[0]@Hlist[1]@Hlist[2]

        # End-effector (EE) position and axes
        self.ee.x = self.points[-1][0]
        self.ee.y = self.points[-1][1]
        self.ee.z = self.points[-1][2]
        rpy = ut.rotm_to_euler(H_ee[:3,:3])
        self.ee.rotx, self.ee.roty, self.ee.rotz = rpy
        
        # EE coordinate axes
        self.EE_axes = np.zeros((3, 3))
        self.EE_axes[0] = H_ee[:3,0] * 0.075 + self.points[-1][0:3]
        self.EE_axes[1] = H_ee[:3,1] * 0.075 + self.points[-1][0:3]
        self.EE_axes[2] = H_ee[:3,2] * 0.075 + self.points[-1][0:3]


class FiveDOFRobotTemplate(BaseRobot):
    """
    Template model of a 5-DOF articulated robot arm.

    Attributes:
        l1, l2, l3, l4, l5 (float): Link lengths (meters).
        joint_values (list[float]): Current joint angles (radians).
        joint_limits (list[list[float]]): Joint angle limits (radians).
        joint_vel_limits (list[list[float]]): Joint velocity limits (rad/s).
        ee (EndEffector): End-effector pose container.
        num_dof (int): Number of degrees of freedom (5).
        points (list[np.ndarray | None]): Joint positions (base + joints + EE).
        EE_axes (np.ndarray | None): End-effector axis endpoints for visualization.
        H_ee (np.ndarray | None): Cumulative end-effector transform (base -> EE).
    """

    
    def __init__(self) -> None:
        """
        Initialize the 5-DOF robot model with default geometry and joint limits.
        """
        super().__init__()

        # Link lengths
        self.l1, self.l2, self.l3, self.l4, self.l5 = 0.155, 0.099, 0.095, 0.055, 0.105 # from hardware measurements
        
        self.joint_values = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Joint limits (in radians)
        self.joint_limits = [
            [-np.pi, np.pi],
            [-np.pi / 3, np.pi],
            [-np.pi + np.pi / 12, np.pi - np.pi / 4],
            [-np.pi + np.pi / 12, np.pi - np.pi / 12],
            [-np.pi, np.pi],
        ]

        self.joint_vel_limits = [
            [-np.pi * 2, np.pi * 2],
            [-np.pi * 2, np.pi * 2],
            [-np.pi * 2, np.pi * 2],
            [-np.pi * 2, np.pi * 2],
            [-np.pi * 2, np.pi * 2],
        ]
        
        self.ee = ut.EndEffector()
        self.num_dof = 5
        self.points = [None] * (self.num_dof + 1)


    def calc_robot_points(
            self, joint_values: List[float], H: List[np.ndarray], radians: bool = True
    ) -> None:
        """ 
        Compute joint and end-effector positions for visualization.

        This method chains a set of **individual link transforms** to compute cumulative
        transforms and determine the positions of all joints and the end effector in the
        base frame.

        Args:
            joint_values (list[float]): Joint angles. Units depend on `radians`.
            H (np.ndarray | None): Array of individual 4x4 transforms with shape (num_dof, 4, 4).
                If None, zero matrices are used.
            radians (bool, optional): If False, joint angles are assumed to be degrees and will be
                converted to radians. Defaults to True.

        Returns:
            None: This method updates internal state. 
        """

        if not radians: # Convert degrees to radians if the input is in degrees
            joint_values = [np.deg2rad(theta) for theta in joint_values]

        self.joint_values = joint_values.copy()

        if H is None:
            H = np.zeros((self.num_dof, 4, 4))

        # Initialize points[0] to the base (origin)
        self.points[0] = np.array([0, 0, 0, 1])

        # Precompute cumulative transformations to avoid redundant calculations
        H_cumulative = [np.eye(4)]
        for i in range(self.num_dof):
            H_cumulative.append(H_cumulative[-1] @ H[i])

        # Calculate the robot points by applying the cumulative transformations
        for i in range(1, len(self.points)):
            self.points[i] = H_cumulative[i] @ self.points[0]

        # Calculate EE position and rotation
        self.EE_axes = H_cumulative[-1] @ np.array([0.075, 0.075, 0.075, 1])  # End-effector axes
        self.H_ee = H_cumulative[-1]  # Final transformation matrix for EE

        # Set the end effector (EE) position
        self.ee.x, self.ee.y, self.ee.z = self.points[-1][:3]
        
        # Extract and assign the RPY (roll, pitch, yaw) from the rotation matrix
        rpy = ut.rotm_to_euler(self.H_ee[:3, :3])
        self.ee.rotx, self.ee.roty, self.ee.rotz = rpy[0], rpy[1], rpy[2]

        # Calculate the EE axes in space (in the base frame)
        self.EE_axes = np.array([self.H_ee[:3, i] * 0.075 + self.points[-1][:3] for i in range(3)])


class KinovaRobotTemplate(BaseRobot):
    """
    Template model of the 6-DOF Kinova robot arm.

    Attributes:
        l1, l2, l3, l4, l5, l6 (float): Link lengths (meters).
        joint_values (list[float]): Current joint angles (radians).
        joint_limits (list[list[float]]): Joint angle limits (radians).
        joint_vel_limits (list[list[float]]): Joint velocity limits (rad/s).
        ee (EndEffector): End-effector pose container.
        num_dof (int): Number of degrees of freedom (5).
        points (list[np.ndarray | None]): Joint positions (base + joints + EE).
        EE_axes (np.ndarray | None): End-effector axis endpoints for visualization.
        H_ee (np.ndarray | None): Cumulative end-effector transform (base -> EE).
    """

    
    def __init__(self) -> None:
        """
        Initialize the 6-DOF robot model with default geometry and joint limits.
        """
        super().__init__()

        # Link lengths
        self.l1, self.l2, self.l3, self.l4, self.l5, self.l6, self.l7 = 0.156, 0.128, 0.410, 0.208, 0.105, 0.105, 0.0615
        
        self.joint_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Joint limits (in radians)
        self.joint_limits = [
            [-2 * np.pi, 2 * np.pi],
            [-0.71 * np.pi, 0.71 * np.pi],
            [-0.82 * np.pi, 0.82 * np.pi],
            [-2 * np.pi, 2 * np.pi],
            [-0.66 * np.pi, 0.66 * np.pi],
            [-2 * np.pi, 2 * np.pi],
        ]

        self.joint_vel_limits = [
            [-np.pi * 2, np.pi * 2],
            [-np.pi * 2, np.pi * 2],
            [-np.pi * 2, np.pi * 2],
            [-np.pi * 2, np.pi * 2],
            [-np.pi * 2, np.pi * 2],
        ]
        
        self.ee = ut.EndEffector()
        self.num_dof = 6
        self.points = [None] * (self.num_dof + 2)


    def calc_robot_points(
            self, joint_values: List[float], Hlist: List[np.ndarray], radians: bool = True
    ) -> None:
        """ 
        Compute joint and end-effector positions for visualization.

        This method chains a set of **individual link transforms** to compute cumulative
        transforms and determine the positions of all joints and the end effector in the
        base frame.

        Args:
            joint_values (list[float]): Joint angles. Units depend on `radians`.
            Hlist (np.ndarray | None): Array of individual 4x4 transforms with shape (num_dof, 4, 4).
                If None, zero matrices are used.
            radians (bool, optional): If False, joint angles are assumed to be degrees and will be
                converted to radians. Defaults to True.

        Returns:
            None: This method updates internal state. 
        """

        if not radians: # Convert degrees to radians if the input is in degrees
            joint_values = [np.deg2rad(theta) for theta in joint_values]

        self.joint_values = joint_values.copy()

        # Initialize points[0] to the base (origin)
        self.points[0] = np.array([0, 0, 0, 1])

        # Precompute cumulative transformations to avoid redundant calculations
        H_cumulative = [np.eye(4)]
        for H in Hlist:
            H_cumulative.append(H_cumulative[-1] @ H)

        # Calculate the robot points by applying the cumulative transformations
        for i in range(1, len(self.points)):
            self.points[i] = H_cumulative[i] @ self.points[0]

        # Calculate EE position and rotation
        self.EE_axes = H_cumulative[-1] @ np.array([0.075, 0.075, 0.075, 1])  # End-effector axes
        self.H_ee = H_cumulative[-1]  # Final transformation matrix for EE

        # Set the end effector (EE) position
        self.ee.x, self.ee.y, self.ee.z = self.points[-1][:3]
        
        # Extract and assign the RPY (roll, pitch, yaw) from the rotation matrix
        rpy = ut.rotm_to_euler(self.H_ee[:3, :3])
        self.ee.rotx, self.ee.roty, self.ee.rotz = rpy[0], rpy[1], rpy[2]

        # Calculate the EE axes in space (in the base frame)
        self.EE_axes = np.array([self.H_ee[:3, i] * 0.075 + self.points[-1][:3] for i in range(3)])