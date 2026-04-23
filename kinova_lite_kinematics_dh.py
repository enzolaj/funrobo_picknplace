from __future__ import annotations

import numpy as np

import utils as ut
from arm_models import KinovaRobotTemplate


class KinovaLiteDH(KinovaRobotTemplate):
    """6-DOF Kinova Gen3 Lite with classic DH parameters."""

    def __init__(self):
        super().__init__()
        self.num_dof = 6
        self.joint_limits = [
            [-2.687, 2.687],
            [-2.618, 2.618],
            [-2.618, 2.618],
            [-2.600, 2.600],
            [-2.530, 2.530],
            [-2.600, 2.600],
        ]


    def _compute_transforms(self, joint_values):
        """
        Helper to calculate cumulative transformation matrices (H_cumulative)
        and individual joint transforms (Hlist) based on the Kinova robot model.
        
        Args:
            joint_values (list/array): Processed joint angles in radians.
        """
        theta = joint_values
        pi = np.pi
        
        # DH parameters based on Standard DH Table (Table 1)
        # Format: [theta, d, a, alpha]
        DH = np.array([
            [theta[0] - pi/2, 0.24325,           0.0,  pi/2], # Joint 1
            [theta[1] + pi/2, 0.032077,          0.28, pi  ], # Joint 2
            [theta[2] + pi/2, 0.022077,          0.0,  pi/2], # Joint 3
            [theta[3] + pi/2, 0.245,             0.0,  pi/2], # Joint 4 
            [theta[4] + pi,   0.057,             0.0,  pi/2], # Joint 5 
            [theta[5] + pi/2, 0.235,             0.0,  0.0 ]  # Joint 6 
        ])

        Hlist = [ut.dh_to_matrix(dh) for dh in DH] # Compute transformation matrices for each joint

        # Compute cumulative transformations
        H_cumulative = [np.eye(4)]
        for H in Hlist:
            H_cumulative.append(H_cumulative[-1] @ H)
            
        return H_cumulative, Hlist


    def calc_forward_kinematics(self, joint_values: list, radians: bool = True):
        q = np.array(joint_values, copy=True, dtype=float)
        if not radians:
            q = np.deg2rad(q)

        H_cumulative, Hlist = self._compute_transforms(q)
        H_ee = H_cumulative[-1]

        ee = ut.EndEffector()
        ee.x, ee.y, ee.z = H_ee[0, 3], H_ee[1, 3], H_ee[2, 3]
        rpy = ut.rotm_to_euler(H_ee[:3, :3])
        ee.rotx, ee.roty, ee.rotz = rpy
        return ee, Hlist

    def jacobian(self, joint_values: list):
        q = np.asarray(joint_values, dtype=float)
        H_cumulative, _ = self._compute_transforms(q)
        p_ee = H_cumulative[-1][:3, 3]
        J = np.zeros((6, 6), dtype=float)
        for i in range(6):
            T = H_cumulative[i]
            z_axis = T[:3, 2]
            p_joint = T[:3, 3]
            J[:3, i] = np.cross(z_axis, (p_ee - p_joint))
            J[3:, i] = z_axis
        return J

    def calc_inverse_kinematics(self, target_ee, q_guess=None, tol=1e-4, ilimit=150):
        q = np.array(q_guess if q_guess is not None else [0.0]*6, dtype=float)
        lambda_sq = 0.01  # Damping factor for singularity robustness
        
        # Target pose extraction
        p_targ = np.array([target_ee.x, target_ee.y, target_ee.z])
        R_targ = ut.euler_to_rotm((target_ee.rotx, target_ee.roty, target_ee.rotz))

        for _ in range(ilimit):
            H_c, _ = self._compute_transforms(q)
            H_ee = H_c[-1]
            
            # Position Error
            dp = p_targ - H_ee[:3, 3]
            
            # Orientation Error (Skew symmetric matrix)
            R_curr = H_ee[:3, :3]
            R_err = R_targ @ R_curr.T
            do = 0.5 * np.array([
                R_err[2, 1] - R_err[1, 2],
                R_err[0, 2] - R_err[2, 0],
                R_err[1, 0] - R_err[0, 1]
            ])
            
            # Combine position and orientation erros
            error = np.hstack([dp, do])
            if np.linalg.norm(error) < tol:
                return [ut.wraptopi(val) for val in q]

            # Damped Least Squares Update
            J = self.jacobian(q)
            JJT = J @ J.T + lambda_sq * np.eye(6)
            dq = J.T @ np.linalg.solve(JJT, error)
            q += dq
            limits = np.array(self.joint_limits)
            q = np.clip(q, limits[:, 0], limits[:, 1])

        return q