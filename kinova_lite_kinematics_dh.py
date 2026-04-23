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
        self.d1 = 0.24325
        self.a2 = 0.28
        self.a3 = 0.14
        self.d4 = 0.02
        self.d5 = 0.105
        self.d6 = 0.105

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
        # Using class attributes for offsets d and lengths a
        DH = np.array([
            [theta[0],           self.d1,            0.0,      pi/2],
            [theta[1] + pi/2,    30.0/1000.0,        0.28,     pi],
            [theta[2] + pi/2,    20.0/1000.0,        0.0,      pi/2],
            [theta[3] + pi/2,    (140.0+105.0)/1000, 0.0,      pi/2],
            [theta[4] + pi,      (28.5+28.5)/1000,   0.0,      pi/2],
            [theta[5] + pi/2,    (105.0+130.0)/1000, 0.0,      0.0]
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

    def calc_inverse_kinematics(self, target_ee, q_guess=None, tol=1e-4, ilimit=200):
        if q_guess is None:
            q_guess = np.array([1.75, 5.76, 2.18, 2.44, 4.54, 0.0])

        q = np.array(q_guess, dtype=float)
        lambda_sq = 0.01

        for _ in range(ilimit):
            curr_ee, _ = self.calc_forward_kinematics(q.tolist(), radians=True)

            dp = np.array(
                [
                    target_ee.x - curr_ee.x,
                    target_ee.y - curr_ee.y,
                    target_ee.z - curr_ee.z,
                ]
            )
            R_curr = ut.euler_to_rotm((curr_ee.rotx, curr_ee.roty, curr_ee.rotz))
            R_targ = ut.euler_to_rotm((target_ee.rotx, target_ee.roty, target_ee.rotz))
            R_err = R_targ @ R_curr.T
            do = 0.5 * np.array(
                [
                    R_err[2, 1] - R_err[1, 2],
                    R_err[0, 2] - R_err[2, 0],
                    R_err[1, 0] - R_err[0, 1],
                ]
            )
            error = np.hstack([dp, do])
            if np.linalg.norm(error) < tol:
                return [ut.wraptopi(float(val)) for val in q]

            J = self.jacobian(q.tolist())
            dq = J.T @ np.linalg.inv(J @ J.T + lambda_sq * np.eye(6)) @ error
            q = q + dq
            for i in range(6):
                q[i] = np.clip(q[i], self.joint_limits[i][0], self.joint_limits[i][1])

        return q
