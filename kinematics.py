import numpy as np
import pybullet as pb
from pybullet_utils import bullet_client

def _slerp(q0, q1, t):
    """Shortest-arc quaternion SLERP. Inputs/output are [x, y, z, w]."""
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)

    dot = float(np.dot(q0, q1))

    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)

    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    s = np.sin(theta)

    res = (np.sin((1.0 - t) * theta) * q0 + np.sin(t * theta) * q1) / s
    return res


class KinovaLiteKinematics:
    """
    FK/IK for the Gen3 Lite 6-DOF.
    Uses a 'Shadow' client to solve math so the 'Live' robot doesn't jitter.
    """

    def __init__(self, pb_client, robot_id, urdf_path, tip_link_name="DUMMY"):
        self.p = pb_client # The Live Client
        self.robot_id = robot_id

        # 1. Create the Shadow Planning Client (DIRECT = No GUI)
        self._plan = bullet_client.BulletClient(connection_mode=pb.DIRECT)
        self._plan_robot = self._plan.loadURDF(
            urdf_path, [0, 0, 0], useFixedBase=True
        )

        # 2. Find the tip link index
        num_joints = self.p.getNumJoints(robot_id)
        self.tip_link_idx = None
        for i in range(num_joints):
            info = self.p.getJointInfo(robot_id, i)
            link_name = info[12].decode()
            if link_name == tip_link_name:
                self.tip_link_idx = i
                break

        if self.tip_link_idx is None:
            raise ValueError("Link not found: " + str(tip_link_name))

        # 3. Identify all movable joints
        self.movable_joints = []
        for i in range(num_joints):
            info = self.p.getJointInfo(robot_id, i)
            if info[2] != self.p.JOINT_FIXED:
                self.movable_joints.append(i)

        # 4. Identify specific arm joints (J0, J1, etc.)
        self.arm_joints = []
        for j in self.movable_joints:
            info = self.p.getJointInfo(robot_id, j)
            if info[1].decode().startswith("J"):
                self.arm_joints.append(j)

        if len(self.arm_joints) != 6:
            raise RuntimeError("Found " + str(len(self.arm_joints)) + " joints, need 6.")

        # 5. Map arm joints to their position in the movable list
        self._arm_idx_in_movable = []
        for j in self.arm_joints:
            idx = self.movable_joints.index(j)
            self._arm_idx_in_movable.append(idx)

        # 6. Store joint limits
        lower_limits = []
        upper_limits = []
        for j in self.arm_joints:
            info = self.p.getJointInfo(robot_id, j)
            lower_limits.append(info[8])
            upper_limits.append(info[9])

        self.arm_lower = np.array(lower_limits)
        self.arm_upper = np.array(upper_limits)

    def close(self):
        """Shut down the hidden planning client."""
        self._plan.disconnect()

    def get_forward_kinematics(self):
        """Gets current tip pose from the LIVE simulation."""
        state = self.p.getLinkState(
            self.robot_id, self.tip_link_idx, computeForwardKinematics=True
        )
        pos = np.array(state[4])
        orn = state[5]
        return pos, orn

    def _sync_plan_to_live(self):
        """Copies the LIVE robot joint positions into the SHADOW robot."""
        for j in self.movable_joints:
            live_q = self.p.getJointState(self.robot_id, j)[0]
            self._plan.resetJointState(self._plan_robot, j, live_q, 0.0)

    def _plan_reset_movable(self, q_movable):
        """Resets the SHADOW robot joints to a specific set of angles."""
        for i in range(len(self.movable_joints)):
            joint_id = self.movable_joints[i]
            angle = q_movable[i]
            self._plan.resetJointState(self._plan_robot, joint_id, angle, 0.0)

    def calculate_ik(self, target_pos, target_orn=None, homotopy_steps=30):
        """Solves IK in the Shadow client using small steps (homotopy)."""
        # Start from where the live robot is currently
        start_pos, start_orn = self.get_forward_kinematics()
        self._sync_plan_to_live()

        # Initial solution is just the current state
        sol = []
        for j in self.movable_joints:
            sol.append(self._plan.getJointState(self._plan_robot, j)[0])

        for i in range(1, homotopy_steps + 1):
            t = i / homotopy_steps
            
            # Interpolate position
            pos_t = (1.0 - t) * np.array(start_pos) + t * np.array(target_pos)

            # Interpolate orientation (SLERP)
            orn_t = None
            if target_orn is not None:
                orn_t = _slerp(start_orn, target_orn, t)

            # Solve IK in the shadow world
            if orn_t is not None:
                sol = self._plan.calculateInverseKinematics(
                    self._plan_robot,
                    self.tip_link_idx,
                    targetPosition=list(pos_t),
                    targetOrientation=list(orn_t),
                    maxNumIterations=200,
                    residualThreshold=1e-8,
                )
            else:
                sol = self._plan.calculateInverseKinematics(
                    self._plan_robot,
                    self.tip_link_idx,
                    targetPosition=list(pos_t),
                    maxNumIterations=200,
                    residualThreshold=1e-8,
                )

            # Move shadow robot to this partial solution to "warm start" the next step
            self._plan_reset_movable(sol)

        # Extract only the 6 arm joints from the full solution
        arm_sol_list = []
        for idx in self._arm_idx_in_movable:
            arm_sol_list.append(sol[idx])

        # Ensure result is within physical limits
        return self._wrap_into_limits(np.array(arm_sol_list))

    def _wrap_into_limits(self, arm_sol):
        """Keeps angles within URDF limits and picks the 'nearest' 2pi rotation."""
        out = np.array(arm_sol, dtype=float)

        for i in range(len(self.arm_joints)):
            joint_idx = self.arm_joints[i]
            q = out[i]
            lo = self.arm_lower[i]
            hi = self.arm_upper[i]

            # Get current position to find the closest rotation
            curr_pos = self.p.getJointState(self.robot_id, joint_idx)[0]

            # Possible rotations: -720, -360, 0, 360, 720 degrees
            candidates = [q - 4*np.pi, q - 2*np.pi, q, q + 2*np.pi, q + 4*np.pi]

            in_range = []
            for c in candidates:
                if lo <= c <= hi:
                    in_range.append(c)

            if len(in_range) > 0:
                # Find the candidate with the smallest distance to current pos
                best_val = in_range[0]
                min_dist = abs(best_val - curr_pos)
                for val in in_range:
                    if abs(val - curr_pos) < min_dist:
                        min_dist = abs(val - curr_pos)
                        best_val = val
                out[i] = best_val
            else:
                out[i] = np.clip(q, lo, hi)
        return out

    def predict_fk(self, arm_angles):
        """Predicts where the tip would be WITHOUT moving the live robot."""
        self._sync_plan_to_live()

        # Get current shadow state
        q_all = []
        for j in self.movable_joints:
            q_all.append(self._plan.getJointState(self._plan_robot, j)[0])

        # Overwrite the arm joints with the angles we want to test
        for i in range(len(arm_angles)):
            target_val = arm_angles[i]
            movable_idx = self._arm_idx_in_movable[i]
            q_all[movable_idx] = float(target_val)

        # Update shadow robot and read tip pose
        self._plan_reset_movable(q_all)
        state = self._plan.getLinkState(
            self._plan_robot, self.tip_link_idx, computeForwardKinematics=True
        )
        pos = np.array(state[4])
        orn = state[5]
        return pos, orn

    def euler_from_quat(self, quat):
        """Helper to get Roll, Pitch, Yaw."""
        return self.p.getEulerFromQuaternion(quat)