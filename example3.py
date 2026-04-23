import numpy as np
import time
import threading
import pybullet as pb
from pybullet_utils import bullet_client
from backend.kinova import BaseApp, SimKinova
from kinematics import KinovaLiteKinematics


URDF_PATH = "visualizer/6dof/urdf/6dof.urdf"


def wrap_to_pi(a):
    """Wraps angles to the range [-pi, pi]."""
    a_array = np.asarray(a)
    result = (a_array + np.pi) % (2 * np.pi) - np.pi
    return result


class KinovaLiteMain(BaseApp):
    def start(self):
        # Explicitly define home angles
        angles = [1.75, 5.76, 2.18, 2.44, 4.54, 0.0]
        self.home_angles = wrap_to_pi(angles)

        self.down_rpy = [np.pi, 0.0, 0.0]
        self.step = 0

        base = self.kinova_robot.base_kinova

        if isinstance(base, SimKinova):
            # Sim backend already has a live PyBullet client/robot.
            pb_client = base.p
            robot_id = base.robot_id
            urdf_path = base.urdf_path
        else:
            # Real robot: build a headless PyBullet "live" client and keep it
            # synced from the real encoders via a background thread.
            pb_client = bullet_client.BulletClient(connection_mode=pb.DIRECT)
            robot_id = pb_client.loadURDF(URDF_PATH, [0, 0, 0], useFixedBase=True)
            urdf_path = URDF_PATH

            # Discover J0..J5 in URDF order to map real angles -> bullet joints.
            arm_joints = []
            for i in range(pb_client.getNumJoints(robot_id)):
                info = pb_client.getJointInfo(robot_id, i)
                if info[2] != pb_client.JOINT_FIXED and info[1].decode().startswith("J"):
                    arm_joints.append(i)

            self._shadow_pb = pb_client
            self._shadow_robot = robot_id
            self._shadow_arm_joints = arm_joints
            self._shadow_running = True

            # Seed once so FK is valid immediately.
            self._sync_shadow_once()

            t = threading.Thread(target=self._shadow_sync_loop, daemon=True)
            t.start()

        self.kin_helper = KinovaLiteKinematics(pb_client, robot_id, urdf_path)

    def _sync_shadow_once(self):
        angles = self.kinova_robot.get_joint_angles()
        n = min(len(angles), len(self._shadow_arm_joints))
        for i in range(n):
            self._shadow_pb.resetJointState(
                self._shadow_robot,
                self._shadow_arm_joints[i],
                float(angles[i]),
                0.0,
            )

    def _shadow_sync_loop(self):
        while self._shadow_running:
            try:
                self._sync_shadow_once()
            except Exception:
                pass
            time.sleep(0.02)

    def _report(self, label, target_pos, target_rpy=None):
        # Get current state from kinematics helper
        reached, orn = self.kin_helper.get_forward_kinematics()
        
        # Calculate error
        target_pos_array = np.array(target_pos)
        err = target_pos_array - reached
        err_norm = np.linalg.norm(err)
        
        # Convert orientation to RPY and round
        current_rpy = self.kin_helper.euler_from_quat(orn)
        rpy_rounded = np.round(current_rpy, 3)
        
        # Build the log message manually for clarity
        msg = "[" + str(label) + "] "
        msg += "reached=" + str(np.round(reached, 4)) + " "
        msg += "rpy=" + str(rpy_rounded) + " "
        msg += "err=" + str(np.round(err, 4)) + " "
        msg += "|err|=" + str(round(err_norm, 4))
        
        if target_rpy is not None:
            target_rpy_rounded = np.round(target_rpy, 3)
            msg += " target_rpy=" + str(target_rpy_rounded)
            
        print(msg)

    def _move_to(self, target_pos, target_rpy=None, gripper=None, label="move"):
        # Determine target quaternion
        if target_rpy is not None:
            orn_quat = self.kin_helper.p.getQuaternionFromEuler(target_rpy)
        else:
            orn_quat = None
            
        # Calculate Inverse Kinematics
        q = self.kin_helper.calculate_ik(target_pos, orn_quat)

        # Check what IK predicts the result will be (Forward Kinematics)
        pred_pos, pred_orn = self.kin_helper.predict_fk(q)
        pred_rpy = self.kin_helper.euler_from_quat(pred_orn)
        pred_rpy_rounded = np.round(pred_rpy, 3)
        
        target_pos_array = np.array(target_pos)
        pred_err = np.linalg.norm(target_pos_array - pred_pos)
        
        # Print status of the IK solution
        print("   IK solution q=" + str(np.round(q, 3)))
        print("   IK predicts pos=" + str(np.round(pred_pos, 4)) + " rpy=" + str(pred_rpy_rounded) + " |err|=" + str(round(pred_err, 4)))

        # Send command to the robot
        self.kinova_robot.set_joint_angles(q, gripper_percentage=gripper)
        
        # Small delay for physics/motion to settle
        time.sleep(0.3)
        
        # Report the final physical state
        self._report(label, target_pos, target_rpy)

    def loop(self):
        # State Machine - Step 0: Move to Home
        if self.step == 0:
            print("Moving to Home...")
            self.kinova_robot.set_joint_angles(self.home_angles, gripper_percentage=100)
            time.sleep(2)

            curr_pos, curr_orn = self.kin_helper.get_forward_kinematics()
            curr_rpy = self.kin_helper.euler_from_quat(curr_orn)
            
            print("Home tip pos: " + str(np.round(curr_pos, 4)))
            print("Home tip rpy: " + str(np.round(curr_rpy, 3)))

            # Define a target point relative to home
            new_x = .3
            new_y = .2
            new_z = .05
            self.target_pos = [new_x, new_y, new_z]
            
            # Draw a green point in the simulator at the target (GUI only)
            try:
                self.kin_helper.p.addUserDebugPoints(
                    [self.target_pos],
                    [[0, 1, 0]],
                    pointSize=10,
                    lifeTime=0,
                )
            except Exception:
                pass
            
            self.step = 1
            return

        # State Machine - Step 1: Move to target
        if self.step == 1:
            print("Moving to target with fingers-down orientation...")
            self._move_to(self.target_pos, self.down_rpy, gripper=20, label="target")
            self.step = 2
            return


if __name__ == "__main__":
    # Initialize project
    final_project = KinovaLiteMain(
        simulate=True, 
        urdf_path="visualizer/6dof/urdf/6dof.urdf"
    )
    
    # Keep the script alive
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        final_project.shutdown()