import sys
import time
import queue
import threading
import numpy as np

# Real Robot Imports
try:
    import backend.utilities as utilities
    from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
    from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
    from kortex_api.autogen.messages import Base_pb2
except ImportError:
    print("[WARNING] Kortex API not found. Only simulation mode will work.")

# ==============================================================================
# 1. KINOVA BACKENDS (Hidden)
# ==============================================================================


class BaseKinova:
    """Physical robot backend using Kortex API."""

    def __init__(self, is_suction=False) -> None:
        self.args = utilities.parseConnectionArguments()
        self.real_angles = []  # Changed from np.zeros(6) to handle 7DOF dynamically
        self.gripper_position = 0.0
        self.is_suction = is_suction

        self.action_queue = queue.Queue()
        self._is_action_running = False

        self._data_lock = threading.Lock()
        self._is_running = False
        self._thread = None

        self._desired_admittance = False
        self._current_admittance = False

    def start(self):
        if not self._is_running:
            self._is_running = True
            self._thread = threading.Thread(target=self._background_loop, daemon=True)
            self._thread.start()
            time.sleep(1)

    def _background_loop(self):
        with utilities.DeviceConnection.createTcpConnection(self.args) as router:
            base = BaseClient(router)
            base_cyclic = BaseCyclicClient(router)

            while self._is_running:
                with self._data_lock:
                    desired_admittance = self._desired_admittance

                if desired_admittance != self._current_admittance:
                    admittance = Base_pb2.Admittance()
                    admittance.admittance_mode = (
                        Base_pb2.JOINT if desired_admittance else Base_pb2.UNSPECIFIED
                    )
                    try:
                        base.SetAdmittance(admittance)
                        self._current_admittance = desired_admittance
                    except Exception:
                        pass

                if not self._is_action_running and not self.action_queue.empty():
                    cmd = self.action_queue.get()
                    self._is_action_running = True

                    if cmd["type"] == "move":
                        threading.Thread(
                            target=self._execute_trajectory_and_grip,
                            args=(base, cmd),
                            daemon=True,
                        ).start()
                    elif cmd["type"] == "grip":
                        threading.Thread(
                            target=self._execute_standalone_grip,
                            args=(base, cmd),
                            daemon=True,
                        ).start()

                self._update_angles(base_cyclic)
                time.sleep(0.01)

    def _check_for_end_or_abort(self, e):
        def check(notification, e=e):
            if notification.action_event in (
                Base_pb2.ACTION_END,
                Base_pb2.ACTION_ABORT,
            ):
                e.set()

        return check

    def _execute_trajectory_and_grip(self, base, cmd):
        action = Base_pb2.Action()
        action.name = "Setting joint angles"
        actuator_count = base.GetActuatorCount()

        for idx, joint_id in enumerate(range(actuator_count.count)):
            if idx < len(cmd["angles"]):
                joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
                joint_angle.joint_identifier = joint_id
                joint_angle.value = np.degrees(cmd["angles"][idx])

        e = threading.Event()
        notification_handle = base.OnNotificationActionTopic(
            self._check_for_end_or_abort(e), Base_pb2.NotificationOptions()
        )

        base.ExecuteAction(action)
        finished = e.wait(20)

        try:
            base.Unsubscribe(notification_handle)
        except Exception:
            pass

        if self._is_running and finished and cmd["gripper"] is not None:
            self._execute_gripper_action(base, cmd["gripper"])

        cmd["event"].set()
        self._is_action_running = False

    def _execute_standalone_grip(self, base, cmd):
        self._execute_gripper_action(base, cmd["value"])
        cmd["event"].set()
        self._is_action_running = False

    def _execute_gripper_action(self, base, ratio):
        # [Unchanged from your original logic]
        if self.is_suction:
            try:
                gripper_command = Base_pb2.GripperCommand()
                gripper_command.mode = Base_pb2.GRIPPER_POSITION
                finger = gripper_command.gripper.finger.add()
                finger.finger_identifier = 1
                finger.value = ratio
                base.SendGripperCommand(gripper_command)
                time.sleep(1.0 if ratio > 0.5 else 3.0)
                with self._data_lock:
                    self.gripper_position = ratio
            except Exception as e:
                print(f"\n[BaseKinova] Suction failed: {e}")
        else:
            action = Base_pb2.Action()
            action.name = "Gripper Action"
            gripper_cmd = action.send_gripper_command
            gripper_cmd.mode = Base_pb2.GRIPPER_POSITION
            finger = gripper_cmd.gripper.finger.add()
            finger.finger_identifier = 1
            finger.value = ratio

            e = threading.Event()
            notification_handle = base.OnNotificationActionTopic(
                self._check_for_end_or_abort(e), Base_pb2.NotificationOptions()
            )
            try:
                base.ExecuteAction(action)
                e.wait(5)
                with self._data_lock:
                    self.gripper_position = ratio
            except Exception as e:
                print(f"\n[BaseKinova] Gripper failed: {e}")
            finally:
                try:
                    base.Unsubscribe(notification_handle)
                except Exception:
                    pass

    def _update_angles(self, base_cyclic):
        feedback = base_cyclic.RefreshFeedback()
        new_real_angles = [np.radians(a.position) for a in feedback.actuators]
        with self._data_lock:
            self.real_angles = new_real_angles

    def set_joint_angles(self, angles, gripper_percentage=None, wait=True):
        completion_event = threading.Event()
        safe_gripper = (
            None
            if gripper_percentage is None
            else max(0.0, min(100.0, float(gripper_percentage))) / 100.0
        )
        self.action_queue.put(
            {
                "type": "move",
                "angles": np.array(angles),
                "gripper": safe_gripper,
                "event": completion_event,
            }
        )
        if wait:
            completion_event.wait()

    def set_gripper(self, percentage: float, wait=True):
        completion_event = threading.Event()
        safe_ratio = max(0.0, min(100.0, float(percentage))) / 100.0
        self.action_queue.put(
            {"type": "grip", "value": safe_ratio, "event": completion_event}
        )
        if wait:
            completion_event.wait()

    def get_joint_angles(self):
        with self._data_lock:
            return list(self.real_angles)

    def open_gripper(self, wait=True):
        self.set_gripper(0.0, wait)

    def close_gripper(self, wait=True):
        self.set_gripper(100.0, wait)

    def set_torque(self, enable: bool):
        with self._data_lock:
            self._desired_admittance = not enable

    def stop(self):
        self._is_running = False
        try:
            with utilities.DeviceConnection.createTcpConnection(self.args) as router:
                BaseClient(router).Stop()
        except Exception:
            pass
        if self._thread:
            self._thread.join()


class SimKinova:
    """Simulated robot backend using PyBullet. Matches BaseKinova API."""

    def __init__(self, urdf_path, is_suction=False) -> None:
        self.urdf_path = urdf_path
        self.is_suction = is_suction

        self.real_angles = []
        self.gripper_position = 0.0

        self.action_queue = queue.Queue()
        self._is_action_running = False

        self._data_lock = threading.Lock()
        self._is_running = False
        self._thread = None

        # Sim specific
        self.p = None
        self.robot_id = None
        self.arm_joints = []
        self.gripper_joints = []

    def start(self):
        import pybullet as p
        import pybullet_data

        self.p = p

        # Init PyBullet
        self.p.connect(self.p.GUI)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.p.setGravity(0, 0, -9.81)
        self.p.loadURDF("plane.urdf")

        if not self.urdf_path:
            raise ValueError("urdf_path must be provided to use simulation mode.")

        self.robot_id = self.p.loadURDF(self.urdf_path, [0, 0, 0], useFixedBase=True)

        # Dynamically discover joints
        num_joints = self.p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            info = self.p.getJointInfo(self.robot_id, i)
            joint_type = info[2]
            name = info[1].decode("utf-8").lower()

            # If it's a movable joint
            if joint_type in [self.p.JOINT_REVOLUTE, self.p.JOINT_PRISMATIC]:
                if (
                    "finger" in name
                    or "gripper" in name
                    or "right" in name
                    or "left" in name
                ):
                    self.gripper_joints.append(i)
                else:
                    self.arm_joints.append(i)

        self.real_angles = np.zeros(len(self.arm_joints))

        if not self._is_running:
            self._is_running = True
            self._thread = threading.Thread(target=self._background_loop, daemon=True)
            self._thread.start()

    def _background_loop(self):
        while self._is_running:
            # Step the simulation
            self.p.stepSimulation()

            # Read angles natively
            states = self.p.getJointStates(self.robot_id, self.arm_joints)
            with self._data_lock:
                self.real_angles = [state[0] for state in states]

            # Process Actions
            if not self._is_action_running and not self.action_queue.empty():
                cmd = self.action_queue.get()
                self._is_action_running = True
                threading.Thread(
                    target=self._execute_action, args=(cmd,), daemon=True
                ).start()

            time.sleep(1.0 / 240.0)

    def _execute_action(self, cmd):
        if cmd["type"] == "move":
            target_angles = cmd["angles"]

            # 1. Get current angles
            states = self.p.getJointStates(
                self.robot_id, self.arm_joints[: len(target_angles)]
            )
            start_angles = np.array([s[0] for s in states])

            # --- THE SPEED LIMITER (Linear Interpolation) ---
            MAX_SPEED_RAD_PER_SEC = 1.0  # Adjust this! 1.0 is ~60 degrees/sec

            # Calculate the shortest path
            raw_diff = target_angles - start_angles
            shortest_diff = (raw_diff + np.pi) % (2 * np.pi) - np.pi
            continuous_target = start_angles + shortest_diff

            # If the circular shortcut would push a joint past its URDF limit,
            # fall back to the direct path for that joint. Both start and
            # target are already within limits, so the direct path is safe.
            arm_idx = self.arm_joints[: len(target_angles)]
            lo = np.array([self.p.getJointInfo(self.robot_id, j)[8] for j in arm_idx])
            hi = np.array([self.p.getJointInfo(self.robot_id, j)[9] for j in arm_idx])
            violates = (continuous_target > hi) | (continuous_target < lo)
            if np.any(violates):
                continuous_target = np.where(violates, target_angles, continuous_target)
                shortest_diff = continuous_target - start_angles

            # Find the joint that has to travel the furthest
            max_distance = np.max(np.abs(shortest_diff))

            # Calculate how long the move should take
            move_time = max(max_distance / MAX_SPEED_RAD_PER_SEC, 0.1)
            steps = int(move_time / 0.05)  # 20 Hz loop rate

            for step in range(1, steps + 1):
                progress = step / steps
                # Add the progressed shortest difference to the start angles
                current_target = start_angles + (shortest_diff * progress)

                self.p.setJointMotorControlArray(
                    self.robot_id,
                    self.arm_joints[: len(target_angles)],
                    self.p.POSITION_CONTROL,
                    targetPositions=current_target,
                    forces=[200]
                    * len(target_angles),  # Keep force high so it doesn't droop
                )
                time.sleep(0.05)
            # ------------------------------------------------

            # 2. Final precision alignment (Using the continuous target!)
            self.p.setJointMotorControlArray(
                self.robot_id,
                self.arm_joints[: len(target_angles)],
                self.p.POSITION_CONTROL,
                targetPositions=continuous_target,
                forces=[200] * len(target_angles),
            )

            # Wait for it to fully settle physically
            start_time = time.time()
            while time.time() - start_time < 2.0:
                states = self.p.getJointStates(
                    self.robot_id, self.arm_joints[: len(target_angles)]
                )
                current = np.array([s[0] for s in states])

                # Check settlement against the continuous target
                if np.max(np.abs(current - continuous_target)) < 0.05:
                    break
                time.sleep(0.05)

            # 3. Handle Gripper
            if cmd["gripper"] is not None:
                self._set_gripper_internal(cmd["gripper"])
                time.sleep(1.0)  # Wait for gripper to physically actuate

        elif cmd["type"] == "grip":
            self._set_gripper_internal(cmd["value"])
            time.sleep(1.0)

        cmd["event"].set()
        self._is_action_running = False

    def _set_gripper_internal(self, ratio):
        with self._data_lock:
            self.gripper_position = ratio

        if self.is_suction:
            return

        for joint in self.gripper_joints:
            info = self.p.getJointInfo(self.robot_id, joint)
            name = info[1].decode("utf-8").lower()

            # --- THE FIX: Hardcode the Kinova 2F Gripper kinematics ---
            # 0.0 is open (neutral origin), 1.0 is closed (extreme limit)
            if "right_bottom" in name:
                target = ratio * 0.96  # Closes positively
            elif "left_bottom" in name:
                target = ratio * -0.96  # Closes negatively
            elif "right_tip" in name:
                target = (
                    ratio * -1.03
                )  # Bends opposite to bottom joint to stay parallel
            elif "left_tip" in name:
                target = (
                    ratio * -1.03
                )  # Bends opposite to bottom joint (axis is inverted in URDF)
            else:
                # Generic fallback if your students load a different gripper later
                lower, upper = info[8], info[9]
                if lower != upper:
                    target = lower + ratio * (upper - lower)
                else:
                    target = 0.0

            # Apply the calculated target position
            self.p.setJointMotorControl2(
                self.robot_id,
                joint,
                self.p.POSITION_CONTROL,
                targetPosition=target,
                force=50,
            )

    def set_joint_angles(self, angles, gripper_percentage=None, wait=True):
        completion_event = threading.Event()
        safe_gripper = (
            None
            if gripper_percentage is None
            else max(0.0, min(100.0, float(gripper_percentage))) / 100.0
        )

        # FIX: Wrap angles to [-pi, pi] to prevent PyBullet physics explosions
        wrapped_angles = (np.array(angles) + np.pi) % (2 * np.pi) - np.pi

        self.action_queue.put(
            {
                "type": "move",
                "angles": wrapped_angles,
                "gripper": safe_gripper,
                "event": completion_event,
            }
        )
        if wait:
            completion_event.wait()

    def set_gripper(self, percentage: float, wait=True):
        completion_event = threading.Event()
        safe_ratio = max(0.0, min(100.0, float(percentage))) / 100.0
        self.action_queue.put(
            {"type": "grip", "value": safe_ratio, "event": completion_event}
        )
        if wait:
            completion_event.wait()

    def get_joint_angles(self):
        with self._data_lock:
            return list(self.real_angles)

    def open_gripper(self, wait=True):
        self.set_gripper(0.0, wait)

    def close_gripper(self, wait=True):
        self.set_gripper(100.0, wait)

    def set_torque(self, enable: bool):
        pass  # Not applicable for standard position-control simulation

    def stop(self):
        self._is_running = False
        if self.p:
            try:
                self.p.disconnect()
            except:
                pass


class Kinova:
    """Wrapper class choosing between Physical and Simulation"""

    def __init__(self, is_suction=False, simulate=False, urdf_path=None) -> None:
        if simulate:
            self.base_kinova = SimKinova(is_suction=is_suction, urdf_path=urdf_path)
        else:
            self.base_kinova = BaseKinova(is_suction=is_suction)
        self.base_kinova.start()

    def set_joint_angles(self, angles, gripper_percentage=None, wait=True):
        self.base_kinova.set_joint_angles(angles, gripper_percentage, wait)

    def get_joint_angles(self):
        return self.base_kinova.get_joint_angles()

    def set_gripper(self, percentage: float, wait=True):
        self.base_kinova.set_gripper(percentage, wait)

    def open_gripper(self, wait=True):
        self.base_kinova.open_gripper(wait)

    def close_gripper(self, wait=True):
        self.base_kinova.close_gripper(wait)

    def set_torque(self, enable: bool):
        self.base_kinova.set_torque(enable)

    def stop(self):
        self.base_kinova.stop()


# ==============================================================================
# 2. APP BOILERPLATE (Hidden)
# ==============================================================================


class BaseApp:
    def __init__(
        self, loop_rate=20, is_suction=False, simulate=False, urdf_path=None
    ) -> None:
        self.kinova_robot = Kinova(
            is_suction=is_suction, simulate=simulate, urdf_path=urdf_path
        )
        self.LOOP_RATE = 1 / float(loop_rate)
        self.action_queue = queue.Queue()
        self.is_running = True

        self.start()

        self.background_thread = threading.Thread(target=self._start_loop, daemon=True)
        self.background_thread.start()
        print(f"Loop Started (Simulation: {simulate})")

    def _start_loop(self):
        try:
            while self.is_running:
                if not self.action_queue.empty():
                    func, args = self.action_queue.get()
                    print(f"Executing: {func.__name__}")
                    func(*args)
                self.loop()
                time.sleep(self.LOOP_RATE)
        except Exception as e:
            print(f"ERROR Background loop crashed: {e}")

    def shutdown(self):
        print("Shutting down gracefully")
        self.is_running = False
        self.kinova_robot.set_torque(True)
        self.kinova_robot.stop()
        sys.exit(0)

    def start(self):
        pass

    def loop(self):
        pass


if __name__ == "__main__":
    import sys
    import time

    print("Testing Environment...\n")

    try:
        # Attempt to initialize the physical robot to verify the connection
        # The docs state this only works if physically connected to the KINOVA.
        test_robot = Kinova(simulate=False)

        # Give the background thread a moment to spin up and verify connection
        time.sleep(2)

        # Safely shut down the test connection
        test_robot.stop()

        print("Environment is ready to go\n")
        print("Have fun using the Kinova Robot Arm!")
        sys.exit(0)

    except Exception as e:
        print("\n[ERROR] Environment test failed. Could not connect to the Kinova arm.")
        print("Did you follow the Physical and Communication Setup sections?")
        print("Are you inside the UV virtual environment?")
        print(f"\nDetails: {e}")
        sys.exit(1)
