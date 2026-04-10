import backend.utilities as utilities
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2
import numpy as np
import threading
import sys
import time

class BaseKinova:
    def __init__(self) -> None:
        self.args = utilities.parseConnectionArguments()
        self.angles = np.zeros(6)
        self.previous_angles = np.zeros(6)
        self.real_angles = np.zeros(6)
        
        # Threading infrastructure
        self._data_lock = threading.Lock()
        self._is_running = False
        self._thread = None
        
        # Admittance state tracking
        self._desired_admittance = False
        self._current_admittance = False

    def start(self):
        """Starts the background connection and periodic loop."""
        if not self._is_running:
            self._is_running = True
            self._thread = threading.Thread(target=self._background_loop, daemon=True)
            self._thread.start()
            # Give it a second to establish the TCP connection
            time.sleep(1) 

    def _background_loop(self):
        """This runs entirely in the background, managing the connection and arm."""
        with utilities.DeviceConnection.createTcpConnection(self.args) as router:
            base = BaseClient(router)
            base_cyclic = BaseCyclicClient(router)
            
            while self._is_running:
                # 1. Safely read shared states
                with self._data_lock:
                    target_angles = self.angles.copy()
                    desired_admittance = self._desired_admittance
                    
                # 2. Check for Admittance Mode changes
                if desired_admittance != self._current_admittance:
                    admittance = Base_pb2.Admittance()
                    if desired_admittance:
                        admittance.admittance_mode = Base_pb2.JOINT # pyright: ignore[reportAttributeAccessIssue]
                        print("\n[BaseKinova] Entering Joint Admittance (Freedrive) mode.")
                    else:
                        admittance.admittance_mode = Base_pb2.UNSPECIFIED # pyright: ignore[reportAttributeAccessIssue]
                        print("\n[BaseKinova] Exiting Admittance mode.")
                    
                    try:
                        base.SetAdmittance(admittance)
                        self._current_admittance = desired_admittance
                    except Exception as e:
                        print(f"\n[BaseKinova] Failed to set admittance mode: {e}")
                    
                # 3. If target changed, trigger movement non-blockingly
                if not np.array_equal(target_angles, self.previous_angles):
                    # We spawn a mini-thread for the action so 'e.wait(20)' 
                    # doesn't freeze our feedback loop
                    threading.Thread(
                        target=self._set_joint_angles, 
                        args=(base, target_angles), 
                        daemon=True
                    ).start()
                    
                    self.previous_angles = target_angles
                
                # 4. Update real angles from the robot
                self._update_angles(base_cyclic)
                
                # 5. Sleep briefly to run at ~100Hz and prevent 100% CPU usage
                time.sleep(0.01)

    def _check_for_end_or_abort(self, e):
        def check(notification, e=e):
            if notification.action_event == Base_pb2.ACTION_END \
            or notification.action_event == Base_pb2.ACTION_ABORT:
                e.set()
        return check

    def _set_joint_angles(self, base, target_angles):
        """Executes the movement. Now runs in its own isolated thread."""
        action = Base_pb2.Action()
        action.name = "Setting joint angles" # pyright: ignore[reportAttributeAccessIssue]
        action.application_data = "" # pyright: ignore[reportAttributeAccessIssue]
        
        actuator_count = base.GetActuatorCount()

        for idx, joint_id in enumerate(range(actuator_count.count)):
            joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add() # pyright: ignore[reportAttributeAccessIssue]
            joint_angle.joint_identifier = joint_id
            joint_angle.value = np.degrees(target_angles[idx])
            
        e = threading.Event()
        notification_handle = base.OnNotificationActionTopic(
            self._check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )
        
        base.ExecuteAction(action)
        finished = e.wait(20)
        base.Unsubscribe(notification_handle)

    def _update_angles(self, base_cyclic):
        """Fetches feedback and safely stores it."""
        feedback = base_cyclic.RefreshFeedback()
        new_real_angles = [np.radians(a.position) for a in feedback.actuators]
            
        # Safely write the new real angles
        with self._data_lock:
            self.real_angles = new_real_angles

    # ------------------------------------------------------------------
    # PUBLIC API - The only methods you need to use in your main program
    # ------------------------------------------------------------------

    def set_joint_angles(self, angles):
        """Command the arm to new angles."""
        with self._data_lock:
            self.angles = np.array(angles)

    def get_joint_angles(self):
        """Read the current physical angles of the arm."""
        with self._data_lock:
            return list(self.real_angles)
            
    def set_torque(self, enable: bool):
        """
        Puts the arm in or out of joint admittance mode.
        When True, the arm holds its position rigidly.
        When False, the arm can be moved freely by hand. 
        """
        with self._data_lock:
            self._desired_admittance = not enable
            
    def stop(self):
        """Stops the background loop and gracefully closes connections."""
        self._is_running = False
        
        # Stop any active trajectories currently executing on the robot
        try:
            with utilities.DeviceConnection.createTcpConnection(self.args) as router:
                base = BaseClient(router)
                base.Stop() # This sends a universal stop command to the motors
                print("[Kinova] Halted all active motor movements.")
        except Exception:
            pass # Failsafe if the connection is already dead
            
        if self._thread is not None:
            self._thread.join()
        
class Kinova():
    def __init__(self) -> None:
        self.base_kinova = BaseKinova()
        self.base_kinova.start()
        
    def set_joint_angles(self, angles):
        self.base_kinova.set_joint_angles(angles)
        
    def get_joint_angles(self):
        return self.base_kinova.get_joint_angles()
            
    def set_torque(self, enable: bool):
        self.base_kinova.set_torque(enable)
        
    def stop(self):
        self.base_kinova.stop()
        
        
if __name__ == "__main__":
    kinova = BaseKinova()
    kinova.start()
    
    print("Testing Environment...")
    print("Environment is ready to go")
    print("Have fun using the Kinova Robot Arm!")
