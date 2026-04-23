from backend.kinova import BaseApp
import time
import cv2
import numpy as np
from kinova_lite_kinematics_dh import KinovaLiteDH
from utils import EndEffector

STATE_SEARCH = 0
STATE_APPROCH_TARGET = 1
STATE_GRAB = 2
STATE_APPROACH_GOAL = 3
STATE_RELEASE = 4
STATE_STOP = 5

# Parameters for Realsense Color channel, 
RS_INTRINSIC_COLOR_640 = np.array([
    [615.21,0,310.90],[0,614.45,243.97],[0,0,1]
])

RS_DIST_COLOR_640 = np.array([0,0,0,0,0])

class Main(BaseApp):
    """
    
    """

    def start(self):
        """
        
        """
        self.state = STATE_SEARCH
        self.cap = cv2.VideoCapture(0)

        self.model = KinovaLiteDH()

        # Used for detecting aruco markers
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.detect_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict,detectorParams=self.detect_params)

        self.use_gui = True
        marker_length = 25.4 # 100 mm
        self.obj_pt_arr = np.asarray([
            [-marker_length/2.0,marker_length/2.0,0],
            [marker_length/2.0,marker_length/2.0,0],
            [marker_length/2.0,-marker_length/2.0,0],
            [-marker_length/2.0,-marker_length/2.0,0]
        ])

        self.target_ids = [1] # ids of the blocks to move
        self.goal_id = 2 # id of the block that we stack targets on
        self.current_target_id = None
        self.last_known_target_pose = None

        self.camMatrix = RS_INTRINSIC_COLOR_640
        self.distCoeffs = RS_DIST_COLOR_640
    
    def loop(self):
        if self.state == STATE_SEARCH:
            # Look for the target block position 
            # Approach it once we find it 
            poses = self.process_frame()
            detected_ids = poses.keys()
            for detected in detected_ids:
                if detected in self.target_ids:
                    # we detected a target to go get
                    self.current_target_id = detected
                    self.state = STATE_APPROCH_TARGET
                    print(f"Entering STATE_APPROACH_TARGET state")
                    self.last_known_target_pose = poses[detected]
                    return
                
        elif self.state == STATE_APPROCH_TARGET:
            pass
        elif self.state == STATE_GRAB:
            pass
        elif self.state == STATE_APPROACH_GOAL:
            pass
        elif self.state == STATE_RELEASE:
            pass
        else:
            pass
    
    def go_towards_pose(self, pose, max_rad):
        """
        Move a small amount towards a given pose in joint space
        """
        target_pos = pose[0]
        target_rpy = pose[1]

        target = EndEffector() # Make end effector object
        target.x, target.y, target.z = target_pos
        target.rotx, target.roty, target.rotz = target_rpy

        q_guess = np.array(self.kinova_robot.get_joint_angles(), dtype=float)
        q_sol = self.model.calc_inverse_kinematics(target, q_guess=q_guess)
        q_sol = np.asarray(q_sol, dtype=float)

        q_diff = q_sol - q_guess
        q_diff = np.mod(q_diff, 2*np.pi)
        q_diff = np.where(q_diff > np.pi, -1*(q_diff + np.pi), q_diff)

        mag_max_change = np.max(np.abs(q_diff))
        proportion_now = max_rad / mag_max_change # proportion of the total trajectory to do at once
        q_diff_now = q_diff * proportion_now
        q_out = q_guess + q_diff_now

        self.kinova_robot.set_joint_angles(q_sol, gripper_percentage=0)

        
    
    def get_target_pos(self,spoof=False):
        """
        
        """
    
    def 

    def process_frame(self):
        poses = {}
        ret, frame = self.cap.read()
        if not ret:
            print("Could not read frame")
            return

        corners, ids, rejected = self.detector.detectMarkers(frame)

        img_clone = frame.copy()
        has_found_tag = False

        if len(corners) < 1:
            # we haven't found a marker, exit early
            if self.use_gui:
                cv2.imshow("Camera Feed",img_clone)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    del(self)
            return

        n_markers = len(corners[0])
        for i in range(n_markers):
            _, rvecs, tvecs = cv2.solvePnP(
                objectPoints=self.obj_pt_arr,
                imagePoints=corners[i],
                cameraMatrix=self.camMatrix,
                distCoeffs=self.distCoeffs,
            )
            #print(f"tvecs: {tvecs}")
            poses[ids[i]] = [rvecs, tvecs]
        return poses


if __name__ == "__main__":
    final_project = Main(simulate=True, urdf_path="visualizer/6dof/urdf/6dof.urdf")

    try:
        while True:
            time.sleep(0.1)  
    except KeyboardInterrupt:
        final_project.shutdown()