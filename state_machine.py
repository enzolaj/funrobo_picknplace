from backend.kinova import BaseApp
import time
import cv2
import numpy as np
from kinova_lite_kinematics_dh import KinovaLiteDH
from utils import EndEffector, rotm_to_euler

STATE_SEARCH = 0
STATE_APPROCH_TARGET = 1
STATE_GRAB = 2
STATE_APPROACH_GOAL = 3
STATE_RELEASE = 4
STATE_STOP = 5

APPROACH_Z_OFFSET = 0.05

# Parameters for Realsense Color channel, 
RS_INTRINSIC_COLOR_640 = np.array([
    [615.21,0,310.90],[0,614.45,243.97],[0,0,1]
])

RS_DIST_COLOR_640 = np.array([0,0,0,0,0])

# Transformations for our scene
H_CAMERA_TO_ROBOT = np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1],
])

BLOCK_WIDTH = 0.0254
H_TAG_TO_BLOCK_CENTER = np.array([
    [1,0,0,BLOCK_WIDTH * 0.5],
    [0,1,0,BLOCK_WIDTH * 0.5],
    [0,0,1,BLOCK_WIDTH * -0.5],
    [0,0,0,1],
])

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
        self.last_known_goal_pose = None

        self.grab_ee_target = None # EE object that defines where to move the gripper to pick up or release

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
            poses = self.process_frame()
            detected_ids = poses.keys()
            if self.current_target_id not in detected_ids:
                # we lost the target! can't see it anymore
                # TODO maybe do something here? 
                pass
            
            if self.current_target_id in detected_ids:
                # update target pose
                self.last_known_target_pose = poses[self.current_target_id]

            # Map OpenCV output in the camera frame to position in the world frame
            pose_robot_frame = self.camera_to_world_frame(tvecs=self.last_known_target_pose[1],rvecs=self.last_known_target_pose[0])
            target_pose = self.get_approach_pose(pose_robot_frame)

            # go towards the target pose
            at_target = self.go_towards_pose(target_pose, 0.1)
            if at_target:
                # we reached our target (the block)
                self.state = STATE_GRAB

                # get rid of the offset we enforce in the approach
                target_pose_shifted = target_pose
                target_pose_shifted.z = target_pose.z - APPROACH_Z_OFFSET
                self.grab_ee_target = target_pose_shifted
                print(f"Entering STATE_GRAB state")

        elif self.state == STATE_GRAB:
            self.kinova_robot.open_gripper()

            q_guess = np.array(self.kinova_robot.get_joint_angles(), dtype=float)
            q_sol = self.model.calc_inverse_kinematics(self.grab_ee_target, q_guess=q_guess)
            q_sol = np.asarray(q_sol, dtype=float)
            self.kinova_robot.set_joint_angles(q_sol)

            self.kinova_robot.close_gripper()
            self.state = STATE_APPROACH_GOAL
            print(f"Entering STATE_APPROACH_GOAL state")
            
        elif self.state == STATE_APPROACH_GOAL:
            # We've grabbed the target block, now we can go to the goal
            poses = self.process_frame()
            detected_ids = poses.keys()
            if self.goal_id in detected_ids:
                self.last_known_goal_pose = poses[self.goal_id]
            
            # Map OpenCV output in the camera frame to position in the world frame
            pose_robot_frame = self.camera_to_world_frame(tvecs=self.last_known_goal_pose[1],rvecs=self.last_known_goal_pose[0])
            goal_pose = self.get_approach_pose(pose_robot_frame)

            # go towards the target pose
            ## TODO we don't want to match the pose exactly --- instead be looking downwards, and target slightly above it
            at_goal = self.go_towards_pose(goal_pose, 0.1)
            if at_goal:
                # we reached our target (the block)
                target_pose_shifted = goal_pose
                target_pose_shifted.z = goal_pose.z - APPROACH_Z_OFFSET
                self.grab_ee_target = target_pose_shifted
                self.state = STATE_RELEASE
                print(f"Entering STATE_RELEASE state")

        elif self.state == STATE_RELEASE:
            q_guess = np.array(self.kinova_robot.get_joint_angles(), dtype=float)
            q_sol = self.model.calc_inverse_kinematics(self.grab_ee_target, q_guess=q_guess)
            q_sol = np.asarray(q_sol, dtype=float)
            self.kinova_robot.set_joint_angles(q_sol)

            self.kinova_robot.open_gripper()

            # Our old goal is covered, our new goal (if stacking multiple blocks) is our earlier target
            self.goal_id = self.current_target_id
            self.target_ids.remove(self.current_target_id)

            ### TODO make the robot retreat a bit, go to some neutral position 
            shifted_goal = self.grab_ee_target
            shifted_goal.z += APPROACH_Z_OFFSET

            q_guess = np.array(self.kinova_robot.get_joint_angles(), dtype=float)
            q_sol = self.model.calc_inverse_kinematics(self.grab_ee_target, q_guess=q_guess)
            q_sol = np.asarray(q_sol, dtype=float)
            self.kinova_robot.set_joint_angles(q_sol)

            self.state = STATE_STOP
            print(f"Entering STATE_STOP state")
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
        proportion_now = min(1.0, proportion_now)

        q_diff_now = q_diff * proportion_now
        q_out = q_guess + q_diff_now

        self.kinova_robot.set_joint_angles(q_out, gripper_percentage=0)

        # return true if we were planning to get all the way to our target
        return proportion_now >= 1.0
        

    def camera_to_world_frame(self,tvecs,rvecs):
        """
        Maps the output of opencv's solvePnP to the world frame
        """
        R_cam, _ = cv2.Rodrigues(rvecs) # marker frame to camera frame
        H_cam = np.eye(4)
        H_cam[:3,:3] = R_cam
        H_cam[:3,3] = tvecs

        H_world = H_CAMERA_TO_ROBOT @ H_cam

        pose = EndEffector()
        pose.rotx, pose.roty, pose.rotz = rotm_to_euler(H_world[:3,:3])
        pose.x = H_world[0]
        pose.y = H_world[1]
        pose.z = H_world[2]

        return pose
    
    def get_approach_pose(self,ee: EndEffector):
        """
        Takes a block pose and returns the pose the arm should go to for picking it up
        """
        ee_out = EndEffector()
        
        # The roll should be the same, so the gripper faces align with the block sides
        ee_out.rotz = ee.rotz

        # The z axis should be facing down (either a rotation by pi about y or x)
        ee_out.rotx = np.pi
        ee_out.roty = 0.0

        # The X and Y positions should be the same as the block, and the Z a bit over it
        ee_out.x = ee.x
        ee_out.y = ee.y
        ee_out.z = ee.z + APPROACH_Z_OFFSET

        return ee_out

    def process_frame(self):
        poses = {}
        ret, frame = self.cap.read()
        if not ret:
            print("Could not read frame")
            return

        corners, ids, rejected = self.detector.detectMarkers(frame)

        img_clone = frame.copy()

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