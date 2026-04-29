from backend.kinova import BaseApp
import time
import cv2
import numpy as np
from kinova_lite_kinematics_dh import KinovaLiteDH
from utils import EndEffector, rotm_to_euler
import sys
import faulthandler
faulthandler.enable()

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
    [0,-1,0,0.33],
    [1,0,0,0.20],
    [0,0,-1,2.8],
    [0,0,0,1],
])

BLOCK_WIDTH = 0.0254
H_TAG_TO_BLOCK_CENTER = np.array([
    [1,0,0,BLOCK_WIDTH * 0.5],
    [0,1,0,BLOCK_WIDTH * 0.5],
    [0,0,1,BLOCK_WIDTH * -0.5],
    [0,0,0,1],
])

TABLE_PTS = np.array([
    [0.406, -0.03, 0.0],   # tag ID 4
    [0.406, 0.438, 0.0],   # tag ID 6
    [-0.02, 0.438, 0.0],   # tag ID 7
], dtype=np.float32)

TAG_IDS = [4,6,7]

HOME_JOINTS = np.array([1.75, 5.76, 2.18, 2.44, 4.54, 0.0])

def rigid_transform_3d(A, B):
    """
    A: Nx3 points in table frame
    B: Nx3 points in camera frame

    Returns:
        A matrix mapping table frame to camrea frame
    """
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # Fix reflection if needed
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Z axis should be negative (flipped camera from world Z)
    #if R[2, 2] > 0:
    #    R[:, 2] *= -1   # flip Z axis
    #    R[:, 1] *= -1 

    t = centroid_B - R @ centroid_A
    H_out = np.eye(4)
    H_out[:3,:3] = R
    H_out[:3,3] = t

    return H_out

class Main(BaseApp):
    """
    
    """
    def __init__(self,simulate=True, urdf_path="visualizer/6dof/urdf/6dof.urdf", video_device=0):
        self.cap = cv2.VideoCapture(video_device)
        super().__init__(simulate=True, urdf_path="visualizer/6dof/urdf/6dof.urdf")

    def start(self):
        """
        
        """
        self.state = STATE_SEARCH

        self.model = KinovaLiteDH()

        # Used for detecting aruco markers
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.detect_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict,detectorParams=self.detect_params)

        self.use_gui = True
        self.marker_length = 0.0254 # 100 mm
        self.obj_pt_arr = np.asarray([
            [-self.marker_length/2.0,self.marker_length/2.0,0],
            [self.marker_length/2.0,self.marker_length/2.0,0],
            [self.marker_length/2.0,-self.marker_length/2.0,0],
            [-self.marker_length/2.0,-self.marker_length/2.0,0]
        ])

        self.target_ids = [1,2] # ids of the blocks to move
        self.goal_id = 3 # id of the block that we stack targets on
        self.current_target_id = None
        self.last_known_target_pose = None
        self.last_known_goal_pose = None

        self.table_marker_poses = [None, None, None]
        self.table_to_cam = np.eye(4)

        self.grab_ee_target = None # EE object that defines where to move the gripper to pick up or release

        self.camMatrix = RS_INTRINSIC_COLOR_640
        self.distCoeffs = RS_DIST_COLOR_640
    
    def loop(self):
        #print(f"self.tablemarkerposes = {self.table_marker_poses}")
        if not any(val is None for val in self.table_marker_poses):
            camera_pts = np.vstack([tvec.reshape(1, 3) for tvec in self.table_marker_poses])
            print(f"camera pts: {camera_pts}, {camera_pts.shape}")
            self.table_to_cam = rigid_transform_3d(TABLE_PTS,camera_pts)
            print(self.table_to_cam)
            H_cam_to_table = np.linalg.inv(self.table_to_cam)
            for marker in self.table_marker_poses:
                p_cam = np.append(marker, 1.0)
                p_table_est = H_cam_to_table @ p_cam
                print(p_table_est[:3])
            print("----------------------------------")

        if self.state == STATE_SEARCH:
            # Look for the target block position 
            # Approach it once we find it 
            ret,poses = self.process_frame()
            if not ret:
                # failed to identify poses
                return
            print(f"poses: {poses}")
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
            ret,poses = self.process_frame()
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
            at_target = self.go_towards_pose(target_pose, 0.01)
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
            ret,poses = self.process_frame()
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
        return 0
        #target_pos = pose[0]
        #target_rpy = pose[1]

        #target = EndEffector() # Make end effector object
        #target.x, target.y, target.z = target_pos
        #target.rotx, target.roty, target.rotz = target_rpy
        target = pose

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

        print(f"setting joint angles {q_out}")
        self.kinova_robot.set_joint_angles(q_out, gripper_percentage=0)
        print(f"got to pose")

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
        #pose.rotx, pose.roty, pose.rotz = rotm_to_euler(H_world[:3,:3])
        rot_xyz = rotm_to_euler(H_world[:3,:3])
        pose.rotx = rot_xyz[0]
        pose.roty = rot_xyz[1]
        pose.rotz = rot_xyz[2]
        pose.x = H_world[0,3]
        pose.y = H_world[1,3]
        pose.z = H_world[2,3]

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
            return False,poses
        corners, ids, rejected = self.detector.detectMarkers(frame)
        print(ids)
        img_clone = frame.copy()

        if len(corners) < 1:
            # we haven't found a marker, exit early
            if self.use_gui:
                cv2.imshow("Camera Feed",img_clone)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    del(self)
                
            return False,poses

        n_markers = len(corners)
        for i in range(n_markers):
            _, rvecs, tvecs = cv2.solvePnP(
                objectPoints=self.obj_pt_arr,
                imagePoints=corners[i],
                cameraMatrix=self.camMatrix,
                distCoeffs=self.distCoeffs,
            )
            #print(f"tvecs: {tvecs}")
            if self.use_gui:
                cv2.aruco.drawDetectedMarkers(img_clone, corners, ids)
                cv2.drawFrameAxes(
                    img_clone,
                    self.camMatrix,
                    self.distCoeffs,
                    rvecs,
                    tvecs,
                    self.marker_length * 0.5,  # axis length
                )

                cv2.imshow("Camera Feed",img_clone)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    del(self)
                
                # convert poses from mm to m
                poses[ids[i][0]] = [rvecs.squeeze(), tvecs.squeeze()]
                print(poses)
            
            for j in range(len(TABLE_PTS)):
                if ids[i].squeeze() == TAG_IDS[j]:
                    self.table_marker_poses[j] = tvecs.squeeze()

        return True,poses


if __name__ == "__main__":
    # video_num = 0
    # print(sys.argv)
    # if len(sys.argv) > 1:
    #     video_num = sys.argv[1]
    #     print(video_num)
    # cap = cv2.VideoCapture(video_num)
    # while True:
    #     if not cap.isOpened():
    #         continue
    #     ret,frame = cap.read()
    #     print("tried reading")
    #     if not ret:
    #         continue
    #     cv2.imshow("frame",frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         pass
    video_num = 0
    print(sys.argv)
    if len(sys.argv) > 1:
        video_num = int(sys.argv[1])
    final_project = Main(simulate=True, urdf_path="visualizer/6dof/urdf/6dof.urdf", video_device=video_num)


    try:
        while True:
            time.sleep(0.1)  
    except KeyboardInterrupt:
        final_project.shutdown()