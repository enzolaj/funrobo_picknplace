from backend.kinova import BaseApp
import time
import cv2
import numpy as np
from kinova_lite_kinematics_dh import KinovaLiteDH
from utils import EndEffector, rotm_to_euler
import sys
import copy
import faulthandler
import pyrealsense2 as pyrs
faulthandler.enable()

STATE_SEARCH = 0
STATE_APPROACH_TARGET = 1
STATE_GRAB = 2
STATE_APPROACH_GOAL = 3
STATE_RELEASE = 4
STATE_STOP = 5

red_lower = np.array([0, 100, 100])
red_upper = np.array([10, 255, 255])
blue_lower = np.array([90, 100, 150]) 
blue_upper = np.array([115, 255, 255])
green_lower = np.array([50, 80, 100]) 
green_upper = np.array([89, 255, 255])

APPROACH_Z_OFFSET = 0.07
RECOMPUTE_IK_TOLERANCE = 0.01

# Camera needs a moment to settle exposure/white-balance on startup.
# During warmup we still show the frame, but ignore detections.
CAMERA_WARMUP_SECONDS = 1.5
CAMERA_WARMUP_DISCARD_FRAMES = 30

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

HOME_POSITION = np.array([1.25, 5.76, 2.18, 2.44, 4.54, 0.0])

TAG_IDS = [4,6,7]

#TAG_IDS = [9,5,8]

TAG_POSITION_IN_ROBOT_FRAME = {}

TAG_POSITIONS_IN_ROBOT_FRAME = {
    4: np.array([0.39, 0.0], dtype=np.float32),
    6: np.array([0.39, -0.415], dtype=np.float32),
    7: np.array([-0.015,  -0.415], dtype=np.float32)
}


def _wrap_to_pi(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return (q + np.pi) % (2.0 * np.pi) - np.pi


def _shortest_angle_diff(q_to: np.ndarray, q_from: np.ndarray) -> np.ndarray:
    return _wrap_to_pi(np.asarray(q_to, dtype=float) - np.asarray(q_from, dtype=float))


def _clip_to_limits(q: np.ndarray, joint_limits) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    limits = np.asarray(joint_limits, dtype=float)
    if limits.shape != (6, 2):
        return q
    return np.clip(q, limits[:, 0], limits[:, 1])

def get_robot_coords(mask, affine_matrix):
    # Find shapes in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    locations = []
    for contour in contours:
        # Filter by area to avoid noise
        if cv2.contourArea(contour) > 200:
            # Calculate the center of the blob
            # Moments count the pixels present in the image
            # m00 = sum of all pixels in the image
            # m10 = sum of all x coordinates of pixels in the image
            # m01 = sum of all y coordinates of pixels in the image
            M = cv2.moments(contour)
            if M["m00"] != 0:
                # px = x coordinate of the center of the blob
                # py = y coordinate of the center of the blob
                # These are flipped because of the way the image is displayed
                px = int(M["m10"] / M["m00"])
                py = int(M["m01"] / M["m00"])

                # pixel_vector has extra 1 for matrix multiplication
                pixel_vector = np.array([px, py, 1.0])
                robot_coords = affine_matrix @ pixel_vector
                locations.append(robot_coords)
    return locations

class Main(BaseApp):
    """
    
    """
    def __init__(self,simulate=False, urdf_path="visualizer/6dof/urdf/6dof.urdf"):
        self.pipeline = pyrs.pipeline()
        self.config = pyrs.config()
        self.config.enable_stream(pyrs.stream.color, 1920, 1080, pyrs.format.bgr8, 30)
        self.pipeline.start(self.config)
        super().__init__(simulate=simulate, urdf_path=urdf_path)

    def start(self):
        """
        
        """
        self.state = STATE_SEARCH

        self.model = KinovaLiteDH()
        self._camera_start_time = time.time()
        self._camera_discard_remaining = CAMERA_WARMUP_DISCARD_FRAMES

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

        self.target_colors = ["Red","Green"] # ids of the blocks to move
        self.goal_color = "Blue" # id of the block that we stack targets on
        self.current_target_color = None
        self.last_known_target_pose = None
        self.last_known_goal_pose = None

        self.pixel_pts = {}
        self.robot_pts = {}

        self.table_marker_poses = [None, None, None]
        self.table_to_cam = np.eye(4)

        self.grab_ee_target = None # EE object that defines where to move the gripper to pick up or release

        self.camMatrix = RS_INTRINSIC_COLOR_640
        self.distCoeffs = RS_DIST_COLOR_640
        self.kinova_robot.set_joint_angles(HOME_POSITION, gripper_percentage=0)

        self.count_stacked = 0
        self.last_IK_joint_target = None
        self.last_IK_target_pose = None
    
    def loop(self):
        if self.state == STATE_SEARCH:
            # Look for the target block position 
            # Approach it once we find it 
            ret,poses = self.process_frame()
            if not ret:
                # failed to identify poses
                return
            #print(f"poses: {poses}")
            print(poses.keys())
            detected_colors = poses.keys()
            for detected in detected_colors:
                if detected in self.target_colors:
                    # we detected a target to go get
                    self.current_target_color = detected
                    self.state = STATE_APPROACH_TARGET
                    print(f"Entering STATE_APPROACH_TARGET state")
                    self.last_known_target_pose = poses[detected]
                    return
                
        elif self.state == STATE_APPROACH_TARGET:
            ret,poses = self.process_frame()
            detected_colors = poses.keys()
            if self.current_target_color not in detected_colors:
                # we lost the target! can't see it anymore
                # TODO maybe do something here? 
                pass
            
            if self.current_target_color in detected_colors:
                # update target pose
                self.last_known_target_pose = poses[self.current_target_color]

            # Map OpenCV output in the camera frame to position in the world frame
            # pose_robot_frame = self.camera_to_world_frame(tvecs=self.last_known_target_pose[1],rvecs=self.last_known_target_pose[0])
            
            if self.last_IK_target_pose is not None:
                diff_x = self.last_IK_target_pose[0] - self.last_known_target_pose[0]
                diff_y = self.last_IK_target_pose[1] - self.last_known_target_pose[1]
                if diff_x ** 2 + diff_y ** 2 > RECOMPUTE_IK_TOLERANCE ** 2:
                    # our target has moved
                    self.last_IK_target_pose = self.last_known_target_pose
                    self.last_IK_joint_target = None # we need to recompute IK
                    if DO_LOGGING:
                        print(f"Our target has moved, recomputing IK")

            target_pose = EndEffector()
            target_pose.rotx = target_pose.roty = target_pose.rotz = 0
            target_pose.z = -0.01
            target_pose.x = self.last_known_target_pose[0]
            target_pose.y = self.last_known_target_pose[1]

            target_pose = self.get_approach_pose(target_pose)
            if DO_LOGGING:
                print(f"Trying to go to {self.current_target_color} at {target_pose.x: .4f}, {target_pose.y: .4f}, {target_pose.z: .4f}")
            #return # comment out to make it run

            # go towards the target pose
            at_target = self.go_towards_pose(target_pose, .3)
            if at_target:
                # we reached our target (the block)
                self.state = STATE_GRAB
                self.last_IK_joint_target = None
                self.last_IK_target_pose = None

                # get rid of the offset we enforce in the approach
                target_pose_shifted = copy.copy(target_pose) # copy it here
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

            self.grab_ee_target = copy.copy(self.grab_ee_target)
            self.grab_ee_target.z += APPROACH_Z_OFFSET
            self.state = STATE_APPROACH_GOAL
            print(f"Entering STATE_APPROACH_GOAL state")
            
        elif self.state == STATE_APPROACH_GOAL:
            # We've grabbed the target block, now we can go to the goal
            ret,poses = self.process_frame()
            detected_colors = poses.keys()
            if self.goal_color in detected_colors:
                self.last_known_goal_pose = poses[self.goal_color]
            
            if self.last_known_goal_pose is None:
                if DO_LOGGING:
                    print(f"We haven't found the goal!")
                return
            
            # Map OpenCV output in the camera frame to position in the world frame
            #pose_robot_frame = self.camera_to_world_frame(tvecs=self.last_known_goal_pose[1],rvecs=self.last_known_goal_pose[0])
            if self.last_IK_target_pose is not None:
                diff_x = self.last_IK_target_pose[0] - self.last_known_goal_pose[0]
                diff_y = self.last_IK_target_pose[1] - self.last_known_goal_pose[1]
                if diff_x ** 2 + diff_y ** 2 > RECOMPUTE_IK_TOLERANCE ** 2:
                    # our goal has moved
                    if DO_LOGGING:
                        print(f"Our goal has moved, recomputing IK")
                    self.last_IK_target_pose = self.last_known_goal_pose
                    self.last_IK_joint_target = None # we need to recompute IK

            goal_pose = EndEffector()
            goal_pose.rotx = goal_pose.roty = goal_pose.rotz = 0
            goal_pose.z = 0.025 + self.count_stacked * 0.025
            goal_pose.x = self.last_known_goal_pose[0]
            goal_pose.y = self.last_known_goal_pose[1]
            goal_pose = self.get_approach_pose(goal_pose)
            self.kinova_robot.close_gripper()

            # go towards the target pose
            ## TODO we don't want to match the pose exactly --- instead be looking downwards, and target slightly above it
            at_goal = self.go_towards_pose(goal_pose, .3)
            
            if at_goal:
                # we reached our target (the block)
                target_pose_shifted = goal_pose
                target_pose_shifted.z = goal_pose.z - APPROACH_Z_OFFSET
                self.last_IK_joint_target = None
                self.last_IK_target_pose = None
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
            self.goal_color = self.current_target_color
            self.target_colors.remove(self.current_target_color)

            ### TODO make the robot retreat a bit, go to some neutral position 
            shifted_goal = copy.copy(self.grab_ee_target)
            shifted_goal.z += APPROACH_Z_OFFSET

            q_guess = np.array(self.kinova_robot.get_joint_angles(), dtype=float)
            q_sol = self.model.calc_inverse_kinematics(shifted_goal, q_guess=q_guess)
            q_sol = np.asarray(q_sol, dtype=float)
            self.kinova_robot.set_joint_angles(q_sol)
            self.kinova_robot.set_joint_angles(HOME_POSITION, gripper_percentage=0)

            # self.state = STATE_STOP
            # print(f"Entering STATE_STOP state")
            self.count_stacked += 1
            self.state = STATE_SEARCH
            print(f"Entering STATE_SEARCH state")
        else:
            pass
    
    def go_towards_pose(self, pose, max_rad):
        """
        Move a small amount towards a given pose in joint space
        """
        success = False
        while not success:
            target = pose
            # Hardware returns angles that may be in [0, 2pi). Normalize to [-pi, pi]
            # so IK and shortest-path stepping stay continuous.
            q_meas = np.array(self.kinova_robot.get_joint_angles(), dtype=float)
            q_guess = _wrap_to_pi(q_meas)

            if self.last_IK_joint_target is None:
                q_sol = self.model.calc_inverse_kinematics(target, q_guess=q_guess)
                q_sol = _wrap_to_pi(np.asarray(q_sol, dtype=float))

                self.last_IK_joint_target = q_sol # target joint vals for IK solution
                self.last_IK_target_pose = [target.x, target.y, target.z]
            else:
                q_sol = self.last_IK_joint_target

            q_diff = _shortest_angle_diff(q_sol, q_guess)

            mag_max_change = np.max(np.abs(q_diff))
            proportion_now = max_rad / (1e-5 + mag_max_change) # proportion of the total trajectory to do at once
            proportion_now = min(1.0, proportion_now)

            q_diff_now = q_diff * proportion_now
            q_out = _wrap_to_pi(q_guess + q_diff_now)
            q_out = _clip_to_limits(q_out, getattr(self.model, "joint_limits", None))

            ee_out,_ = self.model.calc_forward_kinematics(q_out.tolist())
            if DO_LOGGING:
                print(f"EE out, predicted: {ee_out.x}, {ee_out.y}, {ee_out.z}")
            if ee_out.z > 0:
                success = True
                if DO_LOGGING:
                    print(f"setting joint angles {q_out}, with prop {proportion_now}")
                    print(f"my joints are {self.kinova_robot.get_joint_angles()}")
                self.kinova_robot.set_joint_angles(q_out, wait=True)
                if DO_LOGGING:
                    print(f"got to pose")
            
            else:
                if DO_LOGGING:
                    print(f"Tried to move into the table, will try again")

        # Only treat as "at target" if the *measured* robot state is close.
        # The physical backend unblocks on ACTION_ABORT as well as ACTION_END,
        # so we must verify rather than assuming success.
        q_after = _wrap_to_pi(np.array(self.kinova_robot.get_joint_angles(), dtype=float))
        q_err = np.max(np.abs(_shortest_angle_diff(q_out, q_after)))
        ee_after, _ = self.model.calc_forward_kinematics(q_after.tolist())
        pos_err = float(np.linalg.norm(np.array([ee_after.x - pose.x, ee_after.y - pose.y, ee_after.z - pose.z])))

        reached = (q_err < 0.10) and (pos_err < 0.02)
        return bool(reached and proportion_now >= 1.0)
        

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
        if DO_LOGGING:
            print(self.pixel_pts)
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return False,poses
        frame = np.asanyarray(color_frame.get_data())
        img_clone = frame.copy()

        # Warm up camera auto settings; ignore detections briefly.
        if self._camera_discard_remaining > 0:
            self._camera_discard_remaining -= 1
        warming_up = (time.time() - self._camera_start_time) < CAMERA_WARMUP_SECONDS

        # Calling the detect markers
        corners, ids, rejected = self.detector.detectMarkers(frame)
        
        affine_matrix = None
        if ids is not None:
            flat_ids = ids.flatten()
            for i, tag_id in enumerate(flat_ids):
                if tag_id in TAG_POSITIONS_IN_ROBOT_FRAME:
                    # Calculate center of the tag from its 4 corners
                    center = np.mean(corners[i][0], axis=0)
                    self.pixel_pts[tag_id] = center
                    self.robot_pts[tag_id] = TAG_POSITIONS_IN_ROBOT_FRAME[tag_id]

            if len(self.pixel_pts) >= 3:
                pixel_mat = np.vstack([self.pixel_pts[TAG_IDS[0]],self.pixel_pts[TAG_IDS[1]],self.pixel_pts[TAG_IDS[2]]])
                robot_mat = np.vstack([self.robot_pts[TAG_IDS[0]],self.robot_pts[TAG_IDS[1]],self.robot_pts[TAG_IDS[2]]])
                affine_matrix = cv2.getAffineTransform( # 2 x 3 (pixels to xy)
                    np.float32(pixel_mat),
                    np.float32(robot_mat)
                )
                if DO_LOGGING:
                    print(f"Got affine matrix")

        # Ok, so we now have our affine matrix, this will be used to go from pixel to robot frame
        # To extract pixels, we make a color mask
        # We will have a red, blue, and green mask since we need to identify and differentiate cube positions based on color
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)

        # Make masks less noisy
        red_mask = cv2.erode(red_mask, None, iterations=2)
        red_mask = cv2.dilate(red_mask, None, iterations=2)
        blue_mask = cv2.erode(blue_mask, None, iterations=2)
        blue_mask = cv2.dilate(blue_mask, None, iterations=2)
        green_mask = cv2.erode(green_mask, None, iterations=2)
        green_mask = cv2.dilate(green_mask, None, iterations=2)

        if affine_matrix is not None and (not warming_up) and self._camera_discard_remaining <= 0:
            masks = {
                "Red": red_mask,
                "Blue": blue_mask,
                "Green": green_mask
            }

            for color, mask in masks.items():
                # Get robot coordinates
                coords = get_robot_coords(mask, affine_matrix)

                for pos in coords:
                    if DO_LOGGING:
                        print(f"{color} Cube at Robot Frame: X={pos[0]}m, Y={pos[1]}m")
                    poses[color] = pos
        cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
        cv2.imshow('RealSense', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return True,poses

        return True,poses


if __name__ == "__main__":
    video_num = 8
    DO_LOGGING = True
    final_project = Main(simulate=False, urdf_path="visualizer/6dof/urdf/6dof.urdf")


    try:
        while True:
            time.sleep(0.1)  
    except KeyboardInterrupt:
        final_project.shutdown()