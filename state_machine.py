from backend.kinova import BaseApp
import time
import cv2
import numpy as np

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

        self.camMatrix = RS_INTRINSIC_COLOR_640
        self.distCoeffs = RS_DIST_COLOR_640
    
    def loop(self):
        self.process_frame()
        if self.state == STATE_SEARCH:
            pass
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
    
    def get_target_pos(self,spoof=False):
        """
        
        """

    def process_frame(self):
        pose_diffs = {}
        ret, frame = self.cap.read()
        if not ret:
            print("Could not read frame")
            return

        corners, ids, rejected = self.detector.detectMarkers(frame)

        img_clone = frame.copy()
        has_found_tag = False

        if len(corners) < 1:
            # we haven't found a marker, exit early
            # print("No marker found")
            if self.use_gui:
                cv2.imshow("Camera Feed",img_clone)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    del(self)
            return
        else:
            print(f"Got corners!")

        n_markers = len(corners[0])
        for i in range(n_markers):
            _, rvecs, tvecs = cv2.solvePnP(
                objectPoints=self.obj_pt_arr,
                imagePoints=corners[i],
                cameraMatrix=self.camMatrix,
                distCoeffs=self.distCoeffs,
            )
            print(f"tvecs: {tvecs}")


if __name__ == "__main__":
    final_project = Main()

    try:
        while True:
            time.sleep(0.1)  
    except KeyboardInterrupt:
        final_project.shutdown()