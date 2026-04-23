import pyrealsense2 as rs
import numpy as np
import cv2 as cv

aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_APRILTAG_36H11)
_aruco_params = cv.aruco.DetectorParameters()
_aruco_detector = cv.aruco.ArucoDetector(aruco_dict, _aruco_params)


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)
color_stream = profile.get_stream(rs.stream.color)
intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

camera_matrix = np.array([
    [intrinsics.fx, 0, intrinsics.ppx],
    [0, intrinsics.fy, intrinsics.ppy],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float32)

marker_size = 0.0254 


# This was taken from stackexchange
def estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        nada, R, t = cv.solvePnP(marker_points, c, mtx, distortion, False, cv.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash

def get_pose_vectors(image, marker_length):
    corners, ids, rejected = _aruco_detector.detectMarkers(image)
    if ids is None:
        return None, None
    rvecs, tvecs, _ = estimatePoseSingleMarkers(
        corners, marker_length, camera_matrix, dist_coeffs
    )
    return rvecs, tvecs

# Stack
if __name__ == "__main__":
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            rvecs, tvecs = get_pose_vectors(frame, marker_size)
            if rvecs is not None:
                print(rvecs, tvecs)
            cv.imshow("Frame", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv.destroyAllWindows()