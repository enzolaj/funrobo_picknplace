import pyrealsense2 as rs
import cv2
import numpy as np

# This is just setting up our detector, using the specific apriltags
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Positions of tags in robot frame, X distance, Y distance in meters
TAG_POSITIONS_IN_ROBOT_FRAME = {
    9: np.array([0.262, -0.057], dtype=np.float32),
    5: np.array([-.21, -0.057], dtype=np.float32),
    8: np.array([-.21,  -.415], dtype=np.float32)
}

# bounds for colors for mask for each cube
# these arent 100% perfect so maybe something to tweak later
red_lower = np.array([0, 100, 100])
red_upper = np.array([10, 255, 255])
blue_lower = np.array([90, 100, 150]) 
blue_upper = np.array([115, 255, 255])
green_lower = np.array([50, 80, 100]) 
green_upper = np.array([89, 255, 255])


def get_robot_coords(mask, affine_matrix):
    # Find shapes in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    locations = []
    for contour in contours:
        # Filter by area to avoid noise
        if cv2.contourArea(contour) > 100:
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


if __name__ == "__main__":
    # 1. Initialize RealSense Pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # 2. Explicitly request the 1080p stream
    # Note: We use rs.format.bgr8 so the color order matches OpenCV (Blue-Green-Red)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

    # 3. Start streaming
    profile = pipeline.start(config)
    print("RealSense Pipeline Started at 1920x1080")

    try:
        while True:
            # Wait for a coherent set of frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue

            # Convert the RealSense frame to a numpy array for OpenCV
            frame = np.asanyarray(color_frame.get_data())

            # Detect markers
            corners, ids, rejected = detector.detectMarkers(frame)

            pixel_pts = []
            robot_pts = []
            affine_matrix = None

            if ids is not None:
                flat_ids = ids.flatten()
                for i, tag_id in enumerate(flat_ids):
                    if tag_id in TAG_POSITIONS_IN_ROBOT_FRAME:
                        center = np.mean(corners[i][0], axis=0)
                        pixel_pts.append(center)
                        robot_pts.append(TAG_POSITIONS_IN_ROBOT_FRAME[tag_id])

                if len(pixel_pts) >= 3:
                    affine_matrix = cv2.getAffineTransform(
                        np.float32(pixel_pts[:3]),
                        np.float32(robot_pts[:3])
                    )

            # Processing: Blurring and Color Masking
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            
            masks = {
                "Red": cv2.inRange(hsv, red_lower, red_upper),
                "Blue": cv2.inRange(hsv, blue_lower, blue_upper),
                "Green": cv2.inRange(hsv, green_lower, green_upper)
            }

            # Refine masks and get coords
            if affine_matrix is not None:
                for color, mask in masks.items():
                    refined_mask = cv2.erode(mask, None, iterations=2)
                    refined_mask = cv2.dilate(refined_mask, None, iterations=2)
                    
                    coords = get_robot_coords(refined_mask, affine_matrix)
                    for pos in coords:
                        print(f"{color} Cube at Robot Frame: X={pos[0]:.3f}m, Y={pos[1]:.3f}m")

            cv2.imshow("RealSense 1080p Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop the camera even if an error occurs
        pipeline.stop()
        cv2.destroyAllWindows()