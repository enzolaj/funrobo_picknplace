import cv2
import numpy as np

# This is just setting up our detector, using the specific apriltags
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Positions of tags in robot frame, X distance, Y distance in meters
TAG_POSITIONS_IN_ROBOT_FRAME = {
    4: np.array([0.405, 0.0], dtype=np.float32),
    6: np.array([0.405, -0.41], dtype=np.float32),
    7: np.array([0.0,  -0.41], dtype=np.float32)
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
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calling the detect markers
        corners, ids, rejected = detector.detectMarkers(frame)

        pixel_pts = []
        robot_pts = []
        affine_matrix = None
        if ids is not None:
            flat_ids = ids.flatten()
            for i, tag_id in enumerate(flat_ids):
                if tag_id in TAG_POSITIONS_IN_ROBOT_FRAME:
                    # Calculate center of the tag from its 4 corners
                    center = np.mean(corners[i][0], axis=0)
                    pixel_pts.append(center)
                    robot_pts.append(TAG_POSITIONS_IN_ROBOT_FRAME[tag_id])

            if len(pixel_pts) >= 3:
                affine_matrix = cv2.getAffineTransform(
                    np.float32(pixel_pts[:3]),
                    np.float32(robot_pts[:3])
                )

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

        if affine_matrix is not None:
            masks = {
                "Red": red_mask,
                "Blue": blue_mask,
                "Green": green_mask
            }

            for color, mask in masks.items():
                # Get robot coordinates
                coords = get_robot_coords(mask, affine_matrix)

                for pos in coords:
                    print(f"{color} Cube at Robot Frame: X={pos[0]}m, Y={pos[1]}m")

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
