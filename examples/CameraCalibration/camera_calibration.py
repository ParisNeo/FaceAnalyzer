import cv2
from FaceAnalyzer.helpers.calibration import calibrate_camera_from_points
import numpy as np
# Define the number of corners in the checkerboard and the size of each square
board_size = (8, 6)
square_size = 25

# Initialize arrays to store object points and image points
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane

# Initialize the camera capture
cap = cv2.VideoCapture(0)
decimation_counter = 0

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Find the checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, board_size, None)
    
    # If corners are found, add object points and image points to the lists
    if ret == True:
        objp = np.zeros((board_size[0]*board_size[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:board_size[0],0:board_size[1]].T.reshape(-1,2)
        objp *= square_size
        
        if decimation_counter%12==0:
            objpoints.append(objp)
            imgpoints.append(corners)
        
        decimation_counter += 1
        # Draw and display the corners
        cv2.drawChessboardCorners(frame, board_size, corners, ret)
        cv2.imshow('frame', frame)
        
    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera capture
cap.release()

# Calibrate the camera
ret, mtx, dist, rvecs, tvecs = calibrate_camera_from_points(imgpoints, board_size, square_size, camera_matrix=None, dist_coeffs=None)

# Print the calibration results
print("Camera matrix:\n", mtx)
print("\nDistortion coefficients:\n", dist)
