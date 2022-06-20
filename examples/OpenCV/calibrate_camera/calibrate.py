
"""=============
    Example : calibrate.py
    Author  : Saifeddine ALOUI adapted from this tutorial : https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html
    Description :
        Calibrates a calmera in order to account for its distortions
        This can be done once and then you can copy the generated cam_calib.pkl file in the folder of the example you wish to 
        use the calibration for.
        Use the svg asset Checkerboard-A4-25mm-10x7 that yoyu can find in the assets folder in the repository to print the chessboard image needed to do this calibration

<================"""

import pickle
import numpy as np
import cv2
from pathlib import Path
cap = cv2.VideoCapture(0)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

while True:
    success, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7,6), corners2, ret)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        
    cv2.imshow('img', img)
    # Wait for key stroke for 5 ms
    wk = cv2.waitKey(5)
    if wk & 0xFF == 27: # If escape is pressed then return
        break
    if wk & 0xFF == 115: # If s is pressed then take a snapshot
        try:
            output_path = Path(__file__).parent/"cam_calib.pkl"
            with open(str(output_path),"wb") as f:
                pickle.dump({"mtx":mtx,"dist":dist},f)
            print("Calibration file saved")
        except:
            print("!!! Problem")
            
cv2.destroyAllWindows()