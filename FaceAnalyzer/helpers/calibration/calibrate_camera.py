"""=== Face Analyzer Helpers =>
    Module : ui
    Author : Saifeddine ALOUI (ParisNeo)
    Licence : MIT
    Description :
        Calibrate camera helpers
        Here you can find a tool to calibrate your camera using a checherboard
<================"""

import numpy as np
import cv2

import numpy as np
import cv2

def calibrate_camera_from_points(image_points, checkerboard_size, square_size, camera_matrix=None, dist_coeffs=None):
    """
    Calibrates the camera using the provided image points and checkerboard size.
    
    Parameters:
        image_points (list of numpy arrays): List of 2D points in image space.
        checkerboard_size (tuple): Tuple of checkerboard dimensions (width, height).
        square_size (float): The size of each checkerboard square in world units.
        camera_matrix (numpy array): Initial camera matrix estimate.
        dist_coeffs (numpy array): Initial distortion coefficients estimate.
        
    Returns:
        ret (float): The reprojection error of the calibration.
        mtx (numpy array): The camera matrix.
        dist (numpy array): The distortion coefficients.
        rvecs (list of numpy arrays): List of rotation vectors.
        tvecs (list of numpy arrays): List of translation vectors.
    """
    
    # Prepare object points in correct format
    object_points = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    object_points[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    object_points *= square_size
    
    object_points_list = [object_points] * len(image_points)
    
    # Run the calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points_list, image_points, (image_points[0].shape[1], image_points[0].shape[0]), camera_matrix, dist_coeffs)
    
    return ret, mtx, dist, rvecs, tvecs


def calibrate_camera_from_images_list(images, board_size, square_size):
    """Calibrates a camera using a set of checkerboard images
    
    Args:
        images (list): A list of checkerboard images
        board_size (tuple): The number of corners in each row and column of the checkerboard
        square_size (float): The size of each square in the checkerboard, in millimeters
        
    Returns:
        A tuple containing the camera matrix, distortion coefficients, and rotation and translation vectors
    """
    
    # Prepare object points
    objp = np.zeros((board_size[0]*board_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:board_size[0],0:board_size[1]].T.reshape(-1,2)
    objp *= square_size
    
    # Initialize arrays to store object points and image points from all images
    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane
    
    # Loop through all images
    for img in images:
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        
        # If corners are found, add object points and image points to the lists
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            
            # Draw and display the corners
            cv2.drawChessboardCorners(img, board_size, corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
            
    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # Print the calibration results
    print("Camera matrix:\n", mtx)
    print("\nDistortion coefficients:\n", dist)
    
    # Return the camera matrix, distortion coefficients, and rotation and translation vectors
    return mtx, dist, rvecs, tvecs



def calibrate_camera_from_image_files(images, board_size, square_size):
    """Calibrates a camera using a set of checkerboard images
    
    Args:
        images (list): A list of checkerboard images
        board_size (tuple): The number of corners in each row and column of the checkerboard
        square_size (float): The size of each square in the checkerboard, in millimeters
        
    Returns:
        A tuple containing the camera matrix, distortion coefficients, and rotation and translation vectors
    """
    
    # Prepare object points
    objp = np.zeros((board_size[0]*board_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:board_size[0],0:board_size[1]].T.reshape(-1,2)
    objp *= square_size
    
    # Initialize arrays to store object points and image points from all images
    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane
    
    # Loop through all images
    for fname in images:
        # Load the image
        img = cv2.imread(fname)
        
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        
        # If corners are found, add object points and image points to the lists
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            
            # Draw and display the corners
            cv2.drawChessboardCorners(img, board_size, corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
            
    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # Print the calibration results
    print("Camera matrix:\n", mtx)
    print("\nDistortion coefficients:\n", dist)
    
    # Return the camera matrix, distortion coefficients, and rotation and translation vectors
    return mtx, dist, rvecs, tvecs
