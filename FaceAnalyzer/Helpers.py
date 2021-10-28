import numpy as np
import math
from scipy.spatial.transform import Rotation as R

def buildCameraMatrix(focal_length:float=None, center:tuple=None, size=(640,480))->np.ndarray:
    """Builds camera Matrix from the center position and focal length or aproximates it from the image size

    Args:
        focal_length (float, optional): The focal length of the camera. Defaults to None.
        center (tuple, optional): The center position of the camera. Defaults to None.
        size (tuple, optional): The image size in pixels. Defaults to (640,480).

    Returns:
        np.ndarray: The camera matrix
    """
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )
    return camera_matrix

def faceOrientation2Euler(r: np.ndarray, degrees:bool=True) -> np.ndarray:
    """Converts rodriguez representation of a rotation to euler angles

    Args:
        r (np.ndarray): The rodriguez representation vector (angle*u in form x,y,z)
        degrees (bool): If True, the outputs will be in degrees otherwize in radians. Defaults to True.

    Returns:
        np.ndarray: Eyler angles, yaw, pitch and roll
    """
    
    mrp = R.from_rotvec(r[:,0])
    yaw, pitch, roll = mrp.as_euler('yxz', degrees=degrees)
    if degrees:
        return yaw+180 if yaw<0 else yaw-180, pitch, roll+180 if roll<0 else roll-180
    else:
        return yaw+np.pi if yaw<0 else yaw-np.pi, pitch, roll+np.pi if roll<0 else roll-np.pi

def rotationMatrixToEulerAngles(R: np.ndarray) -> np.ndarray:
    """Computes the Euler angles in the form of Pitch yaw roll

    Args:
        R (np.ndarray): The rotation matrix

    Returns:
        np.ndarray: (Pitch, Yaw, Roll)
    """
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = np.math.atan2(R[2, 1], R[2, 2])
        y = np.math.atan2(-R[2, 0], sy)
        z = np.math.atan2(R[1, 0], R[0, 0])
    else:
        x = np.math.atan2(-R[1, 2], R[1, 1])
        y = np.math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])