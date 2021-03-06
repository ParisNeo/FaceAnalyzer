"""=== Face Analyzer Helpers =>
    Module : ui
    Author : Saifeddine ALOUI (ParisNeo)
    Licence : MIT
    Description :
        Orientation helpers
<================"""
import numpy as np
from scipy.spatial.transform import Rotation as R

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

def getDistanceBetweenAngles(angle1: float, angle2: float, is_degrees=True):
    """Computes the distance between two angles (in degrees or radians)

    Args:
        angle1 (float): The first angle
        angle2 (float): The secpnd angle
        is_degrees (bool, optional): Whether the angles are in degrees or radians. Defaults to True.

    Returns:
        float: The real distance between the two angles in degrees or randians depending on is_degrees parameter
    """
    dist = np.abs(angle2-angle1)
    if is_degrees:
        if dist>180:
            dist = 360-dist
    else:
        if dist>np.pi:
            dist = 2*np.pi-dist
    return dist


def rotateLandmarks(landmarks:np.ndarray, r:np.ndarray, invert:bool=False):
    mrp = R.from_rotvec(-r[:,0])
    if invert:
        mrp=mrp.inv()
    return mrp.apply(landmarks)


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


