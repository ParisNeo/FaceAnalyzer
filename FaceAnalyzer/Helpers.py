import numpy as np
import math

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

def rodriguezToRotationMatrix(r: np.ndarray) -> np.ndarray:
    """Converts rodriguez representation of a rotation to eykler angles

    Args:
        r (np.ndarray): The rodriguez representation vector (angle*u in form x,y,z)

    Returns:
        np.ndarray: Eyler angles, yaw, pitch and roll
    """
    
    alpha = np.linalg.norm(r)
    axis = r/alpha
    x,y,z = axis[0,0],axis[1,0],axis[2,0]
    yaw = math.atan2(y * math.sin(alpha)- x * z * (1 - math.cos(alpha)) , 1 - (y**2 + z**2 ) * (1 - math.cos(alpha)))
    pitch = math.atan2(x * math.sin(alpha)-y * z * (1 - math.cos(alpha)) , 1 - (x**2 + z**2) * (1 - math.cos(alpha)))
    roll = math.asin(x * y * (1 - math.cos(alpha)) + z * math.sin(alpha))


    """
    except at the singularities, straight up:
    heading = 2 * math.atan2(x * math.sin(alpha/2),math.cos(alpha/2))
    bank = 0
    straight down:
    heading = -2 * math.atan2(x * math.sin(alpha/2),math.cos(alpha/2))
    bank = 0
    """

    return yaw, pitch, roll
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