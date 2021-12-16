"""=== Face Analyzer =>
    Module : Helpers
    Author : Saifeddine ALOUI (ParisNeo)
    Licence : MIT
    Description :
        A toolbox for geometry, optics and other tools useful for analyzing positions/orientations and converting them from 2d to 3d or detecting interactions etc...
<================"""

from typing import Tuple
import numpy as np
import math
from numpy.lib.type_check import imag
from scipy.spatial.transform import Rotation as R
import cv2
from PIL import Image, ImageDraw

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



def get_z_line_equation(pos: np.ndarray, ori:np.ndarray):
    """A line is defined by x = p0_x+v_xt
                            y = p0_y+v_yt
                            z = p0_z+v_zt
        The z line is the line pointing forward 'if we consider the face as 2d plane, z line is normal to that plane
    Args:
        pos (np.ndarray): reference position (coordinate of the point at t=0)
        ori (np.ndarray): orientation of the line

    Returns:
        (Tuple): The equation of the line through (p0,v)
    """
    rvec_matrix = cv2.Rodrigues(ori)[0]
    vz = rvec_matrix[:,2]

    # p = pos +vz*t
    return (pos[:,0], vz)

def get_plane_infos(p1: np.ndarray, p2:np.ndarray, p3:np.ndarray):
    """Returns the informations of a plane from a 3d region points

    Args:
        pos (np.ndarray): [description]
        ori (np.ndarray): [description]

    Returns:
        tuple: p, e1, e2, n where p is the reference position of the plane, e1,e2 are authonormal vectors defining a reference frame in the plane, and n is the normal vector to the plane
    """
    n = np.cross(p2-p1,p3-p1)
    n = n/np.linalg.norm(n)

    # (p-p1)Xn=0
    # Get unit vectors of the plane
    e1 = (p2-p1)
    e1 = e1/np.linalg.norm(e1)
    e2 = np.cross(n,e1)
    return (p1,e1,e2,n)

def get_plane_line_intersection(plane:Tuple, line:Tuple):
    """
    Returns a 3d and 2d position of intersection between a line and a plane (if the line is parallel to the plane return None)
    """
    p0  = plane[0]
    e1  = plane[1]
    e2  = plane[2]
    n   = plane[3]

    pl0 = line[0]
    v = line[1]
    pl00=pl0-p0
    """
    (p-p0)Xn=0
    pl0+v*t=p

    ((pl0+vt)-p0)Xn=0
    let pl00 = pl0-p0
    (pl00+vt).n=0

    p1 = (pl00+vt)

    p1x*nx+p1y*ny+p1z*nz=0

    (pl00x+vx * t)*nx + (pl00y+vy * t)*ny + (pl00z+vz * t)*nz =0

    pl00x*nx + pl00y*ny + pl00z*nz + vx*t*nx + vy*t*ny + vz*t*nz = 0

    t (vx*nx+vy*ny+vz*nz) + pl00x*nx+ pl00y*ny + pl00z*nz = 0

    t = -(pl00x*nx+ pl00y*ny + pl00z*nz)/(vx*nx+vy*ny+vz*nz)
    t = -(pl00.n)/(v.n)
    """

    if (np.dot(v,n))!=0: # The plan is not parallel to the line
        t   = -np.dot(pl00,n)/np.dot(v,n)
        vt  = v*t
        p   = pl0+vt
        p2d = np.array([np.dot(p,e1),np.dot(p,e2)])
    else: # The vector and the plan are parallel, there is no intersection point
        p   = None
        p2d = None

    return p, p2d


def region_3d_2_region_2d(region:np.ndarray, plane:np.ndarray):
    """Converts a region3d to a region2d by projecting all points in the plane defined by the first three vertices of the region 

    Args:
        region (np.ndarray): a ndarray (3XN) where N is the number of points in the region
        plane (np.ndarray): The plane containing the region

    Returns:
        tuple: Returns the region2d
    """
    region_2d = np.zeros((2,region.shape[1]))
    # First find the pointing line, and the plan on which the region is selected
    _,e1,e2, _ = plane
    # Lets put all the points of the region inside the 2d plane
    for i in range(region.shape[1]):
        region_2d[:,i]=np.array([np.dot(region[:,i], e1), np.dot(region[:,i], e2)]).T
    return region_2d   

def is_point_inside_region(point: np.ndarray, region:np.ndarray):
    """Returns whether a point is inside a convex region

    Args:
        point (np.ndarray): The point to be tested
        region (tuple): A list of points in form of ndarray that represent the region (all points should belong to the same plan)

    Returns:
        boolean: If true then the point is inside the region else false
    """
    # Now let's check that the poit is inside the region
    in_range=True
    for i in range(region.shape[1]):
        AB = region[:, (i+1)%region.shape[1]]-region[:, i]
        AP = point-region[:, i]
        c = np.cross(AB, AP)
        if i==0:
            if c>=0:
                pos=True
            else:
                pos=False
        else:
            if c>=0 and pos==False:
                in_range = False
                break
            elif c<0 and pos==True:
                in_range = False
                break
    
    return in_range

class KalmanFilter():
    """A simple linear Kalman filter
    """
    def __init__(self, Q:np.ndarray, R:np.ndarray, x0:np.ndarray, P0:np.ndarray, F, H) -> None:
        """Initializes a simple linear Kalman Filter

        Args:
            Q (np.ndarray): The state noise covariance matrix (how uncertain our state model is)
            R (np.ndarray): The measurement noise covariance matrix (how uncertain our measurement model is)
            x0 (np.ndarray): The initial state mean
            P0 (np.ndarray): The initial state covariance matrix (how uncertain our initial state is)
            F ([type]): state evolution matrix (x_n+1 = F*x_n)
            H ([type]): measurement matrix z_n = H*x_n
        """
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0
        self.F = F
        self.H = H


    def process(self, z:np.ndarray):
        """Processes a new measuremnt and updates the current state

        Args:
            z (np.ndarray): The measurement at current instant
        """
        # Predict
        self.x_ = self.F@self.x
        self.P_ = self.F@self.P@self.F.T+self.Q
        # Update
        y_ = z-self.H@self.x_                   # innovation
        S = self.R+ self.H@self.P_@self.H.T     # innovation covariance 
        K = self.P_@self.H@np.linalg.inv(S)     # Kalman gain
        self.x = self.x_ + K@y_                 # update state
        self.P = self.P_ - K@self.H@self.P_     # update state covariance

def cvDrawCross(image, pos:np.ndarray, color:tuple=(255,0,0), thickness:int=2):
    cv2.line(image, 
                                    (int(pos[0]-10), 
                                    int(pos[1]-10)),
                                    (int(pos[0]+10),
                                    int(pos[1]+10))
                                    ,color,thickness)
    cv2.line(image, 
                                    (int(pos[0]+10), 
                                    int(pos[1]-10)),
                                    (int(pos[0]-10),
                                    int(pos[1]+10))
                                    ,color,thickness)

def cvShowErrorEllipse(image, chisquare_val:float, mean:np.ndarray, covmat:np.ndarray, color:tuple=(255,0,0), thickness:int=2):
	[retval, eigenvalues, eigenvectors] = cv2.eigen(covmat)

	#Calculate the angle between the largest eigenvector and the x-axis
	angle = np.arctan2(eigenvectors[0,1], eigenvectors[0,0])

	#Shift the angle to the [0, 2pi] interval instead of [-pi, pi]
	if(angle < 0):
		angle += 6.28318530718

	# Conver to degrees instead of radians
	angle = 180*angle/3.14159265359

	# Calculate the size of the minor and major axes
	halfmajoraxissize=chisquare_val*np.sqrt(eigenvalues[0])
	halfminoraxissize=chisquare_val*np.sqrt(eigenvalues[1])

	cv2.ellipse(image,(int(mean[0]),int(mean[1])),(int(halfmajoraxissize), int(halfminoraxissize)), angle, 0, 360, color, thickness)


def cvOverlayImageWirthAlpha(img:np.ndarray, img_overlay:np.ndarray, x:int, y:int, w:int, h:int, alpha_mask:float)->None:
    """Overlays an image on another image (uses Pillow)

    Args:
        img (np.ndarray): Background image
        img_overlay (np.ndarray): Overlay image
        x (int): x position
        y (int): y position
        w (int): Width
        h (int): Height
        alpha_mask (float): Alpha value
    """
    pimg = Image.fromarray(np.uint8(img))
    pimg_overlay = Image.fromarray(np.uint8(img_overlay))
    alpha = Image.fromarray((np.array(pimg_overlay.split()[-1])*alpha_mask).astype(np.uint8))
    pimg_overlay.putalpha(alpha)
    pimg_overlay = pimg_overlay.resize((w,h))
    pimg.paste(pimg_overlay, (int(x), int(y), int(x+w), int(y+h)), pimg_overlay)

    img[:]= np.array(pimg)



def pilDrawCross(image:Image, pos:np.ndarray, color:tuple=(255,0,0), thickness:int=2):
    draw = ImageDraw.Draw(image)

    draw.line([(int(pos[0]-10), int(pos[1]-10)),
               (int(pos[0]+10), int(pos[1]+10))]
                                    ,color,thickness)
    draw.line([(int(pos[0]+10), int(pos[1]-10)),
               (int(pos[0]-10), int(pos[1]+10))]
                                    ,color,thickness)

def pilShowErrorEllipse(pimg:Image, chisquare_val:float, mean:np.ndarray, covmat:np.ndarray, color:tuple=(255,0,0), thickness:int=2):
    [retval, eigenvalues, eigenvectors] = cv2.eigen(covmat)

    #Calculate the angle between the largest eigenvector and the x-axis
    angle = np.arctan2(eigenvectors[0,1], eigenvectors[0,0])
    #Shift the angle to the [0, 2pi] interval instead of [-pi, pi]
    if(angle < 0):
        angle += 6.28318530718

    # Conver to degrees instead of radians
    angle = 180*angle/3.14159265359

    # Calculate the size of the minor and major axes
    halfmajoraxissize=chisquare_val*np.sqrt(eigenvalues[0])
    halfminoraxissize=chisquare_val*np.sqrt(eigenvalues[1])
    cvImage = np.array(pimg)
    cv2.ellipse(cvImage,(int(mean[0]),int(mean[1])),(int(halfmajoraxissize), int(halfminoraxissize)), angle, 0, 360, color, thickness)
    return Image.fromarray(np.uint8(cvImage))



def pilOverlayImageWirthAlpha(pimg:Image, pimg_overlay:Image, x:int, y:int, w:int, h:int, alpha_mask:float)->None:
    """Overlays an image on another image (uses Pillow)

    Args:
        img (Image): Background image
        img_overlay (Image): Overlay image
        x (int): x position
        y (int): y position
        w (int): Width
        h (int): Height
        alpha_mask (float): Alpha value
    """

    #pimg_overlay.putalpha(int(255*alpha_mask))
    pimg_overlay = pimg_overlay.resize((w,h))
    alpha = Image.fromarray((np.array(pimg_overlay.split()[-1])*alpha_mask).astype(np.uint8))
    pimg_overlay.putalpha(alpha)
    pimg.paste(pimg_overlay, (int(x), int(y), int(x+w), int(y+h)), pimg_overlay)

    