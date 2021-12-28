"""=== Face Analyzer Helpers =>
    Module : ui
    Author : Saifeddine ALOUI (ParisNeo)
    Licence : MIT
    Description :
        Euclidian geometry helpers
<================"""
import numpy as np
import cv2


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

def get_z_line_equation(pos: np.ndarray, ori:np.ndarray):
    """A line is defined by x = p0_x+v_xt
                            y = p0_y+v_yt
                            z = p0_z+v_zt
        The z line is the line pointing forward 'if we consider the face as 2d plane, z line is normal to that plane
    Args:
        pos (np.ndarray): reference position (coordinate of the point at t=0)
        ori (np.ndarray): orientation of the line

    Returns:
        (tuple): The equation of the line through (p0,v)
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

def get_plane_line_intersection(plane:tuple, line:tuple):
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

def is_point_inside_rect(point: tuple, rect:tuple):
    """Returns whether a point is inside a rectangular region

    Args:
        point (tuple): The point to be tested
        rect (tuple): A rectangular region coordinates (up_left_x,upleft_y,bottom_right_x,bottom_right_y)
    Returns:
        boolean: If true then the point is inside the rectangle else false
    """
    # Now let's check that the poit is inside the region
    return point[0]>rect[0] and point[0]<rect[2] and point[1]>rect[1] and point[1]<rect[3] 
