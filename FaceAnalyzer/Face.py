# -*- coding: utf-8 -*-
"""=== Face Analyzer =>
    Module : Face
    Author : Saifeddine ALOUI (ParisNeo)
    Licence : MIT
    Description :
        Face data holder (landmarks, posture ...). Allows the extraction of multiple facial features out of landmarks.
<================"""


import re
from typing import NamedTuple, Tuple
import numpy as np
import mediapipe as mp
import cv2
from scipy.signal import butter, filtfilt
import math
import time
from PIL import Image
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R


from .Helpers import buildCameraMatrix, get_plane_infos, get_z_line_equation, get_plane_line_intersection

# Get an instance of drawing specs to be used for drawing masks on faces
DrawingSpec =  mp.solutions.drawing_utils.DrawingSpec
class Face():    
    """Face is the class that provides operations on face landmarks.
    It is extracted by the face analyzer and could then be used for multiple face features extraction purposes
    """


    # Key landmark indices
    nose_tip_index = 4

    left_eyelids_indices = [
                            362,  # right 
                            374,  # bottom
                            263,  # left
                            386   # top
                            ]

    left_eye_contour_indices = [474, 475, 476, 477]
    left_eye_center_index = 473

    right_eyelids_indices = [
                                130, # right
                                145, # bottom
                                133, # left
                                159  # top
                            ]
    right_eye_contour_indices = [
                                    469, 
                                    470, 
                                    471, 
                                    472
                                ]
    right_eye_center_index = 468    

    # Nose, left face_extremety, right_face_extremity
    


    eyes_3d_reference_positions=np.array([
        [-50,50,-70],        # left Eye iris
        [50,50,-70],         # right Eye iris
        [-50,55.5,-70],        # left Eye top
        [50,55.5,-70],        # right Eye top
        [-50,44.5,-70],        # left Eye bottom
        [50,44.5,-70],        # right Eye bottom

        [-55.5,50,-70],        # left Eye left
        [-45.5,50,-70],          # left Eye right

        [45.5,50,-70],        # right Eye left
        [55.5,50,-70],          # right Eye right
    ])

    eyes_orientation_landmarks = [
        473,           # left Eye iris
        468,           # right Eye iris
        475,           # left Eye top
        470,           # right Eye top
        477,           # left Eye bottom
        472,           # left Eye bottom

        474,           # left Eye left        
        476,           # left Eye right        

        469,           # right Eye left
        471,           # right Eye right
        ]    

    # A list of simplified facial features used to reduce computation cost of drawing and morphing faces
    simplified_face_features = [
        10, 67, 54, 162, 127, 234, 93, 132,172,150,176,148,152,377,378,365,435,323,447,454,264,389,251, 332, 338, #Oval
        139, 105, 107, 151, 8, 9, 336, 334, 368,                            #  Eyelids
        130, 145, 155, 6, 382, 374, 359, 159, 386,                  #  Eyes
        129, 219, 79, 238, 2, 458, 457, 439, 358, 1, 4, 5, 197,     #  Nose
        61, 84, 314, 409, 14, 87, 81, 12,37,267, 402, 311, 321, 269, 39, 415, 91, 178, 73, 303, 325,
        50, 207, 280, 427
    ]
    all_face_features = list(range(468))
    def __init__(self, landmarks:NamedTuple = None, image_shape: tuple = (640, 480)):
        """Creates an instance of Face

        Args:
            landmarks (NamedTuple, optional): Landmarks object extracted by mediapipe tools
            image_shape (tuple, optional): The width and height of the image used to extract the face. Required to get landmarks in the right pixel size (useful for face copying and image operations). Defaults to (480, 640).
        """
        self.image_shape = image_shape

        self.update(landmarks)


        self.blinking = False

        self.face_oval = list(set(
            list(sum(list(mp.solutions.face_mesh.FACEMESH_FACE_OVAL), ()))))

        self.face_contours = list(set(
            list(sum(list(mp.solutions.face_mesh.FACEMESH_CONTOURS), ()))[::3]
        ))

        # Initialize face information
        self.pos = None
        self.ori = None
        # Initialize face information
        self.eyes_pos = None
        self.eyes_ori = None

        self.reference_facial_cloud = None

        self.mp_drawing = mp.solutions.drawing_utils
        # Three points were removed from my initial code (I leave them for tests) as they seem to be affected by grimacing (the chin) or are not very accurate (Left and Right)
        self.face_3d_reference_positions=np.array([
        [0,0,0],            # Nose tip
        #[-80,50,-90],       # Left
        #[0,-70,-30],        # Chin
        #[80,50,-90],        # Right
        [-70,50,-70],       # Left left eye
        [70,50,-70],        # Right right eye
        [0,80,-30]        # forehead center
        ])

        # these points was chosen so that the mouth motion and eye closong do not affect them

        self.face_reference_landmark_ids = [
            4,          # Nose tip
            #127,        # Left
            #152,        # Chin
            #264,        # Right
            130,        # Left left eye
            359,        # Right right eye
            151         # forehead center
            ]


    @property
    def ready(self)->bool:
        """Returns if the face has landmarks or not

        Returns:
            bool: True if the face has landmarks
        """
        return self.landmarks is not None

    def update(self, landmarks:NamedTuple)->None:
        """Updates the landmarks of the face

        Args:
            landmarks (NamedTuple): The nex landmarks
        """
        if landmarks is not None:
            self.landmarks = landmarks
            self.npLandmarks = np.array([[lm.x * self.image_shape[0], lm.y * self.image_shape[1], lm.z * self.image_shape[0]] for lm in landmarks.landmark])
        else:
            self.landmarks = None
            self.npLandmarks = np.array([])




    def get_left_eye_width(self)->float:
        """Gets the left eye width

        Returns:
            float: The width of the left eye
        """

        # Assertion to verify that the face object is ready
        assert self.ready, "Face object is not ready. There are no landmarks extracted."

        p1 = self.getlandmark_pos(self.left_eye_contour_indices[2])
        p2 = self.getlandmark_pos(self.left_eye_contour_indices[0])
        return np.abs(p2[0] - p1[0])

    def get_left_eye_height(self):
        """Gets the left eye height

        Returns:
            float: The height of the left eye
        """        

        # Assertion to verify that the face object is ready
        assert self.ready, "Face object is not ready. There are no landmarks extracted."

        p1 = self.getlandmark_pos(self.left_eye_contour_indices[3])
        p2 = self.getlandmark_pos(self.left_eye_contour_indices[1])
        return np.abs(p2[1] - p1[1])

    def get_right_eye_width(self):
        """Gets the right eye width

        Returns:
            float: The width of the right eye
        """        

        # Assertion to verify that the face object is ready
        assert self.ready, "Face object is not ready. There are no landmarks extracted."

        p1 = self.getlandmark_pos(self.right_eye_contour_indices[2])
        p2 = self.getlandmark_pos(self.right_eye_contour_indices[0])
        return np.abs(p2[0] - p1[0])

    def get_right_eye_height(self):
        """Gets the right eye height

        Returns:
            float: The height of the left eye
        """        

        # Assertion to verify that the face object is ready
        assert self.ready, "Face object is not ready. There are no landmarks extracted."

        p1 = self.getlandmark_pos(self.right_eye_contour_indices[3])
        p2 = self.getlandmark_pos(self.right_eye_contour_indices[1])
        return np.abs(p2[1] - p1[1])

    def getlandmark_pos(self, index) -> Tuple:
        """Recovers the position of a landmark from a results array

        Args:
            index (int): Index of the landmark to recover

        Returns:
            Tuple: Landmark 3D position in image space
        """

        # Assertion to verify that the face object is ready
        assert self.ready, "Face object is not ready. There are no landmarks extracted."

        lm = self.npLandmarks[index, ...]
        return (lm[0], lm[1], lm[2])


    def getlandmarks_pos(self, indices: list) -> np.ndarray:
        """Recovers the position of a landmark from a results array

        Args:
            indices (list): List of indices of landmarks to extract

        Returns:
            np.ndarray: A nX3 array where n is the number of landmarks to be extracted and 3 are the 3 cartesian coordinates
        """

        # Assertion to verify that the face object is ready
        assert self.ready, "Face object is not ready. There are no landmarks extracted."

        return self.npLandmarks[indices,...]

    def draw_landmark_by_index(self, image: np.ndarray, index: int, color: tuple = (255, 0, 0), radius: int = 5, thickness:int=1) -> np.ndarray:
        """Draw a landmark on an image from landmark index

        Args:
            image (np.ndarray): Image to draw the landmark on
            index (int): Index of the landmark
            color (tuple, optional): Color of the landmark. Defaults to (255, 0, 0).
            radius (int, optional): Radius of the circle to draw the landmark. Defaults to 5.
            thickness (int, optional): Thickness of the line to draw the landmark. Defaults to 5.

        Returns:
            np.ndarray: Output image
        """
        pos = self.npLandmarks[index,:]
        return cv2.circle(
            image,(int(pos[0]), int(pos[1])), radius, color, thickness
        )


    def draw_landmarks(self, image: np.ndarray, landmarks: np.ndarray, radius:int=1, color: tuple = (255, 0, 0), thickness: int = 1) -> np.ndarray:
        """Draw a list of landmarks on an image

        Args:
            image (np.ndarray): Image to draw the contour on
            landmarks (np.ndarray): a nX3 ndarray containing the positions of the landmarks
            radius (int, optional): Radius of the circle to draw the landmark. Defaults to 5.
            color (tuple, optional): Color of the landmark. Defaults to (255, 0, 0).
            thickness (int, optional): Thickness of the line to draw the landmark. Defaults to 5.


        Returns:
            np.ndarray: The image with the contour drawn on it
        """
        for i in range(landmarks.shape[0]):
            image = cv2.circle(image, (int(landmarks[i,0]), int(landmarks[i,1])), radius,color, thickness)
        return image

    def draw_landmark(self, image: np.ndarray, pos: tuple, color: tuple = (255, 0, 0), radius: int = 5, thickness:int=1) -> np.ndarray:
        """Draw a landmark on an image

        Args:
            image (np.ndarray): Image to draw the landmark on
            pos (tuple): Position of the landmark
            color (tuple, optional): Color of the landmark. Defaults to (255, 0, 0).
            radius (int, optional): Radius of the circle to draw the landmark. Defaults to 5.
            thickness (int, optional): Thickness of the line to draw the landmark. Defaults to 5.

        Returns:
            np.ndarray: Output image
        """
        return cv2.circle(
            image,(int(pos[0]), int(pos[1])), radius, color, thickness
        )

    def draw_contour(self, image: np.ndarray, contour: np.ndarray, color: tuple = (255, 0, 0), thickness: int = 1) -> np.ndarray:
        """Draw a contour on an image

        Args:
            image (np.ndarray): Image to draw the contour on
            contour (np.ndarray): a nX3 ndarray containing the positions of the landmarks
            color (tuple, optional): Color of the landmark. Defaults to (255, 0, 0).
            thickness (int, optional): Thickness of the line to draw the landmark. Defaults to 5.


        Returns:
            np.ndarray: The image with the contour drawn on it
        """

        pts = np.array([[int(p[0]), int(p[1])] for p in contour.tolist()]).reshape((-1, 1, 2))
        return cv2.polylines(image, [pts], True, color, thickness)

    def draw_overlay_on_left_iris(self, image:np.ndarray, overlay:np.ndarray)->np.ndarray:
        """Draws an overlay image on the left iris of the face

        Args:
            image (np.ndarray): Image to draw the overlay on
            overlay (np.ndarray): The overlay image to be drawn (support rgba format for transparency)

        Returns:
            np.ndarray: The image with the overlay drawn
        """

        # Assertion to verify that the face object is ready
        assert self.ready, "Face object is not ready. There are no landmarks extracted."



        pImage = Image.fromarray(image)
        pos = self.getlandmark_pos(self.left_eye_center_index)[0:2]

        w = int(self.get_left_eye_width())
        h = int(self.get_left_eye_height())

        if w > 0 and h > 0:
            overlay_ = overlay.resize((w, h), Image.ANTIALIAS)
            x = int(pos[0] - overlay_.size[0] / 2)
            y = int(pos[1] - overlay_.size[1] / 2)
            pImage.paste(overlay_, (x, y), overlay_)
        return np.array(pImage).astype(np.uint8)

    def draw_overlay_on_right_iris(self, image:np.ndarray, overlay:np.ndarray)->np.ndarray:
        """Draws an overlay image on the right iris of the face

        Args:
            image (np.ndarray): Image to draw the overlay on
            overlay (np.ndarray): The overlay image to be drawn (support rgba format for transparency)

        Returns:
            np.ndarray: The image with the overlay drawn
        """

        # Assertion to verify that the face object is ready
        assert self.ready, "Face object is not ready. There are no landmarks extracted."

        pImage = Image.fromarray(image)
        pos = self.getlandmark_pos(self.right_eye_center_index)[0:2]

        w = int(self.get_right_eye_width())
        h = int(self.get_right_eye_height())

        if w > 0 and h > 0:
            overlay_ = overlay.resize((w, h), Image.ANTIALIAS)
            x = int(pos[0] - overlay_.size[0] / 2)
            y = int(pos[1] - overlay_.size[1] / 2)
            pImage.paste(overlay_, (x, y), overlay_)
        return np.array(pImage).astype(np.uint8)

    def get_head_posture(self, camera_matrix:np.ndarray = None, dist_coeffs:np.ndarray=np.zeros((4,1)))->tuple:
        """Gets the posture of the head (position in cartesian space and Euler angles)
        Args:
            camera_matrix (int, optional)       : The camera matrix built using buildCameraMatrix Helper function. Defaults to a perfect camera matrix 
            dist_coeffs (np.ndarray, optional)) : The distortion coefficients of the camera
        Returns:
            tuple: (position, orientation) the orientation is either in compact rodriguez format (angle * u where u is the rotation unit 3d vector representing the rotation axis). Feel free to use the helper functions to convert to angles or matrix
        """

        # Assertion to verify that the face object is ready
        assert self.ready, "Face object is not ready. There are no landmarks extracted."

        if camera_matrix is None:
            camera_matrix= buildCameraMatrix()

        # V2 : Use opencv's PnPsolver to solve the rotation problem

        face_2d_positions = self.npLandmarks[self.face_reference_landmark_ids,:2]
        (success, face_ori, face_pos, _) = cv2.solvePnPRansac(
                                                    self.face_3d_reference_positions.astype(np.float),
                                                    face_2d_positions.astype(np.float), 
                                                    camera_matrix, 
                                                    dist_coeffs,
                                                    flags=cv2.SOLVEPNP_ITERATIVE)

        if not success:
            return None, None

        # save posture
        self.pos = face_pos
        self.ori = face_ori

        return face_pos, face_ori


    def get_eyes_position(self)->tuple:
        """Gets the posture of the eyes (position in cartesian space and Euler angles)
        Args:
            camera_matrix (int, optional)       : The camera matrix built using buildCameraMatrix Helper function. Defaults to a perfect camera matrix 
            dist_coeffs (np.ndarray, optional)) : The distortion coefficients of the camera
        Returns:
            tuple: (position, orientation) the orientation is either in compact rodriguez format (angle * u where u is the rotation unit 3d vector representing the rotation axis). Feel free to use the helper functions to convert to angles or matrix
        """

        # Assertion to verify that the face object is ready
        assert self.ready, "Face object is not ready. There are no landmarks extracted."

        # Left eye
        iris = np.array(self.getlandmark_pos(Face.left_eye_center_index))
        left = np.array(self.getlandmark_pos(263))
        right = np.array(self.getlandmark_pos(362))
        top = np.array(self.getlandmark_pos(386))
        bottom = np.array(self.getlandmark_pos(374))

        center = (left+right)/2
        ex = left-right
        ey = top-bottom
        nx = np.linalg.norm(ex)
        ny = np.linalg.norm(ey)
        ex /=nx
        ey /=ny
        nx/=2
        ny/=2
        left_pos = np.array([np.dot((iris-center),ex)/nx,np.dot((iris-center),ey)/ny])

        # right
        iris = np.array(self.getlandmark_pos(Face.right_eye_center_index))
        left = np.array(self.getlandmark_pos(133))
        right = np.array(self.getlandmark_pos(130))
        top = np.array(self.getlandmark_pos(159))
        bottom = np.array(self.getlandmark_pos(145))

        center = (left+right)/2
        ex = left-right
        ey = top-bottom
        nx = np.linalg.norm(ex)
        ny =np.linalg.norm(ey)
        ex /=nx
        ey /=ny
        nx/=2
        ny/=2

        right_pos = [np.dot((iris-center),ex)/nx,np.dot((iris-center),ey)/ny]

        return left_pos, right_pos

    def compose_eye_rot(self, eye_pos:list, face_orientation:np.ndarray, x2ang: int=180, y2ang:int=30)->np.ndarray:
        """Composes eye position with face rotation to produce eye orientation in world coordinates

        Args:
            eye_pos (list): The local normalized eye position
            face_orientation (np.ndarray): The orientation of the face in compressed axis angle format
            x2ang (int, optional): A coefficient to convert normalized position to angle in X axis. Defaults to 180.
            y2ang (int, optional): A coefficient to convert normalized position to angle in Y axis. Defaults to 30.

        Returns:
            np.ndarray: [description]
        """
        fo = R.from_rotvec(face_orientation[:,0])
        ypr = R.from_euler('yxz',[-eye_pos[0]*x2ang,-eye_pos[1]*y2ang,0], degrees=True)
        return np.array((ypr*fo).as_rotvec()).reshape((3,1))


    def getEyesDist(self)->int:
        """Gets the distance between the two eyes

        Returns:
            int: The distance between the two eyes
        """

        # Assertion to verify that the face object is ready
        assert self.ready, "Face object is not ready. There are no landmarks extracted."


        pos = self.getlandmarks_pos([self.left_eye_center_index, self.right_eye_center_index])
        return np.linalg.norm(pos[1,:]-pos[0,:])

    def process_eyes(self, image: np.ndarray, normalize:bool=False, detect_blinks: bool = False, blink_th:float=5, blinking_double_threshold_factor:float=1.05, draw_landmarks: bool = False)->tuple:
        """Process eye information and extract eye opening value, normalized eye opening and detect blinks

        Args:
            image (np.ndarray): Image to draw on when landmarks are to be drawn
            normalize (bool, optional): If True, the eye opening will be normalized by the distance between the eyes. Defaults to False.
            detect_blinks (bool, optional): If True, blinks will be detected. Defaults to False.
            blink_th (float, optional): Blink threshold. Defaults to 5.
            blinking_double_threshold_factor (float, optional): a factor for double blinking threshold detection. 1 means that the threshold is the same for closing and opening. If you put 1.2, it means that after closing, the blinking is considered finished only when the opening surpssess the blink_threshold*1.2. Defaults to 1.05.
            draw_landmarks (bool, optional): If True, the landmarks will be drawn on the image. Defaults to False.

        Returns:
            tuple: Depending on what configuration was chosen in the parameters, the output is:
            left_eye_opening, right_eye_opening, is_blink if blinking detection is activated
            left_eye_opening, right_eye_opening if blinking detection is deactivated
        """

        # Assertion to verify that the face object is ready
        assert self.ready, "Face object is not ready. There are no landmarks extracted."


        left_eye_center = self.getlandmark_pos(self.left_eye_center_index)
        left_eyelids_contour = self.getlandmarks_pos(self.left_eyelids_indices)
        left_eye_upper = left_eyelids_contour[3, ...]
        left_eye_lower = left_eyelids_contour[1, ...]

        left_eye_contour = self.getlandmarks_pos(self.left_eye_contour_indices)
        left_eye_iris_upper = left_eye_contour[3, ...]
        left_eye_iris_lower = left_eye_contour[1, ...]

        right_eye_center = self.getlandmark_pos(self.right_eye_center_index)
        right_eyelids_contour = self.getlandmarks_pos(self.right_eyelids_indices)
        right_eye_upper = right_eyelids_contour[3, ...]
        right_eye_lower = right_eyelids_contour[1, ...]

        right_eye_contour = self.getlandmarks_pos(self.right_eye_contour_indices)
        right_eye_iris_upper = right_eye_contour[1, ...]
        right_eye_iris_lower = right_eye_contour[3, ...]




        if draw_landmarks:

            image = self.draw_landmark(image, left_eye_upper, (0, 0, 255),1)
            image = self.draw_landmark(image, left_eye_lower, (0, 0, 255),1)


            image = self.draw_landmark(image, left_eye_iris_upper, (255, 0, 0),1)
            image = self.draw_landmark(image, left_eye_iris_lower, (255, 0, 0),1)

            image = self.draw_landmark(image, right_eye_upper, (0, 0, 255),1)
            image = self.draw_landmark(image, right_eye_lower, (0, 0, 255),1)

            image = self.draw_landmark(image, right_eye_iris_upper, (255, 0, 0),1)
            image = self.draw_landmark(image, right_eye_iris_lower, (255, 0, 0),1)            

            image = self.draw_contour(image, left_eyelids_contour, (0, 0, 0),2)
            image = self.draw_contour(image, left_eye_contour, (0, 0, 0),2)

            image = self.draw_landmark(image, left_eye_center, (255, 0, 255),1)

            image = self.draw_contour(image, right_eyelids_contour, (0, 0, 0),2)
            image = self.draw_contour(image, right_eye_contour, (0, 0, 0),2)

            image = self.draw_landmark(image, right_eye_center, (255, 0, 255),1)



        # Compute eye opening
        left_eye_opening = np.linalg.norm(left_eye_upper[0:2]-left_eye_lower[0:2])/np.linalg.norm(left_eye_iris_upper[0:2]-left_eye_iris_lower[0:2])
        right_eye_opening = np.linalg.norm(right_eye_upper[0:2]-right_eye_lower[0:2])/np.linalg.norm(right_eye_iris_upper[0:2]-right_eye_iris_lower[0:2])


        if normalize:
            ed = self.getEyesDist()
            left_eye_opening /= ed
            right_eye_opening /= ed
            th = blink_th / ed
        else:
            th = blink_th

        if detect_blinks:
            is_blink = False
            eye_opening = (left_eye_opening+right_eye_opening)/2
            if eye_opening < th and not self.blinking:
                self.blinking = True
                is_blink = True
            elif eye_opening > th*blinking_double_threshold_factor:
                self.blinking = False

            return left_eye_opening, right_eye_opening, is_blink
        else:
            return left_eye_opening, right_eye_opening


    # ======================== Face copying, and morphing ====================

    def triangulate(self, landmark_indices:list=None)->list:
        """Builds triangles using Denaulay triangulation algorithm

        Args:
            landmark_indices (list, optional): List of landmark indices to be used for triangulation. If None, all landmarks will be used. Defaults to None.

        Returns:
            list: A list of triangles extracted by Denaulay algorithm
        """

        # Assertion to verify that the face object is ready
        assert self.ready, "Face object is not ready. There are no landmarks extracted."

        if landmark_indices is None:
            landmarks = self.npLandmarks[:, :2]
        else:
            landmarks = self.npLandmarks[landmark_indices, :2]
        tri = Delaunay(landmarks)
        self.triangles = tri.simplices
        return tri.simplices

    def rect_contains(self, rect:tuple, point:tuple)->bool:
        """Tells whether a point is inside a rectangular region

        Args:
            rect (tuple): The rectangle coordiantes (topleft , bottomright)
            point (tuple): The point position (x,y)

        Returns:
            bool: True if the point is inside the rectangular region
        """
        if point[0] < rect[0]:
            return False
        elif point[1] < rect[1]:
            return False
        elif point[0] > rect[2]:
            return False
        elif point[1] > rect[3]:
            return False
        return True

    def draw_delaunay(
                        self, 
                        img:np.ndarray, 
                        landmark_indices:list=None, 
                        delaunay_colors:list=[[255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255]], 
                        thickness:int=1
                    )->np.ndarray:

        """Draws denaulay triangles aon an image
        Args:
            image (np.ndarray): Image to draw on
            landmark_indices (list, optional): List of landmark indices to be used for triangulation. If None, all landmarks will be used. Defaults to None.
            delaunay_colors (list, optional): List of colors to use for drawing the triangles. Defaults to None.
            thickness (int): The point position (x,y)

        Returns:
            np.ndarray: Image with triangles drawn on it
        """

        # Assertion to verify that the face object is ready
        assert self.ready, "Face object is not ready. There are no landmarks extracted."

                
        if landmark_indices is None:
            landmarks = self.npLandmarks[:, :2]
        else:
            landmarks = self.npLandmarks[landmark_indices, :2]

        triangleList = self.triangles
        size = img.shape
        r = (0, 0, size[1], size[0])
        ncolors = len(delaunay_colors)

        for (i, t) in enumerate(triangleList):

            pt1 = (int(landmarks[t[0], 0]), int(landmarks[t[0], 1]))
            pt2 = (int(landmarks[t[1], 0]), int(landmarks[t[1], 1]))
            pt3 = (int(landmarks[t[2], 0]), int(landmarks[t[2], 1]))

            delaunay_color = delaunay_colors[i % ncolors]
            if self.rect_contains(r, pt1) and self.rect_contains(r, pt2) and self.rect_contains(r, pt3):
                img = cv2.line(img, pt1, pt2, delaunay_color, thickness)
                img = cv2.line(img, pt2, pt3, delaunay_color, thickness)
                img = cv2.line(img, pt3, pt1, delaunay_color, thickness)

        return img

    def getFace(self, image:np.ndarray, src_triangles:list(), landmark_indices:list=None)->np.ndarray:
        """Gets an image of the face extracted from the original image

        Args:
            image (np.ndarray): Image to extract the face from
            src_triangles (list): The delaulay triangles indices (look at triangulate)
            landmark_indices (list, optional): The list of landmarks to be used (the same list used for the triangulate method that allowed the extraction of the triangles). Defaults to None.

        Returns:
            np.ndarray: Face drawn on a black background (the size of the image is equal of that of the face in the original image)
        """

        # Assertion to verify that the face object is ready
        assert self.ready, "Face object is not ready. There are no landmarks extracted."

        if landmark_indices is None:
            landmarks = self.npLandmarks[:, :2]
        else:
            landmarks = self.npLandmarks[landmark_indices, :2]
        p1 = landmarks.min(axis=0)
        p2 = landmarks.max(axis=0)

        landmarks -= p1
        croped = image[int(p1[1]):int(
            p2[1]), int(p1[0]):int(p2[0])]

        dest = np.zeros_like(croped)
        for tr in src_triangles:
            mask = np.zeros_like(dest)
            n_mask = np.ones_like(dest)
            try:
                t_dest = np.array([landmarks[tr[0], 0:2], landmarks[tr[1], 0:2], landmarks[tr[2], 0:2]])
                cv2.fillConvexPoly(mask, t_dest.astype(np.int), [1, 1, 1])
                cv2.fillConvexPoly(n_mask, t_dest.astype(np.int), [0, 0, 0])
                dest = cv2.multiply(dest, n_mask) + cv2.multiply(mask, croped)
            except Exception as ex:
                pass
        return dest

    def copyToFace(self, dst_face, src_image, dst_image:np.ndarray, landmark_indices:list=None, opacity:float=1.0)->np.ndarray:
        """Copies the face to another image (used for face copy or face morphing)

        Args:
            dst_face (Face): A face object describing the face in the destination image (triangulate function should have been called on this object with the same landmark_indices argument)
            src_image ([type]): [description]
            dst_image (np.ndarray): [description]
            landmark_indices (list, optional): The list of landmarks to be used (the same list used for the triangulate method that allowed the extraction of the triangles). Defaults to None.
            opacity (int, optional): the opacity level of the face (between 0 and 1)

        Returns:
            np.ndarray: An image containing only the face
        """

        # Assertion to verify that the face object is ready
        assert self.ready, "Face object is not ready. There are no landmarks extracted."

        # Get landmarks
        if landmark_indices is None:
            src_landmarks = self.npLandmarks[:, :2]
            dst_landmarks = dst_face.npLandmarks[:, :2]
        else:
            src_landmarks = self.npLandmarks[landmark_indices, :2]
            dst_landmarks = dst_face.npLandmarks[landmark_indices, :2]

        # Crop images
        src_p1 = src_landmarks.min(axis=0)
        src_p2 = src_landmarks.max(axis=0)

        src_landmarks -= src_p1
        src_crop = src_image[int(src_p1[1]):int(
            src_p2[1]), int(src_p1[0]):int(src_p2[0])]

        dst_p1 = dst_landmarks.min(axis=0)
        dst_p2 = dst_landmarks.max(axis=0)

        dst_landmarks -= dst_p1
        dst_crop = dst_image[int(dst_p1[1]):int(
            dst_p2[1]), int(dst_p1[0]):int(dst_p2[0])]

        # Prepare empty image
        dest = np.zeros((int(dst_p2[1])-int(dst_p1[1]),int(dst_p2[0])-int(dst_p1[0]),3))
        dst_h, dst_w, _ = dest.shape
        center = (dst_w//2,dst_h//2)

        for src_tr in self.triangles:
            mask = np.zeros_like(dest)
            n_mask = np.ones_like(dest)
            try:
                t_src = np.array(
                    [src_landmarks[src_tr[0], 0:2], src_landmarks[src_tr[1], 0:2], src_landmarks[src_tr[2], 0:2]])
                t_dest = np.array(
                    [dst_landmarks[src_tr[0], 0:2], dst_landmarks[src_tr[1], 0:2], dst_landmarks[src_tr[2], 0:2]])

                cv2.fillConvexPoly(mask, t_dest.astype(np.int), [1, 1, 1])
                cv2.fillConvexPoly(n_mask, t_dest.astype(np.int), [0, 0, 0])

                M = cv2.getAffineTransform(
                    t_src.astype(np.float32),
                    t_dest.astype(np.float32)
                )
                warped = cv2.warpAffine(src_crop,  # src_image,
                                        M,
                                        (dst_w, dst_h)
                                        )
                dest = cv2.multiply(dest, n_mask) + cv2.multiply(np.float64(warped), mask)
            except Exception as ex:
                pass
        mask = cv2.threshold(dest.astype(np.uint8), 1, opacity*255, cv2.THRESH_BINARY)[1]
        try:
            dst_crop = cv2.seamlessClone(dest.astype(np.uint8), dst_crop, mask, (int(center[0]),int(center[1])), cv2.NORMAL_CLONE)
        except:
            pass
        dst_image[int(dst_p1[1]):int(dst_p2[1]), int(dst_p1[0]):int(dst_p2[0])] = dst_crop
        return dst_image

    def draw_bounding_box(self, image:np.ndarray, color:tuple=(255,0,0), thickness:int=1):
        """Draws a bounding box around the face

        Args:
            image (np.ndarray): The image on which we will draw the bounding box
            color (tuple, optional): The color of the bounding box. Defaults to (255,0,0).
            thickness (int, optional): The line thickness. Defaults to 1.
        """
        pt1 = self.npLandmarks.min(axis=0)
        pt2 = self.npLandmarks.max(axis=0)
        cv2.rectangle(image, (int(pt1[0]),int(pt1[1])), (int(pt2[0]),int(pt2[1])), color, thickness)

    def draw_mask(self, 
                    image:np.ndarray, 
                    landmarks_drawing_spec:DrawingSpec = DrawingSpec(color=(121, 0, 0), thickness=1, circle_radius=1), 
                    contours_drawing_specs:DrawingSpec = DrawingSpec(color=(0, 0, 121), thickness=1, circle_radius=1),
                    contour:frozenset=mp.solutions.face_mesh.FACEMESH_FACE_OVAL
                    )->None:
        """Draws landmarks mask on a face

        Args:
            image (np.ndarray): Image to draw the mask on
        """

        # Assertion to verify that the face object is ready
        assert self.ready, "Face object is not ready. There are no landmarks extracted."


        self.mp_drawing.draw_landmarks(image, self.landmarks , contour,
                                       landmarks_drawing_spec,
                                       contours_drawing_specs
                                       )

    def draw_reference_frame(self, image:np.ndarray, pos: np.ndarray, ori:np.ndarray, origin:np.ndarray, translation:np.ndarray=None, line_length:int=50)->None:
        """Draws a reference frame at a sprecific position

        Args:
            image (np.ndarray): The image to draw the reference frame on.
            pos (np.ndarray): The real 3D position of the frame reference
            ori (np.ndarray): The orientation of the frame in compressed axis angle format
            origin (np.ndarray): The origin in camera frame where to draw the frame
            translation (np.ndarray, optional): A translation vector to draw the frame in a different position tha n the origin. Defaults to None.
            line_length (int, optional): The length of the frame lines (X:red,y:green,z:blue). Defaults to 50.
        """

        #Let's project three vectors ex,ey,ez to form a frame and draw it on the nose
        (end_point2D_x, jacobian) = cv2.projectPoints(np.array([(line_length, 0.0, 0.0)]), ori, pos, buildCameraMatrix(), np.zeros((4,1)))
        (end_point2D_y, jacobian) = cv2.projectPoints(np.array([(0.0, line_length, 0.0)]), ori, pos, buildCameraMatrix(), np.zeros((4,1)))
        (end_point2D_z, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, line_length)]), ori, pos, buildCameraMatrix(), np.zeros((4,1)))

        p1 = ( int(origin[0]), int(origin[1]))
        p2_x = ( int(end_point2D_x[0][0][0]), int(end_point2D_x[0][0][1]))         
        p2_y = ( int(end_point2D_y[0][0][0]), int(end_point2D_y[0][0][1]))         
        p2_z = ( int(end_point2D_z[0][0][0]), int(end_point2D_z[0][0][1]))   

        if translation is not None:
            p1=   (p1[0]+translation[0], p1[1]+translation[1])

            p2_x= (p2_x[0]+translation[0],p2_x[1]+translation[1])

            p2_y= (p2_y[0]+translation[0],p2_y[1]+translation[1])

            p2_z= (p2_z[0]+translation[0],p2_z[1]+translation[1])

        cv2.line(image, p1, p2_x, (255,0,0), 2)   
        cv2.line(image, p1, p2_y, (0,255,0), 2)   
        cv2.line(image, p1, p2_z, (0,0,255), 2)
    

    def is_pointing_to_2d_region(self, region:tuple, pos: np.ndarray, ori:np.ndarray):
        """Returns weather the face or eye is pointing inside a 2d region represented by the polygon 

        Args:
            region (tuple): A list of points in form of ndarray that represent the region (all points should belong to the same plan)
            pos (np.ndarray): The position of the face or eye
            ori (np.ndarray): The orientation of the face or eye

        Returns:
            boolean: If true then the face or eye is pointing to that region else false
        """
        assert(len(region)>=3,"Region should contain at least 3 points")
        # Copy stuff
        region = region.copy()
        # First find the pointing line, and the plan on which the region is selected
        pl = get_plane_infos(region[0],region[1],region[2])
        e1 = pl[2]
        e2 = pl[3]
        ln = get_z_line_equation(pos, ori)
        p, p2d = get_plane_line_intersection(pl, ln)
        # Lets put all the points of the region inside the 2d plane
        for i in range(len(region)):
            region[i]=np.array([np.dot(region[i], e1), np.dot(region[i], e2)])

        # Now let's check that the poit is inside the region
        in_range=True
        for i in range(len(region)):
            AB = region[(i+1)%len(region)]-region[i]
            AP = p2d-region[i]
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

        