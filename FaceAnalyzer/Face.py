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
from numpy import linalg
from numpy.lib.type_check import imag
from scipy.signal import butter, filtfilt
import math
import time
from PIL import Image
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R


from .helpers.geometry.euclidian import buildCameraMatrix, get_plane_infos, get_z_line_equation, get_plane_line_intersection
from .helpers.geometry.orientation import rotateLandmarks

# Get an instance of drawing specs to be used for drawing masks on faces
DrawingSpec =  mp.solutions.drawing_utils.DrawingSpec
class Face():    
    """Face is the class that provides operations on face landmarks.
    It is extracted by the face analyzer and could then be used for multiple face features extraction purposes
    """


    # Key landmark indices
    nose_tip_index = 4

    simplified_left_eyelids_indices = [
                            362,  # right 
                            374,  # bottom
                            263,  # left
                            386   # top
                            ]

    left_eyelids_indices = [
                            362,  # right 
                            382,
                            381,
                            380,
                            374,  # bottom
                            373,
                            390,
                            249,
                            263,  # left
                            466,
                            388,
                            387,
                            386,  # top
                            385,
                            384,
                            398
                            ]                            

    left_eye_contour_indices = [
                                    476, # Right
                                    477, # Bottom
                                    474, # Left
                                    475  # Top
                                ]
    left_eye_left_right_indices = [
                                    362, # Right
                                    263, # Left
                                ]                                
    left_eye_center_index = 473

    left_eye_orientation_landmarks=[442,450,362,263, 473]

    simplified_right_eyelids_indices = [
                                130, # right
                                145, # bottom
                                133, # left
                                159  # top
                            ]

    right_eyelids_indices = [
                                130, # right
                                7,
                                163,
                                144,
                                145, # bottom
                                153,
                                154,
                                155,
                                133, # left
                                173,
                                157,
                                158,
                                159, # top
                                160,
                                161,
                                246,
                                33
                            ]

    right_eye_contour_indices = [
                                    471, # right
                                    472, # bottom
                                    469, # left
                                    470  # top
                                ]
    right_eye_left_right_indices = [
                                    130, # Right
                                    133, # Left
                                ]                                   
    right_eye_center_index = 468    
    
    right_eye_orientation_landmarks=[223,230, 130, 133, 468]
    
    # Mouth
    simplified_mouth_outer_indices = [
                            61,  # right 
                            17,  # bottom
                            291,  # left
                            0   # top
                            ]

    mouth_outer_indices = [
                            61,  # right 
                            146,
                            91,
                            181,
                            84,
                            17,  # bottom
                            314,
                            405,
                            321,
                            375,
                            291,  # left
                            409,
                            270,
                            269,
                            267,
                            0,  # top
                            37,
                            39,
                            40,
                            185
                            ]      

    simplified_mouth_inner_indices = [
                            78,  # right 
                            14,  # bottom
                            308,  # left
                            13   # top
                            ]

    mouth_inner_indices = [
                            78,  # right 
                            95,
                            88,
                            178,
                            87,
                            14,  # bottom
                            317,
                            402,
                            318,
                            324,
                            308,  # left
                            415,
                            310,
                            311,
                            312,
                            13,  # top
                            82,
                            81,
                            80,
                            191
                            ]  


    simplified_nose_indices = [
                            129,  # right 
                            94,  # bottom
                            358,  # left
                            4   # top
                            ]

    nose_indices = [
                            129,  # right 
                            219,
                            166,
                            239,
                            20,
                            242,
                            141,
                            94,  # bottom
                            370,
                            462,
                            250,
                            459,
                            392,
                            439,
                            358,  # left
                            344,
                            440,
                            275,
                            4,  # top
                            45,
                            220,
                            115,
                            49,
                            131,
                            134,
                            51,
                            5,
                            281,
                            363,
                            360,
                            279,
                            429,
                            420,
                            456,
                            248,
                            195,
                            3,
                            236,
                            198,
                            209,
                            217,
                            174,
                            196,
                            197,
                            419,
                            399,
                            437,
                            343,
                            412,
                            351,
                            6,
                            122,
                            188,
                            114,
                            245,
                            193,
                            168,
                            417,
                            465
                            ]

    forehead_center_index = [151]
    
    forehead_indices = [
        301,
        298,
        333,
        299,
        337,
        151,
        108,
        69,
        104,
        68,
        71,
        21,
        54,
        103,
        67,
        109,
        10,
        338,
        297,
        332,
        284,
        251
    ]
    # Eye_brows
    left_eye_brows_indices = [
        
        285,
        295,
        282,
        283,
        276,
        383,
        300,
        293,
        334,
        296,

    ]
    right_eye_brows_indices = [

        156,
        46,
        53,
        52,
        65,
        55,
        66,
        105,
        63,
        70
    ]    
    # A list of simplified facial features used to reduce computation cost of drawing and morphing faces
    simplified_face_features = [
        10, 67, 54, 162, 127, 234, 93, 132,172,150,176,148,152,377,378,365,435,323,447,454,264,389,251, 332, 338, #Oval
        139, 105, 107, 151, 8, 9, 336, 334, 368,                            #  Eyelids
        130, 145, 155, 6, 382, 374, 263, 159, 386,                  #  Eyes
        129, 219, 79, 238, 2, 458, 457, 439, 358, 1, 4, 5, 197,     #  Nose
        61, 84, 314, 409, 14, 87, 81, 12,37,267, 402, 311, 321, 269, 39, 415, 91, 178, 73, 303, 325,
        50, 207, 280, 427
    ]

    face_oval_indices = [
        10,
        109,
        67,
        103,
        54,
        21,
        162,
        127,
        234,
        93,
        132,
        58,
        172,
        136,
        150,
        149,
        176,
        148,
        152,
        377,
        400,
        378,
        379,
        365,
        397,
        288,
        361,
        323,
        454,
        356,
        389,
        251,
        284,
        332,
        297,
        338
    ]

    face_forhead_indices = [
        10,
        109,
        67,
        103,
        104,
        105,
        66,
        107,
        9,
        336,
        296,
        334,
        333,
        332,
        297,
        338
    ]
    all_face_features = list(range(468))
    def __init__(self, landmarks:NamedTuple = None, image_shape: tuple = (640, 480)):
        """Creates an instance of Face

        Args:
            landmarks (NamedTuple, optional): Landmarks object extracted by mediapipe tools
            image_shape (tuple, optional): The width and height of the image used to extract the face. Required to get landmarks in the right pixel size (useful for face copying and image operations). Defaults to (480, 640).
        """
        self.image_shape = image_shape

        if type(landmarks)==np.ndarray:
            self.landmarks= landmarks
            self.npLandmarks=landmarks
        else:
            self.update(landmarks)


        self.blinking = False
        self.perclos_buffer = []
        self.blink_start_time = 0
        self.last_blink_duration = 0


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

        #Using the canonical face coordinates
        noze_tip_pos = [0,0.004632,0.075866]
        self.face_3d_reference_positions=(np.array([
        [0, 0.004632, 0.075866],            # Nose tip        
        [ 0.04671,-0.026645,0.030841],      # Left eye extremety
        [-0.04671,-0.026645,0.030841],      # Right eye extremety
        [0,-0.04886,0.053853],              # forehead center
        #[0,-0.079422,0.051812]             # Chin 
        ])-np.array(noze_tip_pos))*1000      # go to centimeters
        self.face_reference_landmark_ids = [
            4,          # Nose tip
            263,        # Left eye extremety
            130,        # Right eye extremety
            151,        # Forehead
            #199         # Chin
        ]


        """
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
            263,        # Right right eye
            151         # forehead center
            ]
        """


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

        p1 = self.get_landmark_pos(self.left_eye_contour_indices[2])
        p2 = self.get_landmark_pos(self.left_eye_contour_indices[0])
        return np.abs(p2[0] - p1[0])

    def get_left_eye_height(self):
        """Gets the left eye height

        Returns:
            float: The height of the left eye
        """        

        # Assertion to verify that the face object is ready
        assert self.ready, "Face object is not ready. There are no landmarks extracted."

        p1 = self.get_landmark_pos(self.left_eye_contour_indices[3])
        p2 = self.get_landmark_pos(self.left_eye_contour_indices[1])
        return np.abs(p2[1] - p1[1])

    def get_right_eye_width(self):
        """Gets the right eye width

        Returns:
            float: The width of the right eye
        """        

        # Assertion to verify that the face object is ready
        assert self.ready, "Face object is not ready. There are no landmarks extracted."

        p1 = self.get_landmark_pos(self.right_eye_contour_indices[2])
        p2 = self.get_landmark_pos(self.right_eye_contour_indices[0])
        return np.abs(p2[0] - p1[0])

    def get_right_eye_height(self):
        """Gets the right eye height

        Returns:
            float: The height of the left eye
        """        

        # Assertion to verify that the face object is ready
        assert self.ready, "Face object is not ready. There are no landmarks extracted."

        p1 = self.get_landmark_pos(self.right_eye_contour_indices[3])
        p2 = self.get_landmark_pos(self.right_eye_contour_indices[1])
        return np.abs(p2[1] - p1[1])

    def get_landmark_pos(self, index) -> Tuple:
        """Recovers the position of a landmark from a results array

        Args:
            index (int): Index of the landmark to recover

        Returns:
            Tuple: Landmark 3D position in image space
        """

        # Assertion to verify that the face object is ready
        assert self.ready, "Face object is not ready. There are no landmarks extracted."

        lm = self.npLandmarks[index, ...]
        return np.array([lm[0], lm[1], lm[2]])



    def get_landmarks_pos(self, indices: list) -> np.ndarray:
        """Recovers the position of a landmark from a results array

        Args:
            indices (list): List of indices of landmarks to extract

        Returns:
            np.ndarray: A nX3 array where n is the number of landmarks to be extracted and 3 are the 3 cartesian coordinates
        """

        # Assertion to verify that the face object is ready
        assert self.ready, "Face object is not ready. There are no landmarks extracted."

        return self.npLandmarks[indices,...]

    def get_3d_realigned_landmarks_pos(self, indices: list=None,camera_matrix: np.ndarray = None, dist_coeffs: np.ndarray = np.zeros((4, 1))) -> np.ndarray:
        """Returns a realigned version of the landmarks such that the head is looking forwards.
        First the face orientation is computed, then the inverse rotation is applyed so that the face is facing the camera.
        Useful for face recognition

        Args:
            indices (list): Indices of the landmarks to extract. Defaults to None, which means all landmarks

        Returns:
            np.ndarray: A realigned landmars vector of form nX3
        """
        # Correct orientation
        vertices = self.npLandmarks.copy()
        pos, ori = self.get_head_posture(camera_matrix, dist_coeffs)
        if pos is not None :
            center = self.npLandmarks[self.nose_tip_index,...]#vertices.mean(axis=0)
            centered = (vertices-center)
            vertices = rotateLandmarks(centered, ori, True)
            vertices += center
        if indices is not None:
            return vertices[indices,:]
        else:
            return vertices

    def get_realigned_landmarks_pos(self, indices: list=None) -> np.ndarray:
        """Returns a realigned version of the landmarks such that the head top is exactly in the center top and the chin is in the center  bottom

        Args:
            indices (list): Indices of the landmarks to extract. Defaults to None, which means all landmarks

        Returns:
            np.ndarray: A realigned landmars vector of form nX3
        """
        # Correct orientation
        vertices = self.npLandmarks.copy()
        up=vertices[10,:2]
        chin=vertices[152,:2]
        center = (up+chin)/2
        vertical_line=up-chin
        angle=np.arctan2(vertical_line[1],vertical_line[0]) + np.pi/2
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        centered = (vertices[:,:2]-center[None,:])
        vertices[:,:2] = (centered@R)+center
        if indices is not None:
            return vertices[indices,:]
        else:
            return vertices

    def get_3d_realigned_face(self, camera_matrix: np.ndarray = None, dist_coeffs: np.ndarray = np.zeros((4, 1))):
        """Returns a face object using a realigned version of the landmarks such that the head is looking forwards.
        First the face orientation is computed, then the inverse rotation is applyed so that the face is facing the camera.
        Useful for face recognition

        Args:
            indices (list): Indices of the landmarks to extract. Defaults to None, which means all landmarks

        Returns:
            np.ndarray: A realigned landmars vector of form nX3
        """
        # Correct orientation
        vertices = self.npLandmarks.copy()
        pos, ori = self.get_head_posture(camera_matrix, dist_coeffs)
        if pos is not None :
            center = vertices.mean(axis=0)
            centered = (vertices-center)
            vertices = rotateLandmarks(centered, ori, True)
            vertices[:,1]*=-1
            vertices += center

        return Face(vertices,self.image_shape)

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


    def draw_landmarks(self, image: np.ndarray, landmarks: np.ndarray=None, radius:int=1, color: tuple = (255, 0, 0), thickness: int = 1, link=False) -> np.ndarray:
        """Draw a list of landmarks on an image

        Args:
            image (np.ndarray): Image to draw the contour on
            landmarks (np.ndarray): a nX3 ndarray containing the positions of the landmarks. Defaults to None (use all landmarks).
            radius (int, optional): Radius of the circle to draw the landmark. Defaults to 5.
            color (tuple, optional): Color of the landmark. Defaults to (255, 0, 0).
            thickness (int, optional): Thickness of the line to draw the landmark. Defaults to 5.


        Returns:
            np.ndarray: The image with the contour drawn on it
        """
        if landmarks is None:
            landmarks = self.npLandmarks
            
        lm_l=landmarks.shape[0]
        for i in range(lm_l):
            image = cv2.circle(image, (int(landmarks[i,0]), int(landmarks[i,1])), radius,color, thickness)
            if link:
                image = cv2.line(image, (int(landmarks[i,0]), int(landmarks[i,1])),(int(landmarks[(i+1)%lm_l,0]), int(landmarks[(i+1)%lm_l,1])),color, thickness)
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

    def draw_contour(self, image: np.ndarray, contour: np.ndarray, color: tuple = (255, 0, 0), thickness: int = 1, isClosed:bool = True) -> np.ndarray:
        """Draw a contour on an image

        Args:
            image (np.ndarray): Image to draw the contour on
            contour (np.ndarray): a nX3 ndarray containing the positions of the landmarks
            color (tuple, optional): Color of the landmark. Defaults to (255, 0, 0).
            thickness (int, optional): Thickness of the line to draw the landmark. Defaults to 5.
            isClosed (bool, optional): If True, the contour will be closed, otherwize it will be kept open. Defaults to True 


        Returns:
            np.ndarray: The image with the contour drawn on it
        """

        pts = np.array([[int(p[0]), int(p[1])] for p in contour.tolist()]).reshape((-1, 1, 2))
        return cv2.polylines(image, [pts], isClosed, color, thickness)

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
        pos = self.get_landmark_pos(self.left_eye_center_index)[0:2]

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
        pos = self.get_landmark_pos(self.right_eye_center_index)[0:2]

        w = int(self.get_right_eye_width())
        h = int(self.get_right_eye_height())

        if w > 0 and h > 0:
            overlay_ = overlay.resize((w, h), Image.ANTIALIAS)
            x = int(pos[0] - overlay_.size[0] / 2)
            y = int(pos[1] - overlay_.size[1] / 2)
            pImage.paste(overlay_, (x, y), overlay_)
        return np.array(pImage).astype(np.uint8)
    
    def reset_face_3d_reference_positions(self):
        self.noze_tip_pos = [0,0.004632,0.075866]
        self.face_3d_reference_positions = (np.array([
        [0, 0.004632, 0.075866],            # Nose tip        
        [ 0.04671,-0.026645,0.030841],      # Left eye extremety
        [-0.04671,-0.026645,0.030841],      # Right eye extremety
        [0,-0.04886,0.053853],              # forehead center
        #[0,-0.079422,0.051812]             # Chin 
        ])-np.array(self.noze_tip_pos))*1000 #go to centimeters
        
    def lock_face_3d_reference_positions(self):
        
        
        self.face_3d_reference_positions = self.npLandmarks[self.face_reference_landmark_ids]-np.array(self.npLandmarks[Face.nose_tip_index])
        self.face_3d_reference_positions[:,2]=-self.face_3d_reference_positions[:,2]
        print(self.face_3d_reference_positions)

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
            camera_matrix= buildCameraMatrix(size=self.image_shape)

        # Use opencv's PnPsolver to solve the rotation problem
        face_2d_positions = self.npLandmarks[self.face_reference_landmark_ids,:2]
        (success, face_ori, face_pos, _) = cv2.solvePnPRansac(
                                                    self.face_3d_reference_positions.astype(np.float32),
                                                    face_2d_positions.astype(np.float32), 
                                                    camera_matrix, 
                                                    dist_coeffs,
                                                    flags=cv2.SOLVEPNP_ITERATIVE)

        if not success:
            return None, None

        # save posture
        face_pos[0]*=-1
        face_pos[1]*=-1
        self.pos = face_pos
        self.ori = face_ori

        return face_pos, face_ori

    def get_eye_pos(self, iris, left, right, up, down, eye_radius = 10):
        """Computes eye angle compared to a reference
        Each eye is represented as a ball

                 |
        ---------|---------

        Args:
            iris (_type_): The 3D position of the iris
            left (_type_): The left  of the eyelids
            right (_type_): The right of the eyelids
            up (_type_): The upper position of the eyelids
            down (_type_): The lower position of the eyelids
            eye_radius (float, optional): The radius of the eye. Defaults to 0.1.

        Returns:
            list: The eye angles yaw and pitch
        """
        ex = (left-right)
        ey = (up-down)
        nx = np.linalg.norm(ex)
        ny = np.linalg.norm(ey)
        ex/= nx
        ey/= ny

        h_center = (right+left)/2
        v_center = (up+down)/2
        h_pos_iris = 2*(iris-h_center)/nx
        v_pos_iris = 2*(iris-v_center)/nx
        return np.array([np.dot(h_pos_iris,ex),np.dot(v_pos_iris,ey)])

    def get_eyes_position(self, camera_matrix:np.ndarray = None, dist_coeffs:np.ndarray=np.zeros((4,1)))->tuple:
        """Gets the posture of the eyes (position in cartesian space and Euler angles)
        Args:
            camera_matrix (int, optional)       : The camera matrix built using buildCameraMatrix Helper function. Defaults to a perfect camera matrix 
            dist_coeffs (np.ndarray, optional)) : The distortion coefficients of the camera
        Returns:
            tuple: (left_pos, right_pos) the iris position inside the eye 
        """

        # Assertion to verify that the face object is ready
        assert self.ready, "Face object is not ready. There are no landmarks extracted."

        # Left eye

        left_eye_infos = self.get_3d_realigned_landmarks_pos(Face.left_eye_orientation_landmarks, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        # Get landmarks



        up         = left_eye_infos[0, ...]
        down       = left_eye_infos[1, ...]
        right       = left_eye_infos[2, ...]
        left        = left_eye_infos[3, ...]
        iris        = left_eye_infos[4, ...]


        left_ori = self.get_eye_pos(iris, left, right, up, down)

        # Right reye
        right_eye_infos = self.get_3d_realigned_landmarks_pos(Face.right_eye_orientation_landmarks, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        # Get landmarks


        up     = right_eye_infos[0, ...]
        down   = right_eye_infos[1, ...]
        right   = right_eye_infos[2, ...]
        left    = right_eye_infos[3, ...]
        iris    = right_eye_infos[4, ...]


        right_ori = self.get_eye_pos(iris, left, right, up, down)

        return left_ori, right_ori

    def compose_eye_rot(self, eye_pos:list, face_orientation:np.ndarray, offset=np.array([0,0]), x2ang: int=60, y2ang:int=45)->np.ndarray:
        """Composes eye position with face rotation to produce eye orientation in world coordinates

        Args:
            eye_pos (list): The local normalized eye position
            face_orientation (np.ndarray): The orientation of the face in compressed axis angle format
            x2ang (int, optional): A coefficient to convert normalized position to angle in X axis. Defaults to 180.
            y2ang (int, optional): A coefficient to convert normalized position to angle in Y axis. Defaults to 30.

        Returns:
            np.ndarray: [description]
        """
        corrected_eye_pos=eye_pos+offset
        fo = R.from_rotvec(face_orientation[:,0])
        ypr = R.from_euler("yxz",[corrected_eye_pos[0]*x2ang,corrected_eye_pos[1]*y2ang,0], degrees=True)#,,0], degrees=True)
        return np.array((ypr*fo).as_rotvec()).reshape((3,1))
        


    def getEyesDist(self)->int:
        """Gets the distance between the two eyes

        Returns:
            int: The distance between the two eyes
        """

        # Assertion to verify that the face object is ready
        assert self.ready, "Face object is not ready. There are no landmarks extracted."


        pos = self.get_landmarks_pos([self.left_eye_center_index, self.right_eye_center_index])
        return np.linalg.norm(pos[1,:]-pos[0,:])

    def process_eyes(self, image: np.ndarray, detect_blinks: bool = False, blink_th:float=5, blinking_double_threshold_factor:float=1.05)->tuple:
        """Process eye information and extract eye opening value, normalized eye opening and detect blinks

        Args:
            image (np.ndarray): Image to draw on when landmarks are to be drawn
            detect_blinks (bool, optional): If True, blinks will be detected. Defaults to False.
            blink_th (float, optional): Blink threshold. Defaults to 5.
            blinking_double_threshold_factor (float, optional): a factor for double blinking threshold detection. 1 means that the threshold is the same for closing and opening. If you put 1.2, it means that after closing, the blinking is considered finished only when the opening surpssess the blink_threshold*1.2. Defaults to 1.05.

        Returns:
            tuple: Depending on what configuration was chosen in the parameters, the output is:
            left_eye_opening, right_eye_opening, is_blink, last_blink_duration if blinking detection is activated
            left_eye_opening, right_eye_opening if blinking detection is deactivated
        """

        # Assertion to verify that the face object is ready
        assert self.ready, "Face object is not ready. There are no landmarks extracted."

        # 12 ->13  vs 374
        left_eyelids_contour = self.get_landmarks_pos(self.left_eyelids_indices)
        left_eye_upper0 = left_eyelids_contour[12, ...]
        left_eye_upper1 = left_eyelids_contour[13, ...]
        left_eye_lower = left_eyelids_contour[4, ...]
        left_eye_upper = (left_eye_upper0+left_eye_upper1)/2

        ud = left_eye_upper-left_eye_lower
        ex = left_eye_upper1-left_eye_upper0
        ex /= np.linalg.norm(ex)
        ey = np.cross(np.array([0,0,1]),ex)

        left_eye_opening = np.dot(ud,ey)
        if left_eye_opening<0:
            right_eye_opening=0

        right_eyelids_contour = self.get_landmarks_pos(self.right_eyelids_indices)
        
        right_eye_upper0 = right_eyelids_contour[12, ...]
        right_eye_upper1 = right_eyelids_contour[13, ...]
        right_eye_lower = right_eyelids_contour[4, ...]
        right_eye_upper = (right_eye_upper0+right_eye_upper1)/2

        ud = right_eye_upper-right_eye_lower
        ex = right_eye_upper1-right_eye_upper0
        ex /= np.linalg.norm(ex)
        ey = np.cross(np.array([0,0,1]),ex)

        right_eye_opening = np.dot(ud,ey)  
        if right_eye_opening<0:
            right_eye_opening=0

        left_eye_contour = self.get_landmarks_pos(self.left_eye_contour_indices)
        left_eye_iris_upper = left_eye_contour[3, ...]
        left_eye_iris_lower = left_eye_contour[1, ...]

        right_eye_contour = self.get_landmarks_pos(self.right_eye_contour_indices)
        right_eye_iris_upper = right_eye_contour[3, ...]
        right_eye_iris_lower = right_eye_contour[1, ...]

        dl = np.linalg.norm(left_eye_iris_upper-left_eye_iris_lower)
        dr = np.linalg.norm(right_eye_iris_upper-right_eye_iris_lower)

        left_eye_opening /=dl
        right_eye_opening /=dr


        if detect_blinks:
            is_blink = False
            eye_opening = (left_eye_opening+right_eye_opening)/2
            if eye_opening < blink_th and not self.blinking:
                self.blinking = True
                is_blink = True
                self.blink_start_time=time.time()
            elif eye_opening > blink_th*blinking_double_threshold_factor and self.blinking:
                self.blinking = False
                self.last_blink_duration=time.time()-self.blink_start_time

            return left_eye_opening, right_eye_opening, is_blink, self.last_blink_duration
        else:
            return left_eye_opening, right_eye_opening


    def compute_perclos(self,left_eye_opening:float, right_eye_opening:float, perclos_buffer_depth:int=1800, buffer:list = None, threshold=0.2):
        """Computes the perclos on a time window
        The perclos is the percentage of ey closure over a period of time. Generally 1 minute.
        Here the perclos_buffer_depth should be computed taking into consideration the frame rate you are using
        the default value supposes 30 FPS and a window of 1 minute.

        Args:
            left_eye_opening (float): The left eye opening( can bee computed using process_eyes method)
            right_eye_opening (float): The left eye opening( can bee computed using process_eyes method)
            perclos_buffer_depth (int, optional): The depth of the perclos computation buffer. Defaults to 1800. This is the buffer size for 1 minute at 30fps
            buffer (list, optional): A custom buffer to be used instead of the default buffer (useful when using multiple window sizes). Defaults to None.
            threshold (float, optional) : The closing threshold (generally in litterature it is 20% so the default one is 0.2)
        Returns:
            float: The perclos (bertween 0 and 1) Multiply it by 100 to get the percentage
        """
        # Buffer
        mean_eye_closure = (left_eye_opening + right_eye_opening)/2
        if buffer is None:
            buffer = self.perclos_buffer
        buffer.append(mean_eye_closure)
        while len(buffer)>perclos_buffer_depth:
            buffer.pop(0)
        # Compute Perclos
        pb = np.array(buffer)
        perclos = ((pb<threshold).astype(int).sum())/len(buffer)
        return perclos

    def draw_eyes_landmarks(self, image:np.ndarray):
        """Draws eyes landmarks on  the image

        Args:
            image (np.ndarray): The image to draw the landmarks on
        """
        self.draw_contour(image, self.get_landmarks_pos(self.left_eye_contour_indices), (255,255,255))
        self.draw_contour(image, self.get_landmarks_pos(self.left_eyelids_indices), (0,0,0))

        self.draw_contour(image, self.get_landmarks_pos(self.right_eye_contour_indices), (255,255,255))
        self.draw_contour(image, self.get_landmarks_pos(self.right_eyelids_indices), (0,0,0))


    def process_mouth(self, image: np.ndarray, normalize:bool=False, detect_yawning: bool = False, yawning_th:float=5, yawning_double_threshold_factor:float=1.05, draw_landmarks: bool = False)->tuple:
        """Process mouth information and extract moth opening value, normalized mouth opening and detect yawning

        Args:
            image (np.ndarray): Image to draw on when landmarks are to be drawn
            normalize (bool, optional): If True, the eye opening will be normalized by the distance between the eyes. Defaults to False.
            detect_blinks (bool, optional): If True, blinks will be detected. Defaults to False.
            yawn_th (float, optional): yawn threshold. Defaults to 5.
            yawning_double_threshold_factor (float, optional): a factor for double yawning threshold detection. 1 means that the threshold is the same for closing and opening. If you put 1.2, it means that after closing, the yawning is considered finished only when the opening surpssess the yawn_threshold*1.2. Defaults to 1.05.
            draw_landmarks (bool, optional): If True, the landmarks will be drawn on the image. Defaults to False.

        Returns:
            tuple: Depending on what configuration was chosen in the parameters, the output is:
            left_eye_opening, right_eye_opening, is_yawn if yawning detection is activated
            left_eye_opening, right_eye_opening if yawning detection is deactivated
        """

        # Assertion to verify that the face object is ready
        assert self.ready, "Face object is not ready. There are no landmarks extracted."


        left_eye_center = self.get_landmark_pos(self.left_eye_center_index)
        left_eyelids_contour = self.get_landmarks_pos(self.left_eyelids_indices)
        left_eye_upper = left_eyelids_contour[3, ...]
        left_eye_lower = left_eyelids_contour[1, ...]

        left_eye_contour = self.get_landmarks_pos(self.left_eye_contour_indices)
        left_eye_iris_upper = left_eye_contour[3, ...]
        left_eye_iris_lower = left_eye_contour[1, ...]

        right_eye_center = self.get_landmark_pos(self.right_eye_center_index)
        right_eyelids_contour = self.get_landmarks_pos(self.right_eyelids_indices)
        right_eye_upper = right_eyelids_contour[3, ...]
        right_eye_lower = right_eyelids_contour[1, ...]

        right_eye_contour = self.get_landmarks_pos(self.right_eye_contour_indices)
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
            th = yawning_th / ed
        else:
            th = yawning_th

        if detect_yawning:
            is_blink = False
            eye_opening = (left_eye_opening+right_eye_opening)/2
            if eye_opening < th and not self.blinking:
                self.blinking = True
                is_blink = True
            elif eye_opening > th*yawning_double_threshold_factor:
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
        try:
            triangleList = self.triangles
        except:
            self.triangulate(landmark_indices)
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

    def getFaceBox(self, image:np.ndarray, landmark_indices:list=None, margins=(0,0,0,0))->np.ndarray:
        """Gets an image of the face extracted from the original image (simple box extraction which will extract some of the background)

        Args:
            image (np.ndarray): Image to extract the face from
            src_triangles (list): The delaulay triangles indices (look at triangulate)
            landmark_indices (list, optional): The list of landmarks to be used (the same list used for the triangulate method that allowed the extraction of the triangles). Defaults to None.
            margins (tuple, optional): Margin around the face (left, top, right, bottom). Defaults to (0,0,0,0).

        Returns:
            np.ndarray: Face drawn on a black background (the size of the image is equal of that of the face in the original image)
        """

        # Assertion to verify that the face object is ready
        assert self.ready, "Face object is not ready. There are no landmarks extracted."

        if landmark_indices is None:
            landmarks = self.npLandmarks[:, :2]
        else:
            landmarks = self.npLandmarks[landmark_indices, :2]

        # Calculate original dimensions of image
        height, width = image.shape[:2]

        # Calculate new dimensions of cropped image
        x1, y1 = landmarks.min(axis=0) - np.array(margins[0:2])
        x2, y2 = landmarks.max(axis=0) + np.array(margins[2:4])
        new_width = int(x2 - x1)
        new_height = int(y2 - y1)

        # Check if new dimensions are within original image frame
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > width:
            x2 = width
        if y2 > height:
            y2 = height

        # Crop the image using adjusted dimensions
        return image[int(y1):int(y2), int(x1):int(x2),...]


    def getFace(self, image:np.ndarray, src_triangles:list(), landmark_indices:list=None)->np.ndarray:
        """Gets an image of the face extracted from the original image (only the face with no background)

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
            try:
                t_dest = np.array([landmarks[tr[0], 0:2], landmarks[tr[1], 0:2], landmarks[tr[2], 0:2]])
                cv2.fillConvexPoly(mask, t_dest.astype(int), [1, 1, 1])
                mask=mask.astype(bool)
                dest = dest*~mask + mask* croped
            except Exception as ex:
                pass
        return dest

    def getLeftEye(self, image:np.ndarray, get_full_rect:bool=False)->np.ndarray:
        """Gets an image of the left eye

        Args:
            image (np.ndarray): Image to extract the face from
            
        Returns:
            np.ndarray: Face drawn on a black background (the size of the image is equal of that of the face in the original image)
        """

        # Assertion to verify that the face object is ready
        assert self.ready, "Face object is not ready. There are no landmarks extracted."

        landmarks = self.npLandmarks[self.left_eyelids_indices, :2]
        p1 = landmarks.min(axis=0)
        p2 = landmarks.max(axis=0)

        landmarks -= p1
        croped = image[int(p1[1]):int(
            p2[1]), int(p1[0]):int(p2[0])]
        if get_full_rect:
            return croped
        src_triangles = self.triangulate(self.left_eyelids_indices)
        dest = np.zeros_like(croped)
        for tr in src_triangles:
            mask = np.zeros_like(dest)
            try:
                t_dest = np.array([landmarks[tr[0], 0:2], landmarks[tr[1], 0:2], landmarks[tr[2], 0:2]])
                cv2.fillConvexPoly(mask, t_dest.astype(int), [1, 1, 1])
                mask=mask.astype(bool)
                dest = dest*~mask + mask* croped
            except Exception as ex:
                pass
        return dest

    def getRightEye(self, image:np.ndarray, get_full_rect:bool=False)->np.ndarray:
        """Gets an image of the left eye

        Args:
            image (np.ndarray): Image to extract the face from
            
        Returns:
            np.ndarray: Face drawn on a black background (the size of the image is equal of that of the face in the original image)
        """

        # Assertion to verify that the face object is ready
        assert self.ready, "Face object is not ready. There are no landmarks extracted."

        landmarks = self.npLandmarks[self.right_eyelids_indices, :2]
        p1 = landmarks.min(axis=0)
        p2 = landmarks.max(axis=0)

        landmarks -= p1
        croped = image[int(p1[1]):int(
            p2[1]), int(p1[0]):int(p2[0])]
        if get_full_rect:
            return croped

        src_triangles = self.triangulate(self.right_eyelids_indices)
        dest = np.zeros_like(croped)
        for tr in src_triangles:
            mask = np.zeros_like(dest)
            try:
                t_dest = np.array([landmarks[tr[0], 0:2], landmarks[tr[1], 0:2], landmarks[tr[2], 0:2]])
                cv2.fillConvexPoly(mask, t_dest.astype(int), [1, 1, 1])
                mask=mask.astype(bool)
                dest = dest*~mask + mask* croped
            except Exception as ex:
                pass
        return dest

    def copyToFace(
                    self, 
                    dst_face, 
                    src_image:np.ndarray, 
                    dst_image:np.ndarray, 
                    landmark_indices:list=None, 
                    opacity:float=1.0, 
                    min_input_triangle_cross:float=20, 
                    min_output_triangle_cross:float=1, 
                    retriangulate:bool=False,
                    seemless_cloning:bool=False,
                    empty_fill_color:Tuple=None
                    )->np.ndarray:
        """Copies the face to another image (used for face copy or face morphing)

        Args:
            dst_face (Face): A face object describing the face in the destination image (triangulate function should have been called on this object with the same landmark_indices argument)
            src_image (np.ndarray): [description]
            dst_image (np.ndarray): [description]
            landmark_indices (list, optional): The list of landmarks to be used (the same list used for the triangulate method that allowed the extraction of the triangles). Defaults to None.
            opacity (int, optional): the opacity level of the face (between 0 and 1)
            retriangulate (bool): if true, then triangles will be computed avery time (needed if the source face is moving)

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

        # Clip landmarks to image size
        src_landmarks = np.clip(src_landmarks,np.array([0,0]),np.array([src_image.shape[1],src_image.shape[0]])).astype(int)
        dst_landmarks = np.clip(dst_landmarks,np.array([0,0]),np.array([src_image.shape[1],src_image.shape[0]])).astype(int)


        # Crop images
        src_p1 = src_landmarks.min(axis=0)
        src_p2 = src_landmarks.max(axis=0)

        src_landmarks -= src_p1
        src_crop = src_image[src_p1[1]:src_p2[1], src_p1[0]:src_p2[0]]

        dst_p1 = dst_landmarks.min(axis=0)
        dst_p2 = dst_landmarks.max(axis=0)

        dst_landmarks -= dst_p1
        dst_crop = dst_image[dst_p1[1]:dst_p2[1], dst_p1[0]:dst_p2[0]]

        # Prepare empty image
        dest = dst_crop.copy()# np.zeros_like(dst_crop)


        dst_h, dst_w, _ = dest.shape
        center = (dst_w//2,dst_h//2)

        if retriangulate:
            self.triangulate(landmark_indices)
            dst_face.triangulate(landmark_indices)

        final_mask=np.zeros_like(dst_crop,dtype=np.uint8)
        # Prepare masks
        for src_tr in self.triangles:
            #try:
                fill=False
                # Get source triangles
                t_src = np.array(
                    [
                        src_landmarks[src_tr[0], 0:2], 
                        src_landmarks[src_tr[1], 0:2], 
                        src_landmarks[src_tr[2], 0:2]
                    ])
                # Get destination triangles
                t_dest = np.array(
                    [
                        dst_landmarks[src_tr[0], 0:2], 
                        dst_landmarks[src_tr[1], 0:2], 
                        dst_landmarks[src_tr[2], 0:2]
                    ])
                
                # Test triangles are not empty
                v1 = t_src[1,:]-t_src[0,:]
                v2 = t_src[2,:]-t_src[1,:]
                cross = np.abs(np.cross(v1,v2))
                if cross<min_input_triangle_cross:
                    if empty_fill_color is not None:
                        fill=True
                    else:
                        continue
                v1 = t_dest[1,:]-t_dest[0,:]
                v2 = t_dest[2,:]-t_dest[1,:]
                cross = np.abs(np.cross(v1,v2))
                if cross<min_output_triangle_cross:
                    continue
                

                #Crop more to just get the triangle zone
                """
                """
                src_min_pos = np.clip(t_src.min(axis=0)-np.array([2,2]),np.array([0,0]),np.array([src_image.shape[1],src_image.shape[0]]))
                src_max_pos = np.clip(t_src.max(axis=0)+np.array([2,2]),np.array([0,0]),np.array([src_image.shape[1],src_image.shape[0]]))

                dst_min_pos = np.clip(t_dest.min(axis=0)-np.array([2,2]),np.array([0,0]),np.array([dest.shape[1],dest.shape[0]]))
                dst_max_pos = np.clip(t_dest.max(axis=0)+np.array([2,2]),np.array([0,0]),np.array([dest.shape[1],dest.shape[0]]))
                
                t_src -= src_min_pos
                t_dest -= dst_min_pos

                #Find transformation matrix from source triangle to destination triangle
                M = cv2.getAffineTransform(
                    t_src.astype(np.float32),
                    t_dest.astype(np.float32)
                )

                src_w,src_h = src_max_pos[0]-src_min_pos[0], src_max_pos[1]-src_min_pos[1]
                dest_w,dest_h = dst_max_pos[0]-dst_min_pos[0], dst_max_pos[1]-dst_min_pos[1]

                # If the cropped size is zero, go to next triangle
                if src_w<=1 or src_h<=1 or dest_w<=1 or dest_h<=1:
                    continue

                if fill:
                    stem = np.ones((src_max_pos[1]-src_min_pos[1],
                                    src_max_pos[0]-src_min_pos[0]))
                    warped_t = cv2.warpAffine(np.dstack(
                                                    [
                                                        (stem*empty_fill_color[0])[:,:,None],
                                                        (stem*empty_fill_color[1])[:,:,None],
                                                        (stem*empty_fill_color[2])[:,:,None]
                                                    ]),
                                            M,
                                            (   dest_w,dest_h
                                            ),flags=cv2.INTER_LINEAR 
                                            )
                else:
                    warped_t = cv2.warpAffine(src_crop[ src_min_pos[1]:src_max_pos[1],
                                                        src_min_pos[0]:src_max_pos[0]], 
                                            M,
                                            (   dest_w,dest_h
                                            ),flags=cv2.INTER_LINEAR
                                            )
                #Build masks
                mask = np.zeros_like(warped_t)
                # Prepare masks
                cv2.fillConvexPoly(mask, t_dest, [1, 1, 1])
                mask = mask.astype(bool)

                # Build global mask
                final_mask[dst_min_pos[1]:dst_max_pos[1],dst_min_pos[0]:dst_max_pos[0],:] = final_mask[dst_min_pos[1]:dst_max_pos[1],dst_min_pos[0]:dst_max_pos[0],:]*~mask+ mask*255

                # Copy warped triangle to destination
                warped_masked = warped_t* mask # cv2.dilate(warped_t* mask,np.ones((2,2)))* mask
                dest_masked = dest[dst_min_pos[1]:dst_max_pos[1],dst_min_pos[0]:dst_max_pos[0],:]* ~mask#cv2.dilate(dest[dst_min_pos[1]:dst_max_pos[1],dst_min_pos[0]:dst_max_pos[0],:]* ~mask,np.ones((2,2)))* ~mask
                dest[dst_min_pos[1]:dst_max_pos[1],dst_min_pos[0]:dst_max_pos[0],:] =  dest_masked + warped_masked

            #except Exception as ex:
            #    pass

        if seemless_cloning:
            dst_crop = cv2.seamlessClone(dest, dst_crop, final_mask, (int(center[0]),int(center[1])), cv2.NORMAL_CLONE)
        else:
            final_mask= final_mask.astype(bool)
            dst_crop = dest*final_mask + dst_crop*~final_mask
        #dst_crop = final_mask 
        dst_image[
            int(dst_p1[1]):int(dst_p2[1]),
            int(dst_p1[0]):int(dst_p2[0]),
            :
            ] = dst_crop
        return dst_image
    

    @property
    def bounding_box(self):
        """
        Calculate the bounding box for a set of landmarks.

        Returns:
            Tuple of (x, y, width, height) representing the bounding box of the landmarks.
        """        
        pt1 = self.npLandmarks.min(axis=0)
        pt2 = self.npLandmarks.max(axis=0)
        return pt1[0], pt1[1], pt2[0], pt2[1]

    def draw_bounding_box(self, image:np.ndarray, color:tuple=(255,0,0), thickness:int=1, text=None):
        """Draws a bounding box around the face

        Args:
            image (np.ndarray): The image on which we will draw the bounding box
            color (tuple, optional): The color of the bounding box. Defaults to (255,0,0).
            thickness (int, optional): The line thickness. Defaults to 1.
        """
        pt1 = self.npLandmarks.min(axis=0)
        pt2 = self.npLandmarks.max(axis=0)
        cv2.rectangle(image, (int(pt1[0]),int(pt1[1])), (int(pt2[0]),int(pt2[1])), color, thickness)
        if text is not None:
            cv2.putText(image, text, (int(pt1[0]),int(pt1[1]-20)),cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)

    def get_face_outer_vertices(self):
        """ Draws a bounding box around the face that rotates wit the face
        Returns
            list : list containing indices of vertices that define the boundaries of the face
        """
        original = [x for x in range(self.npLandmarks.shape[0])]
        left = [x for x in range(self.npLandmarks.shape[0])]
        for next_id in original:
            # Find if there are points in all quadrants or not
            q=[0 for i in range(4)]
            p = self.npLandmarks[next_id,:]
            for other in left:
                if other != next_id:
                    v = self.npLandmarks[other,:]-p
                    if v[0]>0 and v[1]>0:
                        q[0] = 1
                    if v[0]<0 and v[1]>0:
                        q[1] = 1
                    if v[0]<0 and v[1]<0:
                        q[2] = 1
                    if v[0]>0 and v[1]<0:
                        q[3] = 1
                else:
                    continue
                if sum(q)==4:
                    left.remove(next_id)
                    break
        return left

    def draw_oriented_bounding_box(self, image:np.ndarray, color:tuple=(255,0,0), thickness:int=1):
        """Draws a bounding box around the face that rotates wit the face

        Args:
            image (np.ndarray): The image on which we will draw the bounding box
            color (tuple, optional): The color of the bounding box. Defaults to (255,0,0).
            thickness (int, optional): The line thickness. Defaults to 1.
        """
        vertex_ids = self.get_face_outer_vertices()
      
        pts= np.array([self.npLandmarks[v,:2].astype(int) for v in vertex_ids])
        for pt in pts:
            cv2.circle(image, pt, 1, color, thickness)

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

    def draw_reference_frame(self, image:np.ndarray, pos: np.ndarray, ori:np.ndarray, origin:np.ndarray=None, line_length:int=50, camera_matrix=None, dist_coeffs = np.zeros((4,1)))->None:
        """Draws a reference frame at a sprecific position

        Args:
            image (np.ndarray): The image to draw the reference frame on.
            pos (np.ndarray): The real 3D position of the frame reference
            ori (np.ndarray): The orientation of the frame in compressed axis angle format
            origin (np.ndarray): The origin in camera frame where to draw the frame
            translation (np.ndarray, optional): A translation vector to draw the frame in a different position tha n the origin. Defaults to None.
            line_length (int, optional): The length of the frame lines (X:red,y:green,z:blue). Defaults to 50.
        """
        if camera_matrix is None:
            camera_matrix = buildCameraMatrix(size=self.image_shape)
        #Let's project three vectors ex,ey,ez to form a frame and draw it on the nose
        (center_point2D_x, jacobian) = cv2.projectPoints(np.array([(0, 0.0, 0.0)]), ori, pos, camera_matrix, dist_coeffs)

        (end_point2D_x, jacobian) = cv2.projectPoints(np.array([(line_length, 0.0, 0.0)]), ori, pos, camera_matrix, dist_coeffs)
        (end_point2D_y, jacobian) = cv2.projectPoints(np.array([(0.0, -line_length, 0.0)]), ori, pos, camera_matrix, dist_coeffs)
        (end_point2D_z, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, line_length)]), ori, pos, camera_matrix, dist_coeffs)

        p1 = ( int(center_point2D_x[0][0][0]), int(center_point2D_x[0][0][1]))         
        p2_x = ( int(end_point2D_x[0][0][0]), int(end_point2D_x[0][0][1]))         
        p2_y = ( int(end_point2D_y[0][0][0]), int(end_point2D_y[0][0][1]))         
        p2_z = ( int(end_point2D_z[0][0][0]), int(end_point2D_z[0][0][1]))   

        """
        """
        if origin is not None:
            do = (int(origin[0]- p1[0]), int(origin[1]- p1[1]))
            p1=  (int(origin[0]), int(origin[1]))

            p2_x= (p2_x[0]+do[0],p2_x[1]+do[1])

            p2_y= (p2_y[0]+do[0],p2_y[1]+do[1])

            p2_z= (p2_z[0]+do[0],p2_z[1]+do[1])

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

        
