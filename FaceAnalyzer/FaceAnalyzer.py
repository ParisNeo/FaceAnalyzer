# -*- coding: utf-8 -*-
"""=== Face Analyzer =>
    Module : FaceAnalyzer
    Author : Saifeddine ALOUI (ParisNeo)
    Licence : MIT
    Description :
        Main module. FaceAnalyzer analyzes an image and extract faces by creating instances of Face in its attribute Faces
<================"""

from typing import NamedTuple, Tuple
import numpy as np
import mediapipe as mp
import cv2
from scipy.signal import butter, filtfilt
import math
import time
from PIL import Image
from scipy.spatial import Delaunay

from .Face import Face

class FaceAnalyzer():
    """A class that analyzes the facial components
    """

    def __init__(self, max_nb_faces=1, image_shape: tuple = (480, 640)):
        """Creates an instance of the FaceAnalyzer object

        Args:
            max_nb_faces (int,optional) : The maximum number of faces to be detected by the mediapipe library
            image_shape (tuple, optional): The shape of the image to be processed. Defaults to (480, 640).
        """
        self.mp_drawing = mp.solutions.drawing_utils
        self.fmd = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=max_nb_faces)
        self.max_nb_faces = max_nb_faces

        self.faces = [Face(), Face()]
        self.image_shape = image_shape
        self.image = None
        self.results = None
        self.found_faces = False
        self.found_faces = False
        self.nb_faces = 0        

    def process(self, image: np.ndarray, draw_mask: bool = False) -> NamedTuple:
        """Processes an image and extracts the faces

        Args:
            image (np.ndarray): The image to extract faces from
            draw_mask (bool, optional): If true, a mask will be drawn on each detected face. Defaults to False.

        Returns:
            NamedTuple: The result of extracting the image
        """
        # Process the image
        results = self.fmd.process(image)

        # Keep a local reference to the image
        self.image = image

        # If faces found
        if results.multi_face_landmarks is not None:
            self.found_faces = True
            self.nb_faces = len(results.multi_face_landmarks)
        else:
            self.found_faces = False
            self.nb_faces = 0
            return

        # Update faces
        for i, lm in enumerate(results.multi_face_landmarks):
            self.faces[i].update(lm)
        for i in range(len(results.multi_face_landmarks),self.max_nb_faces):
            self.faces[i].update(None)

        if draw_mask:
            for face in self.faces:
                if face.ready:
                    face.draw_mask(image)

        self.results = results
    @staticmethod
    def from_image(file_name:str, max_nb_faces:int=1, image_shape:tuple=(640, 480)):
        """Opens an image and extracts a face from it

        Args:
            file_name (str)                 : The name of the image file containing one or multiple faces
            max_nb_faces (int, optional)    : The maximum number of faces to extract. Defaults to 1
            image_shape (tuple, optional)   : The image shape. Defaults to (640, 480)

        Returns:
            An instance of FaceAnalyzer: A face analyzer containing all processed faces out of the image. Ile image can be found at fa.image
        """
        fa = FaceAnalyzer(max_nb_faces=max_nb_faces)
        image = Image.open(file_name)
        image = image.resize(image_shape)
        npImage = np.array(image)[...,:3]
        fa.process(npImage)
        return fa
