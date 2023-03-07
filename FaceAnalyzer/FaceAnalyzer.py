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
import math
import time
from PIL import Image
from scipy.spatial import Delaunay
from scipy.signal import butter, filtfilt

from .Face import Face

class FaceAnalyzer():
    """A class that analyzes the facial components
    """

    def __init__(self, max_nb_faces=1, image_shape: tuple = (640, 480), static_image_mode=False):
        """Creates an instance of the FaceAnalyzer object

        Args:
            max_nb_faces (int,optional) : The maximum number of faces to be detected by the mediapipe library
            image_shape (tuple, optional): The shape of the image to be processed. Defaults to (480, 640).
        """
        self.mp_drawing = mp.solutions.drawing_utils
        self.fmd = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=max_nb_faces, static_image_mode=static_image_mode)
        self.max_nb_faces = max_nb_faces

        self.faces = [Face(image_shape=image_shape) for i in range(max_nb_faces)]
        self.image_shape = image_shape
        self.image = None
        self.results = None
        self.found_faces = False
        self.found_faces = False
        self.nb_faces = 0        

    @property
    def image_size(self)->tuple:
        """A property to image size

        Returns:
            tuple: The image size
        """
        return self.image_shape

    @image_size.setter
    def image_size(self,new_shape:tuple):
        self.image_shape=new_shape
        for face in self.faces:
            face.image_shape=new_shape

    def process(self, image: np.ndarray) -> NamedTuple:
        """Processes an image and extracts the faces

        Args:
            image (np.ndarray): The image to extract faces from

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
            if i>=len(self.faces):
                continue
            self.faces[i].update(lm)
        for i in range(len(results.multi_face_landmarks),self.max_nb_faces):
            self.faces[i].update(None)

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


    def draw_names_on_bboxes(self, image, bboxes_and_names, font=cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale=1, thickness=2, color=(0, 0, 255), upscale=1):
        # Upscale the image to avoid pixelization of the text
        height, width = image.shape[:2]
        new_height, new_width = int(height * upscale), int(width * upscale)
        resized_image = cv2.resize(image, (new_width, new_height))

        # Upscale the bounding boxes
        bboxes = [bbox_and_name[0] for bbox_and_name in bboxes_and_names]
        scaled_bboxes = [[int(val * upscale) for val in bbox] for bbox in bboxes]

        # Draw the bounding boxes and names
        for i, bbox_and_name in enumerate(bboxes_and_names):
            bbox = scaled_bboxes[i]
            name = bbox_and_name[1]
            x1, y1, x2, y2 = bbox
            cv2.rectangle(resized_image, (x1, y1), (x2, y2), color, thickness)
            text_size, _ = cv2.getTextSize(name, font, font_scale, thickness)
            text_x = x1 + (x2 - x1 - text_size[0]) // 2
            text_y = y1 - text_size[1] - 5
            cv2.rectangle(resized_image, (text_x, text_y-text_size[1]-5), (text_x+text_size[0], text_y+text_size[1]), color, -1)
            cv2.putText(resized_image, name, (text_x, text_y), font, font_scale,(255-color[0], 255-color[1], 255-color[2]) , thickness)

        # Downscale the image back to its original size
        final_image = cv2.resize(resized_image, (width, height))

        return final_image