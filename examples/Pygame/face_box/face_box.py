"""=============
    Example : face_chacer.py
    Author  : Saifeddine ALOUI
    Description :
        A simple program to show how to integrate Face_Analyzer with pygame
<================"""

import pygame
from numpy.lib.type_check import imag
from pygame.constants import QUIT
from scipy.ndimage.measurements import label
from FaceAnalyzer import FaceAnalyzer, Face,  DrawingSpec, buildCameraMatrix, faceOrientation2Euler
from FaceAnalyzer.Helpers import get_z_line_equation, get_plane_infos, get_plane_line_intersection, KalmanFilter, pilDrawCross, pilShowErrorEllipse, pilOverlayImageWirthAlpha, region_3d_2_region_2d, is_point_inside_region
import numpy as np
import cv2
import time
from pathlib import Path
import sys
import pyqtgraph as pg
from PIL import Image, ImageDraw

# open camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
width = 640#width = 1920
height = 480#height = 1080
image_size = [width, height]
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Build face analyzer while specifying that we want to extract just a single face
fa = FaceAnalyzer(max_nb_faces=3, image_shape=(width, height))



box_colors=[
    (255,0,0),
    (255,0,255),
    (255,0,255),
    
]

pygame.init()
screen = pygame.display.set_mode((width,height))
pygame.display.set_caption("Face box")
Running = True

#  Main loop
while Running:
    screen.fill((255,0,0))
    success, image = cap.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#
    # Process the image to extract faces and draw the masks on the face in the image
    fa.process(image)

    if fa.nb_faces>0:
        for i in range(fa.nb_faces):
            face = fa.faces[i]
            # Get head position and orientation compared to the reference pose (here the first frame will define the orientation 0,0,0)
            pos, ori = face.get_head_posture()
            if pos is not None:
                yaw, pitch, roll = faceOrientation2Euler(ori, degrees=True)
                face.draw_bounding_box(image, color=box_colors[i%3], thickness=5)
                face.draw_reference_frame(image, pos, ori, origin=face.getlandmark_pos(Face.nose_tip_index))


    image = np.swapaxes(image,0,1)#cv2.flip(, 1)
    my_surface = pygame.pixelcopy.make_surface(image)
    screen.blit(my_surface,(0,0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print("Done")
            Running=False
    # Update UI
    pygame.display.update()