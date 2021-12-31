"""=============
    Example : face_chacer.py
    Author  : Saifeddine ALOUI
    Description :
        A game of chacing objects using face orientation based on FaceAnalyzer. You use blinking to shoot them
<================"""

import pygame
from numpy.lib.type_check import imag
from pygame.constants import QUIT
from scipy.ndimage.measurements import label
from FaceAnalyzer import FaceAnalyzer, Face,  DrawingSpec, buildCameraMatrix
from FaceAnalyzer.helpers.geometry.orientation import faceOrientation2Euler
from FaceAnalyzer.helpers.geometry.euclidian import get_z_line_equation, get_plane_infos, get_plane_line_intersection, region_3d_2_region_2d, is_point_inside_region
from FaceAnalyzer.helpers.ui.opencv import cvDrawCross
from FaceAnalyzer.helpers.ui.pillow import pilShowErrorEllipse, pilOverlayImageWirthAlpha
from FaceAnalyzer.helpers.ui.pygame import WindowManager, ImageBox, Label

from FaceAnalyzer.helpers.estimation import KalmanFilter

import numpy as np
import cv2
import time
from pathlib import Path
import sys
import pyqtgraph as pg
from PIL import Image, ImageDraw
from Chaceable import Chaceable
import pickle

# open camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
width = 640#width = 1920
height = 480#height = 1080
image_size = [width, height]
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Build face analyzer while specifying that we want to extract just a single face
fa = FaceAnalyzer(max_nb_faces=3, image_shape=(width, height))


fn = Path(__file__).parent/"calib.pkl"

if fn.exists():
    with open(str(fn),"rb") as f:
        v = pickle.load(f) 
        p00 = v["P00"]
        p11 = v["P11"]
else:
    p00 = [-176.01920468,  -46.66348955] 
    p11 = [250.42943629, 154.68334261]


box_colors=[
    (255,0,0),
    (255,0,255),
    (255,0,255),
    
]

# ===== Build pygame window and populate with widgets ===================
pygame.init()
wm = WindowManager("Face box", (width,height))
infoObject = pygame.display.Info()
screensize = infoObject.current_w, infoObject.current_h
feedImage = ImageBox(rect=[0,0,width,height])
lbl_score = Label("Score : 0",[0,0,10,10],style=
"""label{
    color:white;
}""")
wm.addWidget(feedImage)
wm.addWidget(lbl_score)
# =======================================================================


Running = True
# Kalman filter
kalman = KalmanFilter(10*np.eye(2), 1*np.eye(2), np.array([0,0]), 10*np.eye(2),np.eye(2),np.eye(2))
main_plane  =    get_plane_infos(np.array([0,0, 0]),np.array([100,0, 0]),np.array([0, 100,0]))

chaceables=[]
chaceables.append(Chaceable(Path(__file__).parent/"assets/pika.png", [150,150], np.array([-90,0]), image_size))   
chaceables.append(Chaceable(Path(__file__).parent/"assets/pika.png", [150,150], np.array([90,0]), image_size))   
#Score
score_value=0
is_blink = False
p2d = None
#  Main loop
while Running:
    success, image = cap.read()
    image = cv2.cvtColor(image[:,::-1,:], cv2.COLOR_BGR2RGB)#cv2.flip(, 1)
    # Process the image to extract faces and draw the masks on the face in the image
    fa.process(image)
    game_ui_img =np.zeros((600,800,3))
    if fa.nb_faces>0:
        #for i in range(fa.nb_faces):
            i=0
            face = fa.faces[i]
            # Get head position and orientation compared to the reference pose (here the first frame will define the orientation 0,0,0)
            face_pos, face_ori = face.get_head_posture()
            if face_pos is not None:

                
                # First let's get the forward line (a virtual line that goes from the back of the head through tho nose towards the camera)
                li  =    get_z_line_equation(face_pos, face_ori)
                # Now we find the intersection point between the line and the plan. p is the 3d coordinates of the intersection pount, and p2d is the coordinates of this point in the plan
                p, p2d   =    get_plane_line_intersection(main_plane, li)
                kalman.process(p2d)
                # Filtered p2d
                p2d = kalman.x
                x = int((p2d[0]-p00[0])*screensize[0]/(p11[0]-p00[0]))
                y = int((p2d[1]-p00[1])*screensize[1]/(p11[1]-p00[1]))
                p2d = np.array([x,y])


                # Detect blinking
                left_eye_opening, right_eye_opening, is_blink = face.process_eyes(image, detect_blinks=True,  blink_th=0.35)
    for ch in chaceables:
        ch.draw(game_ui_img)
        if p2d is not None:
            is_contact = ch.check_contact(p2d)
            if is_contact and is_blink:
                ch.move_to(np.array([np.random.randint(-image_size[0]//2,image_size[0]//2-20), np.random.randint(-image_size[1]//2,image_size[1]//2-20)]))
                score_value += 1
    if p2d is not None:
        cvDrawCross(game_ui_img, (p2d+np.array(image_size)//2).astype(np.int), (200,0,0), 3)

    wm.process()
    feedImage.setImage(cv2.cvtColor(np.swapaxes(game_ui_img,0,1).astype(np.uint8),cv2.COLOR_BGR2RGB))
    lbl_score.setText(f"score : {score_value}")

    for event in wm.events:
        if event.type == pygame.QUIT:
            print("Done")
            Running=False