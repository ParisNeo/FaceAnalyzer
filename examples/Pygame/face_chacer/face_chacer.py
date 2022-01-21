"""=============
    Example : calibrator.py
    Author  : Saifeddine ALOUI
    Description :
        A tool to calibrate gaze motion inside the screen
        For a better experience, please start by calibrating the software by using the calibrator script.
        Once calibrated, the experience will be better
<================"""

import pygame
import win32api, win32con
from FaceAnalyzer import FaceAnalyzer
from FaceAnalyzer.helpers.geometry.euclidian import is_point_inside_rect, get_z_line_equation, get_plane_infos, get_plane_line_intersection
from FaceAnalyzer.helpers.estimation import KalmanFilter
import numpy as np
import cv2
import time
import ctypes
from pathlib import Path

from FaceAnalyzer.helpers.ui.pygame import Widget, Button, Label, ProgressBar, ImageBox, WindowManager, Sprite
from pygame.mixer import Sound, get_init, pre_init
import array
import pickle

from Chaceable import Chaceable

global click, is_calibrating, calibration_step, calibration_buffer, is_active

score  = 0
pika_width=75
pika_height=100

# open camera
cap = cv2.VideoCapture(0)
width = 640#width = 1920
height = 480#height = 1080
image_size = [width, height]
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Build face analyzer while specifying that we want to extract just a single face
fa = FaceAnalyzer(max_nb_faces=3, image_shape=(width, height))

# If use eyes then the eyes orientation will be used instead of face (Still experimental)
use_eyes =False

# If true, the mouse will not be controlled, you will only need to look first at 0,0 then at w,h points and get the values that would be printed down
is_calibrating = False
calibration_step=0
calibration_buffer = [[],[]]
is_active = False

fn = Path(__file__).parent/"plane_calib.pkl"

if fn.exists():
    with open(str(fn),"rb") as f:
        v = pickle.load(f) 
        p00 = v["P00"]
        p11 = v["P11"]
else:
    if use_eyes:
        p00 = [-114.35650583,  -44.76599764]
        p11 = [84.01429488, 98.52181435]
    else:
        p00 = [-176.01920468,  -46.66348955] 
        p11 = [250.42943629, 154.68334261]

box_colors=[
    (255,0,0),
    (255,0,255),
    (255,0,255),
    
]

pygame.init()

Running = True
kalman = KalmanFilter(1*np.eye(2), 100*np.eye(2), np.array([0,0]), 10*np.eye(2),np.eye(2),np.eye(2))
main_plane  =    get_plane_infos(np.array([0,0, 0]),np.array([100,0, 0]),np.array([0, 100,0]))

is_blink = False
p2d = None
x = None
closed = False
t = time.time()
# By default clicking is deactivated (you need to close your eyes for 2s at least to activate eye clicking)
click = False
waiting = False

# white color
color = (255,255,255)
  
# light shade of the button
color_light = (170,170,170)
  
# dark shade of the button
color_dark = (100,100,100)



piew_sound = pygame.mixer.Sound(Path(__file__).parent/"assets/piew.wav")

# Build a window
wm = WindowManager("Face chacer",None)
infoObject = pygame.display.Info()
screensize = infoObject.current_w, infoObject.current_h

# Simple button 
def template_label(title,rect):
    label_image = str(Path(__file__).parent/"assets/buttons/label.png").replace("\\","/")
    
    # build button
    return Label(
        title,
        rect,
    style="""
        label{
            align:left;
            x-margin:10;
            color:white;
    """+
    f"""
            background-color:transparent;
    """+
    """
        }
    """)

def template_button(title, rect, is_togle=False, clicked_event_handler=None):
    not_pressed_image = str(Path(__file__).parent/"assets/buttons/not_pressed.png").replace("\\","/")
    hovered_image = str(Path(__file__).parent/"assets/buttons/hovered.png").replace("\\","/")
    pressed_image = str(Path(__file__).parent/"assets/buttons/pressed.png").replace("\\","/")
    
    # build button
    return Button(
        title,
        rect,
    style="""
        btn.normal{
            color:white;
    """+
    f"""
            background-color:darkgray;
    """+
    """
        }
        btn.hover{
            color:red;
    """+
    f"""
            background-color:gray;
    """+
    """
        }
        btn.pressed{
            color:red;
    """+
    f"""
            background-color:gray;
    """+
    """
        }
    """,
    is_toggle=is_togle,clicked_event_handler=clicked_event_handler)

def template_progressbar(rect):
    
    # build button
    return ProgressBar(
        rect,
    style="""
        brogressbar.outer{
            color:white;
            background-color:gray;
        }
        brogressbar.inner{
            color:white;
            background-color:red;
        }
    """
    )


def stop():
    exit(0)

# infoObject.current_w, infoObject.current_h
btn_stop        = template_button("Stop",(10,infoObject.current_h-50,100,40), is_togle=True,clicked_event_handler=stop)
lbl_score        = template_label("Score :0",(0,infoObject.current_h-90,infoObject.current_w,40))
pb_advance      = template_progressbar((500,infoObject.current_h-50,200,20))
img_feed        = ImageBox(rect=[infoObject.current_w//2-320,infoObject.current_h//2-240,640,480])

cross_image             = str(Path(__file__).parent/"assets/cross.png").replace("\\","/")
pika_image              = str(Path(__file__).parent/"assets/pika.png").replace("\\","/")
cross_surface           = Sprite(cross_image, rect=[infoObject.current_w//2,infoObject.current_h//2,20,20])
pika_srface             = Chaceable(
                            cv2.cvtColor(cv2.imread(pika_image), cv2.COLOR_BGR2RGB), 
                            rect=[np.random.randint(0,screensize[0]-pika_width),np.random.randint(0,screensize[1]-pika_height),pika_width,pika_height]
                        )

wm.addWidget(lbl_score)
wm.addWidget(btn_stop)
wm.addWidget(img_feed)

wm.addWidget(pika_srface)
wm.addWidget(cross_surface)
wm.addWidget(pb_advance)


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
            if use_eyes:
                left_pos, right_pos = face.get_eyes_position()
                left_eye_ori = face.compose_eye_rot(left_pos, face_ori,np.array([-0.04,-0.07]),90,60)
                #right_eye_ori = face.compose_eye_rot(right_pos, face_ori,np.array([-0.06,-0.14]))
                face_ori = left_eye_ori
            if face_pos is not None:
                # Detect blinking
                left_eye_opening, right_eye_opening, is_blink = face.process_eyes(image, detect_blinks=True, blink_th=0.35)
                eye_opening=(left_eye_opening+right_eye_opening)/2
                
                # =========================================================================================
                # First let's get the forward line (a virtual line that goes from the back of the head through tho nose towards the camera)
                li  =    get_z_line_equation(face_pos, face_ori)
                # Now we find the intersection point between the line and the plan. p is the 3d coordinates of the intersection pount, and p2d is the coordinates of this point in the plan
                p, p2d   =    get_plane_line_intersection(main_plane, li)
                kalman.process(p2d)
                # Filtered p2d
                p2d = kalman.x


                # Need to calibrate the screen 
                # Look at top left of the screen and save the p2d value for that position
                # Look at the bottom right of the screen and  save the p2d value for that position
                # 0,0 -> [-176.01920468  -46.66348955] 
                # w,h -> [250.42943629 154.68334261]

                x = int((p2d[0]-p00[0])*screensize[0]/(p11[0]-p00[0]))
                y = int((p2d[1]-p00[1])*screensize[1]/(p11[1]-p00[1]))
                contact = pika_srface.check_contact((x,y))
                if contact and is_blink:
                    pika_srface.setPosition((np.random.randint(0,screensize[0]-pika_width),np.random.randint(0,screensize[1]-pika_height)))
                    score += 1
                    lbl_score.setText(f"Score:{score}")

                cross_surface.setPosition((x,y))
                if is_blink:
                    pygame.mixer.Sound.play(piew_sound)


    mouse = pygame.mouse.get_pos()
    img_feed.setImage(image)
    wm.process()
    for event in wm.events:
        if event.type == pygame.QUIT:
            print("Done")
            Running=False
