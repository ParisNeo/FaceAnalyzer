"""=============
    Example : face_chacer.py
    Author  : Saifeddine ALOUI
    Description :
        A tool to cntrol the PC mouse only with your head motion and blinck to click
        At the beginning clicking is deactivated. You need to close your eyes for at least 2seconds to activate blink clicking
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

from OOPyGame import Widget, Button, Label, ProgressBar, ImageBox, WindowManager
from pygame.mixer import Sound, get_init, pre_init
import array
import pickle

global click, is_calibrating, calibration_step, calibration_buffer, is_active



user32 = ctypes.windll.user32
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

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
piew_sound = pygame.mixer.Sound(Path(__file__).parent/"assets/audio/piew.wav")

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



# Build a window
wm = WindowManager("Face Mouse controller")

def template_statusbar(rect):
    label_image = str(Path(__file__).parent/"assets/buttons/label.png").replace("\\","/")
    
    # build button
    return Widget(
        rect,
        wm,
    style="""
        widget{
            align:left;
            color:black;
    """+
    f"""
            background-image:url('file:///{label_image}')
    """+
    """
        }
    """)
# Simple button 
def template_label(title,rect):
    label_image = str(Path(__file__).parent/"assets/buttons/label.png").replace("\\","/")
    
    # build button
    return Label(
        title,
        wm,
        rect,
    style="""
        label{
            align:left;
            x-margin:10;
            color:black;
    """+
    f"""
            background-image:url('file:///{label_image}')
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
        wm,
        rect,
    style="""
        btn.normal{
            color:white;
    """+
    f"""
            background-image:url('file:///{not_pressed_image}')
    """+
    """
        }
        btn.hover{
            color:red;
    """+
    f"""
            background-image:url('file:///{hovered_image}')
    """+
    """
        }
        btn.pressed{
            color:red;
    """+
    f"""
            background-image:url('file:///{pressed_image}')
    """+
    """
        }
    """,
    is_toggle=is_togle,clicked_event_handler=clicked_event_handler)

def template_progressbar(rect):
    
    # build button
    return ProgressBar(
        wm,
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

def activate():
    """Activates face mouse control
    """
    global click, is_active
    is_active = btn_activate.pressed
    click = btn_activate.pressed
    lbl_info.setText(f"Eye blinking status : {click}")

def calibrate():
    """Starts calibration or move to next step in the calibration process
    """
    global is_calibrating, calibration_step, calibration_buffer
    if not is_calibrating:
        is_calibrating=True
        calibration_step=0
        calibration_buffer = [[],[]]
        btn_calibrate.setText("Next")
    elif calibration_step==0 or calibration_step==2:
        calibration_step+=1
        calibration_buffer = [[],[]]

    
btn_calibrate   = template_button("Calibrate",(10,560,100,40),clicked_event_handler=calibrate)
btn_activate    = template_button("Activate",(110,560,100,40), is_togle=True,clicked_event_handler=activate)
lbl_info        = template_label("Eye blinking status : False",(0,520,800,40))
bg              = template_label("",(0,560,800,40))
pb_advance      = template_progressbar((500,575,290,10))
img_feed        = ImageBox(rect=[80,0,640,480])

wm.addWidget(img_feed)

wm.addWidget(lbl_info)
wm.addWidget(bg)
wm.addWidget(pb_advance)
wm.addWidget(btn_calibrate)
wm.addWidget(btn_activate)

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
                # ======================= A code to toggle eye clicking using 2s eye closing ============
                if is_active:
                    if eye_opening<0.35 and not closed:
                        t = time.time()
                        closed = True
                    else:
                        if eye_opening>0.35:
                            closed=False
                            waiting = False
                        else:
                            if not waiting:
                                if time.time()-t>2: # 2 seconds to consider it a click
                                    click=not click
                                    pygame.mixer.Sound.play(piew_sound)
                                    waiting = True                
                
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
                if is_calibrating:
                    if calibration_step==0:
                        lbl_info.setText(f"Look at Left Top corner : {p2d} <STAND BY>")
                    elif calibration_step==1:
                        lbl_info.setText(f"Look at Left Top corner : {p2d} <RECORDING>")
                        calibration_buffer[0].append(p2d[0])
                        calibration_buffer[1].append(p2d[1])
                        pb_advance.setValue(len(calibration_buffer[0])/100)
                        if len(calibration_buffer[0])==100:
                            p00 = [np.mean(calibration_buffer[0]), np.mean(calibration_buffer[1])]
                            calibration_step=2
                            pygame.mixer.Sound.play(piew_sound)
                    elif calibration_step==2:
                        lbl_info.setText(f"Look at Right Bottom corner : {p2d}<STAND BY>")
                    elif calibration_step==3:
                        lbl_info.setText(f"Look at Right Bottom corner : {p2d}<RECORDING>")
                        calibration_buffer[0].append(p2d[0])
                        calibration_buffer[1].append(p2d[1])
                        pb_advance.setValue(len(calibration_buffer[0])/100)
                        if len(calibration_buffer[0])==100:
                            p11 = [np.mean(calibration_buffer[0]), np.mean(calibration_buffer[1])]
                            calibration_step=0
                            is_calibrating=False
                            pygame.mixer.Sound.play(piew_sound)
                            lbl_info.setText(f"<Done>")
                            btn_calibrate.setText("Calibrate")
                            fn = Path(__file__).parent/"plane_calib.pkl"
                            with open(str(fn),"wb") as f:
                                pickle.dump({"P00":p00, "P11":p11},f) 
                            


                elif is_active:
                    win32api.SetCursorPos((x,y))

                if click and is_blink and x is not None:
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)

    mouse = pygame.mouse.get_pos()
    img_feed.setImage(image)
    wm.process()
    for event in wm.events:
        if event.type == pygame.QUIT:
            print("Done")
            Running=False
