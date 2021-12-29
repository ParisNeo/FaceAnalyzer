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

from FaceAnalyzer.helpers.ui.pygame import Button
user32 = ctypes.windll.user32
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

# open camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
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

box_colors=[
    (255,0,0),
    (255,0,255),
    (255,0,255),
    
]

pygame.init()
screen = pygame.display.set_mode((800,600))
pygame.display.set_caption("Face chacer")
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

# Simple button 


def template_button(title, rect):
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
    """)

btn_calibrate=template_button("Calibrate",(10,560,100,40))
btn_activate=template_button("Activate",(110,560,100,40))

#  Main loop
while Running:
    screen.fill((0,0,0))
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
                                waiting = True                
                
                # =========================================================================================
                if eye_opening>0.35:
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
                    if use_eyes:
                        p00 = [-114.35650583,  -44.76599764]
                        p11 = [84.01429488, 98.52181435]
                    else:
                        p00 = [-176.01920468,  -46.66348955] 
                        p11 = [250.42943629, 154.68334261]
                    x = int((p2d[0]-p00[0])*screensize[0]/(p11[0]-p00[0]))
                    y = int((p2d[1]-p00[1])*screensize[1]/(p11[1]-p00[1]))
                    if is_calibrating:
                        print(p2d) 
                    else:
                        win32api.SetCursorPos((x,y))

                if click and is_blink and x is not None:
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)

    my_surface = pygame.pixelcopy.make_surface(np.swapaxes(image,0,1).astype(np.uint8))
    screen.blit(my_surface,(50,0))
    btn_calibrate.paint(screen)
    btn_activate.paint(screen)
    mouse = pygame.mouse.get_pos()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print("Done")
            Running=False
    btn_calibrate.handle_events(event)
    btn_activate.handle_events(event)
    # Update UI
    pygame.display.update()