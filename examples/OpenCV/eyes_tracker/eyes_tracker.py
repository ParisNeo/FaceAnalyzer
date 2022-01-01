"""=============
    Example : face_box.py
    Author  : Saifeddine ALOUI
    Description :
        A code to test FaceAnalyzer detects a face, draws a mask around it and measure head position and orientation
<================"""
from os import link
from FaceAnalyzer import FaceAnalyzer, Face,  DrawingSpec, buildCameraMatrix
from FaceAnalyzer.helpers.geometry.orientation import faceOrientation2Euler
import numpy as np
import cv2
import time
from pathlib import Path
import pickle

# open camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Build a window
cv2.namedWindow('Face Mesh', flags=cv2.WINDOW_NORMAL)
cv2.resizeWindow('Face Mesh', (640,480))

# Build face analyzer while specifying that we want to extract just a single face
fa = FaceAnalyzer(max_nb_faces=3)

# FPS processing
prev_frame_time = time.time()
curr_frame_time = time.time()

box_colors=[
    (255,0,0),
    (255,0,255),
    (255,0,255),
    
]
# Get camera calibration parameters
calibration_file_name = Path(__file__).parent/"cam_calib.pkl"
if calibration_file_name.exists():
    with open(str(calibration_file_name),"rb") as f:
        calib = pickle.load(f)
    mtx = calib["mtx"]
    dist = calib["dist"]
else:
    mtx = None
    dist = np.zeros((4, 1))

# Main Loop
while cap.isOpened():
    # Read image
    success, image = cap.read()
    
    # Opencv uses BGR format while mediapipe uses RGB format. So we need to convert it to RGB before processing the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image to extract faces and draw the masks on the face in the image
    fa.process(image)

    if fa.nb_faces>0:
        for i in range(fa.nb_faces):
            face = fa.faces[i]
            # Get head position and orientation compared to the reference pose (here the first frame will define the orientation 0,0,0)
            pos, ori = face.get_head_posture(camera_matrix=mtx, dist_coeffs=dist)
            if pos is not None:
                yaw, pitch, roll = faceOrientation2Euler(ori, degrees=True)
                face.draw_bounding_box(image, color=box_colors[i%3], thickness=5)
                face.draw_reference_frame(image, pos, ori, origin=face.get_landmark_pos(Face.nose_tip_index))

                # Show 
                #ori = Face.rotationMatrixToEulerAngles(ori)
                if i==0:
                    cv2.putText(
                        image, f"Yaw : {yaw:2.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
                    cv2.putText(
                        image, f"Pitch : {pitch:2.0f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
                    cv2.putText(
                        image, f"Roll : {roll:2.0f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
                    cv2.putText(
                        image, f"Position : {pos[0,0]:2.2f},{pos[1,0]:2.2f},{pos[2,0]:2.2f}", (10, 120), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))
            
                left_pos, right_pos = face.get_eyes_position(camera_matrix=mtx, dist_coeffs=dist)
                left_eye_opening, right_eye_opening, is_blink = face.process_eyes(image, detect_blinks=True, blink_th=0.6)
                print(f"left_eye_opening :{left_eye_opening}, right_eye_opening:{right_eye_opening}")

                #print(f'left : {left_pos}, right : {right_pos}')
                face.draw_landmarks(image,face.get_landmarks_pos(face.left_eyelids_indices),1)
                face.draw_landmarks(image,face.get_landmarks_pos(face.left_eye_contour_indices),1,(0,0,0),1, link=True)

                face.draw_landmarks(image,face.get_landmarks_pos(face.right_eyelids_indices),1)
                face.draw_landmarks(image,face.get_landmarks_pos(face.right_eye_contour_indices),1,(0,0,0),1, link=True)

                left_eye_ori = face.compose_eye_rot(left_pos, ori,np.array([-0.04,-0.07]),90,60)
                right_eye_ori = face.compose_eye_rot(right_pos, ori,np.array([-0.02,-0.15]),90,60)

                left_eye = face.get_landmark_pos(Face.left_eye_center_index)
                right_eye = face.get_landmark_pos(Face.right_eye_center_index)


                face.draw_reference_frame(image, pos, left_eye_ori, origin=left_eye)
                face.draw_reference_frame(image, pos, right_eye_ori, origin=right_eye)
            
    # Process fps
    curr_frame_time = time.time()
    dt = curr_frame_time-prev_frame_time
    prev_frame_time = curr_frame_time
    fps = 1/dt
    # Show FPS
    cv2.putText(
        image, f"FPS : {fps:2.2f}", (10, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))

    # Show the image
    try:
        cv2.imshow('Face Mesh', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    except Exception as ex:
        print(ex)
    
    # Wait for key stroke for 5 ms
    wk = cv2.waitKey(5)
    if wk & 0xFF == 27: # If escape is pressed then return
        break
    if wk & 0xFF == 115: # If s is pressed then take a snapshot
        sc_dir = Path(__file__).parent/"screenshots"
        if not sc_dir.exists():
            sc_dir.mkdir(exist_ok=True, parents=True)
        i = 1
        file = sc_dir /f"sc_{i}.jpg"
        while file.exists():
            i+=1
            file = sc_dir /f"sc_{i}.jpg"
        cv2.imwrite(str(file),cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        print("Shot")

# Close the camera properly
cap.release()
