"""=============
    Example : face_mesh.py
    Author  : Saifeddine ALOUI
    Description :
        A code to test FaceAnalyzer detects a face, draws a mask around it and measure head position and orientation
<================"""
from FaceAnalyzer import FaceAnalyzer, Face, faceOrientation2Euler
import numpy as np
import cv2
import time
from pathlib import Path

# open camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Build a window
cv2.namedWindow('Face Mesh', flags=cv2.WINDOW_NORMAL)
cv2.resizeWindow('Face Mesh', (640,480))

# Build face analyzer while specifying that we want to extract just a single face
fa = FaceAnalyzer(max_nb_faces=1)

# FPS processing
prev_frame_time = time.time()
curr_frame_time = time.time()

# Main Loop
while cap.isOpened():
    # Read image
    success, image = cap.read()
    
    # Opencv uses BGR format while mediapipe uses RGB format. So we need to convert it to RGB before processing the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image to extract faces and draw the masks on the face in the image
    fa.process(image)

    if fa.nb_faces>0:
        # Get head position and orientation compared to the reference pose (here the first frame will define the orientation 0,0,0)
        fa.faces[0].draw_mask(image)
        pos, ori = fa.faces[0].get_head_posture()
        if pos is not None:

            yaw, pitch, roll = faceOrientation2Euler(ori, degrees=True)
            # Show 
            #ori = Face.rotationMatrixToEulerAngles(ori)
            cv2.putText(
                image, f"Yaw : {yaw:2.2f}, Pitch : {pitch:2.2f}, Roll : {roll:2.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),1)
            cv2.putText(
                image, f"Position : {pos[0,0]:2.2f},{pos[1,0]:2.2f},{pos[2,0]:2.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # Process fps
    curr_frame_time = time.time()
    dt = curr_frame_time-prev_frame_time
    prev_frame_time = curr_frame_time
    fps = 1/dt
    # Show FPS
    cv2.putText(
        image, f"FPS : {fps:2.2f}", (10, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 255))

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
