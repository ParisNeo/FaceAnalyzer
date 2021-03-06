"""=============
    Example : face_off.py
    Author  : Saifeddine ALOUI
    Description :
        An example to show how to get the eyes opening as well as the blinking information
<================"""

import numpy as np
from pathlib import Path
import cv2
import time
from FaceAnalyzer import FaceAnalyzer

cap = cv2.VideoCapture(0)

# Blinks counter
n_blinks = 0

# Build a window
cv2.namedWindow('Eye processing', flags=cv2.WINDOW_NORMAL)
cv2.resizeWindow('Eye processing', (640,480))
# FPS processing
prev_frame_time = time.time()
curr_frame_time = time.time()

# Prepare perclos buffers
short_perclos_buffer = []
long_perclos_buffer = []
short_perclos_ready = False
long_perclos_ready = False

# Build face analyzer
fa = FaceAnalyzer(max_nb_faces=1)
# Main Loop
while cap.isOpened():
    # Process fps
    curr_frame_time = time.time()
    dt = curr_frame_time-prev_frame_time
    prev_frame_time = curr_frame_time
    fps = 1/dt
    # Read an image from the camera
    success, image = cap.read()
    # Convert it to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Process it
    fa.process(image)
    #Now if we find a face
    if fa.nb_faces==1:
        # Computes eyes opening level and blinks
        left_eye_opening, right_eye_opening, is_blink, last_blink_duration = fa.faces[0].process_eyes(image, detect_blinks=True, blink_th=0.35) #, normalize=True

        # Compute perclos over two spans
        short_perclos = fa.faces[0].compute_perclos(left_eye_opening, right_eye_opening,5*int(fps),short_perclos_buffer)*100
        long_perclos = fa.faces[0].compute_perclos(left_eye_opening, right_eye_opening,60*int(fps),long_perclos_buffer)*100
        if len(short_perclos_buffer)>=5*int(fps):
            short_perclos_ready = True
        if len(long_perclos_buffer)>=60*int(fps):
            long_perclos_ready = True
        fa.faces[0].draw_eyes_landmarks(image)
        # Get eyes positions
        left_eye_pos  = fa.faces[0].get_landmark_pos(fa.faces[0].left_eye_center_index)
        right_eye_pos = fa.faces[0].get_landmark_pos(fa.faces[0].right_eye_center_index)
        
        # Plot eye opening on each eye
        cv2.putText(image, f"{left_eye_opening:2.2f}", (int(left_eye_pos[0]+30), int(left_eye_pos[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255) if left_eye_opening>0.5 else (255,0,0),2)
        cv2.putText(image, f"{right_eye_opening:2.2f}", (int(right_eye_pos[0]-150), int(right_eye_pos[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255) if right_eye_opening>0.5 else (255,0,0),2)

        if is_blink:
            cv2.putText(image, f"Blinking : {n_blinks}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),4)
            print(f"Blinking : {n_blinks}")
            n_blinks += 1
        else:
            cv2.putText(image, f"Blinking : {n_blinks}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)

        # Blink duration
        cv2.putText(image, f"Duration (s) : {last_blink_duration:2.2f}s", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),2)


        # Only after 5 seconds that we can use this perclos
        if short_perclos_ready:
            if short_perclos<20:
                cv2.putText(image, f"Perclos (5 seconds) : {short_perclos:2.2f}%", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)
            else:
                cv2.putText(image, f"Perclos (5 seconds) : {short_perclos:2.2f}%", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),4)

        # Only after 1 minute that we can use this perclos
        if long_perclos_ready:
            if long_perclos<20:
                cv2.putText(image, f"Perclos (1Minute) : {long_perclos:2.2f}%", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)
            else:
                cv2.putText(image, f"Perclos (1Minute) : {long_perclos:2.2f}%", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),4)


    # Show the output 
    try:
        cv2.imshow('Eye processing', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
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
