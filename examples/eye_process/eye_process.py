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

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Blinks counter
n_blinks = 0

# Build face analyzer
fa = FaceAnalyzer(max_nb_faces=1)
# Main Loop
while cap.isOpened():
    # Read an image from the camera
    success, image = cap.read()
    # Convert it to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Process it
    fa.process(image, draw_mask=False)
    #Now if we find a face
    if fa.nb_faces==1:
        left_eye_opening, right_eye_opening, is_blink = fa.faces[0].process_eyes(image, detect_blinks=True, normalize=True, draw_landmarks=True, blink_th=3)
        
        # Get eyes positions
        left_eye_pos = fa.faces[0].getlandmark_pos(fa.faces[0].left_eye_center_index)
        right_eye_pos = fa.faces[0].getlandmark_pos(fa.faces[0].right_eye_center_index)
        # Plot eye opening on each eye
        cv2.putText(image, f"{left_eye_opening:2.2f}", (int(left_eye_pos[0])-150, int(left_eye_pos[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),1)
        cv2.putText(image, f"{right_eye_opening:2.2f}", (int(right_eye_pos[0])+50, int(right_eye_pos[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),1)

        if is_blink:
            cv2.putText(image, f"Blinking : {n_blinks}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),3)
            print(f"Blinking : {n_blinks}")
            n_blinks += 1
        else:
            cv2.putText(image, f"Blinking : {n_blinks}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    # Show the output 
    try:
        cv2.imshow('Eye processing', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    except Exception as ex:
        print(ex)
            # Wait for key stroke for 5 ms
    wk = cv2.waitKey(5)
    if wk & 0xFF == 27: # If escape is pressed then return
        break

# Close the camera properly
cap.release()