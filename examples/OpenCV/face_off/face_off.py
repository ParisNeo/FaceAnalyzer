"""=============
    Example : face_off.py
    Author  : Saifeddine ALOUI
    Description :
        A code to test FaceAnalyzer Copies a face from person to person on the same image using webcam
<================"""

import numpy as np
from pathlib import Path
import cv2
import time
from FaceAnalyzer import FaceAnalyzer

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Build a window
cv2.namedWindow('Face Off', flags=cv2.WINDOW_NORMAL)
cv2.resizeWindow('Face Off', (640,480))

# Build face analyzer
fa = FaceAnalyzer(max_nb_faces=2)
# Main Loop
while cap.isOpened():
    # Read an image from the camera
    success, image = cap.read()
    # Convert it to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Process it
    fa.process(image)
    #Now if we find two faces, we switch them
    if fa.nb_faces==2:
        # Extract face triangles from Face 1
        #fa.faces[0].triangulate(landmark_indices=fa.faces[0].simplified_face_features)
        # Set face 2 triangles as those of face1
        #fa.faces[1].triangles=fa.faces[0].triangles
        # Make a copy of the original image (because we will be switching twice)
        img = image.copy()
        # Put face0 in face 1
        image = fa.faces[0].copyToFace(fa.faces[1], img, image, opacity = 0.8, landmark_indices=fa.faces[0].simplified_face_features)
        # Put face 1 in face 0
        image = fa.faces[1].copyToFace(fa.faces[0], img, image, opacity = 0.8, landmark_indices=fa.faces[0].simplified_face_features)
    # Show the output 
    try:
        cv2.imshow('Face Off', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
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
