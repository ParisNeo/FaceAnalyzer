"""=============
    Example : face_box.py
    Author  : Saifeddine ALOUI
    Description :
        This code allows snapping landmarks and saving them to a reference pickle file 
<================"""
from FaceAnalyzer import FaceAnalyzer, Face,  DrawingSpec

import numpy as np
import cv2
import time
from pathlib import Path
import mediapipe as mp

# open camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Build face analyzer while specifying that we want to extract just a single face
fa = FaceAnalyzer(max_nb_faces=1)

# Main Loop
while cap.isOpened():
    # Read image
    success, image = cap.read()
    
    # Opencv uses BGR format while mediapipe uses RGB format. So we need to convert it to RGB before processing the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image to extract faces and draw the masks on the face in the image
    fa.process(image)

    # If there is a face then grub it
    if fa.nb_faces>0:
        face = fa.faces[0]
        face.draw_mask(image, 
                        landmarks_drawing_spec= None,
                        contours_drawing_specs= DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=0),
                        contour = mp.solutions.face_mesh.FACEMESH_TESSELATION)
        face.draw_bounding_box(image, thickness=5)
    # Show the image
    try:
        cv2.imshow('Face Mesh', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    except Exception as ex:
        print(ex)
    
    # Wait for key stroke for 5 ms
    wk = cv2.waitKey(5)
    if wk & 0xFF == 27: # If escape is pressed then return
        break
    if wk & 0xFF == 115: # If s ispressed a snapshot is taken
        i = 0
        fn = Path(__file__).parent / f"snapshots/fss_{i}.npy"
        while fn.exists():
            i += 1
            fn = Path(f"fss_{i}.npy")

        np.save(str(fn),face.npLandmarks)
        print("landmarks Saved")
# Close the camera properly
cap.release()
