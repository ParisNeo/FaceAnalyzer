"""=== main Title =>

    Author : Saifeddine ALOUI
    Description :
        Builds a reference triangulation face to be used for face masking (regenerate another one if you change the landmarks you are using)
        open your mouth and eyes and press s to generate the file
<================"""

import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import time
from PIL import Image
from FaceAnalyzer import FaceAnalyzer, Face
from pathlib import Path
import pickle

# Select landmark indices to be used to copy faces (if None, all landmarks will be used)
lm_indices = list(set(Face.simplified_face_features+Face.mouth_inner_indices+Face.mouth_outer_indices)) # list(range(468)) #
#lm_indices = Face.all_face_features

# open an image and recover all faces inside it (here there is a single face)
fa_mask = FaceAnalyzer.from_image(str(Path(__file__).parent/"assets/mlk.jpg"))



cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Build a window
cv2.namedWindow('Face Mask', flags=cv2.WINDOW_NORMAL)
cv2.resizeWindow('Face Mask', (640,480))

# Smask_face between viewing the original image with the face triangles on, or applying the mask to the video stream
view_original = False
# Plot
start_time = time.time_ns()
fa = FaceAnalyzer(max_nb_faces=3)
while cap.isOpened():
    success, image = cap.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    files = []
    fa.process(image)
    if fa.found_faces:
        for i, face in enumerate(fa.faces):
            if face.ready:
                triangles = face.triangulate(lm_indices)
                face.draw_delaunay(image, lm_indices)
    try:
        cv2.imshow('Face Mask', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    except Exception as ex:
        print(ex)
    # Wait for key stroke
    wk = cv2.waitKey(5)
    if wk & 0xFF == 27: # If escape is pressed then return
        break
    if wk & 0xFF == 115: # If s is pressed then take a snapshot
        file = Path(__file__).parent/"reference.pkl"
        with open(str(file),"wb") as f:
            pickle.dump(triangles,f)
        print("Saved as reference file")
cap.release()
