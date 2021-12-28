"""=== main Title =>

    Author : Saifeddine ALOUI
    Description :
        A code to test FaceAnalyzer: puts a face mask on a realtime video feed. Can be used to build Halloween masks...
        Feel free to use other assets if you want.
        In the new versions, a reference triangulation file is used. Dont forget to regenerate another one if you change the landmarks you use.
        The image file used in this code is under creative commons licence.
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
# Parameters
use_simplified_face = True  # If true then the simplified faster version of the face landmarks will be used instead of the full more accurate but slower version
seemless_cloning=False #If true, seemless cloning will be used to blend the mask onto the face

# Select landmarks
if use_simplified_face:
    # Simplified facial features (fast)
    lm_indices = list(set(
                        Face.simplified_face_features+
                        Face.mouth_inner_indices+
                        Face.mouth_outer_indices+
                        Face.left_eyelids_indices+
                        Face.right_eyelids_indices+
                        Face.left_eye_contour_indices+
                        Face.right_eye_contour_indices)) # list(range(468)) #
else:
    # Full face (Slower)
    lm_indices = list(range(478))

# open reference triangulation file
file = Path(__file__).parent/"reference.pkl"
with open(str(file),"rb") as f:
    triangles = pickle.load(f)


# open an image and recover all faces inside it (here there is a single face)
fa_mask = FaceAnalyzer.from_image(str(Path(__file__).parent/"assets/pennywize.jpg"))
mask_face = fa_mask.faces[0]
mask_face.triangles=triangles


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Build a window
cv2.namedWindow('Face Mask', flags=cv2.WINDOW_NORMAL)
cv2.resizeWindow('Face Mask', (640,480))
# Smask_face between viewing the original image with the face triangles on, or applying the mask to the video stream
view_original = True
if view_original:
    cv2.namedWindow('Original', flags=cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Original', (640,480))

# Plot
start_time = time.time_ns()
fa = FaceAnalyzer(max_nb_faces=3)
for face in fa.faces:
    face.triangles = mask_face.triangles

if view_original:
    image = fa_mask.image.copy()
    mask_face.draw_delaunay(image, landmark_indices=lm_indices)
    cv2.imshow('Original', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

while cap.isOpened():
    success, image = cap.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    files = []
    fa.process(image)
    if fa.found_faces:
        for i, face in enumerate(fa.faces):
            if face.ready:                    
                image = mask_face.copyToFace(
                    face, 
                    fa_mask.image, 
                    image, 
                    opacity = 1, 
                    landmark_indices=lm_indices, 
                    min_output_triangle_cross=0.01, 
                    min_input_triangle_cross=20,
                    seemless_cloning=seemless_cloning)
    try:
        cv2.imshow('Face Mask', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    except Exception as ex:
        print(ex)
    # Wait for key stroke
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
cap.release()
