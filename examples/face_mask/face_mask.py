"""=== main Title =>

    Author : Saifeddine ALOUI
    Description :
        A code to test FaceAnalyzer: puts a face mask on a realtime video feed. Can be used to build Halloween masks...
        Feel free to use other assets if you want.
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

# Select landmark indices to be used to copy faces (if None, all landmarks will be used)
#lm_indices = Face.simplified_face_features
lm_indices = Face.all_face_features

# open an image and recover all faces inside it (here there is a single face)
fa_mask = FaceAnalyzer.from_image(str(Path(__file__).parent/"assets/pennywize.jpg"))
mask_face = fa_mask.faces[0]
mask_face.triangulate(lm_indices)


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Build a window
cv2.namedWindow('Face Mask', flags=cv2.WINDOW_NORMAL)
cv2.resizeWindow('Face Mask', (640,480))

# Smask_face between viewing the original image with the face triangles on, or applying the mask to the video stream
view_original = False
# Plot
start_time = time.time_ns()
fa = FaceAnalyzer()
for face in fa.faces:
    face.triangles = mask_face.triangles
while cap.isOpened():
    success, image = cap.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if view_original:
        image = fa_mask.image.copy()
        mask_face.draw_delaunay(image, landmark_indices=lm_indices)
    else:
        files = []
        fa.process(image)
        if fa.found_faces:
            for i, face in enumerate(fa.faces):
                if face.ready:
                    
                    image = mask_face.copyToFace(face, fa_mask.image, image, opacity = 1, landmark_indices=lm_indices)
    try:
        cv2.imshow('Face Mask', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    except Exception as ex:
        print(ex)
    # Wait for key stroke
    wk = cv2.waitKey(5)
    if wk & 0xFF == 27: # If escape is pressed then return
        break
    if wk & 0xFF == 115: # If s ispressed a snapshot is taken
        i = 0
        fn = Path(f"ss_{i}.jpg")
        while fn.exists():
            i += 1
            fn = Path(f"ss_{i}.jpg")

        cv2.imwrite(str(fn), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print("Shot")
cap.release()
