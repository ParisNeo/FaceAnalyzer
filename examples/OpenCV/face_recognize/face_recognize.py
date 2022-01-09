"""=============
    Example : extract_face.py
    Author  : Saifeddine ALOUI
    Description :
        An example to show how we can recognize faces.
        Use face_record to record faces and name them, then use this software to recognize them really fast using your webcam
        This is avery fast face recognition tool. But not robust to rotations. It needs to have multiple images from each person to perform well
        
        The image file used in this code is under creative commons licence.

<================"""

import numpy as np
from pathlib import Path
import cv2
import time
from FaceAnalyzer import FaceAnalyzer, Face

import matplotlib.pyplot as plt
from pathlib import Path
import pickle


# Se the maximum distance between the image and the reference (more means less strict)
max_dist = 2

landmark_indices = list(range(468)) # Face.left_eye_brows_indices  + Face.right_eye_brows_indices + Face.nose_indices + Face.forehead_indices + Face.left_eye_left_right_indices + Face.right_eye_left_right_indices
# list(range(468)) # 


# If faces path is empty then 
faces_path = Path(__file__).parent/"faces"
if not faces_path.exists():
    faces_path.mkdir(parents=True, exist_ok=True)

# open camera
cap = cv2.VideoCapture(0)

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

# Load faces
known_faces=[]
known_faces_names=[]
face_files = [f for f in faces_path.iterdir() if f.name.endswith("pkl")]
for file in face_files:
    with open(str(file),"rb") as f:
        faces = pickle.load(f)
    for vertices in faces:
        vertices = vertices[landmark_indices,...]
        ref = vertices[::-1,:]
        distances_list = np.linalg.norm(ref-vertices, axis=1)
        known_faces.append(distances_list)
        known_faces_names.append(file.stem)

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
            face.draw_landmarks(image, face.npLandmarks[landmark_indices, ...], color=(0,0,0))
            vertices = face.get_3d_realigned_landmarks_pos()[:,:2]
            face.draw_landmarks(image,vertices[landmark_indices, ...], color=(255,0,0))
            vertices-=vertices.min(axis=0)
            vertices/=vertices.max(axis=0)
            vertices = vertices[landmark_indices,...]
            ref = vertices[::-1,:]
            # Let's normalize everything

            d = np.linalg.norm(ref-vertices, axis=1)
            nearest_distance = 1e100 # far 
            nearest = 0 # Nearest one
            for i, known_face in enumerate(known_faces):
                distance = np.mean((d-known_face)**2)
                if distance<nearest_distance:
                    nearest_distance = distance
                    nearest = i
            #face.draw_bounding_box(image, thickness=5,text=f"{known_faces_names[nearest]}: {100*(max_dist-nearest_distance)/max_dist:.2f}%" )
            face.draw_bounding_box(image, thickness=5,text=f"{known_faces_names[nearest]}: {100*(max_dist-nearest_distance)/max_dist:.2f}%" if nearest_distance<max_dist else "unknown")
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
