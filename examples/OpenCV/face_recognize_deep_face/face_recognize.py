"""=============
    Example : extract_face.py
    Author  : Saifeddine ALOUI (ParisNeo)
    Description :
        An example to show how to combine face analyzer with facenet and use it to recognier faces
        Make sure you install deepface
        pip install deepface


<================"""

import numpy as np
from pathlib import Path
import cv2
import time
from FaceAnalyzer import FaceAnalyzer

import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import tensorflow as tf
from tqdm import tqdm  # used to draw a progress bar pip install tqdm
from deepface import DeepFace

# Important to set. If higher than this distance, the face is considered unknown
threshold = 4e-1


def cosine_distance(u, v):
    """
    Computes the cosine distance between two vectors.

    Parameters:
        u (numpy array): A 1-dimensional numpy array representing the first vector.
        v (numpy array): A 1-dimensional numpy array representing the second vector.

    Returns:
        float: The cosine distance between the two vectors.
    """
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    return 1 - (dot_product / (norm_u * norm_v))



# If faces path is empty then make it
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
        finger_print = pickle.load(f)
        known_faces.append(finger_print)
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
            try:
                face = fa.faces[i]
                vertices = face.get_face_outer_vertices()
                face_image = face.getFaceBox(image, vertices, margins=(40,40,40,40))
                embedding = DeepFace.represent(face_image)[0]["embedding"]
                face.draw_landmarks(image, color=(0,0,0))
                vertices = face.get_realigned_landmarks_pos()[:,:2]
                face.draw_landmarks(image,vertices, color=(255,0,0))
                nearest_distance = 1e100
                nearest = 0
                for i, known_face in enumerate(known_faces):
                    # absolute distance
                    distance = np.abs(known_face["mean"]-embedding).sum()
                    # euclidian distance
                    #diff = known_face["mean"]-embedding
                    #distance = np.sqrt(diff@diff.T)
                    # Cosine distance
                    distance = cosine_distance(known_face["mean"], embedding)
                    if distance<nearest_distance:
                        nearest_distance = distance
                        nearest = i
                        
                if nearest_distance>threshold:
                    face.draw_bounding_box(image, thickness=5,text=f"Unknown:{nearest_distance:.3e}")
                else:
                    face.draw_bounding_box(image, thickness=5,text=f"{known_faces_names[nearest]}:{nearest_distance:.3e}")
            except Exception as ex:
                pass
            #face.draw_bounding_box(image, thickness=5,text=f"{known_faces_names[nearest]}: {100*(max_dist-nearest_distance)/max_dist:.2f}%" if nearest_distance<max_dist else "unknown")
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
