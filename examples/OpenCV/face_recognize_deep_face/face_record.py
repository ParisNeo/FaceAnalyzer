"""=============
    Example : extract_record.py
    Author  : Saifeddine ALOUI (ParisNeo)
    Description :
        Make sure you install deepface
        pip install deepface

<================"""

import numpy as np
from pathlib import Path
import cv2
import time

from numpy.lib.type_check import imag
from FaceAnalyzer import FaceAnalyzer

import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import tensorflow as tf
from tqdm import tqdm  # used to draw a progress bar pip install tqdm
import tkinter as tk
from tkinter import simpledialog
from deepface import DeepFace

# create Tkinter root window
root = tk.Tk()
root.withdraw()


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
fa = FaceAnalyzer(max_nb_faces=1)

# FPS processing
prev_frame_time = time.time()
curr_frame_time = time.time()

box_colors=[
    (255,0,0),
    (0,255,0),
    (0,0,255),
    (255,255,0),
    (255,0,255),
    
]
# Main Loop
while cap.isOpened():
    # Read image
    success, image = cap.read()
    
    # Opencv uses BGR format while mediapipe uses RGB format. So we need to convert it to RGB before processing the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image to extract faces and draw the masks on the face in the image
    fa.process(image)

    if fa.nb_faces>0:
        face = fa.faces[0]
        # Get a realigned version of the landmarksx
        vertices = face.get_face_outer_vertices()
        image = face.getFaceBox(image, vertices)
        
        #face.draw_landmarks(image,vertices, color=(255,0,0))
        face.draw_landmarks(image,color=(0,0,0))


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
        embeddings_cloud = []
        i = 0
        for i in tqdm(range(10)):
            # Read image
            success, image = cap.read()
            
            # Opencv uses BGR format while mediapipe uses RGB format. So we need to convert it to RGB before processing the image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image to extract faces and draw the masks on the face in the image
            fa.process(image)
            if fa.nb_faces>0:
                try:
                    face = fa.faces[0]
                    vertices = face.get_face_outer_vertices()
                    image = face.getFaceBox(image, vertices, margins=(40,40,40,40))
                    embedding = DeepFace.represent(image)[0]["embedding"]
                    embeddings_cloud.append(embedding)
                    time.sleep(1)
                except Exception as ex:
                    print(ex)
            try:
                cv2.imshow('Face Mesh', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            except Exception as ex:
                print(ex)
            wk = cv2.waitKey(5)
        # Now let's find out where the face lives inside the latent space (128 dimensions space)

        embeddings_cloud = np.array(embeddings_cloud)
        embeddings_cloud_mean = embeddings_cloud.mean(axis=0)
        embeddings_cloud_inv_cov = embeddings_cloud.std(axis=0)
        # Now we save it.
        # create a dialog box to ask for the subject name
        name = simpledialog.askstring(title="Subject Name", prompt="Enter name:")
        with open(str(faces_path/f"{name}.pkl"),"wb") as f:
            pickle.dump({"mean":embeddings_cloud_mean, "inv_cov":embeddings_cloud_inv_cov},f)
        print("Saved")

# Close the camera properly
cap.release()
