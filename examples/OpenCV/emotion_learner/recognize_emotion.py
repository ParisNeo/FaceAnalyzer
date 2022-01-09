"""=============
    Example : recognize_emotion.py
    Author  : Saifeddine ALOUI (ParisNeo)
    Description :
        An example to show how to use faceanalyzer to reognize face emotions

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

classifications_strings=["Neutral","Happy","Sad","Ungry","Surprized"]
landmark_indices = list(range(468)) # Face.left_eye_brows_indices  + Face.right_eye_brows_indices + Face.nose_indices + Face.forehead_indices + Face.left_eye_left_right_indices + Face.right_eye_left_right_indices
# list(range(468)) # 

emotionnet_path = Path(__file__).parent/"emotion_recognition_model.h5"
if not emotionnet_path.exists():
    print("Couldn't fine emotionnet.\nPlease train a model first")
    exit()
# 

emotionnet = tf.keras.models.load_model(str(emotionnet_path))

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

emotions_colors=[
    (255,255,255),
    (255,255,0),
    (0,0,255),
    (255,0,0),
    (255,0,255),
]
# Get camera calibration parameters
calibration_file_name = Path(__file__).parent/"cam_calib.pkl"
if calibration_file_name.exists():
    with open(str(calibration_file_name),"rb") as f:
        calib = pickle.load(f)
    mtx = calib["mtx"]
    dist = calib["dist"]
else:
    mtx = None
    dist = np.zeros((4, 1))

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
            vertices = face.npLandmarks[landmark_indices,:2]
            vertices-=vertices.min(axis=0)
            vertices/=vertices.max(axis=0)
            classification = emotionnet.predict(vertices[None,...])[0,...]
            classification_id = classification.argmax()
            #face.draw_landmarks(image, color=(0,0,0))
            vertices = face.get_3d_realigned_face(camera_matrix=mtx, dist_coeffs=dist)
            #face.draw_landmarks(image,vertices, color=(255,0,0))
            face.draw_bounding_box(image, thickness=2, color=emotions_colors[classification_id],text=f"{classifications_strings[classification_id]}: {100*classification[classification_id]:.2f}%")
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
