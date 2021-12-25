"""=============
    Example : extract_face.py
    Author  : Saifeddine ALOUI
    Description :
        Records a person's face(s) in order to tell its emotion.
        0 : Neutral
        1 : Happy
        2 : Sad
        3 : Angry
        4 : Surprized
        Press s to start recording an emotion. Then you will be asked to put a number (0 for neutral etc). When you press return the app will start recording so make sure you start making the emotion before the recording starts.
        To stop recording press e
        You may record multiple emotions in the same session. just press s to start a new emotion

<================"""

import numpy as np
from pathlib import Path
import cv2
import time

from FaceAnalyzer import FaceAnalyzer

import matplotlib.pyplot as plt
from pathlib import Path
import pickle

# If faces path is empty then 
faces_path = Path(__file__).parent/"faces"
if not faces_path.exists():
    faces_path.mkdir(parents=True, exist_ok=True)


# open camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

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
    (255,0,255),
    (255,0,255),
    
]
# A variable to tell if we are recording or not
recording=False
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
        # Get a realigned version of the landmarks
        vertices = face.get_3d_realigned_landmarks_pos()[:,:2] 
        vertices-= vertices.min(axis=0)
        # If you want to hide the face and only show the realigned landmarks
        #image = np.zeros_like(image)
        face.draw_landmarks(image,vertices, color=(255,0,0))
        face.draw_landmarks(image,color=(0,0,0))
        if recording:
            # Let's normalize everything
            vertices-=vertices.min(axis=0)
            vertices/=vertices.max(axis=0)
            # Now we save it.
            try:
                if fn.exists():
                    with open(str(fn),"rb") as f:
                        faces = pickle.load(f)
                        faces.append(vertices)
                else:
                    faces=[vertices]
                with open(str(fn),"wb") as f:
                    pickle.dump(faces,f)
                print("Saved")
            except:
                print("!!! Problem")


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
        name = input("Emotion ID ?")
        try:
            name=int(name)
            fn = faces_path/f"{name}.pkl"
            recording=True
        except:
            print("!!! Problem")
    if wk & 0xFF == 101: # If s is pressed then take a snapshot
        recording=False
# Close the camera properly
cap.release()
