"""=============
    Example : extract_face.py
    Author  : Saifeddine ALOUI
    Description :
        Records a person's face in order to recognize it later.
        Press s to take a face shot and then give the name of the person.
        Next, use face_recognize to recognize the face you have saved.
        You may take multiple images for the same face. Just give the same name and the reference will be added to the other faces of the same person

<================"""

from pathlib import Path
import cv2
import time

from FaceAnalyzer import FaceAnalyzer

from pathlib import Path
import pickle
import tkinter as tk
from tkinter import simpledialog

# create Tkinter root window
root = tk.Tk()
root.withdraw()

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
        # Get a realigned version of the landmarks
        vertices = face.get_3d_realigned_landmarks_pos()[:,:2] 
        vertices-= vertices.min(axis=0)
        # If you want to hide the face and only show the realigned landmarks
        #image = np.zeros_like(image)
        face.draw_landmarks(image,vertices, color=(255,0,0))
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
        # Let's normalize everything
        vertices-=vertices.min(axis=0)
        vertices/=vertices.max(axis=0)
        # Now we save it.
        # create a dialog box to ask for the subject name
        name = simpledialog.askstring(title="Subject Name", prompt="Enter name:")
        fn = faces_path/f"{name}.pkl"
        if fn.exists():
            with open(str(fn),"rb") as f:
                faces = pickle.load(f)
                faces.append(vertices)
        else:
            faces=[vertices]
        with open(str(fn),"wb") as f:
            pickle.dump(faces,f)
        print("Saved")

# Close the camera properly
cap.release()
