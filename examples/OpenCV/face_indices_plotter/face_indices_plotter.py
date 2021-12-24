"""=============
    Example : face_box.py
    Author  : Saifeddine ALOUI
    Description :
        A code to allow viewing landmarks indices and save them to a high resolution image
<================"""
from FaceAnalyzer import FaceAnalyzer, Face,  DrawingSpec, buildCameraMatrix, faceOrientation2Euler
import numpy as np
import cv2
import time
from pathlib import Path
# open camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Build a window
cv2.namedWindow('Landmarks indices', flags=cv2.WINDOW_NORMAL)
cv2.resizeWindow('Landmarks indices', (640,480))

cv2.namedWindow('OUTPUT', flags=cv2.WINDOW_NORMAL)
cv2.resizeWindow('OUTPUT', (640,480))

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
def draw_contour(image, landmarks, mulx, muly, color=(255,0,0), thickness=2):
    n_lm=len(landmarks)
    for i in range(n_lm):
        p= (int((landmarks[i,0]-p0[0])*mulx), int((landmarks[i,1]-p0[1])*muly))
        p1= (int((landmarks[(i+1)%n_lm,0]-p0[0])*mulx), int((landmarks[(i+1)%n_lm,1]-p0[1])*muly))
        cv2.line(image, p, p1,color, thickness)


def draw_contours(image, face:Face, landmarks_indices, mulx, muly, color=(255,255,255)):
    landmarks = face.get_landmarks_pos(landmarks_indices)
    face.draw_contour(image, landmarks, color)
    draw_contour(output_image, landmarks, mulx, muly, color)


# Main Loop
while cap.isOpened():
    # Read image
    success, image = cap.read()
    
    # Opencv uses BGR format while mediapipe uses RGB format. So we need to convert it to RGB before processing the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output_image = np.zeros((2560*6,1440*8,3), dtype=np.uint8)
    # Process the image to extract faces and draw the masks on the face in the image
    fa.process(image)

    if fa.nb_faces>0:
        for i in range(fa.nb_faces):
            face = fa.faces[i]
            # Get head position and orientation compared to the reference pose (here the first frame will define the orientation 0,0,0)
            p0 = face.npLandmarks.min(axis=0)-np.array([5,5,0])
            p1=face.npLandmarks.max(axis=0)+np.array([10,10,0])

            mulx = output_image.shape[1]/(p1[0]-p0[0])
            muly = output_image.shape[0]/(p1[1]-p0[1])
            for i in range(face.npLandmarks.shape[0]):
                p= (int((face.npLandmarks[i,0]-p0[0])*mulx), int((face.npLandmarks[i,1]-p0[1])*muly))
                output_image = cv2.circle(output_image, p, 1, (255,0,0), 2)
                output_image = cv2.putText(output_image,f"{i}",  p,cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 2)

            face.draw_landmarks(image, face.npLandmarks,color=(0,0,0))            

            draw_contours(image, face, face.left_eye_contour_indices, mulx, muly, (0,255,255))
            draw_contours(image, face, face.left_eyelids_indices, mulx, muly, (255,0,0))

            draw_contours(image, face, face.right_eye_contour_indices, mulx, muly, (0,255,255))
            draw_contours(image, face, face.right_eyelids_indices, mulx, muly, (255,0,0))

            draw_contours(image, face, face.mouth_outer_indices, mulx, muly, (0,255,255))
            draw_contours(image, face, face.mouth_inner_indices, mulx, muly, (255,0,255))
            
            draw_contours(image, face, face.nose_indices, mulx, muly, (255,0,255))


    # Process fps
    curr_frame_time = time.time()
    dt = curr_frame_time-prev_frame_time
    prev_frame_time = curr_frame_time
    fps = 1/dt
    # Show FPS
    cv2.putText(
        image, f"FPS : {fps:2.2f}", (10, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))

    # Show the image
    try:
        cv2.imshow('Landmarks indices', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
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
        cv2.imwrite(str(file),cv2.cvtColor(output_image,cv2.COLOR_BGR2RGB))
        cv2.imshow('OUTPUT', cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        print("Shot")

# Close the camera properly
cap.release()
