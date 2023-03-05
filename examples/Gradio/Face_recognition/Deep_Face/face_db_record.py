"""=============
    Example : face_db_record.py
    Author  : Saifeddine ALOUI (ParisNeo)
    Description :
        Builds face signature out of a list of images 
        This requires that you put a list of photos of the face you are trying to learn inside a subfolder with the name of the person inside the subforlder faces_db.
        For example if I have two persons Bob and Alice, My treez looks like this
        faces_db
          |-Alice
             | Photos of Alice
          |-Bob
             | Photos of Bob

        Make sure the images does contain only the face of the person to learn  
        This code should build a new file in faces. The file is called the name of the person .pkl

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
from deepface import DeepFace


# If faces path is empty then make it
faces_path = Path(__file__).parent/"faces"
if not faces_path.exists():
    faces_path.mkdir(parents=True, exist_ok=True)

faces_db_preprocessed_path = Path(__file__).parent/"faces_db_preprocessed"
if not faces_db_preprocessed_path.exists():
    faces_db_preprocessed_path.mkdir(parents=True, exist_ok=True)


# Build face analyzer while specifying that we want to extract just a single face
fa = FaceAnalyzer(max_nb_faces=1)

faces_db_path = Path(__file__).parent/"faces_db"

for person in  faces_db_path.iterdir():
    if person.is_dir():
        embeddings_cloud = []
        for image_path in person.iterdir():
             print(f"Processing : {image_path}")
             if image_path.is_file() and image_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                image = cv2.imread(str(image_path))
                # Opencv uses BGR format while mediapipe uses RGB format. So we need to convert it to RGB before processing the image
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (640, 480))
                # Process the image to extract faces and draw the masks on the face in the image
                fa.process(image)

                if fa.nb_faces>0:
                    if fa.nb_faces>1:
                        print("Found too many faces!!")
                    face = fa.faces[0]
                    try:
                        # Get a realigned version of the landmarksx
                        vertices = face.get_face_outer_vertices()
                        image = face.getFaceBox(image, vertices,margins=(30,30,30,30))
                        embedding = DeepFace.represent(image)[0]["embedding"]
                        embeddings_cloud.append(embedding)
                        cv2.imwrite(str(faces_db_preprocessed_path/f"im_{image_path.stem}.png"), cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 
                    except Exception as ex:
                        print(ex)
        embeddings_cloud = np.array(embeddings_cloud)
        embeddings_cloud_mean = embeddings_cloud.mean(axis=0)
        embeddings_cloud_inv_cov = np.linalg.inv(np.cov(embeddings_cloud.T))
        # Now we save it.
        # create a dialog box to ask for the subject name
        name = person.stem
        with open(str(faces_path/f"{name}.pkl"),"wb") as f:
            pickle.dump({"mean":embeddings_cloud_mean, "inv_cov":embeddings_cloud_inv_cov},f)
        print(f"Saved {name}")


