"""=============
    Example : extract_face.py
    Author  : Saifeddine ALOUI
    Description :
        An example to show how we can recognize emotions from facial landmarks.
        This is only the script that does the learning. So you first need to build a little facial emotions database using face_record then run this script

<================"""

import numpy as np
from pathlib import Path
import cv2
import time

from numpy.core.numeric import indices
from FaceAnalyzer import FaceAnalyzer, Face

import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import tensorflow as tf

# Parameters
nb_classes = 5
continue_training = True
n_epochs = 100

landmark_indices = list(range(468)) # Face.left_eye_brows_indices  + Face.right_eye_brows_indices + Face.nose_indices + Face.forehead_indices + Face.left_eye_left_right_indices + Face.right_eye_left_right_indices
# list(range(468)) # 


# If faces path is empty then 
faces_path = Path(__file__).parent/"faces"
if not faces_path.exists():
    faces_path.mkdir(parents=True, exist_ok=True)

# Load faces
known_faces=[]
known_faces_label=[]
face_files = [f for f in faces_path.iterdir() if f.name.endswith("pkl")]
for file in face_files:
    with open(str(file),"rb") as f:
        faces = pickle.load(f)
    for vertices in faces:
        vertices = vertices[landmark_indices,...]
        known_faces.append(vertices[None,...])
        known_faces_label.append(int(file.stem))

known_faces = np.vstack(known_faces)
known_faces_label = np.vstack(known_faces_label)
known_faces_label = tf.keras.utils.to_categorical(known_faces_label)
print(known_faces.shape)
print(known_faces_label.shape)

indices=np.array(list(range(known_faces_label.shape[0])))
np.random.shuffle(indices)
known_faces=known_faces[indices,...]
known_faces_label=known_faces_label[indices,...]
# Build the neural network
net_input = tf.keras.layers.Input((468,2))
x = tf.keras.layers.Flatten()(net_input)
x = tf.keras.layers.Dense(50,"relu")(x)
x0 = tf.keras.layers.Dense(25,"relu")(x)
x = tf.keras.layers.Dense(25,"relu")(x)
x = tf.keras.layers.Concatenate()([x,x0])
x = tf.keras.layers.Dense(20,"relu")(x)
x = tf.keras.layers.Dense(10,"relu")(x)

classification = tf.keras.layers.Dense(nb_classes,"softmax")(x)

model = tf.keras.models.Model(net_input, classification)
model.compile(optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(),metrics=["acc"])
model.summary()

path= Path(__file__).parent/"emotion_recognition_model.h5"
if path.exists() and continue_training:
    try:
        model.load_weights(str(path))
        print("Model loaded successfuly")
    except:
        print("Incompatible model file. New model will be created")
model.fit(known_faces, known_faces_label, epochs=n_epochs, shuffle=True)

model.save(str(path))


