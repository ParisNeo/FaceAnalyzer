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

# Number of images to use to build the embedding
nb_images=50

# create Tkinter root window
root = tk.Tk()
root.withdraw()


# If faces path is empty then make it
faces_path = Path(__file__).parent/"faces"
if not faces_path.exists():
    faces_path.mkdir(parents=True, exist_ok=True)


# Build face analyzer while specifying that we want to extract just a single face
fa = FaceAnalyzer(max_nb_faces=1)


box_colors=[
    (255,0,0),
    (0,255,0),
    (0,0,255),
    (255,255,0),
    (255,0,255),
    
]


import gradio as gr
import numpy as np
class UI():
    def __init__(self) -> None:
        self.i=0
        self.embeddings_cloud = []
        self.is_recording=False
        self.face_name=None
        self.nb_images = 20
        with gr.Blocks(css='style.css') as demo:
            gr.Markdown("## FaceAnalyzer face recognition test")
            with gr.Tabs():
                with gr.TabItem('Recognize'):
                    with gr.Blocks() as demo2:
                        with gr.Row():
                            with gr.Column():
                                self.img = gr.Image(label="Input Image", source="webcam", streaming=True)
                                self.txtFace_name = gr.Textbox(label="face_name")
                                self.txtFace_name.change(self.set_face_name, inputs=self.txtFace_name)
                                self.status = gr.Label(label="face_name")
                                self.img.change(self.record, inputs=self.img, outputs=self.status)
                            with gr.Column():
                                self.btn_start = gr.Button("Start Recording face")
                                self.btn_start.click(self.start_stop)

        demo.queue().launch()

    def set_face_name(self, face_name):
        self.face_name=face_name

    def start_stop(self):
        self.is_recording=True

    def record(self, image):
        if self.face_name is None:
            return "Please input a face name"
        if self.is_recording and image is not None:
            if self.i < self.nb_images:
                # Process the image to extract faces and draw the masks on the face in the image
                fa.process(image)
                if fa.nb_faces>0:
                    try:
                        face = fa.faces[0]
                        vertices = face.get_face_outer_vertices()
                        image = face.getFaceBox(image, vertices, margins=(40,40,40,40))
                        embedding = DeepFace.represent(image)[0]["embedding"]
                        self.embeddings_cloud.append(embedding)
                        self.i+=1
                        cv2.imshow('Face Mesh', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    except Exception as ex:
                        print(ex)
                return f"Processing frame {self.i}/{self.nb_images}..."
            else:
                # Now let's find out where the face lives inside the latent space (128 dimensions space)

                embeddings_cloud = np.array(self.embeddings_cloud)
                embeddings_cloud_mean = embeddings_cloud.mean(axis=0)
                embeddings_cloud_inv_cov = embeddings_cloud.std(axis=0)
                # Now we save it.
                # create a dialog box to ask for the subject name
                name = self.face_name
                with open(str(faces_path/f"{name}.pkl"),"wb") as f:
                    pickle.dump({"mean":embeddings_cloud_mean, "inv_cov":embeddings_cloud_inv_cov},f)
                print(f"Saved {name} embeddings")
                self.i=0
                self.embeddings_cloud=[]
                self.is_recording=False

                return f"Saved {name} embeddings"
        else:
            return "Waiting"
ui = UI()

