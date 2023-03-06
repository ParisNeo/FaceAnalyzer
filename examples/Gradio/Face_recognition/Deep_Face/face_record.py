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

from numpy.lib.type_check import imag
from FaceAnalyzer import FaceAnalyzer

from pathlib import Path
import pickle
from deepface import DeepFace

# Number of images to use to build the embedding
nb_images=50


# If faces path is empty then make it
faces_path = Path(__file__).parent/"faces"
if not faces_path.exists():
    faces_path.mkdir(parents=True, exist_ok=True)


# Build face analyzer while specifying that we want to extract just a single face
fa = FaceAnalyzer(max_nb_faces=3)


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
        self.nb_faces = 3
        # Important to set. If higher than this distance, the face is considered unknown
        self.threshold = 4e-1
        self.faces_db_preprocessed_path = Path(__file__).parent/"faces_db_preprocessed"
        self.current_name = None
        self.current_face_files = []
        self.draw_landmarks = True
        self.webcam_process = False
        self.upgrade_faces()
        try:
            DeepFace.represent(np.zeros((100,100,3)), enforce_detection=False)
        except Exception as ex:
            pass

        with gr.Blocks() as demo:
            gr.Markdown("## FaceAnalyzer face recognition test")
            with gr.Tabs():
                with gr.TabItem('Realtime Recognize'):
                    with gr.Blocks():
                        with gr.Row():
                            with gr.Column():
                                self.rt_webcam = gr.Image(label="Input Image", source="webcam", streaming=True)
                                self.start_streaming = gr.Button("Start webcam")
                                self.start_streaming.click(self.start_webcam, [], [self.start_streaming])

                            with gr.Column():
                                self.rt_rec_img = gr.Image(label="Output Image")
                                self.rt_webcam.change(self.process_webcam, inputs=self.rt_webcam, outputs=self.rt_rec_img, show_progress=False)
                with gr.TabItem('Image Recognize'):
                    with gr.Blocks():
                        with gr.Row():
                            with gr.Column():
                                self.rt_inp_img = gr.Image(label="Input Image")
                            with gr.Column():
                                self.rt_rec_img = gr.Image(label="Output Image")
                                self.rt_inp_img.change(self.process_image, inputs=self.rt_inp_img, outputs=self.rt_rec_img, show_progress=True)
                with gr.TabItem('Add face from webcam'):
                    with gr.Blocks():
                        with gr.Row():
                            with gr.Column():
                                self.img = gr.Image(label="Input Image", source="webcam", streaming=True)
                                self.txtFace_name = gr.Textbox(label="face_name")
                                self.status = gr.Label(label="Status")
                                self.txtFace_name.change(self.set_face_name, inputs=self.txtFace_name, outputs=self.status, show_progress=False)
                                self.img.change(self.record_from_webcam, inputs=self.img, outputs=self.status, show_progress=False)
                            with gr.Column():
                                self.btn_start = gr.Button("Start Recording face")
                                self.btn_start.click(self.start_stop)
                with gr.TabItem('Add face from files'):
                    with gr.Blocks():
                        with gr.Row():
                            with gr.Column():
                                self.gallery = gr.Gallery(
                                    label="Uploaded Images", show_label=True, height=300, elem_id="gallery"
                                ).style(grid=[2], height="auto")
                                self.btn_clear = gr.Button("Clear Gallery")

                                self.add_file = gr.Files(label="Files",file_types=["image"])
                                self.add_file.change(self.add_files, self.add_file, self.gallery)
                                self.txtFace_name2 = gr.Textbox(label="face_name")
                                self.btn_start = gr.Button("Build face embeddings")
                                self.status = gr.Label(label="Status")
                                self.txtFace_name2.change(self.set_face_name, inputs=self.txtFace_name2, outputs=self.status, show_progress=False)
                                self.btn_start.click(self.record_from_files, inputs=self.gallery, outputs=self.status, show_progress=True)
                                self.btn_clear.click(self.clear_galery,[],[self.gallery, self.add_file])
                with gr.TabItem('Known Faces List'):
                    with gr.Blocks():
                        with gr.Row():
                            with gr.Column():
                                if len(self.known_faces_names)>0:
                                    self.faces_list = gr.Dataframe(
                                        headers=["Face Name"],
                                        datatype=["str"],
                                        label="Faces",
                                        value=[[n] for n in self.known_faces_names]
                                    )
                                else:
                                    self.faces_list = gr.Dataframe(
                                        headers=["Face Name"],
                                        datatype=["str"],
                                        label="Faces"
                                    )
            with gr.Row():
                with gr.Accordion(label="Options", open=False):
                    self.sld_threshold = gr.Slider(1e-2,10,4e-1,step=1e-2,label="Recognition threshold")
                    self.sld_threshold.change(self.set_th,inputs=self.sld_threshold)
                    self.sld_nb_images = gr.Slider(2,50,20,label="Number of images")
                    self.sld_nb_images.change(self.set_nb_images, self.sld_nb_images)
                    self.cb_draw_landmarks = gr.Checkbox(label="Draw landmarks", value=True)
                    self.cb_draw_landmarks.change(self.set_draw_landmarks, self.cb_draw_landmarks)
                    self.sld_nb_faces = gr.Slider(1,50,3,label="Maximum number of faces")
                    self.sld_nb_faces.change(self.set_nb_faces, self.sld_nb_faces)
                    

        demo.queue().launch(share=True)

    def clear_galery(self):
        return self.gallery.update(value=[]), self.add_file.update(value=[])

    def start_webcam(self):
        self.webcam_process=not self.webcam_process
        return self.start_streaming.update(value="Stop webcam") if self.webcam_process else self.start_streaming.update(value="Start webcam")


    def add_files(self, files):
        for file in files:
            img = cv2.cvtColor(cv2.imread(file.name), cv2.COLOR_BGR2RGB)
            self.current_face_files.append(img)
        return self.current_face_files
    
    def set_th(self, value):
        self.threshold=value

    def set_nb_images(self, value):
        self.nb_images=value

    def set_draw_landmarks(self, value):
        self.draw_landmarks=value

    def set_nb_faces(self,nb_faces):
        self.nb_faces = nb_faces
        fa.nb_faces = nb_faces

    def cosine_distance(self, u, v):
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

    def upgrade_faces(self):
        # Load faces
        self.known_faces=[]
        self.known_faces_names=[]
        face_files = [f for f in faces_path.iterdir() if f.name.endswith("pkl")]
        for file in face_files:
            with open(str(file),"rb") as f:
                finger_print = pickle.load(f)
                self.known_faces.append(finger_print)
            self.known_faces_names.append(file.stem)
            
        if hasattr(self, "faces_list"):
            self.faces_list.update([[n] for n in self.known_faces_names])

    def set_face_name(self, face_name):
        self.face_name=face_name
        return f"face name set to {self.face_name}"

    def start_stop(self):
        self.is_recording=True
        

    def process_db(self, images):
        for i,image in enumerate(images):
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
                    embedding = DeepFace.represent(image, enforce_detection=False)[0]["embedding"]
                    embeddings_cloud.append(embedding)
                    cv2.imwrite(str(self.faces_db_preprocessed_path/f"im_{i}.png"), cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 
                except Exception as ex:
                    print(ex)
        embeddings_cloud = np.array(embeddings_cloud)
        embeddings_cloud_mean = embeddings_cloud.mean(axis=0)
        embeddings_cloud_inv_cov = np.linalg.inv(np.cov(embeddings_cloud.T))
        # Now we save it.
        # create a dialog box to ask for the subject name
        name = self.face_name
        with open(str(faces_path/f"{name}.pkl"),"wb") as f:
            pickle.dump({"mean":embeddings_cloud_mean, "inv_cov":embeddings_cloud_inv_cov},f)
        print(f"Saved {name}")

    def record_from_webcam(self, image):
        if self.face_name is None:
            self.embeddings_cloud=[]
            self.is_recording=False
            return "Please input a face name"
        
        if self.is_recording and image is not None:
            if self.i < self.nb_images:
                fa.image_size=(640, 480, 3)

                # Process the image to extract faces and draw the masks on the face in the image
                fa.process(image)
                if fa.nb_faces>0:
                    try:
                        face = fa.faces[0]
                        vertices = face.get_face_outer_vertices()
                        image = face.getFaceBox(image, vertices, margins=(40,40,40,40))
                        embedding = DeepFace.represent(image, enforce_detection=False)[0]["embedding"]
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
                self.upgrade_faces()

                return f"Saved {name} embeddings"
        else:
            return "Waiting"
        
    def record_from_files(self, images):
        if self.face_name is None:
            self.embeddings_cloud=[]
            self.is_recording=False
            return "Please input a face name"
        
        if images is not None:
            for entry in images:
                image = cv2.cvtColor(cv2.imread(entry["name"]), cv2.COLOR_BGR2RGB)
                if image is None:
                    return None
                # Process the image to extract faces and draw the masks on the face in the image
                if image.shape[1]>640:
                    image = cv2.resize(image,(int(640*(image.shape[1]/image.shape[0])),640))
                fa.image_size=(image.shape[1],image.shape[0],3)
                # Process the image to extract faces and draw the masks on the face in the image
                fa.process(image)
                if fa.nb_faces>0:
                    try:
                        face = fa.faces[0]
                        vertices = face.get_face_outer_vertices()
                        image = face.getFaceBox(image, vertices, margins=(40,40,40,40))
                        embedding = DeepFace.represent(image, enforce_detection=False)[0]["embedding"]
                        self.embeddings_cloud.append(embedding)
                        self.i+=1
                        cv2.imshow('Face Mesh', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    except Exception as ex:
                        print(ex)
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
            self.upgrade_faces()

            return f"Saved {name} embeddings"
        else:
            return "Waiting"

    def process_webcam(self, image):
        if not self.webcam_process:
            return None
        
        fa.image_size=(640, 480, 3)
        # Process the image to extract faces and draw the masks on the face in the image
        fa.process(image)

        if fa.nb_faces>0:
            for i in range(fa.nb_faces):
                try:
                    face = fa.faces[i]
                    vertices = face.get_face_outer_vertices()
                    face_image = face.getFaceBox(image, vertices, margins=(40,40,40,40))
                    embedding = DeepFace.represent(face_image, enforce_detection=False)[0]["embedding"]
                    if self.draw_landmarks:
                        face.draw_landmarks(image, color=(0,0,0))
                    nearest_distance = 1e100
                    nearest = 0
                    for i, known_face in enumerate(self.known_faces):
                        # absolute distance
                        distance = np.abs(known_face["mean"]-embedding).sum()
                        # euclidian distance
                        #diff = known_face["mean"]-embedding
                        #distance = np.sqrt(diff@diff.T)
                        # Cosine distance
                        distance = self.cosine_distance(known_face["mean"], embedding)
                        if distance<nearest_distance:
                            nearest_distance = distance
                            nearest = i
                            
                    if nearest_distance>self.threshold:
                        face.draw_bounding_box(image, thickness=1,text=f"Unknown:{nearest_distance:.3e}")
                    else:
                        face.draw_bounding_box(image, thickness=1,text=f"{self.known_faces_names[nearest]}:{nearest_distance:.3e}")
                except Exception as ex:
                    pass

        # Return the resulting frame
        return image      
        
    def process_image(self, image):
        if image is None:
            return None
        # Process the image to extract faces and draw the masks on the face in the image
        if image.shape[1]>640:
            image = cv2.resize(image,(int(640*(image.shape[1]/image.shape[0])),640))
        fa.image_size=(image.shape[1],image.shape[0],3)
        
        fa.process(image)

        if fa.nb_faces>0:
            for i in range(fa.nb_faces):
                try:
                    face = fa.faces[i]
                    vertices = face.get_face_outer_vertices()
                    face_image = face.getFaceBox(image, vertices, margins=(40,40,40,40))
                    embedding = DeepFace.represent(face_image, enforce_detection=False)[0]["embedding"]
                    if self.draw_landmarks:
                        face.draw_landmarks(image, color=(0,0,0))
                    nearest_distance = 1e100
                    nearest = 0
                    for i, known_face in enumerate(self.known_faces):
                        # absolute distance
                        distance = np.abs(known_face["mean"]-embedding).sum()
                        # euclidian distance
                        #diff = known_face["mean"]-embedding
                        #distance = np.sqrt(diff@diff.T)
                        # Cosine distance
                        distance = self.cosine_distance(known_face["mean"], embedding)
                        if distance<nearest_distance:
                            nearest_distance = distance
                            nearest = i
                            
                    if nearest_distance>self.threshold:
                        face.draw_bounding_box(image, thickness=1,text=f"Unknown:{nearest_distance:.3e}")
                    else:
                        face.draw_bounding_box(image, thickness=1,text=f"{self.known_faces_names[nearest]}:{nearest_distance:.3e}")
                except Exception as ex:
                    image=face_image

        # Return the resulting frame
        return image        
ui = UI()

