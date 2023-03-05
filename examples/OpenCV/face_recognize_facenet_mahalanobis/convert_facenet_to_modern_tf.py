"""=============
    Example : convert_facenet_to_modern_tf.py
    Author  : Saifeddine ALOUI (ParisNeo)
    Description :
        An example to show how to combine face analyzer with facenet and use it to recognier faces with model tensorflow
        The pretrained facenet weights are provided by Hiroki Taniai in the following link but the model version is provided in an old keras format
        This code converts the file to modern format and makes it simple to use
        Download the facenet model from here : https://drive.google.com/drive/folders/1-Frhel960FIv9jyEWd_lwY5bVYipizIT?usp=sharing
        Put the file facenet_keras_weights.h5 in facenet subfolder 
        you need to install tensorflow first
        pip install tensorflow
        On windows we advise installing tensorflow 2.10 if you want to use the gpu
        then run this script to convert facenet to modern tensorflow
<================"""
from pathlib import Path
old_weights_path = Path(__file__).parent/"facenet/facenet_keras_weights.h5"
new_weights_path = Path(__file__).parent/"facenet/facenet.h5"

from inception_resnet_v1 import InceptionResNetV1
def load_weigths_and_save_as_model():
    """
    Arguments:
    None
    Output:
    a model instance
    This method is not meant to be called outside class
    """
    # load model from source
    model = InceptionResNetV1()
    model.load_weights(str(old_weights_path))
    model.save(str(new_weights_path))

print("Converting facenet weights to a model...")
load_weigths_and_save_as_model()
print("OK")