"""=============
    Example : face_off.py
    Author  : Saifeddine ALOUI
    Description :
        An example of loading a face from an image and draw the mask on the face
        Requires matplotlib and pathlib.
        Feel free to use and test on other images
        
        The image file used in this code is under creative commons licence.

<================"""

import numpy as np
from pathlib import Path
import cv2
import time
from FaceAnalyzer import FaceAnalyzer

import matplotlib.pyplot as plt
from pathlib import Path

# open an image and recover all faces inside it (here there is a single face)
fa = FaceAnalyzer.from_image(str(Path(__file__).parent/"assets/corneille.jpg"))
# verify that there is a face found
if fa.nb_faces>0:
    # Draw a mask on the face
    fa.faces[0].draw_mask(fa.image)
# Show the face using matplotlib
plt.figure()
plt.imshow(fa.image)
plt.show()
