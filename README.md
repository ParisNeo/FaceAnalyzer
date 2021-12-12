# FaceAnalyzer
A python library for face detection and features extraction based on mediapipe library

## Introduction
FaceAnalyzer is a library based on mediapipe library and is provided under MIT Licence. It provides an object oriented tool to play around with faces.
It can be used to :
1. Extract faces from an image
2. Measure the face position and orientation
3. Measure eyes openings
4. Detect blinks
5. Extract the face from an image (useful for face learning applications)
6. Compute face triangulation (builds triangular surfaces that can be used to build 3D models of the face)
7. Copy a face from an image to another.

## Requirements
This library requires :
1. mediapipe (used for facial landmarks extraction)
2. opencv used for drawing and image morphing
3. scipy used for efficient delaulay triangulation
4. numpy, as any thing that uses math


## How to install
Just install from pipy. 
```bash
pip install FaceAnalyzer
```
Make sure you upgrade the library from time to time as I am adding new features so frequently those days.

```bash
pip install FaceAnalyzer --upgrade
```

## How to use

```python
# Import the two main classes FaceAnalyzer and Face 
from FaceAnalyzer import FaceAnalyzer, Face

fa = FaceAnalyzer()
# ... Recover an image in RGB format as numpy array (you can use pillow opencv but if you use opencv make sure you change the color space from BGR to RGB)
# Now process the image
fa.process(image)

# Now you can find faces in fa.faces which is a list of instances of object Face
if fa.nb_faces>0:
    print(f"{fa.nb_faces} Faces found")
    # We can get the landmarks in numpy format NX3 where N is the number of the landmarks and 3 is x,y,z coordinates 
    print(fa.faces[0].npLandmarks)
    # We can draw all landmarks
    # Get head position and orientation compared to the reference pose (here the first frame will define the orientation 0,0,0)
    pos, ori = fa.faces[0].get_head_posture(orientation_style=1)

```
Make sure you look at the examples folder in the repository for more details.
# Structure
The library is structured as follow:

 - Helpers : A module containing Helper functions, namely geometric transformation between rotation formats, or generation of camera matrix etc
 - FaceAnalyzer : A module to process images and extract faces
 - Face         : The main module that represents a face. Allows doing multiple operations such as copying the face and put it on another one or estimate eye opening, head position/orientation in space etc.
 
## Examples
### face_mesh :
A basic simple example of how to use webcam to get video and process each frame to extract faces and draw face landmarks on the face.
### from_image :
A basic simple example of how to extract faces from an image file.
### eye_process :
An example of how to extract faces from a video (using webcam) then process eyes and return eyes openings as well as detecting blinks.
### face_off :
An example of how to use webcam to switch faces between two persons.
### face_mask :
An example of how to use webcam to put a mask on a face.

### q_face_infos_graph :
An example to view face and eye information over time (uses pyqt or pySide through SQTUI library + pyqtgraph)
please install sqtui using pip:

```
pip install sqtui pyqt5
```

or

```
pip install sqtui pyside2
```
Please notice that pyqt is a GPL3 library so if you need your code t be closed at some level, don't use it or consider paying a licence to pyQt to buy a comercial licence.
As of pySide, it is a LGPL library which contaminates your code only if you link it statically.

Using sqtui allows you to select pyqt5 or pyside2 by setting an environment variable at the beginning of your python code. The rest of the coding will be transparent.

```python
os.environ['PYQTGRAPH_QT_LIB']="PySide2"
```

We use the same environment variable used by PYQTGRAPH to avoid having two different environment variables and to synchronize stqui and pyqtgraph on the basme backbone.

