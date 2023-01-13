# FaceAnalyzer
[![GitHub release](https://badgen.net/github/release/ParisNeo/FaceAnalyzer)](https://github.com/ParisNeo/FaceAnalyzer/releases)
[![GitHub license](https://badgen.net/github/license/ParisNeo/FaceAnalyzer)](https://github.com/ParisNeo/FaceAnalyzer/blob/master/LICENSE)


A python library for face detection and features extraction based on mediapipe library

## Introduction
FaceAnalyzer is a library based on mediapipe library and is provided under MIT Licence. It provides an object oriented tool to play around with faces.
It can be used to :
0. Detect faces using the mediapipe library
1. Extract faces from an image (either a box around the face or a face contour cut without background)
2. Measure the face position and orientation
3. Measure eyes openings, number of blinks, blink duration and perclos
4. Measure eye orientation in 3D space
5. Get the 2D gaze position on a predefined 3D plan(s) allowing to understand what the user is looking at
6. Compute face triangulation (builds triangular surfaces that can be used to build 3D models of the face)
7. Copy a face from an image to another.
8. With the help of facenet model, you can use FaceAnalyzer to recognize faces in an image (a ful example is provided under examples/OpenCV/face_recognizer_facenet)
9. A simple face recognition algorithm based on face landmarks is also presented as an example. 
9. A neural network based emotion recognition algorithm is integrated to the examples section at examples/OpenCv/emotion_learner. 

## Requirements
This library requires :
1. mediapipe (used for facial landmarks extraction)
2. opencv used for drawing and image morphing
3. scipy used for efficient delaulay triangulation
4. numpy, as any thing that uses math
5. For some examples, you may need some additional libraries:
    - For face_recognizer_facenet (under opencv examples set) you need to install tensorflow 2.0 or later
    - For pygame examples, install pygame
    - For SQTUI you need to install SQTUI with either PyQT5 or PySide2

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
## Simple example

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
    #We can get the face rectangle image like this
    face_image = face.getFaceBox(frame)
    # We can get the face forehead image like this
    forehead_image = face.getFaceBox(frame, face.face_forhead_indices)

```

## Examples
### OpenCV
Here are opencv based programs
#### face_mesh :
A basic simple example of how to use webcam to get video and process each frame to extract faces and draw face landmarks on the face.
#### from_image :
A basic simple example of how to extract faces from an image file.
#### eye_process :
An example of how to extract faces from a video (using webcam) then process eyes and return eyes openings as well as detecting blinks.
#### face_off :
An example of how to use webcam to switch faces between two persons.
#### face_mask :
An example of how to use webcam to put a mask on a face.
#### animate_image :
An example of how to use webcam to animate a still face using the user's face motion. Just put an image file containing a face in the assets subfolder and select the file in the parameters and you're good to go.
#### extract_face :
An example of how to use webcam to extract only the face (generates a black image with only the face).
#### eyes_tracker :
An example to show how we can get the eyes orientation in space.
#### face_recognizer :
An example to record and then recognize faces in a video stream using facial landmarks. This is a very fast but not robust face recognition tool. Multiple images are needed for a single person on multiple angles to perform better.

The code starts by extracting landmarks. Then reorients the face so that the forehead is up and the chin is down, then normalizes the landmarks positions. Finally, distances between landmarks and their opposite landmarks are computed. This is done for each reference image, and for each frame from the video stream. Then a simple distance is computed between this vector and all the vectors from reference faces, and we take the face that is most close to the one we are watching. If the distance is higher than a threshold, the algorithm just says Unknown.
#### face_recognizer_facenet :
An example to record and then recognize faces in a video stream using facenet neural network.

Here an embedding representation of each reference face is computed. We record multiple frames for each face and get a 128 dimensions vector for each one. The means and standard deviation are computed and saved.

At inference time, each face is extracted and sent to the facenet network. We obtain an embedding. We compute the distance between this embedding and all our database. We take the closest one. If the distance is higher than a threshold, the algorithm says unknown.

This is a more robust tool. Bust requires more resources. It is advised to use a GPU to have a decent framerate.
### Pygame
Here you can find all examples using pygame library
#### win_face_mouse_controller (Only works on Microsoft Windows for now)
A software to control a mouse using the face and blink to press. 
The software provides a tool to calibrate the mouse control using the face by asking the user to look at the top left of the screen, then to the down right.
You can activate the motion using activate button and deactivate it by just pressing it again.
Finally, you can pause eye press control by closing the eye for 2 seconds. To reactivate this close the eye for 2 seconds. After 2 seconds of eye closing, you'll hear a beep that confirms that you have changed the mode.


### SQTUI

#### q_face_infos_graph :
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

#### q_face_pointing_pos_graph :
An example on how we can track face pointing vector and find the position of intersection between the line guided by this vector and a plane defined by at least 3 points in space. We demonstrate how it is possible to detect the intersection of this vector with a region that can be convex or not. This can also be done using gaze vector.

This example allows us to define regions in a 3d space and determine if the user is looking at on object or another. Very useful for example for controlling stuff using gaze or face motion. This can help people with disability to use their gaze to interact with the screen. (A calibration may be required to determine how to position elements in the reference frame of the camera).

The module shows how to use the kalman filter helper to enhance the tracking and remove noise.

#### face_chacer :

A little game where you use your face top chace some animals on the screen. You need to point on them and blink to shoot.
Uses Kalman filter to filter motion which makes it interesting.
