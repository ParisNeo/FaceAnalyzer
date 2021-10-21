# -*- coding: utf-8 -*-
"""=== Face Analyzer =>
    Face analyzer module
    Author : Saifeddine ALOUI (ParisNeo)
    Licence : MIT
    Description :
        An object oriented library for face operations.
        Allows the extraction of face landmarks using mediapipe library
        Extracted faces are then placed in object of type Face.
        Each face object allow to do multiple operations of the face:
            1 - get face position and orientation in 3D space
            2 - process eyes and detect blinks, eye opening ...
            3 - do some fun face copying and pasting between images 
<================"""
from .Face import Face, DrawingSpec
from .FaceAnalyzer import FaceAnalyzer
from .Helpers import buildCameraMatrix, rodriguezToRotationMatrix