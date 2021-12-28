"""=== Face Analyzer =>
    Module : Helpers
    Author : Saifeddine ALOUI (ParisNeo)
    Licence : MIT
    Description :
        A toolbox for geometry, optics and other tools useful for analyzing positions/orientations and converting them from 2d to 3d or detecting interactions etc...
<================"""

from typing import Tuple
import numpy as np
import math
from numpy.lib.type_check import imag
from scipy.spatial.transform import Rotation as R
import cv2
from PIL import Image, ImageDraw

