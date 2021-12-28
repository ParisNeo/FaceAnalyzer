from sqtui import QtWidgets, QtCore
import pyqtgraph as pg
from PIL import Image, ImageDraw
from pathlib import Path
import numpy as np
from FaceAnalyzer.helpers.geometry.euclidian import  is_point_inside_region
from FaceAnalyzer.helpers.ui.pillow import pilOverlayImageWirthAlpha


class Chaceable():
    """An object that can be chaced in space
    """
    def __init__(self, image_path:Path, size:np.ndarray, position_2d:list, image_size:list=[640,480], normal_color:tuple=(255,255,255), highlight_color:tuple=(0,255,0))->None:
        """Builds the chaceable

        Args:
            image (np.ndarray): Image representing the chaceable to chace
            size (np.ndarray): The width and height of the chaceable
            position_2d (list): The 2d position of the chaceable
            image_size (list, optional): The size of the image on which to plot the chaceable. Defaults to [640,480].
            normal_color (tuple, optional): The normal color of the cheaceable. Defaults to (255,255,255).
            highlight_color (tuple, optional): The hilight color of the chaceable. Defaults to (0,255,0).
        """
        self.image_size = image_size
        self.overlay = Image.open(str(image_path))
        self.size =size
        self.shape=np.array([[0,0],[size[0],0],[size[0],size[1]],[0,size[1]]]).T
        self.pos= position_2d.reshape((2,1))
        self.normal_color = normal_color
        self.highlight_color = highlight_color
        self.is_contact=False
        self.curr_shape = self.shape+self.pos
    
    def move_to(self, position_2d:np.ndarray)->None:
        """Moves the object to a certain position

        Args:
            position_2d (np.ndarray): The new position to move to
        """
        self.pos= position_2d.reshape((2,1))
        self.curr_shape = self.shape+self.pos

    def check_contact(self, p2d:np.ndarray)->bool:
        """Check if a point is in contact with the chaceable

        Args:
            p2d (np.ndarray): The point to check

        Returns:
            bool: True if the point is inside the object
        """
        self.is_contact=is_point_inside_region(p2d, self.curr_shape)
        return self.is_contact

    def draw(self, pImage:Image)->None:
        """Draws the chaceable on an image

        Args:
            image (np.ndarray): The image on which to draw the chaceable
        """
        npstyle_region_porel_pos = self.pos+np.array([self.image_size]).T//2
        if self.is_contact:
            pilOverlayImageWirthAlpha(pImage, self.overlay, npstyle_region_porel_pos[0], npstyle_region_porel_pos[1], self.size[0], self.size[1], 0.5)
        else:
            pilOverlayImageWirthAlpha(pImage, self.overlay, npstyle_region_porel_pos[0], npstyle_region_porel_pos[1], self.size[0], self.size[1], 1.0)