from sqtui import QtWidgets, QtCore
import pyqtgraph as pg
import cv2
from pathlib import Path
import numpy as np
from FaceAnalyzer.helpers.geometry.euclidian import  is_point_inside_rect
from FaceAnalyzer.helpers.ui.opencv import cvOverlayImageWithAlpha
from OOPyGame import ImageBox
import pygame

class Chaceable(ImageBox):
    """An object that can be chaced in space
    """
    def __init__(self, image:np.array, rect:list=[0,0,10,10])->None:
        """Builds the chaceable

        Args:
            image (np.ndarray): Image representing the chaceable to chace
            size (np.ndarray): The width and height of the chaceable
            position_2d (list): The 2d position of the chaceable
            image_size (list, optional): The size of the image on which to plot the chaceable. Defaults to [640,480].
            normal_color (tuple, optional): The normal color of the cheaceable. Defaults to (255,255,255).
            highlight_color (tuple, optional): The hilight color of the chaceable. Defaults to (0,255,0).
        """
        ImageBox.__init__(self, image, None, rect, color_key=(0,0,0), alpha=100)
        self.is_contact=False
    
    def move_to(self, position_2d:np.ndarray)->None:
        """Moves the object to a certain position

        Args:
            position_2d (np.ndarray): The new position to move to
        """
        self.rect[0]=position_2d[0]
        self.rect[1]=position_2d[1]
        self.setRect(self.rect)

    def check_contact(self, p2d:np.ndarray)->bool:
        """Check if a point is in contact with the chaceable

        Args:
            p2d (np.ndarray): The point to check

        Returns:
            bool: True if the point is inside the object
        """
        self.is_contact=is_point_inside_rect(p2d, self.rect)
        if self.is_contact:
            self.alpha=50
        return self.is_contact

    def paint(self, screen):
        if self.surface is not None:
            if self.is_contact:
                self.surface.set_alpha(100)
                screen.blit(pygame.transform.scale(self.surface, (self.rect[2], self.rect[3])),(self.rect[0],self.rect[1]))
            else:
                self.surface.set_alpha(200)
                screen.blit(pygame.transform.scale(self.surface, (self.rect[2], self.rect[3])),(self.rect[0],self.rect[1]))
