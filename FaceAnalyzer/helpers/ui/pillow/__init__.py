# -*- coding: utf-8 -*-
"""=== Face Analyzer Helpers =>
    Module : ui
    Author : Saifeddine ALOUI (ParisNeo)
    Licence : MIT
    Description :
        User interface helpers using pillow
<================"""
from PIL import Image, ImageDraw
import numpy as np

def pilDrawCross(image:Image, pos:np.ndarray, color:tuple=(255,0,0), thickness:int=2):
    """Draws a cross on an image at a specific position

    Args:
        image (Image): The image on which to put the cross.
        pos (np.ndarray): The position where to put the cross.
        color (tuple, optional): The color of the cross. Defaults to (255,0,0).
        thickness (int, optional): The thickness of the cross. Defaults to 2.
    """
    draw = ImageDraw.Draw(image)

    draw.line([(int(pos[0]-10), int(pos[1]-10)),
               (int(pos[0]+10), int(pos[1]+10))]
                                    ,color,thickness)
    draw.line([(int(pos[0]+10), int(pos[1]-10)),
               (int(pos[0]-10), int(pos[1]+10))]
                                    ,color,thickness)

def pilShowErrorEllipse(pimg:Image, chisquare_val:float, mean:np.ndarray, covmat:np.ndarray, color:tuple=(255,0,0), thickness:int=2)->Image:
    """Shows error ellipse on a pillow image using opencv to build the ellipse

    Args:
        pimg (Image): The pillow image to draw the ellipse on
        chisquare_val (float): Value of the xi square
        mean (np.ndarray): The mean of the distraibution (the center of the ellipse)
        covmat (np.ndarray): Covariance matrix from which to build the ellipse
        color (tuple, optional): The color of the ellipse. Defaults to (255,0,0).
        thickness (int, optional): The thickness of the ellipse drawing. Defaults to 2.

    Returns:
        Image: A pillow image where the ellipse is drawn
    """
    import cv2
    [retval, eigenvalues, eigenvectors] = cv2.eigen(covmat)

    #Calculate the angle between the largest eigenvector and the x-axis
    angle = np.arctan2(eigenvectors[0,1], eigenvectors[0,0])
    #Shift the angle to the [0, 2pi] interval instead of [-pi, pi]
    if(angle < 0):
        angle += 6.28318530718

    # Conver to degrees instead of radians
    angle = 180*angle/3.14159265359

    # Calculate the size of the minor and major axes
    halfmajoraxissize=chisquare_val*np.sqrt(eigenvalues[0])
    halfminoraxissize=chisquare_val*np.sqrt(eigenvalues[1])
    cvImage = np.array(pimg)
    cv2.ellipse(cvImage,(int(mean[0]),int(mean[1])),(int(halfmajoraxissize), int(halfminoraxissize)), angle, 0, 360, color, thickness)
    return Image.fromarray(np.uint8(cvImage))



def pilOverlayImageWirthAlpha(pimg:Image, pimg_overlay:Image, x:int, y:int, w:int, h:int, alpha_mask:float)->None:
    """Overlays an image on another image (uses Pillow)

    Args:
        img (Image): Background image
        img_overlay (Image): Overlay image
        x (int): x position
        y (int): y position
        w (int): Width
        h (int): Height
        alpha_mask (float): Alpha value
    """

    #pimg_overlay.putalpha(int(255*alpha_mask))
    pimg_overlay = pimg_overlay.resize((w,h))
    alpha = Image.fromarray((np.array(pimg_overlay.split()[-1])*alpha_mask).astype(np.uint8))
    pimg_overlay.putalpha(alpha)
    pimg.paste(pimg_overlay, (int(x), int(y), int(x+w), int(y+h)), pimg_overlay)
