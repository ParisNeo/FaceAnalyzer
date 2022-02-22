# -*- coding: utf-8 -*-
"""=== Face Analyzer Helpers =>
    Module : ui
    Author : Saifeddine ALOUI (ParisNeo)
    Licence : MIT
    Description :
        User interface helpers using opencv
<================"""

import cv2
import numpy as np

def cvDrawCross(image, pos:np.ndarray, color:tuple=(255,0,0), thickness:int=2):
    cv2.line(image, 
                                    (int(pos[0]-10), 
                                    int(pos[1]-10)),
                                    (int(pos[0]+10),
                                    int(pos[1]+10))
                                    ,color,thickness)
    cv2.line(image, 
                                    (int(pos[0]+10), 
                                    int(pos[1]-10)),
                                    (int(pos[0]-10),
                                    int(pos[1]+10))
                                    ,color,thickness)

def cvShowErrorEllipse(image, chisquare_val:float, mean:np.ndarray, covmat:np.ndarray, color:tuple=(255,0,0), thickness:int=2):
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

	cv2.ellipse(image,(int(mean[0]),int(mean[1])),(int(halfmajoraxissize), int(halfminoraxissize)), angle, 0, 360, color, thickness)


def cvOverlayImageWithAlpha(img:np.ndarray, img_overlay:np.ndarray, x:int, y:int, w:int, h:int, alpha_mask:float)->None:
    """Overlays an image on another image (uses Pillow)

    Args:
        img (np.ndarray): Background image
        img_overlay (np.ndarray): Overlay image
        x (int): x position
        y (int): y position
        w (int): Width
        h (int): Height
        alpha_mask (float): Alpha value
    """
    from PIL import Image
    pimg = Image.fromarray(np.uint8(img))
    pimg_overlay = Image.fromarray(np.uint8(img_overlay))
    alpha = Image.fromarray((np.array(pimg_overlay.split()[-1])*255*alpha_mask).astype(np.uint8))
    pimg_overlay.putalpha(alpha)
    pimg_overlay = pimg_overlay.resize((w,h))
    pimg.paste(pimg_overlay, (int(x), int(y), int(x+w), int(y+h)), pimg_overlay)

    img[:]= np.array(pimg)
    return img


def cvOverlayImage(img:np.ndarray, img_overlay:np.ndarray, x:int, y:int, w:int, h:int)->None:
    """Overlays an image on another image (uses Pillow)

    Args:
        img (np.ndarray): Background image
        img_overlay (np.ndarray): Overlay image
        x (int): x position
        y (int): y position
        w (int): Width
        h (int): Height
        alpha_mask (float): Alpha value
    """
    img_overlay_resized = cv2.resize(img_overlay,(w,h))
    img[y:y+h,x:x+w]= img_overlay_resized
    return img