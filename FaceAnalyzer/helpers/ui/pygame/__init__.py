# -*- coding: utf-8 -*-
"""=== Face Analyzer Helpers =>
    Module : ui
    Author : Saifeddine ALOUI (ParisNeo)
    Licence : MIT
    Description :
        User interface helpers
<================"""
import io
import pygame
import cssutils
from FaceAnalyzer.helpers.geometry.euclidian import is_point_inside_rect
from FaceAnalyzer.helpers.ui.pygame.colors import get_color
# Widgets
from dataclasses import dataclass
from urllib.request import urlopen


# Initialize font
pygame.font.init()

@dataclass
class ButtonStateStyle:
    """Class for keeping track of an item in inventory."""
    font : pygame.font.Font = pygame.font.Font('freesansbold.ttf', 14)
    bg_color: tuple = (100,100,100)
    border_color: tuple =(0,0,0)
    text_color: tuple = (0,0,0)
    border_size: int = 1
    font_name: str = 'freesansbold'
    font_size: int = 24
    x_margin: int = 0
    y_margin: int = 0
    align:str = 'center'
    img:str = None

class Button():
    def __init__(
                    self,
                    text, 
                    rect:tuple=[0,0,100,50], 
                    style:str="btn.normal{color:red; background:#ffffff;}\nbtn.hover{color:red; background:#ff0000};",
                    clicked_event_handler=None
                ):
        self.text = text
        self.rect = rect
        self.rect2 = (rect[0],rect[1],rect[0]+rect[2],rect[1]+rect[3]) 
        self.hovered=False
        self.pressed=False
        self.clicked_event_handler = clicked_event_handler
        self.setStyleSheet(style)


    def setStyleSheet(self, style:str):
        """Sets the button stylesheet

        Args:
            style (str): A css stylesheet to specify the button caracteristics
        """
        self.style_normal = ButtonStateStyle()
        self.style_hovered = ButtonStateStyle()
        self.style_pressed = ButtonStateStyle()
        self.style = cssutils.parseString(style)
        
        for rule in self.style:
            if rule.type == rule.STYLE_RULE:
                if rule.selectorText=="btn.normal":
                    style = self.style_normal
                elif rule.selectorText=="btn.hover":
                    style = self.style_hovered
                elif rule.selectorText=="btn.pressed":
                    style = self.style_pressed
                # find property
                for property in rule.style:
                    if property.name == 'color':
                        v = get_color(property.value)
                        if v is not None:
                            style.text_color = v
                                                 
                    if property.name == 'background-image':
                        bgi = property.value.strip()
                        if bgi.startswith("url"):
                            image_url = bgi[4:-1]
                            print(image_url)
                            image_str = urlopen(image_url).read()
                            # create a file object (stream)
                            image_file = io.BytesIO(image_str)
                            image = pygame.image.load(image_file)
                        if image is not None:
                            style.img = image
                    if property.name == 'background':
                        v = get_color(property.value)
                        if v is not None:
                            style.bg_color = v

                    if property.name == 'font-size':
                        style.font_size=property.value
                        style.font = pygame.font.Font(style.font_name+'.ttf', style.font_size)

    def blit_text(self, style:ButtonStateStyle, screen):
        """Blits button text using a css style

        Args:
            style (ButtonStateStyle): The style to be used
            screen ([type]): The screen on which to blit
        """
        text_render = style.font.render(self.text,True, style.text_color)
        if style.align =='center':
            screen.blit(text_render,(self.rect[0]+self.rect[2]//2-text_render.get_width()//2,self.rect[1]+self.rect[3]//2-text_render.get_height()//2))   
        elif style.align =='left':
            screen.blit(text_render,(self.rect[0]+style.x_margin,self.rect[1]+self.rect[3]//2-text_render.get_height()//2))   
        elif style.align =='right':
            screen.blit(text_render,(self.rect[0]+self.rect[2]-text_render.get_width(),self.rect[1]+self.rect[3]//2-text_render.get_height()//2))   

    def paint(self, screen):
        """Paints the button

        Args:
            screen ([type]): The screen on which to blit
        """
                          
        if self.pressed:
            style = self.style_pressed
        elif self.hovered:
            style = self.style_hovered
        else:
            style = self.style_normal

        if style.img is None:
            pygame.draw.rect(screen,style.bg_color,self.rect)
        else:
            screen.blit(pygame.transform.scale(style.img, (self.rect[2], self.rect[3])), (self.rect[0],self.rect[1]))
        self.blit_text(style, screen)

    def handle_events(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = is_point_inside_rect(event.pos,self.rect2)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.hovered == True:
                self.pressed=True
                if self.clicked_event_handler is not None:
                    self.clicked_event_handler()

        elif event.type == pygame.MOUSEBUTTONUP:
            self.pressed=False
