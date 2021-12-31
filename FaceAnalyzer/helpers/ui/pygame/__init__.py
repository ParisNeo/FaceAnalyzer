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
class WidgetStyle:
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




class Widget():
    def __init__(
                    self,
                    rect:tuple=[0,0,100,50], 
                    style:str="widget{background:#ffffff;}\n",
                    extra_styles={}
                ):
        self.rect = rect
        self.rect2 = (rect[0],rect[1],rect[0]+rect[2],rect[1]+rect[3]) 
        self.styles=self.merge_two_dicts({
            "widget":WidgetStyle()
        }, extra_styles)
        self.setStyleSheet(style)

    def merge_two_dicts(self, x, y):
        z = x.copy()   # start with keys and values of x
        z.update(y)    # modifies z with keys and values of y
        return z

    def setStyleSheet(self, style:str):
        """Sets the button stylesheet

        Args:
            style (str): A css stylesheet to specify the button caracteristics
        """
        self.style = cssutils.parseString(style)
        
        for rule in self.style:
            if rule.type == rule.STYLE_RULE:
                try:
                    style = self.styles[rule.selectorText]
                except:
                    continue
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

                    # Text stuff
                    if property.name=='x-margin':
                        style.x_margin = int(property.value)
                    if property.name=='y-margin':
                        style.y_margin = int(property.value)
                    if property.name=='align':
                        style.align = property.value
                    if property.name == 'font-size':
                        style.font_size=property.value
                        style.font = pygame.font.Font(style.font_name+'.ttf', style.font_size)
                    if property.name == 'font-name':
                        style.font_name=property.value
                        style.font = pygame.font.Font(style.font_name+'.ttf', style.font_size)

    def paint(self, screen):
        """Paints the button

        Args:
            screen ([type]): The screen on which to blit
        """
                          
        style = self.widget_style
        if style.img is None:
            pygame.draw.rect(screen, style.bg_color, self.rect)
        else:
            screen.blit(pygame.transform.scale(style.img, (self.rect[2], self.rect[3])), (self.rect[0],self.rect[1]))

    def handle_events(self, events):
        pass
class WindowManager():
    def __init__(self, window_title:str="", resolution:tuple=(800,600)):
        """Builds a window managaer object
        """
        self.screen = pygame.display.set_mode(resolution)
        pygame.display.set_caption(window_title)
        self.widgets = []
        self.events = None

    def addWidget(self, widget:Widget):
        """Adds a new widget to the widgets list

        Args:
            widget (Widget): The widget to be added
        """
        self.widgets.append(widget)
    
    def process(self, background_color:tuple = (0,0,0)):
        self.screen.fill(background_color)
        self.events = pygame.event.get()
        for widget in self.widgets:
            widget.paint(self.screen)
            widget.handle_events(self.events)
class Label(Widget):
    def __init__(
                    self,
                    text, 
                    rect:tuple=[0,0,100,50], 
                    style:str="btn.normal{color:red; background:#ffffff;}\nbtn.hover{color:red; background:#ff0000};",
                    clicked_event_handler=None
                ):
        Widget.__init__(self,rect, style,extra_styles={"label":WidgetStyle()})
        self.text = text
        self.hovered=False
        self.pressed=False
        self.clicked_event_handler = clicked_event_handler
        self.setStyleSheet(style)

    def setText(self,text:str)->None:
        """Changes the text to be displayed inside the label

        Args:
            text (str): The text to be displayed
        """
        self.text = text

    def blit_text(self, style:WidgetStyle, screen):
        """Blits button text using a css style

        Args:
            style (WidgetStyle): The style to be used
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
        style = self.styles["label"]
        if style.img is None:
            pygame.draw.rect(screen, style.bg_color, self.rect)
        else:
            screen.blit(pygame.transform.scale(style.img, (self.rect[2], self.rect[3])), (self.rect[0],self.rect[1]))

        self.blit_text(style, screen)

class Button(Widget):
    def __init__(
                    self,
                    text, 
                    rect:tuple=[0,0,100,50], 
                    style:str="btn.normal{color:red; background:#ffffff;}\nbtn.hover{color:red; background:#ff0000};\nbtn.pressed{color:red; background:#ff0000};",
                    is_toggle=False,
                    clicked_event_handler=None
                ):
        Widget.__init__(
                        self,
                        rect,
                        style,
                        {
                            "btn.normal":WidgetStyle(),
                            "btn.hover":WidgetStyle(),
                            "btn.pressed":WidgetStyle(),
                        }
                        )
        self.text = text
        self.is_toggle = is_toggle
        self.hovered=False
        self.pressed=False
        self.toggled=False
        self.clicked_event_handler = clicked_event_handler


    def setText(self,text:str)->None:
        """Changes the text to be displayed inside the label

        Args:
            text (str): The text to be displayed
        """
        self.text = text

    def blit_text(self, style:WidgetStyle, screen):
        """Blits button text using a css style

        Args:
            style (WidgetStyle): The style to be used
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
            style = self.styles["btn.pressed"]
        elif self.hovered:
            style = self.styles["btn.hover"]
        else:
            style = self.styles["btn.normal"]

        if style.img is None:
            pygame.draw.rect(screen,style.bg_color,self.rect)
        else:
            screen.blit(pygame.transform.scale(style.img, (self.rect[2], self.rect[3])), (self.rect[0],self.rect[1]))
        self.blit_text(style, screen)

    def handle_events(self, events):
        """Handles the events

        """
        for event in events:
            if event.type == pygame.MOUSEMOTION:
                self.hovered = is_point_inside_rect(event.pos,self.rect2)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.hovered == True:
                    if self.is_toggle:
                        if not self.toggled:
                            self.pressed=not self.pressed
                            self.toggled=True
                    else:
                        self.pressed=True
                    if self.clicked_event_handler is not None:
                        self.clicked_event_handler()

            elif event.type == pygame.MOUSEBUTTONUP:
                if not self.is_toggle:
                    self.pressed=False
                self.toggled=False

class ProgressBar(Widget):
    def __init__(
                self, 
                rect: tuple = [0, 0, 100, 50], 
                style: str = "brogressbar.outer{background:#ffffff;}\nbrogressbar.inner{background:#ffffff;}", 
                value=0
            ):
        """Builds a progressbar widget

        Args:
            rect (tuple, optional): Rectangle where to put the progressbar. Defaults to [0, 0, 100, 50].
            style (str, optional): [description]. Defaults to "brogressbar.outer{background:#ffffff;}\nbrogressbar.inner{background:#ffffff;}".
            value (int, optional): [description]. Defaults to 0.
        """
        super().__init__(
                            rect=rect, 
                            style=style, 
                            extra_styles={
                                "brogressbar.outer":WidgetStyle(),
                                "brogressbar.inner":WidgetStyle(),
                            }
                        )
        self.value=value

    def setValue(self, value):
        self.value = value            

    def paint(self, screen):
        """Paints the button

        Args:
            screen ([type]): The screen on which to blit
        """
                          
        outer_style = self.styles["brogressbar.outer"]
        inner_style = self.styles["brogressbar.inner"]
        if outer_style.img is None:
            pygame.draw.rect(screen,outer_style.bg_color,self.rect)
        else:
            screen.blit(pygame.transform.scale(outer_style.img, (self.rect[2], self.rect[3])), (self.rect[0],self.rect[1]))
        
        if inner_style.img is None:
            pygame.draw.rect(screen,inner_style.bg_color,[self.rect[0], self.rect[1], self.rect[2]*self.value, self.rect[3]])
        else:
            screen.blit(pygame.transform.scale(inner_style.img, (self.rect[2], self.rect[3])), (self.rect[0],self.rect[1]))
        
