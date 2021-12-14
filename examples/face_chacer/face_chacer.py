"""=============
    Example : face_chacer.py
    Author  : Saifeddine ALOUI
    Description :
        A game of chacing objects using face orientation based on FaceAnalyzer. You use blinking to shoot them
<================"""
from PySide2 import QtCore
from numpy.lib.type_check import imag
from scipy.ndimage.measurements import label
from FaceAnalyzer import FaceAnalyzer, Face,  DrawingSpec, buildCameraMatrix, faceOrientation2Euler
from FaceAnalyzer.Helpers import get_z_line_equation, get_plane_infos, get_plane_line_intersection, KalmanFilter, showErrorEllipse, drawCross, region_3d_2_region_2d, is_point_inside_region, overlay_image_alpha
import numpy as np
import cv2
import time
from pathlib import Path
import sys
# Important!! if you don't have it, just install it using pip install sqtui pyqt5 (or pyside2) pyqtgraph
from sqtui import QtWidgets, QtCore
import pyqtgraph as pg

# open camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
width = 640#width = 1920
height = 480#height = 1080
image_size = [width, height]
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Build face analyzer while specifying that we want to extract just a single face
fa = FaceAnalyzer(max_nb_faces=3, image_shape=(width, height))



box_colors=[
    (255,0,0),
    (255,0,255),
    (255,0,255),
    
]
class PlotObject():

    def __init__(self, parent:pg.PlotWidget, data_size, **kwargs):
        self.plot = parent.plot(**kwargs)   
        self.data=[]
        self.data_size=data_size

    def add_data(self, data):
        self.data.append(data)
        if len(self.data)>self.data_size:
            self.data.pop(0)

        self.plot.setData(self.data)

class CurveObject():

    def __init__(self, parent:pg.PlotWidget, **kwargs):
        self.plot = parent.plot(**kwargs)   

    def update(self, x, y):
        self.plot.setData(x,y)

class Chaceable():
    """An object that can be chaced in space
    """
    def __init__(self, image_path:Path, size:np.ndarray, position_2d:list, plane:np.ndarray, image_size:list=[640,480], normal_color:tuple=(255,255,255), highlight_color:tuple=(0,255,0))->None:
        """Builds the chaceable

        Args:
            image (np.ndarray): Image representing the chaceable to chace
            size (np.ndarray): The width and height of the chaceable
            position_2d (list): The 2d position of the chaceable
            plane (np.ndarray): Plane where this chaceable resides
            image_size (list, optional): The size of the image on which to plot the chaceable. Defaults to [640,480].
            normal_color (tuple, optional): The normal color of the cheaceable. Defaults to (255,255,255).
            highlight_color (tuple, optional): The hilight color of the chaceable. Defaults to (0,255,0).
        """
        self.overlay = cv2.imread(str(image_path), -1)
        colored = self.overlay[:,:,:-1]
        alpha = self.overlay[:,:,-1]
        self.overlay = np.dstack([colored[:,:,::-1],alpha])
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

    def draw(self, image:np.ndarray)->None:
        """Draws the chaceable on an image

        Args:
            image (np.ndarray): The image on which to draw the chaceable
        """
        npstyle_region_porel_pos = self.pos+np.array([image_size]).T//2
        if self.is_contact:
            overlay_image_alpha(image, self.overlay, npstyle_region_porel_pos[0], npstyle_region_porel_pos[1], self.size[0], self.size[1], 0.5)
        else:
            overlay_image_alpha(image, self.overlay, npstyle_region_porel_pos[0], npstyle_region_porel_pos[1], self.size[0], self.size[1], 1.0)


class WinForm(QtWidgets.QWidget):
    def __init__(self,parent=None):
        """Builds a SQTUI window (either a pyqt5 or pyside window)

        Args:
            parent (QWidget, optional): The parent of the widget. Defaults to None.
        """
        super(WinForm, self).__init__(parent)
        self.setWindowTitle('q_face_pointing_pos_graph')

        # Let's build a kalman filter to filter 2d position of the pointing position 
        # more filtering than tracking, a zero position initial point, and a high uncertainty (F and H are 2d identity)
        self.kalman = KalmanFilter(10*np.eye(2), 1*np.eye(2), np.array([0,0]), 2*np.eye(2),np.eye(2),np.eye(2))


        # FPS processing
        self.prev_frame_time = time.time()
        self.curr_frame_time = time.time()

        self.listFile=QtWidgets.QListWidget()
        self.setMinimumWidth(500)
        self.setMinimumHeight(500)
        self.showMaximized()
        layout=QtWidgets.QGridLayout()

        self.timer=QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        # Create image to view the camera input 
        self.image = pg.ImageView()
        self.image.setMaximumWidth(200)
        self.image.setMaximumHeight(200)

        # Create the plot to plot informations over time

        self.point_pos = pg.ImageView()
        # Now let's define a plane in 3d space using 3 points (here the place is prthogonal to the camera's focal line)
        self.main_plane  =    get_plane_infos(np.array([0,0, 0]),np.array([100,0, 0]),np.array([0, 100,0]))
        # Let's build sume stuff to chace using the pointing vector
        self.chaceables=[]
        self.chaceables.append(Chaceable(Path(__file__).parent/"assets/pika.png", [150,150], np.array([-90,0]), self.main_plane, image_size))   
        self.chaceables.append(Chaceable(Path(__file__).parent/"assets/pika.png", [150,150], np.array([90,0]), self.main_plane, image_size))   
        self.empty_image_view = np.zeros((1000,1000,3))

        self.image_view = self.empty_image_view.copy()
        self.point_pos.setImage(self.image_view)



        self.image.ui.histogram.hide()
        self.image.ui.roiBtn.hide()
        self.image.ui.menuBtn.hide()

        self.point_pos.ui.histogram.hide()
        self.point_pos.ui.roiBtn.hide()
        self.point_pos.ui.menuBtn.hide()

        self.score = 0
        self.infos = QtWidgets.QLabel(f"Shoot with blinks.\nScore: {self.score}")
        self.infos.setStyleSheet("font-size:24px")
        self.infos.setMinimumHeight(100)

        layout.addWidget(self.infos,0,0,1,1)
        layout.addWidget(self.image,1,0,1,1)
        layout.addWidget(self.point_pos,1,1,1,1)

        self.timer.start()

        self.setLayout(layout)


    def update(self):
        # Read image
        success, image = cap.read()
        
        # Opencv uses BGR format while mediapipe uses RGB format. So we need to convert it to RGB before processing the image
        image = cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1)


        # Process the image to extract faces and draw the masks on the face in the image
        fa.process(image)
        if fa.nb_faces>0:
            for i in range(fa.nb_faces):
                face = fa.faces[i]
                # Get head position and orientation compared to the reference pose (here the first frame will define the orientation 0,0,0)
                pos, ori = face.get_head_posture()
                if pos is not None:

                    self.image_view = self.empty_image_view.copy()
                    
                    # First let's get the forward line (a virtual line that goes from the back of the head through tho nose towards the camera)
                    li  =    get_z_line_equation(pos, ori)
                    # Now we find the intersection point between the line and the plan. p is the 3d coordinates of the intersection pount, and p2d is the coordinates of this point in the plan
                    p, p2d   =    get_plane_line_intersection(self.main_plane, li)
                    self.kalman.process(p2d)
                    # Filtered p2d
                    p2d = self.kalman.x


                    # Detect blinking
                    left_eye_opening, right_eye_opening, is_blink = face.process_eyes(image, detect_blinks=True, draw_landmarks=False, blink_th=0.5)
                    for ch in self.chaceables:
                        is_contact = ch.check_contact(p2d)
                        ch.draw(self.image_view)
                        if is_contact and is_blink:
                            ch.move_to(np.array([np.random.randint(-image_size[0]//2,image_size[0]//2-20), np.random.randint(-image_size[1]//2,image_size[1]//2-20)]))
                            self.score += 1
                            self.infos.setText(f"Shoot with blinks.\nScore: {self.score}")

                    drawCross(self.image_view, (p2d+np.array(image_size)//2).astype(np.int), (200,0,0), 3)
                    showErrorEllipse(self.image_view, 10, p2d+np.array(image_size)//2, self.kalman.P,(255,0,0), 2)


                    # Just put a reference on the nose
                    face.draw_reference_frame(image, pos, ori, origin=face.getlandmark_pos(Face.nose_tip_index))
                    self.point_pos.setImage(np.swapaxes(self.image_view,0,1))


        self.image.setImage(np.swapaxes(image,0,1))


if __name__ == '__main__':
    app=QtWidgets.QApplication(sys.argv)
    form=WinForm()
    form.show()
    sys.exit(app.exec_())
