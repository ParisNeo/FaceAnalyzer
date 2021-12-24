"""=============
    Example : face_chacer.py
    Author  : Saifeddine ALOUI
    Description :
        A game of chacing objects using face orientation based on FaceAnalyzer. You use blinking to shoot them
<================"""

from numpy.lib.type_check import imag
from scipy.ndimage.measurements import label
from FaceAnalyzer import FaceAnalyzer, Face,  DrawingSpec, buildCameraMatrix, faceOrientation2Euler
from FaceAnalyzer.Helpers import get_z_line_equation, get_plane_infos, get_plane_line_intersection, KalmanFilter, pilDrawCross, pilShowErrorEllipse, pilOverlayImageWirthAlpha, region_3d_2_region_2d, is_point_inside_region
import numpy as np
import cv2
import time
from pathlib import Path
import sys
# Important!! if you don't have it, just install it using pip install sqtui pyqt5 (or pyside2) pyqtgraph
from sqtui import QtWidgets, QtCore
import pyqtgraph as pg
from PIL import Image, ImageDraw

from Chaceable import Chaceable

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
        self.kalman = KalmanFilter(10*np.eye(2), 1*np.eye(2), np.array([0,0]), 10*np.eye(2),np.eye(2),np.eye(2))


        # FPS processing
        self.prev_frame_time = time.time()
        self.curr_frame_time = time.time()

        self.listFile=QtWidgets.QListWidget()
        self.setMinimumWidth(500)
        self.setMinimumHeight(500)
        self.showMaximized()
        layout=QtWidgets.QGridLayout()

        # timers
        self.process_timer=QtCore.QTimer()
        self.process_timer.setInterval(10)
        self.process_timer.timeout.connect(self.process)

        self.visual_timer=QtCore.QTimer()
        self.visual_timer.setInterval(100)
        self.visual_timer.timeout.connect(self.update_ui)


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
        self.chaceables.append(Chaceable(Path(__file__).parent/"assets/pika.png", [150,150], np.array([-90,0]), image_size))   
        self.chaceables.append(Chaceable(Path(__file__).parent/"assets/pika.png", [150,150], np.array([90,0]), image_size))   
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
        self.eye_opening_pb = QtWidgets.QProgressBar()
        self.eye_opening_pb.setOrientation(QtCore.Qt.Vertical)

        layout.addWidget(self.infos,0,0,1,1)
        layout.addWidget(self.image,1,0,1,1)
        layout.addWidget(self.point_pos,1,1,1,1)
        layout.addWidget(self.eye_opening_pb,1,2,1,1)

        self.process_timer.start()
        self.visual_timer.start()

        self.setLayout(layout)
        self.updated_image = np.zeros((height,width,3))
        self.game_ui = np.zeros((height,width,3))

    def process(self):
        # Read image
        success, image = cap.read()
        
        # Opencv uses BGR format while mediapipe uses RGB format. So we need to convert it to RGB before processing the image
        image = cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1)


        # Process the image to extract faces and draw the masks on the face in the image
        fa.process(image)
        if fa.nb_faces>0:
            #for i in range(fa.nb_faces):
                i=0
                face = fa.faces[i]
                # Get head position and orientation compared to the reference pose (here the first frame will define the orientation 0,0,0)
                self.face_pos, self.face_ori = face.get_head_posture()
                if self.face_pos is not None:

                    
                    # First let's get the forward line (a virtual line that goes from the back of the head through tho nose towards the camera)
                    li  =    get_z_line_equation(self.face_pos, self.face_ori)
                    # Now we find the intersection point between the line and the plan. p is the 3d coordinates of the intersection pount, and p2d is the coordinates of this point in the plan
                    p, self.p2d   =    get_plane_line_intersection(self.main_plane, li)
                    self.kalman.process(self.p2d)
                    # Filtered p2d
                    self.p2d = self.kalman.x


                    # Detect blinking
                    self.left_eye_opening, self.right_eye_opening, self.is_blink = face.process_eyes(image, detect_blinks=True,  blink_th=0.35)
                    

        self.updated_image = image
        #
    def update_ui(self):
        self.eye_opening_pb.setValue(100*(self.left_eye_opening+self.right_eye_opening)/2)
        self.game_ui_img = Image.fromarray(np.uint8(self.empty_image_view.copy()))
        for ch in self.chaceables:
            is_contact = ch.check_contact(self.p2d)
            ch.draw(self.game_ui_img)
            if is_contact and self.is_blink:
                ch.move_to(np.array([np.random.randint(-image_size[0]//2,image_size[0]//2-20), np.random.randint(-image_size[1]//2,image_size[1]//2-20)]))
                self.score += 1
                self.infos.setText(f"Shoot with blinks.\nScore: {self.score}")

        pilDrawCross(self.game_ui_img, (self.p2d+np.array(image_size)//2).astype(np.int), (200,0,0), 3)
        self.game_ui_img = pilShowErrorEllipse(self.game_ui_img, 10, self.p2d+np.array(image_size)//2, self.kalman.P,(255,0,0), 2)        
        # Just put a reference on the nose
        #face.draw_reference_frame(image, pos, ori, origin=face.get_landmark_pos(Face.nose_tip_index))
        self.image.setImage(np.swapaxes(self.updated_image,0,1))
        self.point_pos.setImage(np.swapaxes(np.array(self.game_ui_img),0,1))


if __name__ == '__main__':
    app=QtWidgets.QApplication(sys.argv)
    form=WinForm()
    form.show()
    sys.exit(app.exec_())
