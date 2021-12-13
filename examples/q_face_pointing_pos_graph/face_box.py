"""=============
    Example : face_box.py
    Author  : Saifeddine ALOUI
    Description :
        A code to test FaceAnalyzer by visualizing the evolution of multiple face parameters inside a pyqt5 or pyside2 interface
        (Requires installing sqtui with either pyqt5 or pyside2 and pyqtgraph)
<================"""
from PySide2 import QtCore
from numpy.lib.type_check import imag
from scipy.ndimage.measurements import label
from FaceAnalyzer import FaceAnalyzer, Face,  DrawingSpec, buildCameraMatrix, faceOrientation2Euler
from FaceAnalyzer.Helpers import get_z_line_equation, get_plane_infos, get_plane_line_intersection, KalmanFilter, showErrorEllipse, drawCross
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
        self.image.setMinimumWidth(100)
        self.image.setMinimumHeight(100)

        # Create the plot to plot informations over time

        self.point_pos = pg.ImageView()
        self.region_points=[
                            np.array([-300,-300,0]),
                            np.array([0,-350,0]),
                            np.array([300,-300,0]),
                            np.array([300,300,0]),
                            np.array([-300,300,0])
                            ]
        self.empty_image_view = np.zeros((1000,1000,3))

        self.image_view = self.empty_image_view.copy()
        self.point_pos.setImage(self.image_view)


        # face intersection point with a plan plot
        self.face_pointing_pos = pg.PlotWidget()
        self.face_pointing_pos.addLegend()
        self.face_pointing_pos_x_plot = PlotObject(self.face_pointing_pos,100, pen='r', name="face_pointing x")
        self.face_pointing_pos_y_plot = PlotObject(self.face_pointing_pos,100, pen='g', name="face_pointing y")
        self.face_pointing_pos_z_plot = PlotObject(self.face_pointing_pos,100, pen='b', name="face_pointing z")

        # face plan intersection 2d coordinates 
        self.face_pointing_pos_2d = pg.PlotWidget()
        self.face_pointing_pos_2d.addLegend()
        self.face_pointing_pos_2d_x_plot = PlotObject(self.face_pointing_pos_2d,100, pen='r', name="face_pointing x")
        self.face_pointing_pos_2d_y_plot = PlotObject(self.face_pointing_pos_2d,100, pen='g', name="face_pointing y")

        self.image.ui.histogram.hide()
        self.image.ui.roiBtn.hide()
        self.image.ui.menuBtn.hide()

        self.point_pos.ui.histogram.hide()
        self.point_pos.ui.roiBtn.hide()
        self.point_pos.ui.menuBtn.hide()

        self.infos = QtWidgets.QLabel("Is in plan : False")
        self.infos.setStyleSheet("font-size:24px")
        self.infos.setMinimumHeight(100)
        self.filter_slider = QtWidgets.QSlider()
        self.filter_slider.setOrientation(QtCore.Qt.Horizontal)
        def updated():
            self.kalman.R = 20*(self.filter_slider.value()+1)*np.eye(2)
        self.filter_slider.valueChanged.connect(updated)

        layout.addWidget(self.image,0,0,1,1)
        layout.addWidget(self.point_pos,0,1,1,1)
        layout.addWidget(self.face_pointing_pos,1,0,1,1)
        layout.addWidget(self.face_pointing_pos_2d,1,1,1,1)
        layout.addWidget(self.infos,2,1,1,1)
        layout.addWidget(self.filter_slider,2,0,1,1)

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
                    # Now let's define a plane in 3d space using 3 points (here the place is prthogonal to the camera's focal line)
                    pl  =    get_plane_infos(np.array([0,0, 0]),np.array([100,0, 0]),np.array([0, 100,0]))
                    # Now we find the intersection point between the line and the plan. p is the 3d coordinates of the intersection pount, and p2d is the coordinates of this point in the plan
                    p, p2d   =    get_plane_line_intersection(pl, li)
                    self.kalman.process(p2d)
                    # Filtered p2d
                    p2d = self.kalman.x
                    
                    # Plot 3d coordinates
                    self.face_pointing_pos_x_plot.add_data(p[0])
                    self.face_pointing_pos_y_plot.add_data(p[1])
                    self.face_pointing_pos_z_plot.add_data(p[2])
                    # Plot 2d coordinates
                    self.face_pointing_pos_2d_x_plot.add_data(p2d[0])
                    self.face_pointing_pos_2d_y_plot.add_data(p2d[1])

                    # Use the built in function to determine if the face is pointed to a region
                    region2d = face.region_3d_2_region_2d(self.region_points)
                    is_in=face.is_point_inside_region(p2d, region2d)

                    npstyle_region_points = (np.array(self.region_points, np.int32)[:,:2]+np.array([[500,500]])).reshape((-1, 1, 2))
                    if is_in:
                        cv2.fillPoly(self.image_view, [npstyle_region_points], (0,255,0))
                    else:
                        cv2.fillPoly(self.image_view, [npstyle_region_points], (255,255,255))
                    drawCross(self.image_view, (int(p2d[0]+500), int(p2d[1]+500)), (200,0,0),3)
                    showErrorEllipse(self.image_view,10,p2d+np.array([500,500]), self.kalman.P,(255,0,0),10)

                    # position the point
                    #self.point_plot.update([p2d[0]-0.5,p2d[0]+0.5],[p2d[1]-0.5,p2d[1]+0.5])

                    if is_in:
                        self.infos.setText("Is in region : True")
                    else:
                        self.infos.setText("Is in region : False")
                    # Just put a reference on the nose
                    face.draw_reference_frame(image, pos, ori, origin=face.getlandmark_pos(Face.nose_tip_index))
                    self.point_pos.setImage(np.swapaxes(self.image_view,0,1))

        # Process fps
        self.curr_frame_time = time.time()
        dt = self.curr_frame_time-self.prev_frame_time
        self.prev_frame_time = self.curr_frame_time
        fps = 1/dt
        # Show FPS
        cv2.putText(
            image, f"FPS : {fps:2.2f}", (10, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))


        self.image.setImage(np.swapaxes(image,0,1))


if __name__ == '__main__':
    app=QtWidgets.QApplication(sys.argv)
    form=WinForm()
    form.show()
    sys.exit(app.exec_())
