"""=============
    Example : face_box.py
    Author  : Saifeddine ALOUI
    Description :
        A code to test FaceAnalyzer by visualizing the evolution of multiple face parameters inside a pyqt5 or pyside2 interface
        (Requires installing sqtui with either pyqt5 or pyside2 and pyqtgraph)
<================"""
from PySide2 import QtCore
from scipy.ndimage.measurements import label
from FaceAnalyzer import FaceAnalyzer, Face,  DrawingSpec, buildCameraMatrix, faceOrientation2Euler
from FaceAnalyzer.Helpers import get_z_line_equation, get_plane_infos, get_plane_line_intersection
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

class WinForm(QtWidgets.QWidget):
    def __init__(self,parent=None):
        super(WinForm, self).__init__(parent)
        self.setWindowTitle('q_face_infos_graph')

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

        # Create the plot to plot informations over time
        # Eye orientation plot
        self.eye_ori_plot = pg.PlotWidget()
        self.eye_ori_plot.addLegend()
        self.left_eye_yaw_plot = PlotObject(self.eye_ori_plot,100, pen='r', name="left eye yaw")
        self.left_eye_pitch_plot = PlotObject(self.eye_ori_plot,100, pen='g', name="left eye pitch")
        self.left_eye_roll_plot = PlotObject(self.eye_ori_plot,100, pen='b', name="left eye roll")

        self.right_eye_yaw_plot = PlotObject(self.eye_ori_plot,100, pen=pg.mkPen(color='r',width=2), name="right eye yaw")
        self.right_eye_pitch_plot = PlotObject(self.eye_ori_plot,100, pen=pg.mkPen(color='g',width=2), name="right eye pitch")
        self.right_eye_roll_plot = PlotObject(self.eye_ori_plot,100, pen=pg.mkPen(color='b',width=2), name="right eye roll")

        self.head_ori_plot = pg.PlotWidget()
        self.head_ori_plot.addLegend()
        self.yaw_plot = PlotObject(self.head_ori_plot,100, pen='r', name="Head yaw")
        self.pitch_plot = PlotObject(self.head_ori_plot,100, pen='g', name="Head pitch")
        self.roll_plot = PlotObject(self.head_ori_plot,100, pen='b', name="Head roll")

        self.head_pos_plot = pg.PlotWidget()
        self.head_pos_plot.addLegend()
        self.x_plot = PlotObject(self.head_pos_plot,100, pen='r', name="Head x")
        self.y_plot = PlotObject(self.head_pos_plot,100, pen='g', name="Head y")
        self.z_plot = PlotObject(self.head_pos_plot,100, pen='b', name="Head z")

        self.image.ui.histogram.hide()
        self.image.ui.roiBtn.hide()
        self.image.ui.menuBtn.hide()
        layout.addWidget(self.image,0,0,1,1)
        layout.addWidget(self.eye_ori_plot,0,1,1,1)
        layout.addWidget(self.head_pos_plot,1,0,1,1)
        layout.addWidget(self.head_ori_plot,1,1,1,1)

        self.timer.start()

        self.setLayout(layout)

    def update(self):
        # Read image
        success, image = cap.read()
        
        # Opencv uses BGR format while mediapipe uses RGB format. So we need to convert it to RGB before processing the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image to extract faces and draw the masks on the face in the image
        fa.process(image)
        if fa.nb_faces>0:
            for i in range(fa.nb_faces):
                face = fa.faces[i]
                # Get head position and orientation compared to the reference pose (here the first frame will define the orientation 0,0,0)
                pos, ori = face.get_head_posture()
                if pos is not None:
                    yaw, pitch, roll = faceOrientation2Euler(ori, degrees=True)
                    face.draw_bounding_box(image, color=box_colors[i%3], thickness=5)
                    face.draw_reference_frame(image, pos, ori, origin=face.get_landmark_pos(Face.nose_tip_index))
                    # Show 
                    #ori = Face.rotationMatrixToEulerAngles(ori)
                    if i==0:
                        cv2.putText(
                            image, f"Yaw : {yaw:2.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
                        cv2.putText(
                            image, f"Pitch : {pitch:2.0f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
                        cv2.putText(
                            image, f"Roll : {roll:2.0f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
                        cv2.putText(
                            image, f"Position : {pos[0,0]:2.2f},{pos[1,0]:2.2f},{pos[2,0]:2.2f}", (10, 120), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))
                        self.yaw_plot.add_data(yaw)
                        self.pitch_plot.add_data(pitch)
                        self.roll_plot.add_data(roll)
                        self.x_plot.add_data(pos[0,0])
                        self.y_plot.add_data(pos[1,0])
                        self.z_plot.add_data(pos[2,0])
                    left_pos, right_pos = face.get_eyes_position()
                    left_eye_ori = face.compose_eye_rot(left_pos, ori)
                    right_eye_ori = face.compose_eye_rot(right_pos, ori)
                    left_eye = face.get_landmark_pos(Face.left_eye_center_index)
                    right_eye = face.get_landmark_pos(Face.right_eye_center_index)
                    face.draw_reference_frame(image, pos, left_eye_ori, origin=left_eye)
                    face.draw_reference_frame(image, pos, right_eye_ori, origin=right_eye)
                    left_eye_yaw, left_eye_pitch, left_eye_roll = faceOrientation2Euler(left_eye_ori, degrees=True)
                    right_eye_yaw, right_eye_pitch, right_eye_roll = faceOrientation2Euler(right_eye_ori, degrees=True)
                       
                    self.left_eye_yaw_plot.add_data(left_eye_yaw)
                    self.left_eye_pitch_plot.add_data(left_eye_pitch)
                    self.left_eye_roll_plot.add_data(left_eye_roll)

                    self.right_eye_yaw_plot.add_data(right_eye_yaw)
                    self.right_eye_pitch_plot.add_data(right_eye_pitch)
                    self.right_eye_roll_plot.add_data(right_eye_roll)


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

