"""=============
    Example : face2_3d.py
    Author  : Saifeddine ALOUI
    Description :
        A tool that uses FaceAnalyzer to represent the face position in 3D space
<================"""

from numpy.lib.type_check import imag
from scipy.ndimage.measurements import label
from FaceAnalyzer import FaceAnalyzer, Face,  DrawingSpec, buildCameraMatrix
from FaceAnalyzer.helpers.geometry.orientation import faceOrientation2Euler
from FaceAnalyzer.helpers.geometry.euclidian import get_quaternion_from_euler, get_z_line_equation, get_plane_infos, get_plane_line_intersection, region_3d_2_region_2d, is_point_inside_region
from FaceAnalyzer.helpers.ui.pillow import pilDrawCross, pilShowErrorEllipse, pilOverlayImageWirthAlpha
from FaceAnalyzer.helpers.estimation import KalmanFilter
import numpy as np
import cv2
import time
from pathlib import Path
import sys
# Important!! if you don't have it, just install it using pip install sqtui pyqt5 (or pyside2) pyqtgraph
from sqtui import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from PIL import Image, ImageDraw
import pyqtgraph.opengl as gl



# open camera
cap = cv2.VideoCapture(0)
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
        self.infos = QtWidgets.QLabel(f"Head direction 2D position")
        self.infos.setStyleSheet("font-size:24px")
        self.infos.setMinimumHeight(100)
        self.eye_opening_pb = QtWidgets.QProgressBar()
        self.eye_opening_pb.setOrientation(QtCore.Qt.Vertical)


        # Build gl view
        w = gl.GLViewWidget()
        w.show()
        w.setWindowTitle('pyqtgraph example: GLMeshItem')
        w.setCameraPosition(distance=40)

        # Put a camera

        w.setCameraPosition(QtGui.QVector3D(0, -5, 20),distance=15,elevation=80,azimuth=-90 )
        # Add a grid
        g = gl.GLGridItem()
        g.scale(2,2,1)
        w.addItem(g)

        #Build face reference
        verts = np.array([
            [-1, -1, 0],
            [1, -1, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3]
        ])
        colors = np.array([
            [1, 0, 0, 0.3],
            [0, 1, 0, 0.3],
            [0, 0, 1, 0.3],
            [1, 1, 0, 0.3]
        ])

        ## Mesh item will automatically compute face normals.
        self.face_shape = gl.GLMeshItem(vertexes=verts, faces=faces, faceColors=colors, smooth=False)
        #self.face_shape.setGLOptions('additive')
        w.addItem(self.face_shape)        

        # Gaze position
        md = gl.MeshData.sphere(rows=10, cols=20, radius=0.1)
        self.gaze_pos = gl.GLMeshItem(
            meshdata=md,
            smooth=True,
            color=(1, 0, 0, 0.2),
            shader="balloon",
            glOptions="additive",
        )
        w.addItem(self.gaze_pos)        

        # Make reference
        md = gl.MeshData.cylinder(rows=10, cols=20, radius=[0.1,0.1], length=0.5)
        self.x = gl.GLMeshItem(
            meshdata=md,
            smooth=True,
            color=(1, 0, 0, 1.0),
            shader="balloon",
            glOptions="additive",
        )
        self.x.rotate(90,0,1,0)
        w.addItem(self.x)        
        self.y = gl.GLMeshItem(
            meshdata=md,
            smooth=True,
            color=(0, 1, 0, 1.0),
            shader="balloon",
        )
        self.y.rotate(90,-1,0,0)
        w.addItem(self.y)        
        self.z = gl.GLMeshItem(
            meshdata=md,
            smooth=True,
            color=(0, 0, 1, 1.0),
            shader="balloon",
        )
        w.addItem(self.z)        

        self.gaze_vector = gl.GLLinePlotItem(pos=np.array([[0,0,0],[0,0,1]]), width=1, antialias=False)
        self.view_zone = gl.GLLinePlotItem(pos=np.array([[0,0,0],[0,0,1]]), width=1, antialias=False)
        w.addItem(self.gaze_vector) 
        w.addItem(self.view_zone)

        self.view3d = w

        layout.addWidget(self.infos,0,0,1,1)
        layout.addWidget(self.image,1,0,1,1)
        layout.addWidget(self.point_pos,0,1,1,1)
        layout.addWidget(self.view3d,1,1,1,1)
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
        #image = cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1)


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
                    self.p3d = p

                    # Detect blinking
                    self.left_eye_opening, self.right_eye_opening, self.is_blink, self.blink_duration = face.process_eyes(image, detect_blinks=True,  blink_th=0.35)
                    

        self.updated_image = image
        #
    def update_ui(self):
        self.eye_opening_pb.setValue(int(100*(self.left_eye_opening+self.right_eye_opening)/2))
        self.game_ui_img = Image.fromarray(np.uint8(self.empty_image_view.copy()))

        transform = QtGui.QMatrix4x4()
        transform.setToIdentity()
        q = get_quaternion_from_euler(self.face_ori[0],self.face_ori[1],self.face_ori[2])
        transform.translate(self.face_pos[0]/100, self.face_pos[1]/100, self.face_pos[2]/100)
        transform.rotate(QtGui.QQuaternion(QtGui.QVector4D(q[0],q[1],q[2],q[3])))
        self.face_shape.setTransform(transform)


        transform = QtGui.QMatrix4x4()
        transform.setToIdentity()
        transform.translate(self.p3d[0]/100, self.p3d[1]/100, self.p3d[2]/100)

        self.gaze_pos.setTransform(transform)


        self.p2d[1]=-self.p2d[1]
        pilDrawCross(self.game_ui_img, (self.p2d+np.array(image_size)//2).astype(np.int), (200,0,0), 3)
        self.game_ui_img = pilShowErrorEllipse(self.game_ui_img, 10, self.p2d+np.array(image_size)//2, self.kalman.P,(255,0,0), 2)        
        # Just put a reference on the nose
        #face.draw_reference_frame(image, pos, ori, origin=face.get_landmark_pos(Face.nose_tip_index))
        self.image.setImage(np.swapaxes(self.updated_image,0,1))
        self.point_pos.setImage(np.swapaxes(np.array(self.game_ui_img),0,1))

        self.gaze_vector.setData(pos=np.array([[self.face_pos[0]/100, self.face_pos[1]/100, self.face_pos[2]/100],[self.p3d[0]/100, self.p3d[1]/100, self.p3d[2]/100]]));

        self.infos.setText(f"Head direction 2D projection : {self.p2d}")


if __name__ == '__main__':
    app=QtWidgets.QApplication(sys.argv)
    form=WinForm()
    form.show()
    sys.exit(app.exec_())
