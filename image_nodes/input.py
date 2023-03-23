import os
import time

import requests
from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import QUrl
from PySide6.QtGui import QImage
from qtpy import QtGui
from qtpy import QtCore, QtWidgets
from qtpy.QtWidgets import QLabel, QFileDialog, QVBoxLayout
from qtpy.QtGui import QPixmap
import cv2

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend.node_engine.utils import dumpException
from custom_nodes.ainodes_engine_base_nodes.ainodes_backend import pil_image_to_pixmap, poorman_wget

OP_NODE_IMG_INPUT = get_next_opcode()

class ImageInputWidget(QDMNodeContentWidget):
    fileName = None
    parent_resize_signal = QtCore.Signal()
    def initUI(self):
        self.create_widgets()
        self.create_main_layout()
    def create_widgets(self):
        self.image = self.create_label("Image")
        self.open_button = QtWidgets.QPushButton("Open New Image")
        self.open_button.clicked.connect(self.openFileDialog)
        self.create_button_layout([self.open_button])
        self.firstRun_done = None

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mkv)")
        if file_name:
            # print(file_name)
            self.video.load_video(file_name)
            self.advance_frame()

    def openFileDialog(self):
        print("OPENING")
        # Open the file dialog to select a PNG file
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Image", "",
                                                  "PNG Files (*.png);JPEG Files (*.jpeg *.jpg);All Files(*)")
        # If a file is selected, display the image in the label
        if fileName != None:
            image = Image.open(fileName)
            qimage = ImageQt(image)
            pixmap = QPixmap().fromImage(qimage)
            self.image.setPixmap(pixmap)
            self.parent_resize_signal.emit()
            self.fileName = fileName



@register_node(OP_NODE_IMG_INPUT)
class ImageInputNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/input_image.png"
    op_code = OP_NODE_IMG_INPUT
    op_title = "Input Image"
    content_label_objname = "image_input_node"
    category = "image"
    input_socket_name = ["EXEC"]
    output_socket_name = ["EXEC", "IMAGE"]

    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[5,1])
        self.content.parent_resize_signal.connect(self.resize)
        self.images = []

    def initInnerClasses(self):
        self.content = ImageInputWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 220
        self.busy = False
        self.content.eval_signal.connect(self.evalImplementation)

    def add_urls(self, url_list):
        i = 0
        for url in url_list:
            file_name = url.fileName()
            file_ext = os.path.splitext(file_name)[-1].lower()
            print("FILE", file_name)
            if 'http' not in url.url():
                print("LOCAL")
                if file_ext in ['.png', '.jpg', '.jpeg']:

                    print("PATH", url.toLocalFile())

                    image = Image.open(url.toLocalFile())
                    pixmap = pil_image_to_pixmap(image)
                    self.images.append(pixmap)
                    self.content.image.setPixmap(pixmap)
                    self.resize()
                elif file_ext in ['.mp4', '.avi', '.mov']:
                    pixmaps = self.process_video_file(url.toLocalFile())
                    for pixmap in pixmaps:
                        self.content.image.setPixmap(pixmap)
                        self.resize()
            else:
                temp_path = 'temp'
                os.makedirs(temp_path, exist_ok=True)
                local_file_name = url.toLocalFile()
                file_path = os.path.join(temp_path, f"frame_{i:04}.png")
                self.poormanswget(url.url(), file_path)
                i += 1

    def poormanswget(self, url, filepath):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("NEW_URL", QUrl(filepath))
        url = QUrl.fromLocalFile(filepath)
        self.add_urls([url])

    def process_video_file(self, file_path):
        print("VIDEO", file_path)

        cap = cv2.VideoCapture(file_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Number of frames: {num_frames}")

        frames_dir = f"frames_{time.time():.0f}"
        os.makedirs(frames_dir, exist_ok=True)

        pixmaps = []
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            pixmap = pil_image_to_pixmap(image)
            pixmap.save(os.path.join(frames_dir, f"frame_{i:04}.png"))
            pixmaps.append(pixmap)

        cap.release()
        return pixmaps
    def cv2_to_qimage(self, cv_img):
        height, width, channel = cv_img.shape
        bytes_per_line = channel * width
        return QtGui.QImage(cv_img.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888).rgbSwapped()

    @QtCore.Slot()
    def resize(self):
        self.content.setMinimumHeight(self.content.image.pixmap().size().height())
        self.content.setMinimumWidth(self.content.image.pixmap().size().width())
        self.grNode.height = self.content.image.pixmap().size().height() + 96
        self.grNode.width = self.content.image.pixmap().size().width() + 64
        self.update_all_sockets()

    def init_image(self):
        if self.content.fileName == None:
            self.content.fileName = self.openFileDialog()
        if self.content.fileName != None:
            image = Image.open(self.content.fileName)
            qimage = ImageQt(image)
            pixmap = QPixmap().fromImage(qimage)
            self.content.image.setPixmap(pixmap)
        self.resize()

    def onMarkedDirty(self):
        self.content.fileName = None
    @QtCore.Slot()
    def evalImplementation_thread(self, index=0):
        #self.init_image()
        self.markDirty(False)
        self.markInvalid(False)
        self.grNode.setToolTip("")
        if len(self.images) > 0:
            for pixmap in self.images:
                self.content.image.setPixmap(pixmap)
                time.sleep(0.1)
        self.setOutput(0, self.images)
        if len(self.getOutputs(1)) > 0:
            self.executeChild(output_index=1)
        return self.images

    def openFileDialog(self):
        # Open the file dialog to select a PNG file
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(None, "Select Image", "",
                                                  "PNG Files (*.png);JPEG Files (*.jpeg *.jpg);All Files(*)",
                                                  options=options)
        # If a file is selected, display the image in the label
        if file_name:
            return file_name
        return None
