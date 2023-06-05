import json
import os
import time

import requests
from PIL import Image
from PIL.ImageQt import ImageQt
from qtpy.QtCore import QUrl
from qtpy import QtGui
from qtpy import QtCore, QtWidgets
from qtpy.QtWidgets import QFileDialog
from qtpy.QtGui import QPixmap
import cv2

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from custom_nodes.ainodes_engine_base_nodes.ainodes_backend import pil_image_to_pixmap
from ainodes_frontend import singleton as gs

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
        self.open_graph_button = QtWidgets.QPushButton("Open Graph")
        self.create_button_layout([self.open_button, self.open_graph_button])
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
        fileName, _ = QFileDialog.getOpenFileName(self.node.scene.getView().parent().window(), "Select Image", "",
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
    category = "Image"
    input_socket_name = ["EXEC"]
    output_socket_name = ["EXEC", "IMAGE"]

    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[5,1])
        self.content.parent_resize_signal.connect(self.resize)
        self.images = []

    def initInnerClasses(self):
        self.content = ImageInputWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.icon = self.icon

        self.grNode.height = 220
        self.content.eval_signal.connect(self.evalImplementation)
        self.content.open_graph_button.clicked.connect(self.tryOpenGraph)
        self.video = VideoPlayer()
        self.content_type = None

    def add_urls(self, url_list):

        i = 0
        for url in url_list:
            file_name = url.fileName()
            file_ext = os.path.splitext(file_name)[-1].lower()
            if gs.debug:
                print("FILE", file_name)
            if 'http' not in url.url():
                if gs.debug:
                    print("LOCAL")
                if file_ext in ['.png', '.jpg', '.jpeg']:

                    self.url = url.toLocalFile()

                    image = Image.open(url.toLocalFile())

                    metadata = image.info

                    # Check if the image has metadata
                    if metadata != {}:
                        self.content.open_graph_button.setVisible(True)
                    else:
                        self.content.open_graph_button.setVisible(False)


                    pixmap = pil_image_to_pixmap(image)
                    self.images.append(pixmap)
                    self.content.image.setPixmap(pixmap)
                    self.resize()
                    self.content_type = "image"
                elif file_ext in ['.mp4', '.avi', '.mov']:
                    pixmap = self.process_video_file(url.toLocalFile())
                    #for pixmap in pixmaps:
                    self.content.image.setPixmap(pixmap)
                    self.resize()
                    self.content_type = "video"
            else:
                temp_path = 'temp'
                os.makedirs(temp_path, exist_ok=True)
                local_file_name = url.toLocalFile()
                file_path = os.path.join(temp_path, f"frame_{i:04}.png")
                self.poormanswget(url.url(), file_path)
                i += 1
                self.url = file_path

    def poormanswget(self, url, filepath):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        if gs.debug:
            print("NEW_URL", QUrl(filepath))
        url = QUrl.fromLocalFile(filepath)
        self.add_urls([url])

    def process_video_file(self, file_path):
        if gs.debug:
            print("VIDEO", file_path)
        self.video.load_video(file_path)
        pixmap = self.video.get_frame()
        return pixmap

        """cap = cv2.VideoCapture(file_path)
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
        return pixmaps"""
    def cv2_to_qimage(self, cv_img):
        height, width, channel = cv_img.shape
        bytes_per_line = channel * width
        return QtGui.QImage(cv_img.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888).rgbSwapped()

    #@QtCore.Slot()
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

    def tryOpenGraph(self):
        try:

            image = Image.open(self.url)
            # Get the metadata from the image
            metadata = image.info
            # Check if the image has metadata
            if metadata is not None:
                if 'graph' in metadata:
                    json_data = metadata['graph']
                    # Deserialize the JSON data
                    deserialized_data = json.loads(json_data)
                    # Save the deserialized data as the next available numbered temp file
                    os.makedirs("temp", exist_ok=True)
                    temp_dir = "temp"
                    count = 1
                    while True:
                        temp_filename = os.path.join(temp_dir, f"temp{count}.json")
                        if not os.path.exists(temp_filename):
                            break
                        count += 1

                    with open(temp_filename, 'w') as file:
                        json.dump(deserialized_data, file)
                    meta = temp_filename

                    # Extract the filename from the URL
                    #filename = os.path.basename(self.url)
                    # Strip the extension from the filename
                    #filename_without_extension = os.path.splitext(filename)[0]
                    #meta = os.path.join(gs.metas, f"{filename_without_extension}.json")

                    self.scene.getView().parent().window().file_open_signal.emit(temp_filename)
        except Exception as e:
            print(e)




    #@QtCore.Slot()
    def evalImplementation_thread(self, index=0):
        self.markDirty(False)
        self.markInvalid(False)
        self.grNode.setToolTip("")
        if len(self.images) > 0:
            for pixmap in self.images:
                self.content.image.setPixmap(pixmap)
                time.sleep(0.1)
        if self.content_type == 'video':
            if gs.debug:
                print("GETTING VIDEO FRAME")
            pixmap = self.video.get_frame()
            if pixmap is not None:
                self.content.image.setPixmap(pixmap)
        else:
            pixmap = self.content.image.pixmap()
        return pixmap
    #@QtCore.Slot(object)
    def onWorkerFinished(self, pixmap):
        #super().onWorkerFinished(None)
        if pixmap is not None:
            self.setOutput(0, [pixmap])
            if len(self.getOutputs(1)) > 0:
                self.executeChild(output_index=1)
        else:
            print("End of Video or No Image loaded, stopping execution at", self)


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



class VideoPlayer:
    def __init__(self):
        self.video_file = None
        self.video_capture = None

    def load_video(self, video_file):
        try:
            self.video_capture.release()
        except:
            pass
        self.video_file = video_file
        self.video_capture = cv2.VideoCapture(self.video_file)

    def get_frame(self, skip=1):
        # Skip frames based on the specified interval
        for _ in range(skip - 1):
            self.video_capture.grab()
        try:
            # Read the next frame and convert it to a pixmap
            ret, frame = self.video_capture.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            pixmap = pil_image_to_pixmap(image)
        except:
            ret = None

        # Return the pixmap if the read was successful
        if ret:
            return pixmap
        else:
            return None

    def reset(self):
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)