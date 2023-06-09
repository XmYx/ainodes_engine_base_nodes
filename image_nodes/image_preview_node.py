import datetime
import json
import os
import time

from PIL.PngImagePlugin import PngInfo
from qtpy.QtWidgets import QLabel
from qtpy.QtCore import Qt
from qtpy import QtWidgets, QtGui, QtCore

from ..ainodes_backend import pixmap_to_pil_image, pil_image_to_pixmap

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs
from PIL import Image

OP_NODE_IMG_PREVIEW = get_next_opcode()

class ImagePreviewWidget(QDMNodeContentWidget):
    preview_signal = QtCore.Signal(object)
    def initUI(self):

        self.image = self.create_label("")
        self.fps = self.create_spin_box(min_val=1, max_val=250, default_val=24, step_value=1, label_text="FPS")
        self.checkbox = self.create_check_box("Autosave")
        self.meta_checkbox = self.create_check_box("Embed Node graph in PNG")
        self.button = QtWidgets.QPushButton("Save Image")
        self.next_button = QtWidgets.QPushButton("Show Next")
        self.create_button_layout([self.button, self.next_button])
        self.start_stop = QtWidgets.QPushButton("Play / Pause")
        self.create_button_layout([self.start_stop])
        self.create_main_layout(grid=1)

        # self.image = QLabel(self)
        # self.image.setAlignment(Qt.AlignLeft)
        # self.image.setObjectName(self.node.content_label_objname)
        # self.checkbox = QtWidgets.QCheckBox("Autosave")
        # self.meta_checkbox = QtWidgets.QCheckBox("Embed Node graph in PNG")
        #
        #
        #
        # palette = QtGui.QPalette()
        # palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("white"))
        # palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, QtGui.QColor("black"))
        # self.checkbox.setPalette(palette)
        # self.meta_checkbox.setPalette(palette)
        # button_layout = QtWidgets.QHBoxLayout()
        # button_layout.addWidget(self.button)
        # button_layout.addWidget(self.next_button)
        # layout = QtWidgets.QVBoxLayout()
        # layout.setContentsMargins(15, 30, 15, 35)
        # layout.addWidget(self.image)
        # layout.addWidget(self.checkbox)
        # layout.addWidget(self.meta_checkbox)
        # layout.addLayout(button_layout)
        # self.setLayout(layout)



@register_node(OP_NODE_IMG_PREVIEW)
class ImagePreviewNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/image_preview.png"
    op_code = OP_NODE_IMG_PREVIEW
    op_title = "Image Preview"
    content_label_objname = "image_output_node"
    category = "Image"
    dims = (512,512)

    def __init__(self, scene):
        super().__init__(scene, inputs=[5,6,1], outputs=[5,6,1])

    def initInnerClasses(self):
        self.content = ImagePreviewWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.icon = self.icon
        self.grNode.thumbnail = QtGui.QImage(self.grNode.icon).scaled(64, 64, QtCore.Qt.KeepAspectRatio)

        self.grNode.height = 400
        self.grNode.width = 320
        self.images = []
        self.index = 0
        self.content.preview_signal.connect(self.show_image)
        self.content.eval_signal.connect(self.evalImplementation)
        self.content.button.clicked.connect(self.manual_save)
        self.content.next_button.clicked.connect(self.show_next_image)
        self.timer = QtCore.QTimer()
        self.timer.setInterval(40)
        self.timer.timeout.connect(self.iter_preview)
        self.content.start_stop.clicked.connect(self.start_stop)
        self.content.fps.valueChanged.connect(self.set_interval)
        #self.timer.start()

    def set_interval(self, fps):
        interval = int(1000.0 / fps)
        self.timer.setInterval(interval)


    def remove(self):
        try:
            self.timer.start()
        except:
            pass
        del self.images
        super().remove()
    def start_stop(self):
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start()
    def iter_preview(self):
        self.show_next_image()


    def show_next_image(self):
        if hasattr(self, "images"):
            length = len(self.images)
            if self.index >= length:
                self.index = 0
            if length > 0:
                pixmap = self.images[self.index]
                #pixmap = pil_image_to_pixmap(img)
                self.resize(pixmap)

                self.content.image.setPixmap(pixmap)
                self.setOutput(0, [pixmap])
                self.index += 1
        else:
            self.timer.stop()

    #@QtCore.Slot()
    def evalImplementation_thread(self, index=0):
        self.busy = True
        if len(self.getInputs(0)) > 0:
            images = self.getInputData(0)
            return images

    def show_image(self, image):
        self.content.image.setPixmap(image)
        #self.resize()


    def onWorkerFinished(self, result):
        self.busy = False
        self.images = result
        if self.content.checkbox.isChecked() == True:
            self.save_image(result[0])
        if result is not None:
            for image in result:
                self.content.preview_signal.emit(image)
                #time.sleep(0.1)

        if result is not None:
            self.resize(result[0])
            #self.timer.start()
        self.setOutput(0, self.images)
        self.markInvalid(False)
        self.markDirty(False)
        if gs.should_run:
            self.executeChild(2)

    def manual_save(self):

        for image in self.images:
            self.save_image(image)

    def save_image(self, pixmap):
        try:
            image = pixmap_to_pil_image(pixmap)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f')
            os.makedirs(os.path.join(gs.output, "stills"), exist_ok=True)
            filename = f"{gs.output}/stills/{timestamp}.png"

            meta_save = self.content.meta_checkbox.isChecked()
            if meta_save:

                metadata = PngInfo()

                json_data = self.scene.serialize()

                metadata.add_text("graph", json.dumps(json_data))


                image.save(filename, pnginfo=metadata, compress_level=4)
            else:
                image.save(filename)

            if gs.logging:
                print(f"IMAGE PREVIEW NODE: File saved at {filename}")
        except Exception as e:
            print(f"IMAGE PREVIEW NODE: Image could not be saved because: {e}")

    def resize(self, pixmap):
        self.grNode.setToolTip("")
        self.grNode.height = pixmap.size().height() + 300
        self.grNode.width = pixmap.size().width() + 32
        self.content.image.setMinimumHeight(pixmap.size().height())
        self.content.image.setMinimumWidth(pixmap.size().width())

        self.content.setMaximumHeight(pixmap.size().height() + 200)
        self.content.setMaximumWidth(pixmap.size().width())

        self.update_all_sockets()
        #self.content.setGeometry(0, 0, pixmap.size().width(),
        #                         pixmap.size().height())
        #for socket in self.outputs + self.inputs:
        #    socket.setSocketPosition()
        #self.updateConnectedEdges()