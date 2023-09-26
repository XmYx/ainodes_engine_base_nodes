import datetime
import json
import os
import time

import PIL.Image
import numpy as np
import torch
from PIL.PngImagePlugin import PngInfo
from PyQt6.QtGui import QGuiApplication, QImage
from qtpy.QtWidgets import QLabel
from qtpy.QtCore import Qt
from qtpy import QtWidgets, QtGui, QtCore

from ..ainodes_backend import pixmap_to_tensor, tensor_image_to_pixmap, tensor2pil

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
        self.fps = self.create_spin_box(min_val=1, max_val=250, default_val=24, step=1, label_text="FPS")
        # self.checkbox = QtWidgets.QCheckBox("Autosave")
        # self.meta_checkbox = QtWidgets.QCheckBox("Embed Node graph in PNG")
        # self.clipboard = QtWidgets.QCheckBox("Copy to Clipboard")

        self.create_check_box("Autosave", spawn="autosave_checkbox")
        self.create_check_box("Embed Node graph in PNG", spawn="meta_checkbox")
        self.create_check_box("Copy to Clipboard", spawn="clipboard")


        self.create_button_layout([self.autosave_checkbox, self.meta_checkbox, self.clipboard])
        self.button = QtWidgets.QPushButton("Save Image")
        self.next_button = QtWidgets.QPushButton("Show Next")
        self.create_button_layout([self.button, self.next_button])
        self.start_stop = QtWidgets.QPushButton("Play / Pause")
        self.create_button_layout([self.start_stop])
        self.create_main_layout(grid=1)


@register_node(OP_NODE_IMG_PREVIEW)
class ImagePreviewNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/image_preview.png"
    op_code = OP_NODE_IMG_PREVIEW
    op_title = "Image Preview"
    content_label_objname = "image_output_node"
    category = "aiNodes Base/Image"
    dims = (512,512)

    def __init__(self, scene):
        super().__init__(scene, inputs=[5,6,1], outputs=[5,6,1])

    def initInnerClasses(self):
        self.content = ImagePreviewWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.icon = self.icon
        self.grNode.thumbnail = QtGui.QImage(self.grNode.icon).scaled(64, 64, QtCore.Qt.KeepAspectRatio)

        self.grNode.height = 350
        self.grNode.width = 400
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
                pixmap = tensor_image_to_pixmap(self.images[self.index])
                #pixmap = pil_image_to_pixmap(img)
                self.resize(pixmap)

                self.content.preview_signal.emit(pixmap)
                self.setOutput(0, [pixmap])
                self.index += 1
        else:
            self.timer.stop()

    #@QtCore.Slot()
    def evalImplementation_thread(self, index=0):
        self.busy = True
        if len(self.getInputs(0)) > 0:

            image = self.getInputData(0)

            self.images.append(image)
            return image

    def show_image(self, image):

        #pixmap = tensor_image_to_pixmap(image)

        self.content.image.setPixmap(image)
        self.resize(image)


    def onWorkerFinished(self, result, exec=True):
        if self.content.autosave_checkbox.isChecked() == True:
            if result is not None:
                self.save_image(result)
        if result is not None:

            for item in result:

                if not isinstance(item, QtGui.QPixmap):

                    pixmap = tensor_image_to_pixmap(item)
                else:
                    pixmap = item
                self.content.preview_signal.emit(pixmap)
                self.resize(pixmap)
                # for image in result:
                #     self.content.preview_signal.emit(image)
                    #time.sleep(0.1)

        self.setOutput(0, result)
        self.markInvalid(False)
        self.markDirty(False)
        if exec:
            if gs.should_run:
                self.executeChild(2)

    def manual_save(self):
        for image in self.images:
            self.save_image(image)

    def save_image(self, pixmap):
        try:
            image = tensor2pil(pixmap)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f')
            os.makedirs(os.path.join(gs.prefs.output, "stills"), exist_ok=True)
            filename = f"{gs.prefs.output}/stills/{timestamp}.png"

            meta_save = self.content.meta_checkbox.isChecked()

            clipboard = self.content.clipboard.isChecked()

            if meta_save:

                filename = f"{gs.prefs.output}/stills/{timestamp}_i.png"

                metadata = PngInfo()

                json_data = self.scene.serialize()

                metadata.add_text("graph", json.dumps(json_data))

                image.save(filename, pnginfo=metadata, compress_level=4)


            else:
                image.save(filename)
            if clipboard:
                print("Copied to clipboard")
                clipboard = QGuiApplication.clipboard()
                clipboard.setImage(QImage(filename))

            if gs.logging:
                print(f"IMAGE PREVIEW NODE: File saved at {filename}")
        except Exception as e:
            print(f"IMAGE PREVIEW NODE: Image could not be saved because: {e}")

    def resize(self, pixmap):
        self.grNode.setToolTip("")
        self.grNode.height = pixmap.size().height() + 255
        self.grNode.width = pixmap.size().width() + 30
        self.content.setGeometry(0, 25, pixmap.size().width(), pixmap.size().height() + 150)
        self.update_all_sockets()