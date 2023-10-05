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

        # Create checkboxes and store the horizontal layout
        checkbox_layout = self.create_horizontal_layout([
            self.create_check_box("Autosave", spawn="autosave_checkbox"),
            self.create_check_box("Embed Node graph in PNG", spawn="meta_checkbox"),
            self.create_check_box("Copy to Clipboard", spawn="clipboard")
        ])

        # Add any additional widgets/buttons to the checkbox_layout if needed
        self.button = QtWidgets.QPushButton("Save Image")
        self.next_button = QtWidgets.QPushButton("Show Next")
        checkbox_layout.addWidget(self.button)
        checkbox_layout.addWidget(self.next_button)

        self.start_stop = QtWidgets.QPushButton("Play / Pause")
        checkbox_layout.addWidget(self.start_stop)


        self.create_main_layout()

        # Add the checkbox_layout to the main layout
        self.main_layout.addLayout(checkbox_layout)


@register_node(OP_NODE_IMG_PREVIEW)
class ImagePreviewNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/image_preview.png"
    op_code = OP_NODE_IMG_PREVIEW
    op_title = "Image Preview"
    content_label_objname = "image_output_node"
    category = "aiNodes Base/Image"
    output_data_ports = [0]
    NodeContent_class = ImagePreviewWidget
    dim = (340, 460)

    make_dirty = True



    def __init__(self, scene):
        super().__init__(scene, inputs=[5,6,1], outputs=[5,6,1])
        self.images = []
        self.pixmaps = []

        self.index = 0
        self.content.preview_signal.connect(self.show_image)
        self.content.button.clicked.connect(self.manual_save)
        self.content.next_button.clicked.connect(self.show_next_image)
        self.timer = QtCore.QTimer()
        self.timer.setInterval(40)
        self.timer.timeout.connect(self.iter_preview)
        self.content.start_stop.clicked.connect(self.start_stop)
        self.content.fps.valueChanged.connect(self.set_interval)

    def set_interval(self, fps):
        interval = int(1000.0 / fps)
        self.timer.setInterval(interval)

    def remove(self):
        try:
            self.timer.stop()
        except:
            pass
        del self.images
        super().remove()
    def start_stop(self):
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start()

    def add_image(self, image_tensor, show=False, reset=True):
        if reset:
            self.images.clear()
            self.pixmaps.clear()
        self.images.append(image_tensor)
        pixmap = tensor_image_to_pixmap(image_tensor)
        if show:
            self.content.preview_signal.emit(pixmap)
        self.pixmaps.append(pixmap)

    def iter_preview(self):
        self.show_next_image()

    def show_next_image(self):
        length = len(self.pixmaps)
        if self.index >= length:
            self.index = 0
        if length > 0:
            pixmap = self.pixmaps[self.index]
            self.resize(pixmap)
            self.content.preview_signal.emit(pixmap)
            self.setOutput(0, [pixmap])
            self.index += 1
        else:
            self.timer.stop()

    def evalImplementation_thread(self, index=0):
        start_time = time.time()  # Start the timer

        result = None
        self.busy = True
        if len(self.getInputs(0)) > 0:
            image = self.getInputData(0)

            if image.shape[0] > 1:  # Assuming the tensor shape is [channels, height, width]
                for img in image:
                    print(img.shape)
                    self.add_image(img.unsqueeze(0), show=True)
            else:
                self.add_image(image, show=True)
        if self.content.autosave_checkbox.isChecked() == True:
            if image is not None:
                self.save_image(image)
        # if image is not None:
        #     for item in image:
        #         if not isinstance(item, QtGui.QPixmap):
        #             pixmap = tensor_image_to_pixmap(item)
        #         else:
        #             pixmap = item
        #         self.content.preview_signal.emit(pixmap)
        #         self.resize(pixmap)
        if gs.debug:
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            print(f"Image Preview Node executed in {elapsed_time:.4f} seconds.")  # Print the time taken
        return [image]

    def show_image(self, image):

        #pixmap = tensor_image_to_pixmap(image)

        self.content.image.setPixmap(image)
        self.resize(image)


    def manual_save(self):
        #for image in self.images:
        self.save_image(self.images[len(self.images) - 1][0])

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
        dims = [pixmap.size().height() + 255, pixmap.size().width() + 30]
        if self.dim != dims:
            self.dim = dims
            self.grNode.height = dims[0]
            self.grNode.width = dims[1]
            self.content.setGeometry(0, 25, pixmap.size().width(), pixmap.size().height() + 150)
            self.update_all_sockets()

    def onInputChanged(self, socket=None):
        self.markDirty(True)