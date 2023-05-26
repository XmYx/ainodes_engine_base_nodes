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
        self.image = QLabel(self)
        self.image.setAlignment(Qt.AlignRight)
        self.image.setObjectName(self.node.content_label_objname)
        self.checkbox = QtWidgets.QCheckBox("Autosave")
        self.meta_checkbox = QtWidgets.QCheckBox("Embed Node graph in PNG")
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("white"))
        palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, QtGui.QColor("black"))
        self.checkbox.setPalette(palette)
        self.meta_checkbox.setPalette(palette)
        self.button = QtWidgets.QPushButton("Save Image")
        self.next_button = QtWidgets.QPushButton("Show Next")
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.button)
        button_layout.addWidget(self.next_button)
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(15, 30, 15, 35)
        layout.addWidget(self.image)
        layout.addWidget(self.checkbox)
        layout.addWidget(self.meta_checkbox)
        layout.addLayout(button_layout)
        self.setLayout(layout)

@register_node(OP_NODE_IMG_PREVIEW)
class ImagePreviewNode(AiNode):
    icon = "icons/out.png"
    op_code = OP_NODE_IMG_PREVIEW
    op_title = "Image Preview"
    content_label_objname = "image_output_node"
    category = "Image"


    def __init__(self, scene):
        super().__init__(scene, inputs=[5,6,1], outputs=[5,6,1])
        self.content.button.clicked.connect(self.manual_save)
        self.content.next_button.clicked.connect(self.show_next_image)
        pass

    def initInnerClasses(self):
        self.content = ImagePreviewWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 400
        self.grNode.width = 320
        self.images = []
        self.index = 0
        self.content.preview_signal.connect(self.show_image)
        self.content.eval_signal.connect(self.evalImplementation)


    def show_next_image(self):
        length = len(self.images)
        if self.index >= length:
            self.index = 0
        if length > 0:
            img = self.images[self.index]
            #print(img)
            # Create a new RGBA image with the same dimensions as the greyscale image
            mask_image = Image.new("RGBA", img.size, (0, 0, 0, 0))

            # Get the pixel data from the greyscale image
            pixels = img.load()

            # Loop over all the pixels in the image
            for x in range(img.width):
                for y in range(img.height):
                    # Get the pixel value
                    value = pixels[x, y]

                    # Set the RGBA value of the corresponding pixel in the mask image
                    mask_image.putpixel((x, y), (255, 255, 255, value))


            pixmap = pil_image_to_pixmap(mask_image)
            self.content.image.setPixmap(pixmap)
            self.setOutput(0, [pixmap])
            self.index += 1
            self.resize()

    @QtCore.Slot()
    def evalImplementation_thread(self, index=0):
        self.busy = True
        if len(self.getInputs(0)) > 0:
            input_images = self.getInputData(0)
            if input_images is not None:
                for image in input_images:
                    self.content.preview_signal.emit(image)
                    if len(input_images) > 1:
                        time.sleep(0.1)
                return input_images

        """elif len(self.getInputs(1)) > 0:
            data_node, other_index = self.getInput(1)
            data = data_node.getOutput(other_index)
            for key, value in data.items():
                if key[1] == 'list':
                    if key[0] == 'images':
                        for image in value:
                            self.images.append(image)
                            # Create a new QPixmap object with the same size as the original image
                            #print(key[0], image)

        return self.images"""
    @QtCore.Slot(object)
    def show_image(self, image):
        self.content.image.setPixmap(image)
        self.resize()
        self.resize()


    @QtCore.Slot(object)
    def onWorkerFinished(self, result):
        super().onWorkerFinished(None)
        if self.content.checkbox.isChecked() == True:
            #for image in val:
            self.save_image(result[0])
        self.setOutput(0, result)
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

                #os.makedirs(os.path.join(gs.output, "metas"), exist_ok=True)
                #filename = f"{gs.output}/metas/{timestamp}.json"
                #self.scene.saveToFile(filename)
            if gs.logging:
                print(f"IMAGE PREVIEW NODE: File saved at {filename}")
        except Exception as e:
            print(f"IMAGE PREVIEW NODE: Image could not be saved because: {e}")

    def onInputChanged(self, socket=None):
        pass


    def resize(self):
        self.grNode.setToolTip("")
        self.grNode.height = self.content.image.pixmap().size().height() + 190
        self.grNode.width = self.content.image.pixmap().size().width() + 32
        self.content.image.setMinimumHeight(self.content.image.pixmap().size().height())
        self.content.image.setMinimumWidth(self.content.image.pixmap().size().width())
        self.content.setGeometry(0, 0, self.content.image.pixmap().size().width(),
                                 self.content.image.pixmap().size().height())
        for socket in self.outputs + self.inputs:
            socket.setSocketPosition()
        self.updateConnectedEdges()