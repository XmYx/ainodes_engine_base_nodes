import datetime
import os
import threading
import time

from qtpy.QtWidgets import QLabel
from qtpy.QtCore import Qt
from qtpy import QtWidgets, QtGui, QtCore

from ..ainodes_backend import pixmap_to_pil_image, pil_image_to_pixmap

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend.node_engine.utils import dumpException

from PIL import Image

OP_NODE_IMG_PREVIEW = get_next_opcode()

class ImageOutputWidget(QDMNodeContentWidget):
    preview_signal = QtCore.Signal(object)
    def initUI(self):
        self.image = QLabel(self)
        self.image.setAlignment(Qt.AlignRight)
        self.image.setObjectName(self.node.content_label_objname)
        self.checkbox = QtWidgets.QCheckBox("Autosave")

        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("white"))
        palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, QtGui.QColor("black"))

        self.checkbox.setPalette(palette)

        self.button = QtWidgets.QPushButton("Save Image")
        self.next_button = QtWidgets.QPushButton("Show Next")
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.button)
        button_layout.addWidget(self.next_button)
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(15, 30, 15, 35)
        layout.addWidget(self.image)
        layout.addWidget(self.checkbox)
        layout.addLayout(button_layout)
        self.setLayout(layout)


    def serialize(self):
        res = super().serialize()
        #res['value'] = self.edit.text()
        return res

    def deserialize(self, data, hashmap={}):
        res = super().deserialize(data, hashmap)
        try:
            #value = data['value']
            #self.image.setPixmap(value)
            return True & res
        except Exception as e:
            dumpException(e)
        return res


@register_node(OP_NODE_IMG_PREVIEW)
class ImagePreviewWidget(AiNode):
    icon = "icons/out.png"
    op_code = OP_NODE_IMG_PREVIEW
    op_title = "Image Preview"
    content_label_objname = "image_output_node"
    category = "image"


    def __init__(self, scene):
        super().__init__(scene, inputs=[5,6,1], outputs=[5,6,1])
        #self.eval()
        #self.content.eval_signal.connect(self.evalImplementation)
        self.content.button.clicked.connect(self.save_image)
        self.content.next_button.clicked.connect(self.show_next_image)

    def initInnerClasses(self):
        self.content = ImageOutputWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.output_socket_name = ["EXEC", "DATA", "IMAGE"]
        self.input_socket_name = ["EXEC", "DATA", "IMAGE"]
        self.grNode.height = 200
        self.images = []
        self.index = 0
        self.content.preview_signal.connect(self.show_image)


    """def evalImplementation(self, index=0):
        thread0 = threading.Thread(target=self.evalImplementation_thread)
        thread0.start()"""


    def show_next_image(self):
        length = len(self.images)
        if self.index >= length:
            self.index = 0
        if length > 0:
            img = self.images[self.index]
            print(img)

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
            """mask_pixmap = QtGui.QPixmap(pixmap.size())

            # Use a QPainter object to paint the alpha values onto the new pixmap
            painter = QtGui.QPainter(mask_pixmap)

            image = pixmap.toImage()
            for x in range(pixmap.width()):
                for y in range(pixmap.height()):
                    color = image.pixelColor(x, y)
                    intensity = color.red()  # assumes the image is in grayscale
                    alpha = 255 - intensity
                    painter.setPen(QtGui.QColor(0, 0, 0, alpha))
                    painter.drawPoint(x, y)
            painter.end()
            mask_image = mask_pixmap.toImage().convertToFormat(QtGui.QImage.Format_ARGB32_Premultiplied)
            mask_pixmap = QtGui.QPixmap.fromImage(mask_image)"""
            self.content.image.setPixmap(pixmap)
            self.setOutput(0, pixmap)
            self.index += 1
            self.resize()
    def evalImplementation(self, index=0):
        #self.markDirty(True)
        if self.getInput(0) is not None:
            input_node, other_index = self.getInput(0)
            if not input_node:
                self.grNode.setToolTip("Input is not connected")
                self.markInvalid()
                return

            val = input_node.getOutput(other_index)

            if val is None:
                self.grNode.setToolTip("Input is NaN")
                self.markInvalid()
                return
            #print("Preview Node Value", val)
            self.content.preview_signal.emit(val)
            #self.content.image.setPixmap(val)

            self.setOutput(0, val)
            self.markInvalid(False)
            self.markDirty(False)
            #print("Reloaded")
            if self.content.checkbox.isChecked() == True:
                self.save_image()
            if len(self.getOutputs(2)) > 0:
                self.executeChild(output_index=2)
        elif self.getInput(1) is not None:
            data_node, other_index = self.getInput(1)
            data = data_node.getOutput(other_index)
            print(data)
            self.images = []
            for key, value in data.items():
                if key[1] == 'list':
                    if key[0] == 'images':
                        for image in value:
                            self.images.append(image)
                            # Create a new QPixmap object with the same size as the original image
                            print(key[0], image)
            val = None
        else:
            val = self.value
        return val
    @QtCore.Slot(object)
    def show_image(self, image):
        self.content.image.setPixmap(image)
        self.resize()

    def save_image(self):
        try:
            pixmap = self.content.image.pixmap()
            image = pixmap_to_pil_image(pixmap)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            os.makedirs("output/stills", exist_ok=True)
            filename = f"output/stills/{timestamp}.png"
            image.save(filename)
            print(f"IMAGE PREVIEW NODE: File saved at {filename}")
        except Exception as e:
            print(f"IMAGE PREVIEW NODE: Image could not be saved because: {e}")
    def onInputChanged(self, socket=None):
        #super().onInputChanged(socket=socket)
        self.markDirty(True)
        self.markInvalid(True)
        #self.eval()
    def eval(self):
        self.evalImplementation(0)
        #self.content.eval_signal.emit()

    def resize(self):
        self.grNode.setToolTip("")
        self.grNode.height = self.content.image.pixmap().size().height() + 155
        self.grNode.width = self.content.image.pixmap().size().width() + 32
        self.content.image.setMinimumHeight(self.content.image.pixmap().size().height())
        self.content.image.setMinimumWidth(self.content.image.pixmap().size().width())
        self.content.setGeometry(0, 0, self.content.image.pixmap().size().width(),
                                 self.content.image.pixmap().size().height())
        for socket in self.outputs + self.inputs:
            socket.setSocketPosition()
        self.updateConnectedEdges()

