import time

import numpy as np
from PIL import Image
#from qtpy.QtWidgets import QLineEdit, QLabel, QPushButton, QFileDialog, QVBoxLayout
from qtpy import QtWidgets, QtCore, QtGui

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import CalcNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend.node_engine.utils import dumpException
from ..ainodes_backend import pixmap_to_pil_image, pil_image_to_pixmap, \
    pixmap_composite_method_list
from ..ainodes_backend.FILM.inference import FilmModel

OP_NODE_FILM = get_next_opcode()

from ainodes_frontend import singleton as gs

class FILMWidget(QDMNodeContentWidget):
    def initUI(self):
        # Create a label to display the image
        self.text_label = QtWidgets.QLabel("Image Operator:")

        self.film = QtWidgets.QSpinBox()
        self.film.setMinimum(1)
        self.film.setSingleStep(1)
        self.film.setMaximum(4096)
        self.film.setValue(25)

        self.composite_method = QtWidgets.QComboBox()
        self.composite_method.addItems(pixmap_composite_method_list)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.composite_method)
        layout.addWidget(self.film)

        self.setLayout(layout)


    def serialize(self):
        res = super().serialize()
        res['film'] = self.film.value()
        return res

    def deserialize(self, data, hashmap={}):
        res = super().deserialize(data, hashmap)
        try:
            self.film.setValue(int(data['film']))
            #self.image.setPixmap(value)
            return True & res
        except Exception as e:
            dumpException(e)
        return res


@register_node(OP_NODE_FILM)
class FILMNode(CalcNode):
    icon = "icons/in.png"
    op_code = OP_NODE_FILM
    op_title = "FILM"
    content_label_objname = "FILM_node"
    category = "video"


    def __init__(self, scene):
        super().__init__(scene, inputs=[5,5,1], outputs=[5,1,1])
        self.painter = QtGui.QPainter()

        self.FILM_temp = []

        if "FILM" not in gs.models:
            gs.models["FILM"] = FilmModel()
        #self.eval()
        #self.content.eval_signal.connect(self.eval)

    def initInnerClasses(self):
        self.content = FILMWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.output_socket_name = ["EXEC", "EXEC/F", "IMAGE"]
        self.input_socket_name = ["EXEC", "IMAGE1", "IMAGE2"]

        self.grNode.height = 220
    @QtCore.Slot(int)
    def evalImplementation(self, index=0):
        if self.getInput(1) != None:
            node, index = self.getInput(1)
            pixmap1 = node.getOutput(index)
        else:
            pixmap1 = None
        if self.getInput(0) != None:
            node, index = self.getInput(0)
            pixmap2 = node.getOutput(index)
        else:
            pixmap2 = None
        if pixmap1 != None and pixmap2 != None:

            image1 = pixmap_to_pil_image(pixmap1)
            image2 = pixmap_to_pil_image(pixmap2)
            np_image1 = np.array(image1)
            np_image2 = np.array(image2)
            frames = gs.models["FILM"].inference(np_image1, np_image2, inter_frames=25)
            print(f"FILM NODE:  {len(frames)}")
            for frame in range(len(frames) - 2):
                image = Image.fromarray(frame)
                pixmap = pil_image_to_pixmap(image)
                self.setOutput(0, pixmap)
                if len(self.getOutputs(1)) > 0:
                    self.executeChild(output_index=1)
                time.sleep(0.05)
            self.markDirty(False)
            self.markInvalid(False)
        elif pixmap1 != None:
            try:
                image = pixmap_to_pil_image(pixmap1)
                np_image = np.array(image)
                self.FILM_temp.append(np_image)
                if len(self.FILM_temp) == 2:
                    frames = gs.models["FILM"].inference(self.FILM_temp[0], self.FILM_temp[1], inter_frames=self.content.film.value())
                    print(f"FILM NODE:  {len(frames)}")
                    for frame in frames:
                        image = Image.fromarray(frame)
                        pixmap = pil_image_to_pixmap(image)
                        self.setOutput(0, pixmap)
                        if len(self.getOutputs(1)) > 0:
                            self.executeChild(output_index=1)
                        time.sleep(0.05)
                    self.FILM_temp = [self.FILM_temp[1]]
                print(f"FILM NODE: Using only First input")
            except:
                if len(self.getOutputs(2)) > 0:
                    self.executeChild(output_index=2)

                pass
        elif pixmap1 != None:
            try:
                self.setOutput(0, pixmap1)
                print(f"FILM NODE: Using only Second input - Passthrough")
                if len(self.getOutputs(2)) > 0:
                    self.executeChild(output_index=2)

            except:
                pass
        if len(self.getOutputs(2)) > 0:
            self.executeChild(output_index=2)
        return None
    def onMarkedDirty(self):
        self.value = None
    def eval(self):
        self.markDirty(True)
        self.evalImplementation()
