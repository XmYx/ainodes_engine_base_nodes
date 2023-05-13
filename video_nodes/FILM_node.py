import copy
import threading
import time

import numpy as np
from PIL import Image
#from qtpy.QtWidgets import QLineEdit, QLabel, QPushButton, QFileDialog, QVBoxLayout
from qtpy import QtWidgets, QtCore, QtGui

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
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
        self.text_label = QtWidgets.QLabel("FILM Interpolation:")
        self.film = self.create_spin_box("FRAMES", 1, 4096, 10, 1)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(15,15,15,15)
        layout.addWidget(self.film)
        self.setLayout(layout)



@register_node(OP_NODE_FILM)
class FILMNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/film.png"
    op_code = OP_NODE_FILM
    op_title = "FILM"
    content_label_objname = "FILM_node"
    category = "Interpolation"


    def __init__(self, scene):
        super().__init__(scene, inputs=[5,5,1], outputs=[5,1])
        self.painter = QtGui.QPainter()

        self.FILM_temp = []
        self.content.eval_signal.connect(self.evalImplementation)
        if "FILM" not in gs.models:
            gs.models["FILM"] = FilmModel()
        self.busy = False
        #self.eval()
    def __del__(self):
        if "FILM" in gs.models:
            print("Cleaned FILM")
            del gs.models["FILM"]

    def initInnerClasses(self):
        self.content = FILMWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.output_socket_name = ["EXEC", "EXEC/F", "IMAGE"]
        self.input_socket_name = ["EXEC", "IMAGE1", "IMAGE2"]

        self.grNode.height = 220

    @QtCore.Slot()
    def evalImplementation_thread(self):
        return_frames = []
        if "FILM" not in gs.models:
            gs.models["FILM"] = FilmModel()

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
            image1 = pixmap_to_pil_image(pixmap1[0])
            image2 = pixmap_to_pil_image(pixmap2[0])
            np_image1 = np.array(image1)
            np_image2 = np.array(image2)
            frames = gs.models["FILM"].inference(np_image1, np_image2, inter_frames=25)
            print(f"FILM NODE:  {len(frames)}")
            for frame in range(len(frames) - 2):
                image = Image.fromarray(frame)
                pixmap = pil_image_to_pixmap(image)
                return_frames.append(pixmap)
        elif pixmap1 != None:
            for pixmap in pixmap1:
                image = pixmap_to_pil_image(pixmap)
                np_image = np.array(image.convert("RGB"))
                self.FILM_temp.append(np_image)
                if len(self.FILM_temp) == 2:
                    frames = gs.models["FILM"].inference(self.FILM_temp[0], self.FILM_temp[1], inter_frames=self.content.film.value())

                    skip_first, skip_last = False, True
                    if skip_first:
                        frames.pop(0)
                    if skip_last:
                        frames.pop(-1)

                    for frame in frames:
                        image = Image.fromarray(copy.deepcopy(frame))
                        pixmap = pil_image_to_pixmap(image)
                        return_frames.append(pixmap)
                    self.FILM_temp = [self.FILM_temp[1]]
        print(f"FILM NODE: Using only First input, created {len(return_frames) - 2} between frames, returning {len(return_frames)} frames.")
        return return_frames
    @QtCore.Slot(object)
    def onWorkerFinished(self, return_frames):
        self.setOutput(0, return_frames)
        if len(self.getOutputs(1)) > 0:
            self.executeChild(output_index=1)
        self.busy = False

    def iterate_frames(self, frames):
        self.iterating = True
        for frame in frames:
            node = None
            if len(self.getOutputs(1)) > 0:
                node = self.getOutputs(1)[0]
            if node is not None:
                image = Image.fromarray(copy.deepcopy(frame))
                pixmap = pil_image_to_pixmap(image)
                self.setOutput(0, pixmap)
                node.eval()
        self.iterating = False
    def onMarkedDirty(self):
        self.value = None
    def eval(self, index=0):
        self.markDirty()
        self.content.eval_signal.emit()



