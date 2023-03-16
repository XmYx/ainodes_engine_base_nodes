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
from ..ainodes_backend.RIFE.infer_rife import RIFEModel

OP_NODE_RIFE = get_next_opcode()

from ainodes_frontend import singleton as gs

class RIFEWidget(QDMNodeContentWidget):
    def initUI(self):
        # Create a label to display the image
        self.text_label = QtWidgets.QLabel("Image Operator:")

        self.exp = QtWidgets.QSpinBox()
        self.exp.setMinimum(1)
        self.exp.setSingleStep(1)
        self.exp.setMaximum(100)
        self.exp.setValue(5)

        self.ratio = QtWidgets.QDoubleSpinBox()
        self.ratio.setMinimum(0.00)
        self.ratio.setSingleStep(0.01)
        self.ratio.setMaximum(1.00)
        self.ratio.setValue(0.00)

        self.rthreshold = QtWidgets.QDoubleSpinBox()
        self.rthreshold.setMinimum(0.00)
        self.rthreshold.setSingleStep(0.01)
        self.rthreshold.setMaximum(100.00)
        self.rthreshold.setValue(0.02)

        self.rmaxcycles = QtWidgets.QSpinBox()
        self.rmaxcycles.setMinimum(1)
        self.rmaxcycles.setSingleStep(1)
        self.rmaxcycles.setMaximum(4096)
        self.rmaxcycles.setValue(8)



        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.exp)
        layout.addWidget(self.ratio)
        layout.addWidget(self.rthreshold)
        layout.addWidget(self.rmaxcycles)

        self.setLayout(layout)




@register_node(OP_NODE_RIFE)
class RIFENode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/interpolation.png"
    op_code = OP_NODE_RIFE
    op_title = "RIFE"
    content_label_objname = "rife_node"
    category = "video"


    def __init__(self, scene):
        super().__init__(scene, inputs=[5,5,1], outputs=[5,1,1])
        self.painter = QtGui.QPainter()

        self.rife_temp = []

        if "rife" not in gs.models:
            gs.models["rife"] = RIFEModel()

    def initInnerClasses(self):
        self.content = RIFEWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.output_socket_name = ["EXEC", "EXEC/F", "IMAGE"]
        self.input_socket_name = ["EXEC", "IMAGE1", "IMAGE2"]

        self.grNode.height = 220
    @QtCore.Slot(int)
    def evalImplementation(self, index=0):
        exp = self.content.exp.value()
        ratio = self.content.ratio.value()
        if ratio == 0.0:
            ratio = None
        rthreshold = self.content.rthreshold.value()
        rmaxcycles = self.content.rmaxcycles.value()

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
            frames = gs.models["rife"].infer(image1=np_image1, image2=np_image2, exp=exp, ratio=ratio, rthreshold=rthreshold, rmaxcycles=rmaxcycles)
            print(f"RIFE NODE:  {len(frames)}")
            for frame in frames:
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
                self.rife_temp.append(np_image)

                if len(self.rife_temp) == 2:
                    frames = gs.models["rife"].infer(image1=self.rife_temp[0], image2=self.rife_temp[1], exp=exp, ratio=ratio, rthreshold=rthreshold, rmaxcycles=rmaxcycles)
                    print(f"RIFE NODE:  {len(frames)}")
                    for frame in frames:
                        image = Image.fromarray(frame)
                        pixmap = pil_image_to_pixmap(image)
                        self.setOutput(0, pixmap)
                        if len(self.getOutputs(1)) > 0:
                            self.executeChild(output_index=1)
                        time.sleep(0.05)

                    self.rife_temp = [self.rife_temp[1]]

                #self.setOutput(0, pixmap2)
                print(f"RIFE NODE: Using only First input")
            except:
                if len(self.getOutputs(2)) > 0:
                    self.executeChild(output_index=2)

                pass
        elif pixmap1 != None:
            try:
                self.setOutput(0, pixmap1)
                print(f"RIFE NODE: Using only Second input - Passthrough")
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
