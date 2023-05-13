import copy
import threading
import time
import numpy as np
import torch
from qtpy import QtWidgets, QtCore

from ..ainodes_backend import torch_gc, pixmap_to_pil_image

from ainodes_frontend import singleton as gs
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend.node_engine.utils import dumpException
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode

OP_NODE_CN_APPLY = get_next_opcode()
class CNApplyWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_widgets()
        self.create_main_layout()
        self.main_layout.setContentsMargins(15, 15, 15, 25)
    def create_widgets(self):
        self.strength = self.create_double_spin_box("Strength", 0.01, 100.00, 0.01, 1.00)
        self.control_net_selector = self.create_combo_box(["controlnet", "t2i"], "Control Style")
        self.button = QtWidgets.QPushButton("Run")
        self.create_button_layout([self.button])

@register_node(OP_NODE_CN_APPLY)
class CNApplyNode(AiNode):
    icon = "ainodes_frontend/icons/in.png"
    op_code = OP_NODE_CN_APPLY
    op_title = "Apply ControlNet"
    content_label_objname = "CN_apply_node"
    category = "ControlNet"

    def __init__(self, scene):
        super().__init__(scene, inputs=[5,3,1], outputs=[3,1])

        self.content.button.clicked.connect(self.evalImplementation)
        self.busy = False
        # Create a worker object
    def initInnerClasses(self):
        self.content = CNApplyWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 340
        self.grNode.width = 256
        self.content.setMinimumWidth(256)
        self.content.setMinimumHeight(256)
        self.content.eval_signal.connect(self.evalImplementation)
    @QtCore.Slot()
    def evalImplementation_thread(self, index=0):
        cond_node, index = self.getInput(1)
        conditioning_list = cond_node.getOutput(index)
        latent_node, index = self.getInput(0)
        image_list = latent_node.getOutput(index)
        self.markDirty(True)
        self.markInvalid(True)
        return_list = []
        if len(conditioning_list) == 1:
            for image in image_list:
                result = self.add_control_image(conditioning_list[0], image)
                return_list.append(result)
        elif len(conditioning_list) == len(image_list):
            x = 0
            for image in image_list:
                result = self.add_control_image(conditioning_list[x], image)
                return_list.append(result)
        return return_list

    def onMarkedDirty(self):
        self.value = None
    def add_control_image(self, conditioning, image, progress_callback=None):
        image = pixmap_to_pil_image(image)
        image = image.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        c = []
        control_hint = image.movedim(-1,1)
        for t in conditioning:
            n = [t[0], t[1].copy()]
            n[1]['control_hint'] = control_hint
            n[1]['control_strength'] = self.content.strength.value()
            c.append(n)
        return c
    @QtCore.Slot(object)
    def onWorkerFinished(self, result):
        # Update the node value and mark it as dirty
        self.markDirty(False)
        self.markInvalid(False)
        self.busy = False
        self.setOutput(0, result)
        if len(self.getOutputs(1)) > 0:
            self.executeChild(1)
        return
    def onInputChanged(self, socket=None):
        pass
