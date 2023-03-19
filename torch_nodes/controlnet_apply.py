import time
import numpy as np
import torch
from qtpy import QtWidgets, QtCore

from ..ainodes_backend import torch_gc, pixmap_to_pil_image

from ainodes_frontend import singleton
gs = singleton.Singleton.instance()
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
    category = "controlnet"

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
    def evalImplementation(self, index=0):
        self.markDirty(True)
        self.markInvalid(True)
        self.busy = False
        if self.value is None:
            result = self.apply_control_net()
            self.setOutput(0, result)
            self.busy = False
            if len(self.getOutputs(1)) > 0:
                self.executeChild(1)

            #self.scene.threadpool.start(self.worker)
            return None
        else:
            self.markDirty(False)
            self.markInvalid(False)
            return self.value

    def onMarkedDirty(self):
        self.value = None
    def apply_control_net(self, progress_callback=None):
        if self.content.control_net_selector.currentText() == 'controlnet':
            print(f"CONTROLNET APPLY NODE: Applying {gs.models['loaded_controlnet']}")
        start_time = time.time()
        try:
            cond_node, index = self.getInput(1)
            conditioning = cond_node.getOutput(index)
        except:
            conditioning = None
        try:
            latent_node, index = self.getInput(0)
            image = latent_node.getOutput(index)
        except:
            image = None
        cnet_string = self.content.control_net_selector.currentText()
        image = pixmap_to_pil_image(image)
        image = image.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        c = []
        control_hint = image.movedim(-1,1)
        for t in conditioning:
            n = [t[0], t[1].copy()]
            c_net = gs.models[cnet_string]
            c_net.set_cond_hint(control_hint, self.content.strength.value())
            if 'control' in t[1]:
                c_net.set_previous_controlnet(t[1]['control'])
            n[1]['control'] = c_net
            c.append(n)
        #self.value = c
        self.setOutput(0, c)
        end_time = time.time()
        time_diff_ms = (end_time - start_time) * 1000
        conditioning = None
        control_hint = None
        torch_gc()
        print("APPLIED")
        return c
    @QtCore.Slot(object)
    def onWorkerFinished(self, result):
        # Update the node value and mark it as dirty
        self.markDirty(False)
        self.markInvalid(False)
        self.setOutput(0, result)
        self.busy = False
        if len(self.getOutputs(1)) > 0:
            self.executeChild(1)
        return

    def onInputChanged(self, socket=None):
        pass
