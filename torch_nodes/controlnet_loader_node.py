import os

from qtpy import QtWidgets, QtCore, QtGui

from ..ainodes_backend import torch_gc, load_controlnet

from ainodes_frontend import singleton as gs
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

OP_NODE_CONTROLNET_LOADER = get_next_opcode()

class ControlnetLoaderWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_widgets()
        self.create_main_layout()

    def create_widgets(self):
        checkpoint_folder = gs.controlnet
        checkpoint_files = [f for f in os.listdir(checkpoint_folder) if f.endswith(('.ckpt', '.pt', '.bin', '.pth', ".safetensors"))]
        self.control_net_name = self.create_combo_box(checkpoint_files, "ControlNet")
        if "loaded_controlnet" not in gs.models:
            gs.models["loaded_controlnet"] = None

class CenterExpandingSizePolicy(QtWidgets.QSizePolicy):
    def __init__(self, parent=None):
        super().__init__(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.parent = parent
        self.setHorizontalStretch(0)
        self.setVerticalStretch(0)
        self.setRetainSizeWhenHidden(True)
        self.setHorizontalPolicy(QtWidgets.QSizePolicy.Expanding)
        self.setVerticalPolicy(QtWidgets.QSizePolicy.Expanding)
        #self.parent.setAlignment(Qt.AlignCenter)

@register_node(OP_NODE_CONTROLNET_LOADER)
class ControlnetLoaderNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/cnet_loader.png"
    op_code = OP_NODE_CONTROLNET_LOADER
    op_title = "ControlNet Loader"
    content_label_objname = "controlnet_loader_node"
    category = "Model Loading"

    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[1])

    def initInnerClasses(self):
        self.content = ControlnetLoaderWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.icon = self.icon
        self.grNode.thumbnail = QtGui.QImage(self.grNode.icon).scaled(64, 64, QtCore.Qt.KeepAspectRatio)

        self.content.control_net_name.currentIndexChanged.connect(self.resize)
        self.grNode.width = 280
        self.grNode.height = 100
        self.content.setMinimumWidth(260)
        self.content.eval_signal.connect(self.evalImplementation)

    def resize(self):
        text = self.content.control_net_name.currentText()
        font_metrics = self.content.control_net_name.fontMetrics()
        text_width = font_metrics.horizontalAdvance(text)
        # Add some extra width for padding and the dropdown arrow
        extra_width = 100
        new_width = text_width + extra_width
        self.grNode.width = new_width + 20 if new_width > 300 else 320
        new_width = new_width if new_width > 280 else 280
        self.content.setMinimumWidth(new_width)
        self.update_all_sockets()

    def evalImplementation_thread(self, index=0):
        model_name = self.content.control_net_name.currentText()
        if gs.models["loaded_controlnet"] != model_name:
            self.markInvalid()
            if model_name != "":
                self.load_controlnet()
                gs.models["loaded_controlnet"] = model_name
                #pass
                return self.value
            else:
                return self.value
        else:
            return self.value

    #@QtCore.Slot(object)
    def onWorkerFinished(self, result):
        self.busy = False
        #super().onWorkerFinished(None)
        self.markDirty(False)
        self.markInvalid(False)
        self.setOutput(0, result)
        if len(self.getOutputs(0)) > 0:
            self.executeChild(output_index=0)

    def load_controlnet(self):
        #if "controlnet" not in gs.models:
        controlnet_dir = gs.controlnet
        controlnet_path = os.path.join(controlnet_dir, self.content.control_net_name.currentText())
        if "controlnet" in gs.models:
            try:
                gs.models["controlnet"].cpu()
                del gs.models["controlnet"]
                gs.models["controlnet"] = None
            except:
                pass
        load_controlnet(controlnet_path)
        return "controlnet"





