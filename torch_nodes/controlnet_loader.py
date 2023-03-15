import os

from qtpy import QtWidgets

from ..ainodes_backend import torch_gc, load_controlnet

from ainodes_frontend import singleton as gs
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend.node_engine.utils import dumpException

OP_NODE_CONTROLNET_LOADER = get_next_opcode()

class ControlnetLoaderWidget(QDMNodeContentWidget):
    def initUI(self):
        # Create a label to display the image
        # Create the dropdown widget
        self.control_net_name = QtWidgets.QComboBox(self)
        #self.dropdown.currentIndexChanged.connect(self.on_dropdown_changed)
        # Populate the dropdown with .ckpt and .safetensors files in the checkpoints folder
        checkpoint_folder = "models/controlnet"
        checkpoint_files = [f for f in os.listdir(checkpoint_folder) if f.endswith(('.ckpt', '.pt', '.bin', '.pth', ".safetensors"))]
        self.control_net_name.addItems(checkpoint_files)
        # Add the dropdown widget to the layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.control_net_name)
        self.setLayout(layout)
        self.setSizePolicy(CenterExpandingSizePolicy(self))
        self.setLayout(layout)
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
    icon = "icons/in.png"
    op_code = OP_NODE_CONTROLNET_LOADER
    op_title = "ControlNet Loader"
    content_label_objname = "controlnet_loader_node"
    category = "controlnet"

    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[1])

        #self.content.eval_signal.connect(self.eval)
        #self.loader = ModelLoader()

    def initInnerClasses(self):
        self.content = ControlnetLoaderWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.width = 280
        self.grNode.height = 100

    def evalImplementation(self, index=0):
        #self.executeChild()
        model_name = self.content.control_net_name.currentText()
        if gs.models["loaded_controlnet"] != model_name:
            self.markInvalid()
            if model_name != "":
                self.setOutput(0, "controlnet")
                self.load_controlnet()
                gs.models["loaded_controlnet"] = model_name
                self.markDirty(False)
                self.markInvalid(False)
                if len(self.getOutputs(0)) > 0:
                    self.executeChild(output_index=0)
                return self.value
            else:
                if len(self.getOutputs(0)) > 0:
                    self.executeChild(output_index=0)

                return self.value
        else:
            self.markDirty(False)
            self.markInvalid(False)
            self.grNode.setToolTip("")
            if len(self.getOutputs(0)) > 0:
                self.executeChild(output_index=0)

            return self.value
    def eval(self, index=0):
        self.markDirty(True)
        self.evalImplementation(0)


    def load_controlnet(self):
        #if "controlnet" not in gs.models:
        controlnet_dir = "models/controlnet"
        controlnet_path = os.path.join(controlnet_dir, self.content.control_net_name.currentText())
        if "controlnet" in gs.models:
            try:
                gs.models["controlnet"].cpu()
                del gs.models["controlnet"]
                gs.models["controlnet"] = None
            except:
                pass
        load_controlnet(controlnet_path)
        torch_gc()
        #gs.models["controlnet"].control_model.cuda()
        return "controlnet"





