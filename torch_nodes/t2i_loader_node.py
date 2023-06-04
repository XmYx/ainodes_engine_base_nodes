import os

from qtpy import QtWidgets, QtCore

from ..ainodes_backend import torch_gc

from ainodes_frontend import singleton as gs
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ..ainodes_backend.t2i import load_t2i_adapter

OP_NODE_T2I_LOADER = get_next_opcode()

class T2ILoaderWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_widgets()
        self.create_main_layout()
    def create_widgets(self):
        checkpoint_folder = gs.t2i_adapter
        checkpoint_files = [f for f in os.listdir(checkpoint_folder) if f.endswith(('.ckpt', '.pt', '.bin', '.pth', ".safetensors"))]
        self.t2i = self.create_combo_box(checkpoint_files, "t2i")
        if "loaded_t2i" not in gs.models:
            gs.models["loaded_t2i"] = None

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

@register_node(OP_NODE_T2I_LOADER)
class T2ILoaderNode(AiNode):
    icon = "ainodes_frontend/icons/in.png"
    op_code = OP_NODE_T2I_LOADER
    op_title = "T2I Loader"
    content_label_objname = "t2i_loader_node"
    category = "Model"

    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[1])

    def initInnerClasses(self):
        self.content = T2ILoaderWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.width = 280
        self.grNode.height = 100

    def evalImplementation_thread(self, index=0):
        #self.executeChild()
        model_name = self.content.t2i.currentText()
        if gs.models["loaded_t2i"] != model_name:
            self.markInvalid()
            if model_name != "":
                self.setOutput(0, "t2i")
                self.load_t2i()
                gs.models["loaded_t2i"] = model_name
        else:
            return self.value

    #@QtCore.Slot(object)
    def onWorkerFinished(self, result):
        super().onWorkerFinished(None)
        self.markDirty(False)
        self.markInvalid(False)
        if len(self.getOutputs(0)) > 0:
            self.executeChild(output_index=0)


    def load_t2i(self):
        #if "controlnet" not in gs.models:
        t2i_dir = gs.t2i_adapter
        t2i_path = os.path.join(t2i_dir, self.content.t2i.currentText())
        if "t2i" in gs.models:
            try:
                gs.models["t2i"].cpu()
                del gs.models["t2i"]
                gs.models["t2i"] = None
            except:
                pass
        gs.models["t2i"] = load_t2i_adapter(t2i_path)

        torch_gc()
        return "t2i"





