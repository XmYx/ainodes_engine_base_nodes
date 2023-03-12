import os

from qtpy import QtWidgets

from ..ainodes_backend import torch_gc, load_controlnet

from ainodes_frontend import singleton as gs
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import CalcNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend.node_engine.utils import dumpException
from ..ainodes_backend.t2i import load_t2i_adapter

OP_NODE_T2I_LOADER = get_next_opcode()

class T2ILoaderWidget(QDMNodeContentWidget):
    def initUI(self):
        # Create a label to display the image
        # Create the dropdown widget
        self.t2i = QtWidgets.QComboBox(self)
        #self.dropdown.currentIndexChanged.connect(self.on_dropdown_changed)
        # Populate the dropdown with .ckpt and .safetensors files in the checkpoints folder
        checkpoint_folder = "models/t2i_adapter"
        checkpoint_files = [f for f in os.listdir(checkpoint_folder) if f.endswith(('.ckpt', '.pt', '.bin', '.pth', ".safetensors"))]
        self.t2i.addItems(checkpoint_files)
        # Add the dropdown widget to the layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.t2i)
        self.setLayout(layout)
        self.setSizePolicy(CenterExpandingSizePolicy(self))
        self.setLayout(layout)
        if "loaded_t2i" not in gs.models:
            gs.models["loaded_t2i"] = None

    def serialize(self):
        res = super().serialize()
        res['t2i'] = self.t2i.currentText()
        return res

    def deserialize(self, data, hashmap={}):
        res = super().deserialize(data, hashmap)
        try:
            #value = data['value']
            self.t2i.setCurrentText(data['t2i'])
            return True & res
        except Exception as e:
            dumpException(e)
        return res
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
class T2ILoaderNode(CalcNode):
    icon = "icons/in.png"
    op_code = OP_NODE_T2I_LOADER
    op_title = "T2I Loader"
    content_label_objname = "t2i_loader_node"
    category = "controlnet"

    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[1])

        #self.content.eval_signal.connect(self.eval)
        #self.loader = ModelLoader()

    def initInnerClasses(self):
        self.content = T2ILoaderWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.width = 280
        self.grNode.height = 100

    def evalImplementation(self, index=0):
        #self.executeChild()
        model_name = self.content.t2i.currentText()
        if gs.models["loaded_t2i"] != model_name:
            self.markInvalid()
            if model_name != "":
                self.setOutput(0, "t2i")
                self.load_t2i()
                gs.models["loaded_t2i"] = model_name
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


    def load_t2i(self):
        #if "controlnet" not in gs.models:
        t2i_dir = "models/t2i_adapter"
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





