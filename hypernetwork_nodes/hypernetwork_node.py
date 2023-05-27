import os
from qtpy import QtCore
from qtpy import QtWidgets

from .load_hypernetwork import load_hypernetwork_patch
from ..ainodes_backend import torch_gc

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget


from ainodes_frontend import singleton as gs
from ..ainodes_backend.sd_optimizations.sd_hijack import valid_optimizations

OP_NODE_HYPERNETWORK = get_next_opcode()


class HypernetworkLoaderWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_widgets()
        self.create_main_layout(grid=1)



    def create_widgets(self):
        hypernetwork_folder = gs.hypernetworks
        hypernetworks = [f for f in os.listdir(hypernetwork_folder) if f.endswith(('.ckpt', '.pt', '.bin', '.pth', '.safetensors'))]
        self.hypernetwork = self.create_combo_box(hypernetworks, "Hypernetwork:")
        self.strength = self.create_double_spin_box("Strength", -10.0, 10.0, 0.1, 1.0, "hypernetwork_strength")
        if hypernetworks == []:
            self.hypernetwork.addItem("Please place a model in models/hypernetworks")
            print(f"TORCH LOADER NODE: No model file found at {os.getcwd()}/{gs.hypernetworks},")
            print(f"TORCH LOADER NODE: please download your favorite ckpt before Evaluating this node.")
        self.button = QtWidgets.QPushButton("Unpatch Model")
        self.create_button_layout([self.button])
        self.force_reload = self.create_check_box("Force Reload")



class CenterExpandingSizePolicy(QtWidgets.QSizePolicy):
    def __init__(self, parent=None):
        super().__init__(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.parent = parent
        self.setHorizontalStretch(0)
        self.setVerticalStretch(0)
        self.setRetainSizeWhenHidden(True)
        self.setHorizontalPolicy(QtWidgets.QSizePolicy.Expanding)
        self.setVerticalPolicy(QtWidgets.QSizePolicy.Expanding)


@register_node(OP_NODE_HYPERNETWORK)
class HypernetworkLoaderNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/torch.png"
    op_code = OP_NODE_HYPERNETWORK
    op_title = "Hypernetwork Loader"
    content_label_objname = "hypernetwork_loader_node"
    category = "Model"
    input_socket_name = ["EXEC"]
    output_socket_name = ["EXEC"]
    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[1])

    def initInnerClasses(self):
        self.content = HypernetworkLoaderWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.width = 340
        self.grNode.height = 240
        self.content.setMinimumHeight(140)
        self.content.setMinimumWidth(340)
        self.content.eval_signal.connect(self.evalImplementation)
        self.content.button.clicked.connect(self.unpatch_model)

    @QtCore.Slot()
    def evalImplementation_thread(self, index=0):
        hypernetwork = self.content.hypernetwork.currentText()
        if hypernetwork in gs.loaded_hypernetworks:
            gs.models["sd"].unpatch_model()
            gs.loaded_hypernetworks.clear()
        if hypernetwork not in gs.loaded_hypernetworks:
            path = os.path.join(gs.hypernetworks, hypernetwork)
            if os.path.isfile(path):
                patch = load_hypernetwork_patch(path, self.content.strength.value())
                if patch is not None:
                    patch.to("cuda")
                    #m = gs.models["sd"].clone()
                    gs.models["sd"].set_model_attn1_patch(patch)
                    gs.models["sd"].set_model_attn2_patch(patch)
                    gs.loaded_hypernetworks.append(hypernetwork)
                #gs.models["sd"] = m
        return True

    def unpatch_model(self):
        m = gs.models["sd"].clone()
        m.unpatch_model()
        gs.models["sd"] = m

    @QtCore.Slot(object)
    def onWorkerFinished(self, result):
        super().onWorkerFinished(None)
        self.executeChild(0)

    def onInputChanged(self, socket=None):
        pass




