import os
from qtpy import QtCore, QtGui
from qtpy import QtWidgets


from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

from ainodes_frontend import singleton as gs

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


@register_node(OP_NODE_HYPERNETWORK)
class HypernetworkLoaderNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/hyper.png"
    op_code = OP_NODE_HYPERNETWORK
    op_title = "Hypernetwork Loader"
    content_label_objname = "hypernetwork_loader_node"
    category = "Model Loading"
    custom_input_socket_name = ["UNET", "EXEC"]
    custom_output_socket_name = ["UNET","EXEC"]
    def __init__(self, scene):
        super().__init__(scene, inputs=[4,1], outputs=[4,1])

    def initInnerClasses(self):
        self.content = HypernetworkLoaderWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.icon = self.icon
        self.grNode.thumbnail = QtGui.QImage(self.grNode.icon).scaled(64, 64, QtCore.Qt.KeepAspectRatio)
        self.grNode.width = 340
        self.grNode.height = 240
        self.content.setMinimumHeight(140)
        self.content.setMinimumWidth(340)
        self.content.eval_signal.connect(self.evalImplementation)
        self.content.button.clicked.connect(self.unpatch_model)
        from .load_hypernetwork import load_hypernetwork_patch
        self.load_hypernetwork_patch = load_hypernetwork_patch
        self.loaded_hypernetworks = []

    def evalImplementation_thread(self, index=0):

        unet = self.getInputData(0)
        assert unet is not None, "Unet Not connected or loaded, plase make sure to input a valid Stable Diffusion unet."

        hypernetwork = self.content.hypernetwork.currentText()
        path = os.path.join(gs.hypernetworks, hypernetwork)
        if hypernetwork not in self.loaded_hypernetworks:
            patch = self.load_hypernetwork_patch(path, self.content.strength.value())
            if patch is not None:
                unet.set_model_attn1_patch(patch)
                unet.set_model_attn2_patch(patch)
                self.loaded_hypernetworks.append(hypernetwork)
        return unet
        # hypernetwork = self.content.hypernetwork.currentText()
        # if hypernetwork in gs.loaded_hypernetworks:
        #     gs.models["sd"].unpatch_model()
        #     gs.loaded_hypernetworks.clear()
        # if hypernetwork not in gs.loaded_hypernetworks:
        #     path = os.path.join(gs.hypernetworks, hypernetwork)
        #     if os.path.isfile(path):
        #
        #         patch = self.load_hypernetwork_patch(path, self.content.strength.value())
        #         print(patch)
        #         if patch is not None:
        #             patch.to("cuda")
        #             gs.models["sd"].set_model_attn1_patch(patch)
        #             gs.models["sd"].set_model_attn2_patch(patch)
        #             gs.loaded_hypernetworks.append(hypernetwork)
        # return True

    def unpatch_model(self):
        m = gs.models["sd"].clone()
        m.unpatch_model()
        gs.models["sd"] = m

    def onWorkerFinished(self, result):
        self.setOutput(0, result)
        self.busy = False
        self.executeChild(0)

    def onInputChanged(self, socket=None):
        self.loaded_hypernetworks = []
        pass




