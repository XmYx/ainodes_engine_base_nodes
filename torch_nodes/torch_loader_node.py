import os
from qtpy import QtCore, QtGui
from qtpy import QtWidgets

from ..ainodes_backend.model_loader import ModelLoader
from ..ainodes_backend import torch_gc

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget


from ainodes_frontend import singleton as gs
from ..ainodes_backend.sd_optimizations.sd_hijack import valid_optimizations

OP_NODE_TORCH_LOADER = get_next_opcode()


class TorchLoaderWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_widgets()
        self.create_main_layout(grid=1)



    def create_widgets(self):
        checkpoint_folder = gs.checkpoints
        checkpoint_files = [f for f in os.listdir(checkpoint_folder) if f.endswith(('.ckpt', '.pt', '.bin', '.pth', '.safetensors'))]
        self.dropdown = self.create_combo_box(checkpoint_files, "Model:")
        if checkpoint_files == []:
            self.dropdown.addItem("Please place a model in models/checkpoints")
            print(f"TORCH LOADER NODE: No model file found at {os.getcwd()}/models/checkpoints,")
            print(f"TORCH LOADER NODE: please download your favorite ckpt before Evaluating this node.")

        config_folder = "models/configs"
        config_files = [f for f in os.listdir(config_folder) if f.endswith((".yaml"))]
        config_files = sorted(config_files, key=str.lower)
        self.config_dropdown = self.create_combo_box(config_files, "Config:")
        self.config_dropdown.setCurrentText("v1-inference_fp16.yaml")

        vae_folder = gs.vae
        vae_files = [f for f in os.listdir(vae_folder) if f.endswith(('.ckpt', '.pt', '.bin', '.pth', '.safetensors'))]
        vae_files = sorted(vae_files, key=str.lower)
        self.vae_dropdown = self.create_combo_box(vae_files, "Vae")
        self.vae_dropdown.addItem("default")
        self.vae_dropdown.setCurrentText("default")
        self.optimization = self.create_combo_box(valid_optimizations, "LDM Optimization")

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


@register_node(OP_NODE_TORCH_LOADER)
class TorchLoaderNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/torch.png"
    op_code = OP_NODE_TORCH_LOADER
    op_title = "Torch Loader"
    content_label_objname = "torch_loader_node"
    category = "Model Loading"
    input_socket_name = ["EXEC"]
    output_socket_name = ["EXEC"]
    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[1])
        self.loader = ModelLoader()

    def initInnerClasses(self):
        self.content = TorchLoaderWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.icon = self.icon
        self.grNode.thumbnail = QtGui.QImage(self.grNode.icon).scaled(64, 64, QtCore.Qt.KeepAspectRatio)

        self.grNode.width = 340
        self.grNode.height = 240
        self.content.setMinimumHeight(140)
        self.content.setMinimumWidth(340)
        self.content.eval_signal.connect(self.evalImplementation)
    def clean_sd(self):
        gs.loaded_loras = []
        if "sd" in gs.models:
            try:
                gs.models["sd"].cpu()
            except:
                pass
            del gs.models["sd"]
            gs.models["sd"] = None
            torch_gc()
        if "inpaint" in gs.models:
            try:
                gs.models["inpaint"].cpu()
            except:
                pass
            del gs.models["inpaint"]
            gs.models["inpaint"] = None
            torch_gc()

    #@QtCore.Slot()
    def evalImplementation_thread(self, index=0):
        self.busy = True
        model_name = self.content.dropdown.currentText()
        config_name = self.content.config_dropdown.currentText()

        #print(gs.loaded_sd, model_name)

        inpaint = True if "inpaint" in model_name else False
        m = "sd_model" if not inpaint else "inpaint"
        if gs.loaded_sd != model_name or self.content.force_reload.isChecked() == True:
            self.clean_sd()
            gs.loaded_hypernetworks.clear()
            self.loader.load_model(model_name, config_name, inpaint, style=self.content.optimization.currentText())
            gs.loaded_sd = model_name
            self.setOutput(0, model_name)
        if self.content.vae_dropdown.currentText() != 'default':
            model = self.content.vae_dropdown.currentText()
            self.loader.load_vae(model)
            gs.loaded_vae = model
        else:
            gs.loaded_vae = 'default'
        if gs.loaded_vae != self.content.vae_dropdown.currentText():
            model = self.content.vae_dropdown.currentText()
            self.loader.load_vae(model)
            gs.loaded_vae = model
        return self.value

    #@QtCore.Slot(object)
    def onWorkerFinished(self, result):
        self.busy = False
        self.markDirty(False)
        self.markInvalid(False)
        self.grNode.setToolTip("")

        #super().onWorkerFinished(None)
        self.executeChild(0)

    def onInputChanged(self, socket=None):
        pass




