import os
from qtpy import QtWidgets, QtCore

from ..ainodes_backend.lora_loader import load_lora_for_models

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend.node_engine.utils import dumpException
from ainodes_frontend import singleton as gs

OP_NODE_LORA_LOADER = get_next_opcode()
class LoraLoaderWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_widgets()
        self.create_main_layout()
    def create_widgets(self):
        lora_folder = gs.loras
        lora_files = [f for f in os.listdir(lora_folder) if f.endswith(('.safetensors', '.ckpt', '.pt', '.bin', '.pth'))]
        if lora_files == []:
            self.dropdown.addItem("Please place a lora in models/loras")
            print(f"LORA LOADER NODE: No model file found at {os.getcwd()}/models/loras,")
            print(f"LORA LOADER NODE: please download your favorite ckpt before Evaluating this node.")
        self.dropdown = self.create_combo_box(lora_files, "Lora")

        self.force_load = self.create_check_box("Force Load")
        self.model_weight = self.create_double_spin_box("Model Weight", 0.0, 1.0, 0.1, 1.0)
        self.clip_weight = self.create_double_spin_box("Clip Weight", 0.0, 1.0, 0.1, 1.0)

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

@register_node(OP_NODE_LORA_LOADER)
class LoraLoaderNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/lora.png"
    op_code = OP_NODE_LORA_LOADER
    op_title = "Lora Loader"
    content_label_objname = "lora_loader_node"
    category = "Model"
    input_socket_name = ["EXEC"]
    output_socket_name = ["EXEC"]
    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[1])
        #self.loader = ModelLoader()

    def initInnerClasses(self):
        self.content = LoraLoaderWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.width = 340
        self.grNode.height = 200
        self.content.setMinimumWidth(320)
        self.content.eval_signal.connect(self.evalImplementation)
        self.current_lora = ""

    def evalImplementation_thread(self, index=0):
        file = self.content.dropdown.currentText()

        if gs.loaded_loras == []:
            self.current_lora = ""
        if self.current_lora != file or self.content.force_load.isChecked() == True:
            if file not in gs.loaded_loras or self.content.force_load.isChecked() == True:
                self.load_lora_to_ckpt(file)
                gs.loaded_loras.append(file)
                self.current_lora = file
        return self.value


    @QtCore.Slot(object)
    def onWorkerFinished(self, result):
        super().onWorkerFinished(None)


        if len(self.getOutputs(0)) > 0:
            self.executeChild(output_index=0)

    def onInputChanged(self, socket=None):
        pass

    def load_lora_to_ckpt(self, lora_name):
        lora_path = os.path.join(gs.loras, lora_name)
        strength_model = self.content.model_weight.value()
        strength_clip = self.content.clip_weight.value()
        load_lora_for_models(lora_path, strength_model, strength_clip)



