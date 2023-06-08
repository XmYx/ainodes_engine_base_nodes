import os

import requests
from qtpy.QtCore import QObject, Signal
from qtpy import QtWidgets, QtCore

from ..ainodes_backend.hash import sha256
from ..ainodes_backend.lora_loader import load_lora_for_models

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs

OP_NODE_LORA_LOADER = get_next_opcode()
class LoraLoaderWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_widgets()
        self.create_main_layout(grid=1)
    def create_widgets(self):
        lora_folder = gs.loras
        lora_files = [f for f in os.listdir(lora_folder) if f.endswith(('.safetensors', '.ckpt', '.pt', '.bin', '.pth'))]
        if lora_files == []:
            self.dropdown.addItem("Please place a lora in models/loras")
            print(f"LORA LOADER NODE: No model file found at {os.getcwd()}/models/loras,")
            print(f"LORA LOADER NODE: please download your favorite ckpt before Evaluating this node.")
        self.dropdown = self.create_combo_box(lora_files, "Lora")

        self.force_load = self.create_check_box("Force Load")
        self.model_weight = self.create_double_spin_box("Model Weight", 0.0, 10.0, 0.1, 1.0)
        self.clip_weight = self.create_double_spin_box("Clip Weight", 0.0, 10.0, 0.1, 1.0)

        self.help_prompt = self.create_label("Trained Words:")

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


class APIHandler(QObject):
    response_received = Signal(dict)

    def get_response(self, hash_value):
        url = f"https://civitai.com/api/v1/model-versions/by-hash/{hash_value}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            self.response_received.emit(data)
        else:
            # Handle error
            self.response_received.emit({})


@register_node(OP_NODE_LORA_LOADER)
class LoraLoaderNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/lora.png"
    op_code = OP_NODE_LORA_LOADER
    op_title = "Lora Loader"
    content_label_objname = "lora_loader_node"
    category = "Model Loading"
    input_socket_name = ["EXEC"]
    output_socket_name = ["EXEC"]
    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[1])
        #self.loader = ModelLoader()

    def initInnerClasses(self):
        self.content = LoraLoaderWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.icon = self.icon
        self.grNode.thumbnail = QtGui.QImage(self.grNode.icon).scaled(64, 64, QtCore.Qt.KeepAspectRatio)

        self.grNode.width = 340
        self.grNode.height = 300
        self.content.setMinimumWidth(320)
        self.content.eval_signal.connect(self.evalImplementation)
        self.current_lora = ""
        self.apihandler = APIHandler()

    def evalImplementation_thread(self, index=0):
        file = self.content.dropdown.currentText()

        sha = sha256(os.path.join(gs.loras, file))

        print("SHA", sha)

        self.apihandler.response_received.connect(self.handle_response)
        self.apihandler.get_response(sha)

        force = None if self.content.force_load.isChecked() == False else True

        strength_model = self.content.model_weight.value()
        strength_clip = self.content.clip_weight.value()

        data = {"m_w": strength_model,
                "m_C": strength_clip
                }

        if self.values != data or self.current_lora != file:
            print("LOADING LORA")
            if not force:
                gs.models["sd"].unpatch_model()
                gs.models["clip"].patcher.unpatch_model()
            self.load_lora_to_ckpt(file)
            self.current_lora = file
            self.values = data


        """if gs.loaded_loras == []:
            self.current_lora = ""
        if self.current_lora != file or force:
            if file not in gs.loaded_loras or force:
                self.load_lora_to_ckpt(file)
                if file not in gs.loaded_loras:
                    gs.loaded_loras.append(file)
                self.current_lora = file"""
        return self.value
    #@QtCore.Slot(object)
    def handle_response(self, data):
        # Process the received data
        if "trainedWords" in data:

            words = "\n".join(data["trainedWords"])
            self.content.help_prompt.setText(f"Trained Words:\n{words}")

    #@QtCore.Slot(object)
    def onWorkerFinished(self, result):
        self.busy = False
        #super().onWorkerFinished(None)


        if len(self.getOutputs(0)) > 0:
            self.executeChild(output_index=0)

    def onInputChanged(self, socket=None):
        pass

    def load_lora_to_ckpt(self, lora_name):
        lora_path = os.path.join(gs.loras, lora_name)
        strength_model = self.content.model_weight.value()
        strength_clip = self.content.clip_weight.value()
        load_lora_for_models(lora_path, strength_model, strength_clip)
