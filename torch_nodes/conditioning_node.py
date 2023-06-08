import os

import requests
import torch
from qtpy.QtCore import Qt
from qtpy.QtCore import QObject, Signal
#from qtpy.QtGui import Qt
from qtpy import QtWidgets, QtCore, QtGui

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.base.settings import handle_ainodes_exception
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

from ainodes_frontend import singleton as gs

from qtpy.QtWidgets import QDialog, QListWidget, QCheckBox, QDoubleSpinBox, QVBoxLayout, QDialogButtonBox, \
    QListWidgetItem, QHBoxLayout, QWidget

from custom_nodes.ainodes_engine_base_nodes.ainodes_backend.hash import sha256


class EmbedDialog(QDialog):
    def __init__(self, embed_files, prev_dict):
        super().__init__()
        self.embed_files = embed_files
        self.embed_values = {}

        self.setWindowTitle("Select Embeddings")
        layout = QVBoxLayout()
        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        for file_name in embed_files:
            item_widget = QWidget()
            item_layout = QHBoxLayout(item_widget)

            check_box = QCheckBox()
            check_box.setText(file_name)
            item_layout.addWidget(check_box)

            spin_box = QDoubleSpinBox()
            spin_box.setRange(0.0, 1.0)
            spin_box.setSingleStep(0.1)
            spin_box.setEnabled(True)
            item_layout.addWidget(spin_box)
            for item in prev_dict:
                print(item)
                if item['embed']['filename'] == file_name:
                    check_box.setChecked(True)
                    spin_box.setValue(item['embed']['value'])
            self.d = prev_dict
            check_box.stateChanged.connect(lambda state, box=spin_box: box.setEnabled(state == Qt.Checked))
            self.embed_values[file_name] = [check_box, spin_box]

            list_widget_item = QListWidgetItem()
            list_widget_item.setSizeHint(item_widget.sizeHint())  # Set size hint for proper layout
            self.list_widget.addItem(list_widget_item)
            self.list_widget.setItemWidget(list_widget_item, item_widget)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self.setLayout(layout)

    def accept(self):
        selected_embeds = []
        for file_name, check_box in self.embed_values.items():
            if check_box[0].isChecked():
                selected_embeds.append({"embed":{"filename":file_name,
                                        "value":check_box[1].value(),
                                        "word":""}})
        self.selected_embeds = selected_embeds
        super().accept()
        return selected_embeds
OP_NODE_CONDITIONING = get_next_opcode()
class ConditioningWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_widgets()
        self.create_main_layout()
    def create_widgets(self):
        self.prompt = self.create_text_edit("Prompt")
        self.embed_checkbox = self.create_check_box("Use embeds")
        self.button = QtWidgets.QPushButton("Get Conditioning")
        self.set_embeds = QtWidgets.QPushButton("Embeddings")
        self.create_button_layout([self.button, self.set_embeds])

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

@register_node(OP_NODE_CONDITIONING)
class ConditioningNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/conditioning.png"
    op_code = OP_NODE_CONDITIONING
    op_title = "Conditioning"
    content_label_objname = "cond_node"
    category = "Conditioning"

    def __init__(self, scene):
        super().__init__(scene, inputs=[6,1], outputs=[6,3,1])
        self.content.eval_signal.connect(self.evalImplementation)
        # Create a worker object
    def initInnerClasses(self):
        self.content = ConditioningWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.icon = self.icon
        self.grNode.thumbnail = QtGui.QImage(self.grNode.icon).scaled(64, 64, QtCore.Qt.KeepAspectRatio)

        self.grNode.height = 300
        self.grNode.width = 320
        self.content.setMinimumHeight(200)
        self.content.setMinimumWidth(320)
        self.content.button.clicked.connect(self.evalImplementation)
        self.content.set_embeds.clicked.connect(self.show_embeds)
        self.input_socket_name = ["EXEC"]
        self.output_socket_name = ["EXEC", "COND"]
        self.embed_dict = []
        self.apihandler = APIHandler()
        self.apihandler.response_received.connect(self.handle_response)
        self.string = ""


    def show_embeds(self):

        embed_files = [f for f in os.listdir(gs.embeddings) if f.endswith(('.ckpt', '.pt', '.bin', '.pth', '.safetensors'))]
        if embed_files is not []:
            # The embedding strings returned as: "embedding:<filename without extension>:<weight> where weight is a float between 0.0 and 1.0"
            self.show_embed_dialog(embed_files)


    def show_embed_dialog(self, embed_files):
        dialog = EmbedDialog(embed_files, self.embed_dict)
        if dialog.exec() == QDialog.Accepted:
            selected_embeds = dialog.selected_embeds
            self.embed_dict = selected_embeds

            """for embed in self.embed_dict:
                print("word", embed["embed"]["filename"])
                file = os.path.join(gs.embeddings, embed["embed"]["filename"])
                sha = sha256(file)
                self.apihandler.response_received.connect(self.handle_response)
                self.apihandler.get_response(sha)"""

        else:
            return None


    #@QtCore.Slot()
    def evalImplementation_thread(self, index=0, prompt_override=None):
        try:
            result = None
            data = None
            prompt = self.content.prompt.toPlainText()
            if prompt_override is not None:
                prompt = prompt_override

            string = ""
            for item in self.embed_dict:
                string = f'{string} embedding:{item["embed"]["filename"]}'

            string = "" if not self.content.embed_checkbox.isChecked() else string

            prompt = f"{prompt} {string}" if (self.content.embed_checkbox.isChecked and string != "") else prompt
            if len(self.getInputs(0)) > 0:
                data_node, index = self.getInput(0)
                data = data_node.getOutput(index)
            if data:
                if "prompt" in data:
                    prompt = data["prompt"]
                else:
                    data["prompt"] = prompt
                if "model" in data:
                    if data["model"] == "deepfloyd_1":
                        result = [gs.models["deepfloyd_1"].encode_prompt(prompt)]
                else:
                    if prompt_override is not None:
                        prompt = prompt_override
                    result = [self.get_conditioning(prompt=prompt)]

            else:
                data = {}
                data["prompt"] = prompt
                result = [self.get_conditioning(prompt=prompt)]
            if gs.logging:
                print(f"CONDITIONING NODE: Applying conditioning with prompt: {prompt}")
            return result, data
        except Exception as e:
            done = handle_ainodes_exception()
            if type(e) is KeyError and 'clip' in str(e):
                print("Clip / SD Model not loaded yet, please place and validate a Torch loader node")
            else:
                print(repr(e))
            return None

    #@QtCore.Slot(object)
    def handle_response(self, data):
        if 'files' in data:
            file = data['files'][0]['name']

            for item in self.embed_dict:
                if item['embed']['filename'] == file:
                    item['embed']['word'] = "\n".join(data["trainedWords"])
                    item['embed']['word'] = item['embed']['filename']
        string = ""
        for item in self.embed_dict:
            if item['embed']['word'] != "":
                string = f'{string} embedding:{item["embed"]["word"]}'
        self.string = string

    def get_conditioning(self, prompt="", progress_callback=None):

        """if gs.loaded_models["loaded"] == []:
            for node in self.scene.nodes:
                if isinstance(node, TorchLoaderNode):
                    node.evalImplementation()
                    #print("Node found")"""
        with torch.autocast("cuda"):
            with torch.no_grad():
                #clip = gs.models["clip"].clone()
                c = gs.models["clip"].encode(prompt)

                uc = {}
                #del clip
                return [[c, uc]]

    #@QtCore.Slot(object)
    def onWorkerFinished(self, result):
        self.busy = False
        #super().onWorkerFinished(None)
        if result is not None:
            self.setOutput(1, result[0])
            self.setOutput(0, result[1])
            self.markDirty(False)
            self.markInvalid(False)
            if gs.should_run:
                self.executeChild(2)


SCHEDULERS = ["karras", "normal", "simple", "ddim_uniform"]
SAMPLERS = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
            "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde",
            "dpmpp_2m", "ddim", "uni_pc", "uni_pc_bh2"]