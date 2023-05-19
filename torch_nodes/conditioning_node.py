import threading
import time

from qtpy import QtWidgets, QtCore

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend.node_engine.utils import dumpException

from ainodes_frontend import singleton as gs

OP_NODE_CONDITIONING = get_next_opcode()
class ConditioningWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_widgets()
        self.create_main_layout()
    def create_widgets(self):
        self.prompt = self.create_text_edit("Prompt")
        self.button = QtWidgets.QPushButton("Get Conditioning")
        self.create_button_layout([self.button])

@register_node(OP_NODE_CONDITIONING)
class ConditioningNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/cond.png"
    op_code = OP_NODE_CONDITIONING
    op_title = "Conditioning"
    content_label_objname = "cond_node"
    category = "Sampling"

    def __init__(self, scene):
        super().__init__(scene, inputs=[6,1], outputs=[6,3,1])
        self.content.eval_signal.connect(self.evalImplementation)
        # Create a worker object
    def initInnerClasses(self):
        self.content = ConditioningWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 300
        self.grNode.width = 320
        self.content.setMinimumHeight(200)
        self.content.setMinimumWidth(320)
        #pass
        self.content.button.clicked.connect(self.evalImplementation)
        self.input_socket_name = ["EXEC"]
        self.output_socket_name = ["EXEC", "COND"]

    @QtCore.Slot()
    def evalImplementation_thread(self, index=0, prompt_override=None):
        try:
            data = None
            prompt = self.content.prompt.toPlainText()
            if prompt_override is not None:
                prompt = prompt_override
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
            print("ERROR:", e)
            #pass
            return None

    def get_conditioning(self, prompt="", progress_callback=None):

        """if gs.loaded_models["loaded"] == []:
            for node in self.scene.nodes:
                if isinstance(node, TorchLoaderNode):
                    node.evalImplementation()
                    #print("Node found")"""


        c = gs.models["clip"].encode(prompt)
        uc = {}
        return [[c, uc]]

    @QtCore.Slot(object)
    def onWorkerFinished(self, result):
        super().onWorkerFinished(None)

        self.setOutput(1, result[0])
        self.setOutput(0, result[1])
        self.markDirty(False)
        self.markInvalid(False)
        #pass
        if len(self.getOutputs(2)) > 0:
            self.executeChild(2)

    def onInputChanged(self, socket=None):
        pass


SCHEDULERS = ["karras", "normal", "simple", "ddim_uniform"]
SAMPLERS = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
            "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde",
            "dpmpp_2m", "ddim", "uni_pc", "uni_pc_bh2"]