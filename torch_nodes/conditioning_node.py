import threading
import time

from qtpy import QtWidgets, QtCore

from ..torch_nodes.torch_loader import TorchLoaderNode

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
    category = "sampling"

    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[3,1])
        self.content.eval_signal.connect(self.evalImplementation)
        # Create a worker object
    def initInnerClasses(self):
        self.content = ConditioningWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 256
        self.grNode.width = 320
        self.content.setMinimumHeight(200)
        self.content.setMinimumWidth(320)
        self.busy = False
        self.content.button.clicked.connect(self.evalImplementation)
        self.input_socket_name = ["EXEC"]
        self.output_socket_name = ["EXEC", "COND"]

    def evalImplementation_thread(self, index=0):
        try:
            self.markDirty(True)
            print(f"CONDITIONING NODE: Applying conditioning with prompt: {self.content.prompt.toPlainText()}")
            result = self.get_conditioning()
            result = [result]
            self.setOutput(0, result)
            self.markDirty(False)
            self.markInvalid(False)
            self.busy = False
            return result
        except:
            self.busy = False
            return None
    def eval(self, index=0):
        self.markDirty(True)
        self.content.eval_signal.emit()
    def onMarkedDirty(self):
        self.value = None
    def get_conditioning(self, progress_callback=None):
        prompt = self.content.prompt.toPlainText()
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
        self.setOutput(0, result)
        self.markDirty(False)
        self.markInvalid(False)
        self.busy = False
        if len(self.getOutputs(1)) > 0:
            node = self.getOutputs(1)[0]
            node.eval()
        return True
    def onInputChanged(self, socket=None):
        pass

    def exec(self):
        self.markDirty(True)
        self.markInvalid(True)
        self.value = None
        self.eval(0)

SCHEDULERS = ["karras", "normal", "simple", "ddim_uniform"]
SAMPLERS = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
            "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde",
            "dpmpp_2m", "ddim", "uni_pc", "uni_pc_bh2"]
